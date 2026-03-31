"""Whole-pattern spectral fitting indexer for LIBS.

This is the only supported indexer in alibz. The legacy
``alibz.peaky_indexer`` and ``alibz.peaky_indexer_v2`` modules are deprecated
compatibility shims that route callers here.

Current status: experimental research prototype. The API and solver behaviour
are still under active development.

Treats the observed peak table as a linear combination of element
spectra and solves for concentrations, temperature, electron density,
and broadening parameters simultaneously.

Architecture:
    Outer loop (Bayesian optimisation over T, nₑ, σ, γ)
        └─ Inner loop (NNLS for non-negative concentrations)
"""

import numpy as np
import scipy.sparse
from scipy.optimize import nnls
from scipy.special import voigt_profile as voigt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from alibz.utils.database import Database
from alibz.utils.constants import BOLTZMANN
from alibz.utils.voigt import voigt_width as _voigt_width
from alibz.utils.sahaboltzmann import SahaBoltzmann


# ---------------------------------------------------------------------------
# Database column indices
# ---------------------------------------------------------------------------

_COL_ION = 0
_COL_WAVELENGTH = 1
_COL_GA = 3
_COL_EI = 4
_COL_EK = 5
_COL_GI = 12
_COL_GK = 13


class PhysicsComputationError(RuntimeError):
    """Raised when the physics layer required by the v3 indexer fails."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PeakVector:
    """Feature vector for one observed peak."""
    idx: int
    wavelength: float
    amplitude: float
    sigma: float
    gamma: float
    fwhm: float
    lorentzian_fraction: float
    pc_scores: Optional[np.ndarray] = None


@dataclass
class Species:
    """An (element, ion) pair."""
    element: str
    ion: int
    Z: int                      # atomic number
    abundance: float            # crustal abundance
    line_start: int             # start index in LineTable arrays
    line_end: int               # end index (exclusive)

    @property
    def n_lines(self):
        return self.line_end - self.line_start


@dataclass
class FitResult:
    """Output of the v3 fitting pipeline."""
    temperature: float
    ne: float
    sigma: float
    gamma: float
    species: List[Species]
    concentrations: np.ndarray
    predicted: np.ndarray       # predicted amplitude per peak
    observed: np.ndarray        # observed amplitude per peak
    residuals: np.ndarray
    cost: float
    r_squared: float
    peak_assignments: List[dict]
    unexplained_peaks: List[int]
    convergence_info: dict


# ---------------------------------------------------------------------------
# LineTable: vectorised database subset
# ---------------------------------------------------------------------------

class LineTable:
    """Pre-filtered, contiguous-array line database for fast forward model.

    All arrays are aligned by index.  Lines are grouped by species
    (element, ion) for efficient slicing.
    """

    def __init__(
        self,
        db: Database,
        sb: SahaBoltzmann,
        wl_range: Tuple[float, float],
        max_ion_stage: int = 2,
        min_gA: float = 100.0,
    ):
        self.db = db
        self.sb = sb

        wl_min, wl_max = wl_range
        species_list: List[Species] = []
        all_wl, all_gA, all_Ei, all_Ek = [], [], [], []
        all_species_idx = []

        for el in db.elements:
            if el in db.no_lines:
                continue
            lines = db.lines(el)
            if lines.size == 0:
                continue

            Z = db.elements.index(el) + 1
            abundance = db.elem_abund.get(el, 0.0)
            data = lines[:, [_COL_ION, _COL_WAVELENGTH, _COL_GA,
                             _COL_EI, _COL_EK]].astype(float)

            for ion_stage in range(1, max_ion_stage + 1):
                ion = float(ion_stage)
                mask = ((data[:, 0] == ion) &
                        (data[:, 1] >= wl_min) &
                        (data[:, 1] <= wl_max) &
                        (data[:, 2] >= min_gA))

                if not np.any(mask):
                    continue

                subset = data[mask]
                start = len(all_wl)

                all_wl.extend(subset[:, 1].tolist())
                all_gA.extend(subset[:, 2].tolist())
                all_Ei.extend(subset[:, 3].tolist())
                all_Ek.extend(subset[:, 4].tolist())

                end = len(all_wl)
                sp = Species(
                    element=el, ion=ion_stage, Z=Z,
                    abundance=abundance,
                    line_start=start, line_end=end,
                )
                all_species_idx.extend([len(species_list)] * (end - start))
                species_list.append(sp)

        self.wavelengths = np.array(all_wl, dtype=np.float64)
        self.gA = np.array(all_gA, dtype=np.float64)
        self.Ei = np.array(all_Ei, dtype=np.float64)
        self.Ek = np.array(all_Ek, dtype=np.float64)
        self.species_idx = np.array(all_species_idx, dtype=np.int32)
        self.species = species_list

        self.n_lines = len(self.wavelengths)
        self.n_species = len(species_list)

    # ----- Partition function cache -----

    def compute_partition_functions(self, temperature: float) -> np.ndarray:
        """Compute Z(T) for each species.  Returns shape (n_species,)."""
        T_arr = np.array([temperature])
        Z = np.ones(self.n_species, dtype=np.float64)

        for i, sp in enumerate(self.species):
            try:
                Zi = self.sb.stage_partition(sp.element, T_arr, ion=sp.ion)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise PhysicsComputationError(
                    f"Failed to compute partition function for "
                    f"{sp.element} ion {sp.ion} at T={temperature:.1f} K."
                ) from exc

            Zi = np.asarray(Zi, dtype=float).reshape(-1)
            if Zi.size != 1 or not np.isfinite(Zi[0]) or Zi[0] <= 0:
                raise PhysicsComputationError(
                    f"Invalid partition function for {sp.element} ion {sp.ion} "
                    f"at T={temperature:.1f} K: {Zi!r}"
                )

            Z[i] = max(float(Zi[0]), 1e-30)
        return Z

    def compute_saha_fractions(
        self, temperature: float, log_ne: float,
    ) -> np.ndarray:
        """Compute Saha ionisation fractions.  Returns shape (n_species,)."""
        T_arr = np.array([temperature])
        fractions = np.ones(self.n_species, dtype=np.float64)

        # Group species by element
        el_species: Dict[str, List[int]] = {}
        for i, sp in enumerate(self.species):
            el_species.setdefault(sp.element, []).append(i)

        for el, sp_indices in el_species.items():
            if len(sp_indices) <= 1:
                continue
            try:
                ci, _ = self.sb.ionization_distribution(el, T_arr, log_ne)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise PhysicsComputationError(
                    f"Failed to compute ionization distribution for {el} "
                    f"at T={temperature:.1f} K and log10(ne)={log_ne:.3f}."
                ) from exc

            ci = np.asarray(ci, dtype=float)
            if ci.ndim != 2 or ci.shape[0] != 1:
                raise PhysicsComputationError(
                    f"Invalid ionization distribution shape for {el}: {ci.shape!r}"
                )

            max_required_ion = max(self.species[i].ion for i in sp_indices)
            if ci.shape[1] < max_required_ion:
                raise PhysicsComputationError(
                    f"Ionization distribution for {el} returned {ci.shape[1]} "
                    f"stages but the line table requires stage {max_required_ion}."
                )

            for idx in sp_indices:
                ion = self.species[idx].ion
                frac = float(ci[0, ion - 1])
                if not np.isfinite(frac) or frac < 0:
                    raise PhysicsComputationError(
                        f"Invalid ion fraction for {el} ion {ion}: {frac!r}"
                    )
                fractions[idx] = max(frac, 1e-30)

        return fractions


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PeakyIndexerV3:
    """Whole-pattern spectral fitting indexer.

    Parameters
    ----------
    peak_array : ndarray, shape (n_peaks, 4)
        ``[amplitude, center, sigma, gamma]`` per peak.
    pca_scores : ndarray, optional
        PCA scores, shape (n_peaks, n_components).
    dbpath : str
        Path to NIST database.
    temperature_init : float
        Initial temperature guess (K).
    ne_init : float
        Initial log10(electron density / cm⁻³).
    """

    def __init__(
        self,
        peak_array: np.ndarray,
        pca_scores: Optional[np.ndarray] = None,
        dbpath: str = "db",
        temperature_init: float = 10_000.0,
        ne_init: float = 17.0,
    ):
        self.peak_array = np.asarray(peak_array, dtype=float)
        self.n_peaks = self.peak_array.shape[0]
        self.pca_scores = pca_scores

        self.db = Database(dbpath)
        self.sb = SahaBoltzmann(dbpath)

        self.T_init = temperature_init
        self.ne_init = ne_init

        # Build peak vectors
        self.peaks: List[PeakVector] = []
        for i in range(self.n_peaks):
            amp, wl, sigma, gamma = self.peak_array[i, :4]
            total = abs(sigma) + abs(gamma)
            lor_frac = abs(gamma) / total if total > 0 else 0.5
            pv = PeakVector(
                idx=i, wavelength=wl, amplitude=amp,
                sigma=sigma, gamma=gamma,
                fwhm=float(_voigt_width(sigma, gamma)),
                lorentzian_fraction=lor_frac,
            )
            if pca_scores is not None and i < pca_scores.shape[0]:
                pv.pc_scores = pca_scores[i]
            self.peaks.append(pv)

        self._obs_wl = self.peak_array[:, 1]
        self._obs_amp = self.peak_array[:, 0]

        self.line_table: Optional[LineTable] = None
        self.peak_line_map: Optional[scipy.sparse.csr_matrix] = None

    # =================================================================
    # Step 2: Candidate matrix
    # =================================================================

    def build_candidate_matrix(
        self,
        shift_tolerance: float = 0.1,
        max_ion_stage: int = 2,
        min_gA: float = 100.0,
    ):
        """Build the LineTable and sparse peak-line overlap matrix.

        ``peak_line_map[i, j]`` is the Voigt overlap between observed
        peak *i* and database line *j*, normalised to [0, 1].
        """
        wl_range = (self._obs_wl.min() - 1.0, self._obs_wl.max() + 1.0)
        self.line_table = LineTable(
            self.db, self.sb, wl_range,
            max_ion_stage=max_ion_stage,
            min_gA=min_gA,
        )

        lt = self.line_table
        rows, cols, vals = [], [], []

        for i in range(self.n_peaks):
            obs_wl = self._obs_wl[i]
            sigma = max(self.peaks[i].sigma, 1e-6)
            gamma = max(self.peaks[i].gamma, 1e-6)

            # Vectorised distance to all lines
            diffs = np.abs(lt.wavelengths - obs_wl)
            within = np.where(diffs <= shift_tolerance)[0]

            if len(within) == 0:
                continue

            # Voigt overlap: how much each line contributes to this peak
            peak_voigt = voigt(0, sigma, gamma)
            if peak_voigt < 1e-30:
                continue
            overlaps = voigt(diffs[within], sigma, gamma) / peak_voigt

            for j, ov in zip(within, overlaps):
                if ov > 1e-6:
                    rows.append(i)
                    cols.append(j)
                    vals.append(ov)

        if rows:
            self.peak_line_map = scipy.sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(self.n_peaks, lt.n_lines),
            )
        else:
            self.peak_line_map = scipy.sparse.csr_matrix(
                (self.n_peaks, lt.n_lines))

    # =================================================================
    # Step 3: Forward model
    # =================================================================

    def _line_weights(
        self,
        temperature: float,
        log_ne: float,
    ) -> np.ndarray:
        """Per-line intensity at unit concentration.  Shape (n_lines,)."""
        lt = self.line_table
        kT = BOLTZMANN * temperature

        # Boltzmann factor
        boltz = lt.gA * np.exp(-lt.Ek / kT)

        # Partition functions
        Z = lt.compute_partition_functions(temperature)
        Z_per_line = Z[lt.species_idx]
        boltz /= Z_per_line

        # Saha fractions
        saha = lt.compute_saha_fractions(temperature, log_ne)
        boltz *= saha[lt.species_idx]

        return boltz

    def _build_design_matrix(
        self,
        line_weights: np.ndarray,
    ) -> np.ndarray:
        """Build NNLS design matrix A.  Shape (n_peaks, n_species).

        ``A[i, s]`` = predicted contribution of species *s* to peak *i*
        at unit concentration.
        """
        lt = self.line_table
        n_species = lt.n_species

        # Weighted line intensities
        weighted = self.peak_line_map.multiply(
            line_weights[np.newaxis, :]).tocsr()

        # Sum by species using a species indicator matrix
        # indicator shape (n_lines, n_species): indicator[j, s] = 1 if line j belongs to species s
        indicator = scipy.sparse.csc_matrix(
            (np.ones(lt.n_lines), (np.arange(lt.n_lines), lt.species_idx)),
            shape=(lt.n_lines, n_species),
        )
        A = np.asarray((weighted @ indicator).todense())

        return A

    def forward_model(
        self,
        concentrations: np.ndarray,
        temperature: float,
        log_ne: float,
    ) -> np.ndarray:
        """Predict peak amplitudes.  Returns shape (n_peaks,)."""
        lw = self._line_weights(temperature, log_ne)
        A = self._build_design_matrix(lw)
        return A @ concentrations

    # =================================================================
    # Step 4: Optimisation
    # =================================================================

    def _solve_concentrations(
        self,
        temperature: float,
        log_ne: float,
    ) -> Tuple[np.ndarray, float]:
        """NNLS solve for concentrations at fixed (T, nₑ).

        Columns of the design matrix are normalised so that
        concentrations are on a comparable scale.  This prevents
        species with a single weak line from getting absurd
        concentrations.

        Returns (concentrations, residual_norm).
        """
        lw = self._line_weights(temperature, log_ne)
        A = self._build_design_matrix(lw)

        # Normalise columns: each species' column scaled so its max
        # predicted contribution to any peak is 1.  Concentrations
        # then represent "how many times the max-contribution" rather
        # than raw physical units.
        col_max = np.max(A, axis=0)
        col_max[col_max < 1e-30] = 1.0  # avoid division by zero
        A_norm = A / col_max[np.newaxis, :]

        active = col_max > 1e-30

        if not np.any(active):
            c = np.zeros(A.shape[1])
            return c, float(np.linalg.norm(self._obs_amp))

        c_norm, residual = nnls(A_norm[:, active], self._obs_amp)

        # Un-normalise: c_real = c_norm / col_max
        # (but we keep c_norm for interpretability — it's the
        # "relative concentration" in normalised units)
        c = np.zeros(A.shape[1])
        c[active] = c_norm

        predicted = A_norm[:, active] @ c_norm
        cost = float(np.sum((self._obs_amp - predicted) ** 2))

        # Store the normalised design matrix for later use
        self._last_A_norm = A_norm
        self._last_col_max = col_max

        return c, cost

    def _outer_objective(self, params: np.ndarray) -> float:
        """Objective for Bayesian optimisation over (T, nₑ, σ, γ).

        Rebuilds overlap matrix if broadening changed, then solves NNLS.
        Per-peak sigma/gamma are not modified; only the shared overlap
        matrix is rebuilt with the trial broadening values.
        """
        T, log_ne, sigma, gamma = params

        self._rebuild_overlap(sigma, gamma)

        _, cost = self._solve_concentrations(T, log_ne)
        return cost

    def _rebuild_overlap(self, sigma: float, gamma: float):
        """Rebuild the peak-line overlap matrix with new broadening."""
        lt = self.line_table
        sigma = max(sigma, 1e-6)
        gamma = max(gamma, 1e-6)
        peak_voigt_0 = voigt(0, sigma, gamma)
        if peak_voigt_0 < 1e-30:
            return

        rows, cols, vals = [], [], []
        for i in range(self.n_peaks):
            obs_wl = self._obs_wl[i]
            diffs = np.abs(lt.wavelengths - obs_wl)
            within = np.where(diffs <= 0.2)[0]  # broader search for shared sigma/gamma
            if len(within) == 0:
                continue
            overlaps = voigt(diffs[within], sigma, gamma) / peak_voigt_0
            for j, ov in zip(within, overlaps):
                if ov > 1e-6:
                    rows.append(i)
                    cols.append(j)
                    vals.append(ov)

        if rows:
            self.peak_line_map = scipy.sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(self.n_peaks, lt.n_lines),
            )

    def fit(
        self,
        T_bounds: Tuple[float, float] = (4_000.0, 25_000.0),
        ne_bounds: Tuple[float, float] = (14.0, 19.0),
        sigma_bounds: Tuple[float, float] = (0.01, 0.3),
        gamma_bounds: Tuple[float, float] = (0.01, 0.3),
        n_calls: int = 40,
        verbose: bool = True,
    ) -> FitResult:
        """Run Bayesian optimisation + NNLS fitting.

        Parameters
        ----------
        T_bounds, ne_bounds, sigma_bounds, gamma_bounds : tuple
            Search bounds for outer parameters.
        n_calls : int
            Number of Bayesian optimisation evaluations.
        verbose : bool
            Print progress.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError as exc:
            raise ImportError(
                "PeakyIndexerV3.fit requires scikit-optimize. "
                "Install alibz with `pip install -e .`."
            ) from exc

        dimensions = [
            Real(T_bounds[0], T_bounds[1], name='T'),
            Real(ne_bounds[0], ne_bounds[1], name='log_ne'),
            Real(sigma_bounds[0], sigma_bounds[1], name='sigma'),
            Real(gamma_bounds[0], gamma_bounds[1], name='gamma'),
        ]

        if verbose:
            print(f"Bayesian optimisation: {n_calls} evaluations over "
                  f"T=[{T_bounds[0]:.0f},{T_bounds[1]:.0f}], "
                  f"ne=[{ne_bounds[0]:.0f},{ne_bounds[1]:.0f}]")

        eval_count = [0]

        def objective(params):
            cost = self._outer_objective(np.array(params))
            eval_count[0] += 1
            if verbose and eval_count[0] % 10 == 0:
                print(f"  eval {eval_count[0]}/{n_calls}: "
                      f"T={params[0]:.0f} K, ne={params[1]:.1f}, "
                      f"σ={params[2]:.4f}, γ={params[3]:.4f}, "
                      f"cost={cost:.2e}")
            return cost

        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=max(10, n_calls // 4),
            random_state=42,
            verbose=False,
        )

        # Extract best parameters
        best_T, best_ne, best_sigma, best_gamma = result.x
        if verbose:
            print(f"\nBest: T={best_T:.0f} K, ne={best_ne:.1f}, "
                  f"σ={best_sigma:.4f}, γ={best_gamma:.4f}")

        # Final solve at best parameters (per-peak sigma/gamma are preserved)
        self._rebuild_overlap(best_sigma, best_gamma)
        concentrations, cost = self._solve_concentrations(best_T, best_ne)

        # Compute predicted and residuals using normalised matrix
        A = self._last_A_norm
        predicted = A @ concentrations
        residuals = self._obs_amp - predicted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self._obs_amp - np.mean(self._obs_amp)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Peak assignments: for each peak, which species contributes most
        peak_assignments = []
        for i in range(self.n_peaks):
            contributions = A[i, :] * concentrations
            if np.max(contributions) > 0:
                best_s = int(np.argmax(contributions))
                sp = self.line_table.species[best_s]
                peak_assignments.append({
                    'peak_idx': i,
                    'element': sp.element,
                    'ion': sp.ion,
                    'contribution': float(contributions[best_s]),
                    'total_predicted': float(predicted[i]),
                    'observed': float(self._obs_amp[i]),
                })
            else:
                peak_assignments.append({
                    'peak_idx': i,
                    'element': None,
                    'ion': None,
                    'contribution': 0.0,
                    'total_predicted': 0.0,
                    'observed': float(self._obs_amp[i]),
                })

        # Unexplained peaks: large residuals
        med_res = np.median(np.abs(residuals))
        threshold = max(3 * med_res, 0.05 * np.max(self._obs_amp))
        unexplained = [i for i in range(self.n_peaks)
                       if abs(residuals[i]) > threshold]

        if verbose:
            print(f"R² = {r_squared:.4f}, cost = {cost:.2e}")
            n_active = int(np.sum(concentrations > 0))
            print(f"Active species: {n_active}/{self.line_table.n_species}")
            print(f"Unexplained peaks: {len(unexplained)}/{self.n_peaks}")

            # Top species by concentration
            order = np.argsort(concentrations)[::-1]
            print(f"\nTop species:")
            for s in order[:15]:
                if concentrations[s] <= 0:
                    break
                sp = self.line_table.species[s]
                ion_label = ['', 'I', 'II', 'III']
                ion_str = (ion_label[sp.ion]
                           if sp.ion < len(ion_label)
                           else f"{sp.ion}")
                n_peaks_for = int(np.sum(
                    np.argmax(A * concentrations, axis=1) == s))
                print(f"  {sp.element:4s} {ion_str:3s}: "
                      f"c={concentrations[s]:.4e}  "
                      f"({n_peaks_for} peaks)")

        return FitResult(
            temperature=best_T,
            ne=best_ne,
            sigma=best_sigma,
            gamma=best_gamma,
            species=self.line_table.species,
            concentrations=concentrations,
            predicted=predicted,
            observed=self._obs_amp,
            residuals=residuals,
            cost=cost,
            r_squared=r_squared,
            peak_assignments=peak_assignments,
            unexplained_peaks=unexplained,
            convergence_info={
                'n_evaluations': eval_count[0],
                'best_params': result.x,
                'all_costs': result.func_vals.tolist(),
            },
        )

    # =================================================================
    # Full pipeline
    # =================================================================

    def run(
        self,
        shift_tolerance: float = 0.1,
        max_ion_stage: int = 2,
        min_gA: float = 100.0,
        T_bounds: Tuple[float, float] = (4_000.0, 25_000.0),
        ne_bounds: Tuple[float, float] = (14.0, 19.0),
        sigma_bounds: Tuple[float, float] = (0.01, 0.3),
        gamma_bounds: Tuple[float, float] = (0.01, 0.3),
        n_calls: int = 40,
        verbose: bool = True,
    ) -> FitResult:
        """Execute the full v3 pipeline.

        1. Build candidate matrix
        2. Bayesian optimisation + NNLS
        3. Return FitResult
        """
        if verbose:
            print(f"Building candidate matrix "
                  f"(tol={shift_tolerance}, max_ion={max_ion_stage}, "
                  f"min_gA={min_gA})...")

        self.build_candidate_matrix(
            shift_tolerance=shift_tolerance,
            max_ion_stage=max_ion_stage,
            min_gA=min_gA,
        )

        lt = self.line_table
        nnz = self.peak_line_map.nnz
        if verbose:
            print(f"  {lt.n_lines} lines, {lt.n_species} species, "
                  f"{nnz} peak-line overlaps "
                  f"({nnz / max(self.n_peaks * lt.n_lines, 1):.4%} dense)\n")

        return self.fit(
            T_bounds=T_bounds,
            ne_bounds=ne_bounds,
            sigma_bounds=sigma_bounds,
            gamma_bounds=gamma_bounds,
            n_calls=n_calls,
            verbose=verbose,
        )


PeakyIndexer = PeakyIndexerV3

__all__ = [
    "FitResult",
    "LineTable",
    "PeakVector",
    "PeakyIndexer",
    "PeakyIndexerV3",
    "PhysicsComputationError",
    "Species",
]
