"""Whole-pattern spectral fitting indexer for LIBS.

This is the only supported indexer in alibz.

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
from dataclasses import dataclass
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

    def filter_species(self, keep_mask: np.ndarray) -> None:
        """Drop species and compact the aligned line arrays in-place."""
        keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if keep_mask.shape != (self.n_species,):
            raise ValueError(
                f"keep_mask must have shape ({self.n_species},), "
                f"got {keep_mask.shape!r}"
            )

        if np.all(keep_mask):
            return

        species_list: List[Species] = []
        all_wl, all_gA, all_Ei, all_Ek = [], [], [], []
        all_species_idx = []

        for old_idx, keep in enumerate(keep_mask):
            if not keep:
                continue

            sp = self.species[old_idx]
            span = slice(sp.line_start, sp.line_end)
            start = len(all_wl)

            all_wl.extend(self.wavelengths[span].tolist())
            all_gA.extend(self.gA[span].tolist())
            all_Ei.extend(self.Ei[span].tolist())
            all_Ek.extend(self.Ek[span].tolist())

            end = len(all_wl)
            species_list.append(
                Species(
                    element=sp.element,
                    ion=sp.ion,
                    Z=sp.Z,
                    abundance=sp.abundance,
                    line_start=start,
                    line_end=end,
                )
            )
            all_species_idx.extend([len(species_list) - 1] * (end - start))

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
        self.pseudo_line_map: Optional[scipy.sparse.csr_matrix] = None
        self._shift_tolerance: Optional[float] = None
        self._pseudo_wavelengths = np.empty(0, dtype=float)
        self._pseudo_species_weights = np.empty(0, dtype=float)
        self._pseudo_obs_weight = 0.0
        self._init_relative_intensity = 0.0
        self._pseudo_line_rel_threshold = 0.0
        self._pseudo_max_lines_per_species = 0
        self._evidence_top_k = 0
        self._evidence_strong_line_rel_threshold = 0.0
        self._evidence_presence_threshold = 0.0
        self._evidence_min_coverage = 0.0
        self._evidence_min_supported_lines = 0
        self._evidence_max_missing_lines = 0
        self._evidence_missing_mass_weight = 0.0
        self._evidence_missing_count_weight = 0.0
        self._evidence_min_net = float("-inf")
        self._evidence_max_refits = 0
        self._last_species_evidence = None

    def _empty_result(self, reason: str) -> FitResult:
        """Return a stable result for inputs that cannot be optimised."""
        return FitResult(
            temperature=float(self.T_init),
            ne=float(self.ne_init),
            sigma=0.0,
            gamma=0.0,
            species=[],
            concentrations=np.empty(0, dtype=float),
            predicted=np.empty(0, dtype=float),
            observed=np.empty(0, dtype=float),
            residuals=np.empty(0, dtype=float),
            cost=0.0,
            r_squared=0.0,
            peak_assignments=[],
            unexplained_peaks=[],
            convergence_info={
                'status': reason,
                'n_evaluations': 0,
                'best_params': None,
                'all_costs': [],
            },
        )

    # =================================================================
    # Step 2: Candidate matrix
    # =================================================================

    def build_candidate_matrix(
        self,
        shift_tolerance: float = 0.1,
        max_ion_stage: int = 2,
        min_gA: float = 100.0,
        min_init_relative_intensity: float = 1e-3,
        pseudo_obs_weight: float = 1.0,
        pseudo_line_rel_threshold: float = 0.25,
        pseudo_max_lines_per_species: int = 2,
        evidence_top_k: int = 20,
        evidence_strong_line_rel_threshold: float = 0.1,
        evidence_presence_threshold: float = 0.25,
        evidence_min_coverage: float = 0.2,
        evidence_min_supported_lines: int = 2,
        evidence_max_missing_lines: int = 6,
        evidence_missing_mass_weight: float = 0.25,
        evidence_missing_count_weight: float = 0.1,
        evidence_min_net: float = 0.0,
        evidence_max_refits: int = 2,
    ):
        """Build the LineTable and sparse peak-line overlap matrix.

        ``peak_line_map[i, j]`` is the Voigt overlap between observed
        peak *i* and database line *j*, normalised to [0, 1].

        ``min_init_relative_intensity`` applies a conservative candidate
        prefilter using the initial Saha-Boltzmann estimate at
        ``(self.T_init, self.ne_init)``. Species whose strongest matched
        initial contribution is negligible relative to the strongest
        candidate are removed before optimisation.

        ``pseudo_*`` arguments control zero-target pseudo-observations for
        strong unmatched lines. They add negative evidence for species that
        would otherwise imply visible lines where no observed peak exists.
        When only a subset of unmatched lines is retained, the penalty is
        reweighted by the total unmatched strong-line mass for that species.
        """
        self._shift_tolerance = float(shift_tolerance)
        self._init_relative_intensity = float(max(min_init_relative_intensity, 0.0))
        self._pseudo_obs_weight = float(max(pseudo_obs_weight, 0.0))
        self._pseudo_line_rel_threshold = float(max(pseudo_line_rel_threshold, 0.0))
        self._pseudo_max_lines_per_species = int(max(pseudo_max_lines_per_species, 0))
        self._evidence_top_k = int(max(evidence_top_k, 0))
        self._evidence_strong_line_rel_threshold = float(
            max(evidence_strong_line_rel_threshold, 0.0)
        )
        self._evidence_presence_threshold = float(
            np.clip(evidence_presence_threshold, 0.0, 1.0)
        )
        self._evidence_min_coverage = float(np.clip(evidence_min_coverage, 0.0, 1.0))
        self._evidence_min_supported_lines = int(max(evidence_min_supported_lines, 0))
        self._evidence_max_missing_lines = int(max(evidence_max_missing_lines, 0))
        self._evidence_missing_mass_weight = float(max(evidence_missing_mass_weight, 0.0))
        self._evidence_missing_count_weight = float(max(evidence_missing_count_weight, 0.0))
        self._evidence_min_net = float(evidence_min_net)
        self._evidence_max_refits = int(max(evidence_max_refits, 0))
        wl_range = (self._obs_wl.min() - 1.0, self._obs_wl.max() + 1.0)
        self.line_table = LineTable(
            self.db, self.sb, wl_range,
            max_ion_stage=max_ion_stage,
            min_gA=min_gA,
        )

        self.peak_line_map = self._build_observed_overlap_map()
        self._prefilter_species_by_initial_strength(self._init_relative_intensity)
        self._prefilter_species_by_line_evidence(self.T_init, self.ne_init)
        self._select_pseudo_wavelengths(self.T_init, self.ne_init)

    # =================================================================
    # Step 3: Forward model
    # =================================================================

    def _build_overlap_map(
        self,
        row_wavelengths: np.ndarray,
        sigma: float,
        gamma: float,
    ) -> scipy.sparse.csr_matrix:
        """Return an overlap map for arbitrary wavelength rows."""
        lt = self.line_table
        row_wavelengths = np.asarray(row_wavelengths, dtype=float).reshape(-1)
        n_rows = row_wavelengths.size
        if n_rows == 0 or lt is None or lt.n_lines == 0:
            shape = (n_rows, 0 if lt is None else lt.n_lines)
            return scipy.sparse.csr_matrix(shape)

        sigma = max(float(sigma), 1e-6)
        gamma = max(float(gamma), 1e-6)
        tolerance = (
            float(self._shift_tolerance)
            if self._shift_tolerance is not None
            else 0.2
        )
        peak_voigt_0 = voigt(0, sigma, gamma)
        if peak_voigt_0 < 1e-30:
            return scipy.sparse.csr_matrix((n_rows, lt.n_lines))

        rows, cols, vals = [], [], []
        for i, obs_wl in enumerate(row_wavelengths):
            diffs = np.abs(lt.wavelengths - obs_wl)
            within = np.where(diffs <= tolerance)[0]
            if len(within) == 0:
                continue
            overlaps = voigt(diffs[within], sigma, gamma) / peak_voigt_0
            for j, ov in zip(within, overlaps):
                if ov > 1e-6:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(ov))

        if rows:
            return scipy.sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(n_rows, lt.n_lines),
            )
        return scipy.sparse.csr_matrix((n_rows, lt.n_lines))

    def _build_observed_overlap_map(self) -> scipy.sparse.csr_matrix:
        """Build the overlap map using the fitted width of each observed peak."""
        lt = self.line_table
        if lt is None or lt.n_lines == 0:
            shape = (self.n_peaks, 0 if lt is None else lt.n_lines)
            return scipy.sparse.csr_matrix(shape)

        rows, cols, vals = [], [], []
        tolerance = (
            float(self._shift_tolerance)
            if self._shift_tolerance is not None
            else 0.2
        )

        for i in range(self.n_peaks):
            obs_wl = self._obs_wl[i]
            sigma = max(self.peaks[i].sigma, 1e-6)
            gamma = max(self.peaks[i].gamma, 1e-6)
            peak_voigt = voigt(0, sigma, gamma)
            if peak_voigt < 1e-30:
                continue

            diffs = np.abs(lt.wavelengths - obs_wl)
            within = np.where(diffs <= tolerance)[0]
            if len(within) == 0:
                continue

            overlaps = voigt(diffs[within], sigma, gamma) / peak_voigt
            for j, ov in zip(within, overlaps):
                if ov > 1e-6:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(ov))

        if rows:
            return scipy.sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(self.n_peaks, lt.n_lines),
            )
        return scipy.sparse.csr_matrix((self.n_peaks, lt.n_lines))

    def _median_peak_widths(self) -> Tuple[float, float]:
        """Return robust shared broadening values from the observed peaks."""
        if self.n_peaks == 0:
            return 0.05, 0.05
        sigmas = np.array([max(p.sigma, 1e-6) for p in self.peaks], dtype=float)
        gammas = np.array([max(p.gamma, 1e-6) for p in self.peaks], dtype=float)
        return float(np.median(sigmas)), float(np.median(gammas))

    def _build_design_matrix_from_map(
        self,
        overlap_map: scipy.sparse.csr_matrix,
        line_weights: np.ndarray,
    ) -> np.ndarray:
        """Aggregate line intensities into one column per species."""
        lt = self.line_table
        n_species = lt.n_species
        weighted = overlap_map.multiply(line_weights[np.newaxis, :]).tocsr()
        indicator = scipy.sparse.csc_matrix(
            (np.ones(lt.n_lines), (np.arange(lt.n_lines), lt.species_idx)),
            shape=(lt.n_lines, n_species),
        )
        return np.asarray((weighted @ indicator).todense())

    def _prefilter_species_by_initial_strength(
        self,
        min_relative_intensity: float,
    ) -> None:
        """Drop species whose strongest matched initial line is negligible."""
        lt = self.line_table
        if lt is None or lt.n_species == 0:
            return

        init_weights = self._line_weights(self.T_init, self.ne_init)
        A_init = self._build_design_matrix_from_map(self.peak_line_map, init_weights)
        if A_init.size == 0:
            return

        species_signal = np.max(A_init, axis=0)
        keep_mask = species_signal > 0
        if np.any(keep_mask) and min_relative_intensity > 0:
            max_signal = float(np.max(species_signal[keep_mask]))
            keep_mask &= species_signal >= max_signal * min_relative_intensity
            if not np.any(keep_mask):
                keep_mask[int(np.argmax(species_signal))] = True

        if np.all(keep_mask):
            return

        lt.filter_species(keep_mask)
        self.peak_line_map = self._build_observed_overlap_map()

    def _line_presence(self) -> np.ndarray:
        """Return per-line support from the observed peak table in ``[0, 1]``."""
        lt = self.line_table
        if (
            lt is None
            or lt.n_lines == 0
            or self.peak_line_map is None
            or self.peak_line_map.shape[1] == 0
        ):
            return np.zeros(0 if lt is None else lt.n_lines, dtype=float)

        presence = self.peak_line_map.max(axis=0)
        if scipy.sparse.issparse(presence):
            presence = presence.toarray()
        presence = np.asarray(presence, dtype=float).reshape(-1)
        return np.clip(presence, 0.0, 1.0)

    def _species_line_evidence(self, line_weights: np.ndarray) -> Dict[str, np.ndarray]:
        """Return per-species support and missing-line evidence."""
        lt = self.line_table
        n_species = 0 if lt is None else lt.n_species
        coverage = np.zeros(n_species, dtype=float)
        matched_mass = np.zeros(n_species, dtype=float)
        missing_mass = np.zeros(n_species, dtype=float)
        total_mass = np.zeros(n_species, dtype=float)
        supported_strong_lines = np.zeros(n_species, dtype=np.int32)
        strong_missing_count = np.zeros(n_species, dtype=np.int32)
        strong_line_count = np.zeros(n_species, dtype=np.int32)
        net_evidence = np.zeros(n_species, dtype=float)

        if lt is None or lt.n_species == 0:
            return {
                "coverage": coverage,
                "matched_mass": matched_mass,
                "missing_mass": missing_mass,
                "total_mass": total_mass,
                "supported_strong_lines": supported_strong_lines,
                "strong_missing_count": strong_missing_count,
                "strong_line_count": strong_line_count,
                "net_evidence": net_evidence,
            }

        presence = self._line_presence()
        top_k = self._evidence_top_k
        strong_rel = self._evidence_strong_line_rel_threshold
        presence_threshold = self._evidence_presence_threshold

        for sp_idx, sp in enumerate(lt.species):
            span = slice(sp.line_start, sp.line_end)
            strengths = np.asarray(line_weights[span], dtype=float).reshape(-1)
            if strengths.size == 0:
                continue

            order = np.argsort(strengths)[::-1]
            if top_k > 0:
                order = order[:top_k]
            strengths = strengths[order]
            if strengths.size == 0:
                continue

            max_strength = float(np.max(strengths))
            if max_strength <= 0:
                continue

            relative_strengths = strengths / max_strength
            presence_slice = presence[span][order]
            total = float(np.sum(relative_strengths))
            matched = float(np.sum(relative_strengths * presence_slice))
            missing = max(total - matched, 0.0)

            strong_mask = relative_strengths >= strong_rel
            strong_presence = presence_slice[strong_mask]
            support_count = int(np.sum(strong_presence >= presence_threshold))
            missing_count = int(np.sum(strong_presence < presence_threshold))
            n_strong = int(np.sum(strong_mask))

            total_mass[sp_idx] = total
            matched_mass[sp_idx] = matched
            missing_mass[sp_idx] = missing
            coverage[sp_idx] = matched / total if total > 0 else 0.0
            supported_strong_lines[sp_idx] = support_count
            strong_missing_count[sp_idx] = missing_count
            strong_line_count[sp_idx] = n_strong
            net_evidence[sp_idx] = (
                matched
                - self._evidence_missing_mass_weight * missing
                - self._evidence_missing_count_weight * float(missing_count)
            )

        return {
            "coverage": coverage,
            "matched_mass": matched_mass,
            "missing_mass": missing_mass,
            "total_mass": total_mass,
            "supported_strong_lines": supported_strong_lines,
            "strong_missing_count": strong_missing_count,
            "strong_line_count": strong_line_count,
            "net_evidence": net_evidence,
        }

    def _species_evidence_keep_mask(
        self,
        evidence: Dict[str, np.ndarray],
        require_net: bool,
    ) -> np.ndarray:
        """Return a boolean keep mask from species evidence metrics."""
        total_mass = np.asarray(evidence["total_mass"], dtype=float)
        strong_line_count = np.asarray(evidence["strong_line_count"], dtype=np.int32)
        required_support = np.minimum(self._evidence_min_supported_lines, strong_line_count)
        allowed_missing = np.minimum(self._evidence_max_missing_lines, strong_line_count)

        keep = total_mass > 0
        keep &= np.asarray(evidence["coverage"], dtype=float) >= self._evidence_min_coverage
        keep &= (
            np.asarray(evidence["supported_strong_lines"], dtype=np.int32)
            >= required_support
        )
        keep &= (
            np.asarray(evidence["strong_missing_count"], dtype=np.int32)
            <= allowed_missing
        )
        if require_net:
            keep &= np.asarray(evidence["net_evidence"], dtype=float) >= self._evidence_min_net
        return keep

    def _prefilter_species_by_line_evidence(
        self,
        temperature: float,
        log_ne: float,
    ) -> None:
        """Drop species that fail multi-line support before optimisation."""
        lt = self.line_table
        if lt is None or lt.n_species == 0 or self._evidence_top_k <= 0:
            return

        evidence = self._species_line_evidence(self._line_weights(temperature, log_ne))
        keep_mask = self._species_evidence_keep_mask(evidence, require_net=False)
        if np.all(keep_mask):
            return
        if not np.any(keep_mask):
            return

        lt.filter_species(keep_mask)
        self.peak_line_map = self._build_observed_overlap_map()

    def _select_pseudo_wavelengths(
        self,
        temperature: Optional[float] = None,
        log_ne: Optional[float] = None,
    ) -> None:
        """Select strong unmatched lines as zero-target pseudo-observations."""
        lt = self.line_table
        self._pseudo_wavelengths = np.empty(0, dtype=float)
        self._pseudo_species_weights = np.ones(
            0 if lt is None else lt.n_species,
            dtype=float,
        )
        self.pseudo_line_map = scipy.sparse.csr_matrix(
            (0, 0 if lt is None else lt.n_lines)
        )

        if (
            lt is None
            or lt.n_species == 0
            or self._pseudo_obs_weight <= 0
            or self._pseudo_line_rel_threshold <= 0
            or self._pseudo_max_lines_per_species <= 0
        ):
            return

        if temperature is None:
            temperature = self.T_init
        if log_ne is None:
            log_ne = self.ne_init

        init_weights = self._line_weights(temperature, log_ne)
        tolerance = (
            float(self._shift_tolerance)
            if self._shift_tolerance is not None
            else 0.2
        )
        pseudo_wavelengths: List[float] = []
        pseudo_species_weights = np.ones(lt.n_species, dtype=float)

        for sp_idx, sp in enumerate(lt.species):
            span = slice(sp.line_start, sp.line_end)
            wl = lt.wavelengths[span]
            strengths = init_weights[span]
            if strengths.size == 0:
                continue

            max_strength = float(np.max(strengths))
            if max_strength <= 0:
                continue

            strong_mask = strengths >= max_strength * self._pseudo_line_rel_threshold
            if not np.any(strong_mask):
                continue

            nearest_peak_diff = np.min(
                np.abs(wl[:, None] - self._obs_wl[None, :]),
                axis=1,
            )
            unmatched = nearest_peak_diff > tolerance
            candidate_idx = np.where(strong_mask & unmatched)[0]
            if candidate_idx.size == 0:
                continue

            relative_strengths = strengths[candidate_idx] / max_strength
            total_relative_mass = float(np.sum(relative_strengths))
            order = candidate_idx[np.argsort(strengths[candidate_idx])[::-1]]
            selected = order[: self._pseudo_max_lines_per_species]
            selected_relative_mass = float(
                np.sum(strengths[selected] / max_strength)
            )
            if selected_relative_mass > 0:
                pseudo_species_weights[sp_idx] = np.sqrt(
                    total_relative_mass / selected_relative_mass
                )
            pseudo_wavelengths.extend(wl[selected].tolist())

        self._pseudo_species_weights = pseudo_species_weights

        if not pseudo_wavelengths:
            return

        self._pseudo_wavelengths = np.unique(
            np.round(np.asarray(pseudo_wavelengths, dtype=float), decimals=10)
        )
        sigma, gamma = self._median_peak_widths()
        self.pseudo_line_map = self._build_overlap_map(
            self._pseudo_wavelengths,
            sigma,
            gamma,
        )

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
        return self._build_design_matrix_from_map(self.peak_line_map, line_weights)

    @staticmethod
    def _species_penalty_block(
        active: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        """Return diagonal penalty rows for active species with positive scales."""
        active_scales = np.asarray(scales, dtype=float)[np.asarray(active, dtype=bool)]
        positive = active_scales > 0
        if not np.any(positive):
            return np.empty((0, int(np.sum(active))), dtype=float)

        penalty = np.zeros((int(np.sum(positive)), int(np.sum(active))), dtype=float)
        penalty[np.arange(int(np.sum(positive))), np.flatnonzero(positive)] = active_scales[positive]
        return penalty

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
        if A.size == 0:
            c = np.zeros(self.line_table.n_species, dtype=float)
            self._last_A_norm = A
            self._last_col_max = np.empty(0, dtype=float)
            self._last_species_evidence = None
            return c, float(np.linalg.norm(self._obs_amp))

        raw_col_max = np.max(A, axis=0)
        active = raw_col_max > 1e-30
        col_max = raw_col_max.copy()
        col_max[col_max < 1e-30] = 1.0  # avoid division by zero
        A_norm = A / col_max[np.newaxis, :]

        if not np.any(active):
            c = np.zeros(A.shape[1])
            self._last_A_norm = A_norm
            self._last_col_max = col_max
            self._last_species_evidence = None
            return c, float(np.linalg.norm(self._obs_amp))

        A_aug = A_norm[:, active]
        y_aug = self._obs_amp

        species_evidence = self._species_line_evidence(lw)
        self._last_species_evidence = species_evidence
        missing_mass_rows = self._species_penalty_block(
            active,
            np.sqrt(
                self._evidence_missing_mass_weight
                * np.maximum(species_evidence["missing_mass"], 0.0)
            ),
        )
        if missing_mass_rows.shape[0] > 0:
            A_aug = np.vstack([A_aug, missing_mass_rows])
            y_aug = np.concatenate(
                [y_aug, np.zeros(missing_mass_rows.shape[0], dtype=float)]
            )

        missing_count_rows = self._species_penalty_block(
            active,
            np.sqrt(
                self._evidence_missing_count_weight
                * np.maximum(species_evidence["strong_missing_count"], 0.0)
            ),
        )
        if missing_count_rows.shape[0] > 0:
            A_aug = np.vstack([A_aug, missing_count_rows])
            y_aug = np.concatenate(
                [y_aug, np.zeros(missing_count_rows.shape[0], dtype=float)]
            )

        pseudo_cost = 0.0
        if self._pseudo_obs_weight > 0 and self.pseudo_line_map is not None:
            if self.pseudo_line_map.shape[0] > 0:
                A_pseudo = self._build_design_matrix_from_map(
                    self.pseudo_line_map,
                    lw,
                )
                A_pseudo_norm = A_pseudo / col_max[np.newaxis, :]
                pseudo_species_weights = np.asarray(
                    self._pseudo_species_weights,
                    dtype=float,
                ).reshape(-1)
                if pseudo_species_weights.shape != (A.shape[1],):
                    pseudo_species_weights = np.ones(A.shape[1], dtype=float)
                A_pseudo_norm *= pseudo_species_weights[np.newaxis, :]
                pseudo_scale = np.sqrt(self._pseudo_obs_weight)
                A_aug = np.vstack([A_aug, pseudo_scale * A_pseudo_norm[:, active]])
                y_aug = np.concatenate(
                    [self._obs_amp, np.zeros(A_pseudo_norm.shape[0], dtype=float)]
                )
            else:
                A_pseudo_norm = np.empty((0, A.shape[1]), dtype=float)
        else:
            A_pseudo_norm = np.empty((0, A.shape[1]), dtype=float)

        c_norm, _residual = nnls(A_aug, y_aug)

        # Un-normalise: c_real = c_norm / col_max
        # (but we keep c_norm for interpretability — it's the
        # "relative concentration" in normalised units)
        c = np.zeros(A.shape[1])
        c[active] = c_norm

        predicted = A_norm[:, active] @ c_norm
        cost = float(np.sum((self._obs_amp - predicted) ** 2))
        if A_pseudo_norm.shape[0] > 0:
            pseudo_predicted = A_pseudo_norm[:, active] @ c_norm
            pseudo_cost = float(self._pseudo_obs_weight * np.sum(pseudo_predicted ** 2))
            cost += pseudo_cost

        # Store the normalised design matrix for later use
        self._last_A_norm = A_norm
        self._last_col_max = col_max
        self._last_pseudo_cost = pseudo_cost

        return c, cost

    def _prune_and_refit(
        self,
        temperature: float,
        log_ne: float,
        sigma: float,
        gamma: float,
        concentrations: np.ndarray,
        cost: float,
    ) -> Tuple[np.ndarray, float]:
        """Iteratively drop active species with weak net evidence and refit."""
        if self._evidence_top_k <= 0 or self._evidence_max_refits <= 0:
            return concentrations, cost

        for _ in range(self._evidence_max_refits):
            lw = self._line_weights(temperature, log_ne)
            evidence = self._species_line_evidence(lw)
            active = np.asarray(concentrations, dtype=float) > 1e-12
            bad_active = active & ~self._species_evidence_keep_mask(
                evidence,
                require_net=True,
            )
            if not np.any(bad_active):
                self._last_species_evidence = evidence
                break

            keep_mask = np.ones(self.line_table.n_species, dtype=bool)
            keep_mask[bad_active] = False
            if not np.any(keep_mask):
                break

            self.line_table.filter_species(keep_mask)
            self._select_pseudo_wavelengths(temperature, log_ne)
            self._rebuild_overlap(sigma, gamma)
            concentrations, cost = self._solve_concentrations(temperature, log_ne)

        return concentrations, cost

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
        if lt is None:
            return
        self.peak_line_map = self._build_overlap_map(self._obs_wl, sigma, gamma)
        self.pseudo_line_map = self._build_overlap_map(
            self._pseudo_wavelengths,
            sigma,
            gamma,
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
        if self.n_peaks == 0:
            if verbose:
                print("No peaks supplied; skipping optimisation.")
            return self._empty_result("empty_peak_table")
        if n_calls < 1:
            raise ValueError("n_calls must be at least 1")

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

        n_initial_points = min(n_calls, max(10, n_calls // 4))

        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
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
        concentrations, cost = self._prune_and_refit(
            best_T,
            best_ne,
            best_sigma,
            best_gamma,
            concentrations,
            cost,
        )

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
        min_init_relative_intensity: float = 1e-3,
        pseudo_obs_weight: float = 1.0,
        pseudo_line_rel_threshold: float = 0.25,
        pseudo_max_lines_per_species: int = 2,
        evidence_top_k: int = 20,
        evidence_strong_line_rel_threshold: float = 0.1,
        evidence_presence_threshold: float = 0.25,
        evidence_min_coverage: float = 0.2,
        evidence_min_supported_lines: int = 2,
        evidence_max_missing_lines: int = 6,
        evidence_missing_mass_weight: float = 0.25,
        evidence_missing_count_weight: float = 0.1,
        evidence_min_net: float = 0.0,
        evidence_max_refits: int = 2,
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

        The candidate build stage supports two false-positive suppression
        heuristics:
        - an initial Saha-Boltzmann prefilter that drops negligible species
        - pseudo-observations at strong unmatched line positions
        """
        if self.n_peaks == 0:
            if verbose:
                print("No peaks supplied; returning empty fit result.")
            return self._empty_result("empty_peak_table")

        if verbose:
            print(f"Building candidate matrix "
                  f"(tol={shift_tolerance}, max_ion={max_ion_stage}, "
                  f"min_gA={min_gA})...")

        self.build_candidate_matrix(
            shift_tolerance=shift_tolerance,
            max_ion_stage=max_ion_stage,
            min_gA=min_gA,
            min_init_relative_intensity=min_init_relative_intensity,
            pseudo_obs_weight=pseudo_obs_weight,
            pseudo_line_rel_threshold=pseudo_line_rel_threshold,
            pseudo_max_lines_per_species=pseudo_max_lines_per_species,
            evidence_top_k=evidence_top_k,
            evidence_strong_line_rel_threshold=evidence_strong_line_rel_threshold,
            evidence_presence_threshold=evidence_presence_threshold,
            evidence_min_coverage=evidence_min_coverage,
            evidence_min_supported_lines=evidence_min_supported_lines,
            evidence_max_missing_lines=evidence_max_missing_lines,
            evidence_missing_mass_weight=evidence_missing_mass_weight,
            evidence_missing_count_weight=evidence_missing_count_weight,
            evidence_min_net=evidence_min_net,
            evidence_max_refits=evidence_max_refits,
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
