"""Physics-driven peak indexing pipeline for LIBS spectra (v2).

Implements a five-stage sequential pipeline:

1. Self-absorption quantification from PCA scores
2. Ground-state anchor identification with Boltzmann temperature
3. Temperature estimation from line ratios (TODO)
4. Candidate ranking with lineshape constraints (TODO)
5. Forward model validation (TODO)
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from alibz.utils.database import Database
from alibz.utils.constants import BOLTZMANN
from alibz.utils.voigt import voigt_width as _voigt_width
from alibz.peaky_maker import PeakyMaker


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PeakRecord:
    """Per-peak data accumulated through the pipeline stages."""

    peak_idx: int
    wavelength: float           # fitted center (nm)
    amplitude: float            # fitted amplitude
    sigma: float                # Gaussian width
    gamma: float                # Lorentzian width
    fwhm: float                 # Voigt FWHM (nm)

    # Stage 1: self-absorption
    self_absorption_index: float = 0.0
    is_self_absorbed: bool = False

    # PCA scores (populated if available)
    pc_scores: Optional[np.ndarray] = None

    # Stage 2: anchor identification
    is_anchor: bool = False
    anchor_element: Optional[str] = None
    anchor_ion: Optional[float] = None
    anchor_ref_wavelength: Optional[float] = None
    anchor_gA: Optional[float] = None
    anchor_Ek: Optional[float] = None

    # Stage 4: candidate ranking (future)
    candidates: List[dict] = field(default_factory=list)
    assigned_element: Optional[str] = None
    assigned_ion: Optional[float] = None
    assignment_score: float = 0.0


@dataclass
class BoltzmannResult:
    """Temperature estimate from a Boltzmann plot of one element/ion."""

    element: str
    ion: float
    temperature_K: float
    n_lines: int
    r_squared: float
    slope: float
    intercept: float


# ---------------------------------------------------------------------------
# Database column indices (NIST el_lines92.pickle schema)
# ---------------------------------------------------------------------------

_COL_ION = 0
_COL_WAVELENGTH = 1
_COL_GA = 3
_COL_EI = 4
_COL_EK = 5
_COL_GI = 12
_COL_GK = 13


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PeakyIndexerV2:
    """Physics-driven peak indexing pipeline for LIBS spectra.

    Parameters
    ----------
    peak_array : ndarray, shape (n_peaks, 4)
        Fitted peak parameters ``[amplitude, center, sigma, gamma]``.
    pca_scores : ndarray, shape (n_peaks, n_components), optional
        PCA scores from :class:`~peaky_pca.PeakyPCA`.  Required for
        Stage 1 (self-absorption quantification).
    dbpath : str
        Path to the NIST atomic line database directory.
    temperature : float
        Default plasma temperature in K for ground-state filtering.
    """

    def __init__(
        self,
        peak_array: np.ndarray,
        pca_scores: Optional[np.ndarray] = None,
        dbpath: str = "db",
        temperature: float = 10_000.0,
    ) -> None:
        if peak_array.ndim != 2 or peak_array.shape[1] < 4:
            raise ValueError(
                "peak_array must have shape (n_peaks, >=4) with columns "
                "[amplitude, center, sigma, gamma]"
            )

        self.peak_array = np.asarray(peak_array, dtype=float)
        self.n_peaks = self.peak_array.shape[0]
        self.pca_scores = pca_scores
        self.temperature = float(temperature)

        # Database and forward model
        self.db = Database(dbpath)
        self.maker = PeakyMaker(dbpath)

        # Build PeakRecord for each peak
        self.peaks: List[PeakRecord] = []
        for i in range(self.n_peaks):
            amp, mu, sigma, gamma = self.peak_array[i, :4]
            fwhm = float(_voigt_width(sigma, gamma))
            rec = PeakRecord(
                peak_idx=i,
                wavelength=mu,
                amplitude=amp,
                sigma=sigma,
                gamma=gamma,
                fwhm=fwhm,
            )
            if pca_scores is not None and i < pca_scores.shape[0]:
                rec.pc_scores = pca_scores[i]
            self.peaks.append(rec)

        # Stage results (populated by each stage)
        self.ground_states: Dict[str, Dict[float, dict]] = {}
        self.anchors: Dict[str, Dict[float, List[dict]]] = {}
        self.boltzmann_results: Dict[str, BoltzmannResult] = {}
        self.consensus_temperature: Optional[float] = None

    # ===================================================================
    # STAGE 1: Self-absorption quantification
    # ===================================================================

    def quantify_self_absorption(
        self,
        pc_indices: Tuple[int, ...] = (2, 5),
        weights: Optional[np.ndarray] = None,
        threshold: float = 2.0,
    ) -> List[PeakRecord]:
        """Compute a self-absorption index for each peak from PCA scores.

        Self-absorption flattens and broadens the tops of strong lines,
        producing characteristic signatures on specific PCs (typically
        PC3 and PC6).  The self-absorption index is a weighted sum of
        the absolute PCA scores on those components.

        Parameters
        ----------
        pc_indices : tuple of int
            Zero-based indices of PCs that capture flat-top / asymmetric
            behaviour.  Default ``(2, 5)`` = PC3 and PC6.
        weights : ndarray, optional
            Per-component weights.  If ``None``, equal weights are used.
        threshold : float
            Peaks with ``self_absorption_index > threshold * std`` are
            flagged.  Default is 2.0 (i.e. 2-sigma outliers).

        Returns
        -------
        list of PeakRecord
            Updated peak records.

        Raises
        ------
        ValueError
            If ``pca_scores`` was not provided at construction time.
        """
        if self.pca_scores is None:
            raise ValueError(
                "PCA scores are required for self-absorption quantification. "
                "Pass pca_scores to the constructor."
            )

        n_comp = self.pca_scores.shape[1]
        for idx in pc_indices:
            if idx >= n_comp:
                raise ValueError(
                    f"PC index {idx} out of range for {n_comp} components"
                )

        if weights is None:
            weights = np.ones(len(pc_indices))
        weights = np.asarray(weights, dtype=float)

        # Compute self-absorption index: weighted sum of |score| on
        # the selected PCs.  Absolute value because both positive and
        # negative deviations on these PCs indicate self-absorption
        # (sign reflects left vs right asymmetry direction).
        selected_scores = self.pca_scores[:, list(pc_indices)]  # (n, k)
        sa_index = np.abs(selected_scores) @ weights             # (n,)

        # Flag outliers above threshold * std
        sa_mean = np.mean(sa_index)
        sa_std = np.std(sa_index)
        cutoff = sa_mean + threshold * sa_std if sa_std > 0 else np.inf

        for i, rec in enumerate(self.peaks):
            if i < len(sa_index):
                rec.self_absorption_index = float(sa_index[i])
                rec.is_self_absorbed = bool(sa_index[i] >= cutoff)

        n_flagged = sum(1 for r in self.peaks if r.is_self_absorbed)
        return self.peaks

    # ===================================================================
    # STAGE 2: Ground-state anchor identification
    # ===================================================================

    def identify_ground_state_lines(
        self,
        temperature: Optional[float] = None,
        occupation_threshold: float = 0.001,
        max_ion_stage: int = 3,
    ) -> Dict[str, Dict[float, dict]]:
        """Identify ground-state transitions for all elements at a given T.

        Uses the proper Boltzmann factor ``exp(-Ek / (k_B * T))`` with
        the partition function for each ion stage.

        Parameters
        ----------
        temperature : float, optional
            Plasma temperature in K.  Defaults to ``self.temperature``.
        occupation_threshold : float
            Minimum relative occupation probability to retain a line.
        max_ion_stage : int
            Maximum ionization stage to consider (1=neutral only,
            2=up to singly ionized, 3=up to doubly ionized).
            Default is 3, suitable for LIBS plasmas at ~10,000 K.

        Returns
        -------
        dict
            ``{element: {ion: {'wavelengths': array, 'gA': array,
            'Ek': array, 'weights': array}}}``
        """
        T = temperature if temperature is not None else self.temperature
        if T <= 0:
            raise ValueError("temperature must be positive")

        kT = BOLTZMANN * T  # eV

        self.ground_states = {}
        for el in self.db.elements:
            if el in self.db.no_lines:
                continue

            lines = self.db.lines(el)
            if lines.size == 0:
                continue

            data = lines[:, [_COL_ION, _COL_WAVELENGTH, _COL_GA,
                             _COL_EI, _COL_EK]].astype(float)
            ionization = data[:, 0]
            peak_loc = data[:, 1]
            gA = data[:, 2]
            Ei = data[:, 3]
            Ek = data[:, 4]

            el_dict: Dict[float, dict] = {}

            for ion in np.unique(ionization):
                if ion > max_ion_stage:
                    continue
                ion_mask = ionization == ion
                ground_mask = ion_mask & (Ei == 0)

                if not np.any(ground_mask):
                    continue

                g_wl = peak_loc[ground_mask]
                g_gA = gA[ground_mask]
                g_Ek = Ek[ground_mask]

                # Boltzmann weight with partition function
                Z = float(np.squeeze(
                    self.maker.sb.stage_partition(
                        el, np.array([T]), ion=ion
                    )
                ))
                Z = max(Z, np.finfo(float).eps)

                weight = g_gA * np.exp(-g_Ek / kT) / Z

                # Normalize and threshold
                w_max = np.max(weight)
                if w_max <= 0:
                    continue
                rel_weight = weight / w_max
                keep = rel_weight > occupation_threshold

                if not np.any(keep):
                    continue

                el_dict[float(ion)] = {
                    'wavelengths': g_wl[keep],
                    'gA': g_gA[keep],
                    'Ek': g_Ek[keep],
                    'weights': weight[keep],
                }

            if el_dict:
                self.ground_states[el] = el_dict

        return self.ground_states

    def find_anchor_peaks(
        self,
        shift_tolerance: float = 0.1,
        require_no_self_absorption: bool = True,
    ) -> Dict[str, Dict[float, List[dict]]]:
        """Find unambiguous single-element ground-state peaks.

        A peak is an anchor if all matching ground-state lines (within
        ``shift_tolerance``) belong to a single element.  Optionally
        rejects anchors flagged as self-absorbed in Stage 1.

        Parameters
        ----------
        shift_tolerance : float
            Maximum wavelength deviation (nm) for a match.
        require_no_self_absorption : bool
            If True, reject self-absorbed candidates from being anchors.
            They are still recorded with metadata but ``is_anchor`` is
            not set.

        Returns
        -------
        dict
            ``{element: {ion: [{'peak_idx': int, 'wavelength': float,
            'ref_wavelength': float, 'gA': float, 'Ek': float,
            'amplitude': float, 'is_self_absorbed': bool}, ...]}}``
        """
        if not self.ground_states:
            self.identify_ground_state_lines()

        self.anchors = {}

        for rec in self.peaks:
            center = rec.wavelength

            # Find all (element, ion, ref_wavelength, gA, Ek) matches
            candidates = []
            for el, ions in self.ground_states.items():
                for ion, info in ions.items():
                    diffs = np.abs(info['wavelengths'] - center)
                    matches = np.where(diffs <= shift_tolerance)[0]
                    for m in matches:
                        candidates.append({
                            'element': el,
                            'ion': ion,
                            'ref_wavelength': float(info['wavelengths'][m]),
                            'gA': float(info['gA'][m]),
                            'Ek': float(info['Ek'][m]),
                            'distance': float(diffs[m]),
                        })

            # Anchor only if all candidates are from the SAME element
            unique_elements = {c['element'] for c in candidates}
            if len(unique_elements) != 1 or not candidates:
                continue

            el = candidates[0]['element']

            # Check self-absorption
            skip_anchor = (require_no_self_absorption
                           and rec.is_self_absorbed)

            # Group by ion
            for cand in candidates:
                ion = cand['ion']
                entry = {
                    'peak_idx': rec.peak_idx,
                    'wavelength': rec.wavelength,
                    'amplitude': rec.amplitude,
                    'ref_wavelength': cand['ref_wavelength'],
                    'gA': cand['gA'],
                    'Ek': cand['Ek'],
                    'distance': cand['distance'],
                    'is_self_absorbed': rec.is_self_absorbed,
                }

                if el not in self.anchors:
                    self.anchors[el] = {}
                if ion not in self.anchors[el]:
                    self.anchors[el][ion] = []
                self.anchors[el][ion].append(entry)

                # Mark the PeakRecord (even if self-absorbed, record
                # the element — just don't set is_anchor)
                if not skip_anchor:
                    rec.is_anchor = True
                    rec.anchor_element = el
                    rec.anchor_ion = ion
                    rec.anchor_ref_wavelength = cand['ref_wavelength']
                    rec.anchor_gA = cand['gA']
                    rec.anchor_Ek = cand['Ek']

        return self.anchors

    # ===================================================================
    # STAGE 3: Temperature estimation from Boltzmann plots
    # ===================================================================

    def estimate_temperature_boltzmann(
        self,
        min_lines: int = 3,
        exclude_self_absorbed: bool = True,
        t_min: float = 3000.0,
        t_max: float = 50000.0,
    ) -> Tuple[Dict[str, BoltzmannResult], Optional[float]]:
        """Estimate plasma temperature from Boltzmann plots of anchored elements.

        For each anchored element with ``>= min_lines`` from the same
        ion stage, constructs a Boltzmann plot:

            ln(I · λ / gA) = −Eₖ / (k_B · T) + const

        where *I* is the observed amplitude, *λ* the wavelength (nm),
        *gA* the NIST transition probability × degeneracy, and *Eₖ*
        the upper energy level (eV).

        The slope of this linear fit gives ``−1 / (k_B · T)``.

        Parameters
        ----------
        min_lines : int
            Minimum number of lines per (element, ion) for a fit.
        exclude_self_absorbed : bool
            If True, omit self-absorbed peaks from the fit.
        t_min, t_max : float
            Physically plausible temperature bounds (K).  Fits outside
            this range are discarded.

        Returns
        -------
        per_element : dict
            ``{element: BoltzmannResult}`` for each element that yielded
            a valid temperature.
        consensus_temperature : float or None
            Weighted average temperature (weighted by r² × n_lines).
            ``None`` if no valid fits.

        Physics notes
        -------------
        - Only valid under LTE.
        - Lines must come from the same ion stage; mixing ion stages
          would require a Saha correction.
        - The partition function cancels when all lines share the same
          ion at the same T: I ~ (gA / λ) · (1/Z) · exp(−Eₖ/kT),
          so ln(I·λ/gA) = −Eₖ/(kT) + ln(1/Z) and Z is the same
          constant for all lines of this ion.
        - Self-absorbed lines have I_obs < I_true, making their
          ln(I·λ/gA) too negative and biasing T upward.
        """
        if not self.anchors:
            raise RuntimeError(
                "No anchor peaks found.  Call find_anchor_peaks() first."
            )

        kB = BOLTZMANN  # eV/K

        self.boltzmann_results = {}

        for el, ions in self.anchors.items():
            for ion, entries in ions.items():
                # Collect usable lines
                Ek_vals = []
                y_vals = []

                for e in entries:
                    # Optionally skip self-absorbed
                    if exclude_self_absorbed and e['is_self_absorbed']:
                        continue

                    amp = e['amplitude']
                    wl = e['wavelength']
                    gA = e['gA']
                    Ek = e['Ek']

                    # Skip if any value is non-physical
                    if amp <= 0 or gA <= 0 or wl <= 0:
                        continue

                    # Boltzmann y-axis: ln(I * lambda / gA)
                    y = np.log(amp * wl / gA)
                    Ek_vals.append(Ek)
                    y_vals.append(y)

                if len(Ek_vals) < min_lines:
                    continue

                Ek_arr = np.array(Ek_vals)
                y_arr = np.array(y_vals)

                # Need spread in Ek for a meaningful fit
                if np.ptp(Ek_arr) < 0.1:  # less than 0.1 eV spread
                    continue

                # Linear least-squares: y = slope * Ek + intercept
                # slope = -1 / (kB * T)
                A = np.vstack([Ek_arr, np.ones(len(Ek_arr))]).T
                result = np.linalg.lstsq(A, y_arr, rcond=None)
                slope, intercept = result[0]

                # Compute T from slope
                if slope >= 0:
                    # Positive slope is non-physical (would give negative T)
                    continue
                T_est = -1.0 / (kB * slope)

                # Reject non-physical temperatures
                if T_est < t_min or T_est > t_max:
                    continue

                # R² goodness of fit
                y_pred = slope * Ek_arr + intercept
                ss_res = np.sum((y_arr - y_pred) ** 2)
                ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                key = f"{el}_{int(ion)}"
                self.boltzmann_results[key] = BoltzmannResult(
                    element=el,
                    ion=ion,
                    temperature_K=T_est,
                    n_lines=len(Ek_arr),
                    r_squared=r_sq,
                    slope=slope,
                    intercept=intercept,
                )

        # Consensus temperature: weighted by r² * n_lines
        if self.boltzmann_results:
            weights = []
            temps = []
            for br in self.boltzmann_results.values():
                w = max(br.r_squared, 0.0) * br.n_lines
                weights.append(w)
                temps.append(br.temperature_K)
            weights = np.array(weights)
            temps = np.array(temps)
            if np.sum(weights) > 0:
                self.consensus_temperature = float(
                    np.sum(weights * temps) / np.sum(weights)
                )
            else:
                self.consensus_temperature = None
        else:
            self.consensus_temperature = None

        return self.boltzmann_results, self.consensus_temperature

    # ===================================================================
    # STAGE 4: Candidate ranking with lineshape constraints
    # ===================================================================

    def _get_candidates(
        self,
        wavelength: float,
        shift_tolerance: float = 0.1,
        max_ion_stage: int = 3,
        element_list: Optional[List[str]] = None,
    ) -> List[dict]:
        """Get all (element, ion, line) candidates near a wavelength.

        Parameters
        ----------
        wavelength : float
            Observed peak centre (nm).
        shift_tolerance : float
            Maximum distance in nm.
        max_ion_stage : int
            Maximum ionization stage to consider.
        element_list : list of str, optional
            Restrict to these elements.  ``None`` = all.

        Returns
        -------
        list of dict
            Each: ``{'element', 'ion', 'Z', 'ref_wavelength', 'gA',
            'Ei', 'Ek', 'gi', 'gk', 'distance'}``.
        """
        elements = element_list if element_list is not None else self.db.elements
        candidates = []

        for el in elements:
            if el in self.db.no_lines:
                continue
            Z = self.db.elements.index(el) + 1

            lines = self.db.lines(el)
            if lines.size == 0:
                continue

            data = lines[:, [_COL_ION, _COL_WAVELENGTH, _COL_GA,
                             _COL_EI, _COL_EK, _COL_GI, _COL_GK]].astype(float)

            # Vectorised pre-filter: wavelength within tolerance
            dists = np.abs(data[:, 1] - wavelength)
            mask = (dists <= shift_tolerance) & (data[:, 0] <= max_ion_stage)
            if not np.any(mask):
                continue

            for row, dist in zip(data[mask], dists[mask]):
                candidates.append({
                    'element': el,
                    'ion': float(row[0]),
                    'Z': Z,
                    'ref_wavelength': float(row[1]),
                    'gA': float(row[2]),
                    'Ei': float(row[3]),
                    'Ek': float(row[4]),
                    'gi': float(row[5]),
                    'gk': float(row[6]),
                    'distance': float(dist),
                })

        return candidates

    @staticmethod
    def _distance_score(distance: float, sigma: float = 0.05) -> float:
        """Gaussian distance decay.  Returns 1.0 at distance=0."""
        return float(np.exp(-distance ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def _fwhm_score(
        observed_gamma: float,
        observed_sigma: float,
        candidate_ion: float,
        candidate_Z: int,
    ) -> float:
        """Score lineshape compatibility with the candidate species.

        Stark broadening (Lorentzian) scales roughly with Z_eff and ion
        stage.  A narrow, Gaussian-dominated peak matching a high-Z ionic
        line is penalised, and vice versa.

        Returns a score in [0, 1].
        """
        # Lorentzian fraction of observed peak
        total_width = abs(observed_sigma) + abs(observed_gamma)
        if total_width < 1e-12:
            obs_lor_frac = 0.5
        else:
            obs_lor_frac = abs(observed_gamma) / total_width

        # Expected Lorentzian fraction heuristic:
        # neutral low-Z → mostly Gaussian (Doppler/instrumental)
        # ionised high-Z → mostly Lorentzian (Stark)
        ion_factor = (candidate_ion - 1) / 3.0   # 0 for neutral, ~0.33 for II, ~0.67 for III
        z_factor = min(candidate_Z / 50.0, 1.0)  # normalise Z to [0, 1]
        expected_lor_frac = 0.1 + 0.6 * (0.5 * ion_factor + 0.5 * z_factor)
        expected_lor_frac = np.clip(expected_lor_frac, 0.05, 0.95)

        # Score: 1 when perfect match, decays with mismatch
        mismatch = abs(obs_lor_frac - expected_lor_frac)
        return float(1.0 - mismatch)

    def _boltzmann_intensity_score(
        self,
        observed_amplitude: float,
        ref_gA: float,
        ref_Ek: float,
        ref_wavelength: float,
        temperature_K: float,
    ) -> float:
        """Score based on expected Boltzmann intensity at estimated T.

        Computes expected relative intensity and compares to observed.
        Returns a score in [0, 1].
        """
        kT = BOLTZMANN * temperature_K
        if kT < 1e-10 or ref_gA <= 0 or ref_wavelength <= 0:
            return 0.5  # uninformative

        log_I_expected = np.log(ref_gA / ref_wavelength) - ref_Ek / kT

        if observed_amplitude <= 0:
            return 0.0

        log_I_observed = np.log(observed_amplitude)

        # The absolute scale doesn't match (unknown concentration), so
        # we compare the *rank order*.  Use a soft penalty on the log-ratio
        # relative to the median across all peaks.
        # For now, just penalise extreme mismatches.
        log_ratio = abs(log_I_observed - log_I_expected)
        # Score decays with log-ratio; half-life at log_ratio = 5
        return float(1.0 / (1.0 + log_ratio / 5.0))

    def _consistency_score(
        self,
        candidate_element: str,
        candidate_ion: float,
    ) -> float:
        """Bonus if other lines of this species are already confirmed.

        Normalized by the number of expected strong lines for this
        element/ion in the observed wavelength range, so elements with
        dense line forests don't dominate.

        Returns a score in [0, 1].
        """
        if candidate_element not in self.anchors:
            return 0.0

        # Count confirmed lines for this element/ion
        n_confirmed = len(
            self.anchors[candidate_element].get(candidate_ion, [])
        )
        # Also count other ion stages (weaker bonus)
        n_other_ion = sum(
            len(entries)
            for ion, entries in self.anchors[candidate_element].items()
            if ion != candidate_ion
        )

        # Normalize by expected line count in our wavelength range
        n_expected = self._count_expected_lines(
            candidate_element, candidate_ion)
        n_expected = max(n_expected, 1)

        # Fraction of expected lines that are confirmed
        frac_confirmed = min(n_confirmed / n_expected, 1.0)
        # Weaker cross-ion bonus
        frac_other = min(n_other_ion / max(n_expected, 5), 0.3)

        return float(min(frac_confirmed + frac_other, 1.0))

    def _count_expected_lines(
        self,
        element: str,
        ion: float,
        gA_threshold_frac: float = 0.01,
    ) -> int:
        """Count how many strong lines this element/ion has in our range.

        Only counts lines with gA above ``gA_threshold_frac`` of the
        element's maximum gA.
        """
        if element in self.db.no_lines:
            return 1

        lines = self.db.lines(element, ion=int(ion))
        if lines.size == 0:
            return 1

        data = lines[:, [_COL_WAVELENGTH, _COL_GA]].astype(float)
        wl, gA = data[:, 0], data[:, 1]

        # Filter to observed wavelength range
        if self.n_peaks > 0:
            wl_min = min(r.wavelength for r in self.peaks) - 1.0
            wl_max = max(r.wavelength for r in self.peaks) + 1.0
        else:
            wl_min, wl_max = 190, 910

        in_range = (wl >= wl_min) & (wl <= wl_max)
        gA_in_range = gA[in_range]

        if len(gA_in_range) == 0:
            return 1

        # Count strong lines (above threshold fraction of max)
        gA_max = np.max(gA_in_range)
        n_strong = int(np.sum(gA_in_range > gA_threshold_frac * gA_max))
        return max(n_strong, 1)

    def _abundance_score(self, candidate_element: str) -> float:
        """Score based on natural crustal abundance.

        Common elements (Li, Na, Ca, Fe, Si, Al, ...) get high scores.
        Rare/radioactive elements (Tc, Fr, Ac, Po, ...) are penalized.

        Returns a score in [0, 1].
        """
        abundance = self.db.elem_abund.get(candidate_element, 0.0)

        # Radioactive/synthetic elements with ~zero abundance
        if abundance < 1e-15:
            return 0.01

        # Log-scale mapping: most abundant ~0.28 (O), least ~1e-12
        # Map log10(abundance) from [-12, -0.5] to [0.1, 1.0]
        log_a = np.log10(max(abundance, 1e-15))
        score = (log_a + 12) / 11.5  # maps -12→0, -0.5→1
        return float(np.clip(score, 0.01, 1.0))

    def _line_strength_score(
        self,
        ref_gA: float,
        ref_Ei: float,
        candidate_element: str,
        candidate_ion: float,
    ) -> float:
        """Score favouring strong resonance lines.

        Lines with high gA and low Ei (ground state or near-ground) are
        more likely to be correctly identified in a LIBS spectrum.

        Returns a score in [0, 1].
        """
        if ref_gA <= 0:
            return 0.0

        # Normalize gA relative to the element's max
        lines = self.db.lines(candidate_element, ion=int(candidate_ion))
        if lines.size == 0:
            return 0.5
        all_gA = lines[:, _COL_GA].astype(float)
        gA_max = np.max(all_gA)
        if gA_max <= 0:
            return 0.5

        # gA fraction (log-scaled since gA spans many orders of magnitude)
        log_ratio = np.log10(ref_gA / gA_max)  # 0 for strongest, negative for weaker
        gA_score = max(0.0, 1.0 + log_ratio / 3.0)  # 3 orders of magnitude → 0

        # Bonus for ground-state transitions (Ei near 0)
        ground_bonus = np.exp(-ref_Ei / 2.0)  # decays with Ei in eV

        return float(np.clip(0.5 * gA_score + 0.5 * ground_bonus, 0.0, 1.0))

    def rank_candidates(
        self,
        shift_tolerance: float = 0.1,
        distance_sigma: float = 0.05,
        max_ion_stage: int = 3,
        w_distance: float = 1.0,
        w_fwhm: float = 0.3,
        w_boltzmann: float = 0.3,
        w_consistency: float = 1.0,
        w_abundance: float = 1.0,
        w_strength: float = 1.0,
        n_iterations: int = 2,
    ) -> List[PeakRecord]:
        """Rank candidate (element, ion) assignments for each non-anchor peak.

        For each unassigned peak, queries the NIST database for all lines
        within ``shift_tolerance`` and scores each candidate on six
        criteria:

        1. **Proximity** — Gaussian distance decay
        2. **FWHM/lineshape** — narrow → low-Z neutral, broad → high-Z ionic
        3. **Boltzmann intensity** — expected intensity at consensus T
        4. **Consistency** — fraction of expected lines confirmed (normalized)
        5. **Abundance** — natural crustal abundance (penalizes Tc, Ac, Fr, ...)
        6. **Line strength** — favours strong resonance lines (high gA, low Ei)

        After the first pass, re-scores with updated consistency (new
        assignments feed back).

        Parameters
        ----------
        shift_tolerance : float
            Wavelength tolerance (nm).
        distance_sigma : float
            Sigma for the Gaussian distance score.
        max_ion_stage : int
            Maximum ion stage to search.
        w_distance, w_fwhm, w_boltzmann, w_consistency,
        w_abundance, w_strength : float
            Relative weights for each scoring criterion.
        n_iterations : int
            Number of scoring passes (iterative refinement).

        Returns
        -------
        list of PeakRecord
            Updated records with ``candidates``, ``assigned_element``,
            ``assigned_ion``, and ``assignment_score`` populated.
        """
        T = (self.consensus_temperature
             if self.consensus_temperature is not None
             else self.temperature)

        for iteration in range(n_iterations):
            n_assigned = 0

            for rec in self.peaks:
                if rec.is_anchor:
                    continue

                candidates = self._get_candidates(
                    rec.wavelength,
                    shift_tolerance=shift_tolerance,
                    max_ion_stage=max_ion_stage,
                )

                if not candidates:
                    rec.candidates = []
                    continue

                # Score each candidate
                scored = []
                for cand in candidates:
                    s_dist = self._distance_score(
                        cand['distance'], distance_sigma)
                    s_fwhm = self._fwhm_score(
                        rec.gamma, rec.sigma,
                        cand['ion'], cand['Z'])
                    s_boltz = self._boltzmann_intensity_score(
                        rec.amplitude, cand['gA'], cand['Ek'],
                        cand['ref_wavelength'], T)
                    s_cons = self._consistency_score(
                        cand['element'], cand['ion'])
                    s_abund = self._abundance_score(
                        cand['element'])
                    s_str = self._line_strength_score(
                        cand['gA'], cand['Ei'],
                        cand['element'], cand['ion'])

                    total = (w_distance * s_dist
                             + w_fwhm * s_fwhm
                             + w_boltzmann * s_boltz
                             + w_consistency * s_cons
                             + w_abundance * s_abund
                             + w_strength * s_str)

                    cand['score'] = total
                    cand['score_distance'] = s_dist
                    cand['score_fwhm'] = s_fwhm
                    cand['score_boltzmann'] = s_boltz
                    cand['score_consistency'] = s_cons
                    cand['score_abundance'] = s_abund
                    cand['score_strength'] = s_str
                    scored.append(cand)

                # Sort descending by score
                scored.sort(key=lambda c: c['score'], reverse=True)
                rec.candidates = scored

                # Assign top candidate
                best = scored[0]
                rec.assigned_element = best['element']
                rec.assigned_ion = best['ion']
                rec.assignment_score = best['score']
                n_assigned += 1

            # After first pass, update anchors dict so consistency
            # scores reflect new assignments in the next iteration.
            if iteration < n_iterations - 1:
                self._update_confirmed_from_assignments()

        return self.peaks

    def _update_confirmed_from_assignments(self, min_score: float = 2.0):
        """Add high-confidence assignments to anchors for consistency feedback.

        Only assignments above ``min_score`` are promoted.  This does NOT
        change ``is_anchor`` on the PeakRecord — it only augments the
        anchors dict so ``_consistency_score`` can see them.
        """
        for rec in self.peaks:
            if rec.is_anchor or rec.assigned_element is None:
                continue
            if rec.assignment_score < min_score:
                continue

            el = rec.assigned_element
            ion = rec.assigned_ion
            if el not in self.anchors:
                self.anchors[el] = {}
            if ion not in self.anchors[el]:
                self.anchors[el][ion] = []

            # Avoid duplicates
            existing_idxs = {e['peak_idx'] for e in self.anchors[el][ion]}
            if rec.peak_idx in existing_idxs:
                continue

            self.anchors[el][ion].append({
                'peak_idx': rec.peak_idx,
                'wavelength': rec.wavelength,
                'amplitude': rec.amplitude,
                'ref_wavelength': rec.candidates[0]['ref_wavelength'] if rec.candidates else rec.wavelength,
                'gA': rec.candidates[0]['gA'] if rec.candidates else 0.0,
                'Ek': rec.candidates[0]['Ek'] if rec.candidates else 0.0,
                'distance': rec.candidates[0]['distance'] if rec.candidates else 0.0,
                'is_self_absorbed': rec.is_self_absorbed,
            })

    # ===================================================================
    # Convenience: run Stages 1-2
    # ===================================================================

    def run_stages_1_2(
        self,
        sa_pc_indices: Tuple[int, ...] = (2, 5),
        sa_threshold: float = 2.0,
        shift_tolerance: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """Execute Stages 1 and 2.

        Parameters
        ----------
        sa_pc_indices : tuple of int
            PC indices for self-absorption (Stage 1).
        sa_threshold : float
            Self-absorption flagging threshold.
        shift_tolerance : float
            Wavelength match tolerance in nm (Stage 2).
        verbose : bool
            Print progress.

        Returns
        -------
        dict with keys 'peaks', 'anchors', 'ground_states',
        'n_self_absorbed', 'n_anchors', 'confirmed_elements'.
        """
        # Stage 1
        if self.pca_scores is not None:
            self.quantify_self_absorption(
                pc_indices=sa_pc_indices,
                threshold=sa_threshold,
            )
            n_sa = sum(1 for r in self.peaks if r.is_self_absorbed)
            if verbose:
                print(f"Stage 1: {n_sa}/{self.n_peaks} peaks flagged "
                      f"as self-absorbed (threshold={sa_threshold}σ)")
        else:
            n_sa = 0
            if verbose:
                print("Stage 1: skipped (no PCA scores provided)")

        # Stage 2
        self.identify_ground_state_lines()
        n_gs_elements = len(self.ground_states)
        n_gs_lines = sum(
            len(info['wavelengths'])
            for el_dict in self.ground_states.values()
            for info in el_dict.values()
        )
        if verbose:
            print(f"Stage 2a: {n_gs_lines} ground-state lines from "
                  f"{n_gs_elements} elements at T={self.temperature:.0f} K")

        self.find_anchor_peaks(shift_tolerance=shift_tolerance)
        n_anchor_peaks = sum(1 for r in self.peaks if r.is_anchor)
        confirmed = set(self.anchors.keys())
        if verbose:
            print(f"Stage 2b: {n_anchor_peaks} anchor peaks → "
                  f"{len(confirmed)} confirmed elements: "
                  f"{sorted(confirmed)}")

            # Show per-element anchor counts
            for el in sorted(confirmed):
                ions = self.anchors[el]
                for ion, entries in sorted(ions.items()):
                    n_total = len(entries)
                    n_sa_in = sum(1 for e in entries if e['is_self_absorbed'])
                    sa_note = f" ({n_sa_in} self-absorbed)" if n_sa_in else ""
                    ion_label = ['', 'I', 'II', 'III', 'IV', 'V']
                    ion_str = ion_label[int(ion)] if int(ion) < len(ion_label) else f"{int(ion)}"
                    print(f"    {el} {ion_str}: {n_total} anchors{sa_note}")

        return {
            'peaks': self.peaks,
            'anchors': self.anchors,
            'ground_states': self.ground_states,
            'n_self_absorbed': n_sa,
            'n_anchors': n_anchor_peaks,
            'confirmed_elements': confirmed,
        }

    def run_stages_1_3(
        self,
        sa_pc_indices: Tuple[int, ...] = (2, 5),
        sa_threshold: float = 2.0,
        shift_tolerance: float = 0.1,
        min_boltzmann_lines: int = 3,
        verbose: bool = True,
    ) -> dict:
        """Execute Stages 1 through 3.

        Returns
        -------
        dict with all keys from run_stages_1_2 plus
        'boltzmann_results', 'consensus_temperature'.
        """
        result = self.run_stages_1_2(
            sa_pc_indices=sa_pc_indices,
            sa_threshold=sa_threshold,
            shift_tolerance=shift_tolerance,
            verbose=verbose,
        )

        # Stage 3
        boltz, T_consensus = self.estimate_temperature_boltzmann(
            min_lines=min_boltzmann_lines,
        )

        if verbose:
            if boltz:
                print(f"\nStage 3: Boltzmann temperature estimates "
                      f"({len(boltz)} fits):")
                for key, br in sorted(boltz.items()):
                    ion_label = ['', 'I', 'II', 'III', 'IV', 'V']
                    ion_str = (ion_label[int(br.ion)]
                               if int(br.ion) < len(ion_label)
                               else f"{int(br.ion)}")
                    print(f"    {br.element} {ion_str}: "
                          f"T = {br.temperature_K:.0f} K  "
                          f"(n={br.n_lines}, R²={br.r_squared:.3f})")
                if T_consensus is not None:
                    print(f"  Consensus T = {T_consensus:.0f} K")
            else:
                print("\nStage 3: no valid Boltzmann fits "
                      f"(need ≥{min_boltzmann_lines} lines per ion)")

        result['boltzmann_results'] = boltz
        result['consensus_temperature'] = T_consensus
        return result

    def run_stages_1_4(
        self,
        sa_pc_indices: Tuple[int, ...] = (2, 5),
        sa_threshold: float = 2.0,
        shift_tolerance: float = 0.1,
        min_boltzmann_lines: int = 3,
        w_distance: float = 1.0,
        w_fwhm: float = 0.3,
        w_boltzmann: float = 0.3,
        w_consistency: float = 1.0,
        w_abundance: float = 1.0,
        w_strength: float = 1.0,
        verbose: bool = True,
    ) -> dict:
        """Execute Stages 1 through 4.

        Returns
        -------
        dict with all keys from run_stages_1_3 plus
        'n_assigned', 'all_elements'.
        """
        result = self.run_stages_1_3(
            sa_pc_indices=sa_pc_indices,
            sa_threshold=sa_threshold,
            shift_tolerance=shift_tolerance,
            min_boltzmann_lines=min_boltzmann_lines,
            verbose=verbose,
        )

        # Stage 4
        self.rank_candidates(
            shift_tolerance=shift_tolerance,
            w_distance=w_distance,
            w_fwhm=w_fwhm,
            w_boltzmann=w_boltzmann,
            w_consistency=w_consistency,
            w_abundance=w_abundance,
            w_strength=w_strength,
        )

        n_assigned = sum(1 for r in self.peaks
                         if r.assigned_element is not None and not r.is_anchor)
        n_unassigned = sum(1 for r in self.peaks
                           if r.assigned_element is None and not r.is_anchor)

        # Collect all identified elements (anchors + assignments)
        all_elements = set(result['confirmed_elements'])
        for rec in self.peaks:
            if rec.assigned_element is not None:
                all_elements.add(rec.assigned_element)

        if verbose:
            print(f"\nStage 4: ranked candidates for "
                  f"{n_assigned + n_unassigned} non-anchor peaks")
            print(f"  {n_assigned} assigned, {n_unassigned} unassigned "
                  f"(no candidates within tolerance)")
            print(f"  All elements ({len(all_elements)}): "
                  f"{sorted(all_elements)}")

            # Top elements by peak count
            from collections import Counter
            el_counts = Counter()
            for rec in self.peaks:
                if rec.assigned_element:
                    el_counts[rec.assigned_element] += 1
                elif rec.anchor_element:
                    el_counts[rec.anchor_element] += 1
            print(f"\n  Peak counts by element (top 20):")
            for el, count in el_counts.most_common(20):
                anchor_count = sum(1 for r in self.peaks
                                   if r.anchor_element == el and r.is_anchor)
                assigned_count = count - anchor_count
                print(f"    {el:4s}: {count:4d} peaks "
                      f"({anchor_count} anchor, {assigned_count} ranked)")

        result['n_assigned'] = n_assigned
        result['all_elements'] = all_elements
        return result
