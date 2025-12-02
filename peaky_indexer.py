import numpy as np
import pulp
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set
from matplotlib import pyplot as plt


from utils.database import Database
from peaky_maker import PeakyMaker

class PeakyIndexer():
    """Tools for indexing and classifying laser plasma spectra.

    Parameters
    ----------
    PeakyFinder : :class:`~peaky_finder.PeakyFinder`
        Initialized peak finding utility.
    dbpath : str, optional
        Path to the database directory. Default is ``"db"``.

    Attributes
    ----------
    finder : :class:`~peaky_finder.PeakyFinder`
        Peak finding utility used for spectral analysis.
    maker : :class:`~peaky_maker.PeakyMaker`
        Spectrum synthesis helper.
    db : :class:`~utils.database.Database`
        Atomic line database.
    k : float
        Boltzmann constant in eV/K.
    h : float
        Planck constant in eV s.
    c : float
        Speed of light in m/s.
    me : float
        Electron mass in eV/c².
    """

    def __init__(self, PeakyFinder, dbpath="db") -> None:
        """Initialize the indexer.

        Parameters
        ----------
        PeakyFinder : :class:`~peaky_finder.PeakyFinder`
            Initialized peak finding utility.
        dbpath : str, optional
            Path to the database directory. Default is ``"db"``.
        """
        self.finder = PeakyFinder
        self.maker = PeakyMaker(dbpath)
        self.db = Database(dbpath)

        # Physical Constants
        self.k = 8.617333262 * 10 ** -5  # Boltzmann constant (eV/K)
        self.h = 4.135667696 * 10 ** -15  # Planck constant eV s
        self.c = 2.99792458 * 10**8  # Speed of light in a vacuum (m/s)
        self.me = 0.51099895000 * 10**6  # electron mass eV / c^2


    def ground_state(self, threshold: float = 0.001) -> None:
        """Identify ground state transitions for each element.

        Parameters
        ----------
        threshold : float, optional
            Minimum relative occupation probability required to keep a line.

        Returns
        -------
        None
            Results are stored in ``self.ground_states``.
        """

        self.ground_states: DefaultDict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        for el in self.db.elements:
            ionization, peak_loc, gA, Ei, Ek, gi, gk = self.db.lines(el)[:, [0, 1, 3, 4, 5, 12, 13]].T.astype(float)
            ions = np.unique(ionization)

            for ion in ions:
                Eind = ionization == ion  # indices of single ionization state
                groundEi = Ei[Eind] == 0  # indices of ground state for ionization state
                groundpeak = peak_loc[Eind][groundEi]  # ground state peak wavelengths
                groundEk = gk[Eind][groundEi] * np.exp(-Ek[Eind][groundEi] / 1000)  # ground state upper level occupation probabilities
                groundgA = 1 / gk[Eind][groundEi] * gA[Eind][groundEi]

                if len(groundEk) > 0 and np.max(groundEk) > 0:
                    groundT = groundgA * groundEk
                    groundEkprob = np.divide(
                        groundT,
                        np.max(groundgA * groundEk),
                        out=np.zeros_like(groundEk),
                        where=groundEk > 0,
                    ) > threshold
                    self.ground_states[el][str(ion)] = (
                        groundpeak[groundEkprob],
                        groundT[groundEkprob],
                    )

    def anchor_peaks(
        self, peak_array: np.ndarray, shift_tolerance: float = 0.1) -> Dict[str, Dict[str, List[float]]]:
        """Identify unambiguous ground-state transitions in fitted peaks.

        Parameters
        ----------
        peak_array : ndarray
            Array of peak parameters with columns ``[amplitude, center, sigma, gamma]``
            produced by :meth:`finder.fit_spectrum_data`.
        shift_tolerance : float, optional
            Maximum allowed deviation in nm between a fitted peak and a ground-state
            transition. Default is ``0.1``.

        Returns
        -------
        dict
            Nested mapping of element names to ionization states and the wavelengths
            of matched peaks.
        """

        if not hasattr(self, "ground_states"):
            self.ground_state()

        peak_centers = peak_array[:, 1]
        anchors: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for center in peak_centers:
            candidates: List[tuple] = []
            for el, ions in self.ground_states.items():
                for ion, (wavelengths, _) in ions.items():
                    if np.any(np.abs(wavelengths - center) <= shift_tolerance):
                        candidates.append((el, ion))

            unique_elements = {el for el, _ in candidates}
            if len(unique_elements) == 1 and candidates:
                el = candidates[0][0]
                for _, ion in candidates:
                    anchors[el][ion].append(float(center))

        return {el: dict(ions) for el, ions in anchors.items()}


    def distance_decay(self, w1, w2, s):
        """Gaussian distance metric for proximate peaks.

        Parameters
        ----------
        w1 : float
            First wavelength.
        w2 : float
            Second wavelength.
        s : float
            Standard deviation of the Gaussian.

        Returns
        -------
        float
            Distance decay factor.
        """
        d = np.exp(-(w1 - w2)**2 / (2 * s**2))
        return d


    def peak_proximity(self, peaks, reference, shift_tolerance):
        """Determine minimum proximity between data and reference peak locations.

        Parameters
        ----------
        peaks : array_like
            Detected peak positions.
        reference : array_like
            Reference peak locations.
        shift_tolerance : float
            Maximum allowed deviation for a match.

        Returns
        -------
        ndarray
            Boolean array indicating peaks within ``shift_tolerance`` of any
            reference line.
        """
        peak_prox = peaks[:, np.newaxis] - reference
        peak_match = np.min(np.abs(peak_prox), axis=-1) <= shift_tolerance
        return peak_match


    def peak_interference(self,
                             x,
                         wid=1,
                        rang=1,
               log_amp_limit=6,
             ground_state=True,
             element_list=None
                               ):
        """Determine ions most likely to interfere at a given peak location.

        Parameters
        ----------
        x : float
            Peak location to evaluate.
        wid : float, optional
            Width parameter for the Gaussian distance metric.
        rang : float, optional
            Maximum distance from ``x`` to consider.
        log_amp_limit : float, optional
            Threshold for logarithmic amplitude difference.
        ground_state : bool, optional
            If ``True``, only ground-state transitions are considered.
        element_list : list of str, optional
            Subset of elements to search. If ``None``, all elements are used.

        Returns
        -------
        list of list
            Each entry is ``[element, ion, location, x, distance, amplitude]``
            for lines within the specified range. If no peaks are found,
            returns ``[[]]``.
        """

        if element_list is None:
            element_list = self.db.elements

        close_peaks = []
        for el in element_list: # elements
            ionization, peak_loc, gA, Ei = self.db.lines(el)[:,[0, 1, 3, 4]].T.astype(float)
            ions = np.sort(np.unique(ionization))
            for ion in ions:
                i_ind = ionization == ion
                if ground_state:
                    E_ind = Ei == 0
                else:
                    E_ind = Ei > 0
                i_peak_loc = peak_loc[i_ind * E_ind]
                i_gA = gA[i_ind * E_ind]

                if len(i_peak_loc) > 0:
                    close_candidates = i_peak_loc - x <= wid
                    i_peak_loc_close = i_peak_loc[close_candidates]
                    i_gA_close = np.log10(i_gA[close_candidates])
                else:
                    i_peak_loc_close = []
                    i_gA_close = []
                
                if len(i_peak_loc_close) > 0:
                    i_peak_loc_round = np.round(i_peak_loc_close, 3)
                    distance_metric = self.distance_decay(x, np.array(i_peak_loc_round), wid)
                    amplitude_metric = np.max(i_gA_close) - log_amp_limit

                    for location, distance, A in zip(i_peak_loc_close, distance_metric, i_gA_close):
                        if np.abs(x-location) < rang:
                            if A > amplitude_metric:
                                close_peaks.append([el, ion, location, x, distance, A]) # log10(A)

        if len(close_peaks) > 0:
            close_peaks.sort(key=lambda dp: dp[-1], reverse=True)
        else:
            close_peaks = [[]]
        
        return close_peaks


    def peak_match(self, peak_array, ground_state=False, element_list=None):
        """Match peaks to candidate element transitions.

        Parameters
        ----------
        peak_array : ndarray
            Array of peak parameters with columns ``[amplitude, center, sigma, gamma]``.
        ground_state : bool, optional
            If ``True``, restrict matches to ground-state transitions.
        element_list : list of str, optional
            Elements to search. If ``None``, all elements in the database are used.

        Returns
        -------
        dict
            Mapping of element names to :class:`Element` instances containing
            matched peak information.
        """
        peak_idx = np.arange(len(peak_array))
        
        peak_match_dictionary = dict({}) # store matched element peak information

        for idx in peak_idx:
            close_peaks = self.peak_interference(peak_array[idx, 1], ground_state=ground_state, element_list=element_list)
            
            if any(close_peaks[0]): # if any close peaks are found, match them
                
                for peak in close_peaks:
                    el, *rest = peak
                    match_data = Element(el)
                    match_data.update(peak, idx)

                    if el not in peak_match_dictionary:
                        peak_match_dictionary[el] = match_data 
                    else:
                        peak_match_dictionary[el].update(peak, idx)
        
        return peak_match_dictionary


    def ion_indexer(self, peak_array, element: str, ion: float, position_tol=None):
        """Compare fitted peaks to a single element/ion reference and return match statistics."""
        if isinstance(peak_array, dict):
            if "sorted_parameter_array" not in peak_array:
                raise ValueError("peak_array dict must contain 'sorted_parameter_array'")
            peak_array = peak_array["sorted_parameter_array"]

        peak_array = np.asarray(peak_array, dtype=float)
        if peak_array.ndim != 2 or peak_array.shape[1] < 2:
            raise ValueError("peak_array must be (n, >=2) with intensity in column 0 and position in column 1")

        I_obs_raw = np.clip(peak_array[:, 0], 0, None)
        I_scale = float(np.max(I_obs_raw)) if I_obs_raw.size else 1.0
        if not np.isfinite(I_scale) or I_scale <= 0:
            I_scale = 1.0
        I_obs = I_obs_raw / I_scale
        p_obs = peak_array[:, 1]

        if position_tol is None:
            sigmas = peak_array[:, 2] if peak_array.shape[1] > 2 else np.array([])
            gammas = peak_array[:, 3] if peak_array.shape[1] > 3 else np.array([])
            if sigmas.size and gammas.size:
                fwhm = self.finder.voigt_width(sigmas, gammas)
                position_tol = float(np.median(fwhm)) if np.any(np.isfinite(fwhm)) else 0.0
            else:
                position_tol = 0.0
            if position_tol <= 0:
                position_tol = 0.1

        lines = self.db.lines(element)
        if lines.size == 0:
            raise ValueError(f"No lines found for element {element}")
        ionization = lines[:, 0].astype(float)
        mask = ionization == ion
        if not np.any(mask):
            raise ValueError(f"No lines found for element {element} ion {ion}")
        predicted_pos = lines[mask, 1].astype(float)
        # Column 3 stored linearly; use directly
        predicted_int = lines[mask, 3].astype(float)
        gk = lines[mask, 13].astype(float) if lines.shape[1] > 13 else np.ones_like(predicted_int)
        predicted_int = predicted_int * np.clip(gk, 1e-12, None)
        p_scale = float(np.max(predicted_int)) if predicted_int.size else 1.0
        if not np.isfinite(p_scale) or p_scale <= 0:
            p_scale = 1.0
        predicted_int_norm = predicted_int / p_scale

        # Exclude predicted peaks outside the observed window (with a small buffer of position_tol)
        if p_obs.size:
            lo, hi = float(np.min(p_obs)) - position_tol, float(np.max(p_obs)) + position_tol
            in_window = (predicted_pos >= lo) & (predicted_pos <= hi)
            predicted_pos = predicted_pos[in_window]
            predicted_int_norm = predicted_int_norm[in_window]

        matches = []
        used_pred = set()
        unmatched_observed = []
        per_predicted_usage = defaultdict(float)
        for i, obs_pos in enumerate(p_obs):
            width = position_tol
            if peak_array.shape[1] > 3:
                width = max(width, float(self.finder.voigt_width(peak_array[i, 2], peak_array[i, 3])))
            mask = np.abs(predicted_pos - obs_pos) <= width
            pred_sum = float(np.sum(predicted_int_norm[mask])) if np.any(mask) else 0.0
            if pred_sum > 0:
                used_pred.update(np.nonzero(mask)[0])
                for idx in np.nonzero(mask)[0]:
                    per_predicted_usage[idx] += predicted_int_norm[idx]
            ratio = I_obs[i] / pred_sum if pred_sum > 0 else float("nan")
            if pred_sum > 0:
                matches.append(
                    {
                        "obs_pos": float(obs_pos),
                        "predicted_positions": predicted_pos[mask].tolist(),
                        "predicted_intensities": predicted_int_norm[mask].tolist(),
                        "predicted_intensity_sum": pred_sum,
                        "obs_int": float(I_obs[i]),
                        "intensity_ratio_obs_over_pred": float(ratio),
                    }
                )
            else:
                unmatched_observed.append({"obs_pos": float(obs_pos), "obs_int": float(I_obs[i])})

        # Combine matches that share the same predicted-position set and renormalize intensities
        combined = defaultdict(lambda: {"obs_int_sum": 0.0, "predicted_positions": None, "predicted_intensities": None, "predicted_intensity_sum": 0.0})
        for m in matches:
            key = tuple(m["predicted_positions"])
            combined[key]["obs_int_sum"] += m["obs_int"]
            if combined[key]["predicted_positions"] is None:
                combined[key]["predicted_positions"] = m["predicted_positions"]
                combined[key]["predicted_intensities"] = m["predicted_intensities"]
                combined[key]["predicted_intensity_sum"] = m["predicted_intensity_sum"]
        merged_matches = []
        for key, val in combined.items():
            pred_sum = val["predicted_intensity_sum"]
            ratio = val["obs_int_sum"] / pred_sum if pred_sum > 0 else float("nan")
            merged_matches.append(
                {
                    "obs_int_sum": float(val["obs_int_sum"]),
                    "predicted_positions": val["predicted_positions"],
                    "predicted_intensities": val["predicted_intensities"],
                    "predicted_intensity_sum": float(pred_sum),
                    "intensity_ratio_obs_over_pred": float(ratio),
                }
            )
        matches = merged_matches

        unmatched_predicted = []
        for idx, (pp, pint) in enumerate(zip(predicted_pos, predicted_int_norm)):
            if idx not in used_pred:
                unmatched_predicted.append({"pred_pos": float(pp), "pred_int": float(pint)})

        ratios = [m["intensity_ratio_obs_over_pred"] for m in matches if np.isfinite(m["intensity_ratio_obs_over_pred"]) and m["predicted_intensity_sum"] > 0]
        summary = {
            "predicted_count": int(len(predicted_pos)),
            "matched_count": int(len(matches)),
            "unmatched_predicted": int(len(unmatched_predicted)),
            "unmatched_observed": int(len(unmatched_observed)),
            "ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
            "ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        }

        return {
            "matches": matches,
            "unmatched_predicted": unmatched_predicted,
            "unmatched_observed": unmatched_observed,
            "summary": summary,
        }


    def _prune_references_for_spectrum(
        self,
        p_obs,
        I_obs,
        refs,
        detection_limit,
        position_tol,
        eta=0.8,
        max_rank=5,
    ):
        """Prune obviously poor references using anchor peaks before MILP."""
        if not refs:
            return refs

        p_obs = np.asarray(p_obs, dtype=float)
        I_obs = np.asarray(I_obs, dtype=float)

        order = np.argsort(I_obs)[::-1]
        bad_refs: Set[int] = set()
        max_rank = min(max_rank, len(order))

        if detection_limit is None:
            detection_limit = -np.inf
        obs_above_limit = I_obs >= detection_limit if np.isfinite(detection_limit) else np.ones_like(I_obs, dtype=bool)
        obs_pos_vis = p_obs[obs_above_limit]

        ref_pos = []
        ref_pos_vis = []
        for ref in refs:
            pos = np.asarray(ref["pos"], dtype=float)
            inten = np.asarray(ref["intensity"], dtype=float)
            ref_pos.append(pos)
            if pos.size == 0:
                ref_pos_vis.append(pos)
                continue
            mask = inten >= detection_limit if np.isfinite(detection_limit) else np.ones_like(inten, dtype=bool)
            ref_pos_vis.append(pos[mask])

        for rank in range(max_rank):
            anchor_idx = int(order[rank])
            anchor_pos = p_obs[anchor_idx]

            candidate_refs = [
                m
                for m, pos in enumerate(ref_pos)
                if m not in bad_refs and pos.size > 0 and np.any(np.abs(pos - anchor_pos) <= position_tol)
            ]

            if not candidate_refs:
                continue

            good_refs = []
            for m in candidate_refs:
                pos_vis = ref_pos_vis[m]
                if pos_vis.size == 0:
                    good_refs.append(m)
                    continue

                if obs_pos_vis.size == 0:
                    bad_refs.add(m)
                    continue

                match = np.any(np.abs(obs_pos_vis[:, None] - pos_vis[None, :]) <= position_tol, axis=0)
                matched = np.count_nonzero(match)
                missing_frac = 1.0 - matched / max(1, pos_vis.size)

                if missing_frac <= eta:
                    good_refs.append(m)
                else:
                    bad_refs.add(m)

        if not bad_refs:
            return refs
        keep = [m for m in range(len(refs)) if m not in bad_refs]
        return [refs[m] for m in keep]


    def spectrum_match(
        self,
        peak_array,
        refs=None,
        position_tol=None,
        detection_limit=None,
        w_pos=1.0,
        w_int=1.0,
        alpha=1.0,
        beta=1.0,
        M_big=None,
        solver=None,
        interference_kwargs=None,
        prune_kwargs=None,
    ):
        """
        Match observed spectrum peaks to a set of reference spectra using a MILP model.

        Parameters
        ----------
        peak_array : array_like or dict
            Observed peak parameters with intensity in column 0 and peak position
            in column 1. Pass ``fit_dict["sorted_parameter_array"]`` from
            :meth:`~peaky_finder.PeakyFinder.fit_spectrum_data` directly; the
            second and first columns provide ``p_obs`` and ``I_obs`` respectively.
        refs : list of dict or None
            Each dict must have keys:
                - "pos": array_like of peak positions (q_{mj})
                - "intensity": array_like of nominal reference intensities (R_{mj})
            Peak locations can come from :meth:`peak_interference` and intensities
            from :meth:`peaky_maker.PeakyMaker.peak_maker`. All arrays are 1-D of
            the same length within each reference. If ``None``, references are
            generated by running :meth:`peak_match` to find candidate elements/ions
            and then scanning those ions with :meth:`peak_interference`.
        position_tol : float or None
            Maximum allowed position difference |p_i - q_{mj}| to consider a candidate match.
            When ``None``, it is set to the median Voigt FWHM computed from the
            sigma and gamma columns of ``peak_array`` using
            :meth:`PeakyFinder.voigt_width`.
        detection_limit : float or None
            Detection limit L for intensities in the observed spectrum. When
            ``None``, it is derived from observed amplitudes using ``n_sigma``
            saved in the finder ``fit_dict`` (mean + n_sigma * std of amplitudes;
            defaults to ``n_sigma=0`` if unavailable).
            Observed intensities are normalized to [0, 1] internally, so this
            value is interpreted in that normalized scale.
        w_pos : float, optional
            Weight for squared position mismatch in the objective.
        w_int : float, optional
            Weight for intensity mismatch residuals in the objective.
        alpha : float, optional
            Penalty for marking an observed peak as noise (unassigned).
        beta : float, optional
            Penalty for marking a reference peak as "hidden" (missing though expected).
        M_big : float, optional
            Big-M constant for linearization. If None, a heuristic value is chosen.
        solver : pulp solver instance, optional
            Custom pulp solver; if None, uses ``pulp.PULP_CBC_CMD()``.
        interference_kwargs : dict, optional
            Passed to :meth:`peak_match` (``ground_state``, ``element_list``) and
            the internal reference builder (e.g. ``wid``, ``rang``,
            ``log_amp_limit``) when ``refs`` is ``None``.
        prune_kwargs : dict, optional
            Parameters for the pre-filtering helper (e.g. ``eta``=0.8,
            ``max_rank``=5); applied before running the MILP.

        Returns
        -------
        result : dict
            Dictionary with fields:
                - "status": MILP solver status string (e.g. "Optimal").
                - "c": array, shape (M,), estimated component fractions.
                - "assignments": list of length n.
                  For each observed peak i, either None (noise) or a dict:
                    {
                    "ref_index": m,
                    "ref_peak_index": j,
                    "ref_peak_global_index": f,
                    }
                - "noise_mask": boolean array, shape (n,), True for noise peaks.
                - "hidden_ref_mask": boolean array, shape (P,), True for hidden reference peaks.
                - "objective_value": float, optimal objective value.
                - "flattened_reference": dict with:
                    "pos", "intensity", "ref_index", "ref_peak_index"
                - "elements": mapping of element symbols to :class:`Element`
                  objects populated from assignments (when reference metadata
                  includes ``element`` and ``ion``)
        """
        # --- Inputs to numpy ---
        n_sigma_val = None
        if isinstance(peak_array, dict):
            if "sorted_parameter_array" not in peak_array:
                raise ValueError(
                    "peak_array dict must contain 'sorted_parameter_array' as returned by PeakyFinder.fit_spectrum_data"
                )
            n_sigma_val = peak_array.get("n_sigma")
            peak_array = peak_array["sorted_parameter_array"]

        peak_array = np.asarray(peak_array, dtype=float)
        if peak_array.ndim != 2 or peak_array.shape[1] < 2:
            raise ValueError("peak_array must be (n, >=2) with intensity in column 0 and position in column 1")

        # Normalize observed intensities to [0, 1]
        I_obs_raw = np.clip(peak_array[:, 0], 0, None)
        I_scale = float(np.max(I_obs_raw)) if I_obs_raw.size else 1.0
        if not np.isfinite(I_scale) or I_scale <= 0:
            I_scale = 1.0
        I_obs = I_obs_raw / I_scale
        p_obs = peak_array[:, 1]
        n = p_obs.shape[0]

        if position_tol is None:
            sigmas = peak_array[:, 2] if peak_array.shape[1] > 2 else np.array([])
            gammas = peak_array[:, 3] if peak_array.shape[1] > 3 else np.array([])
            if sigmas.size and gammas.size:
                fwhm = self.finder.voigt_width(sigmas, gammas) / 2.0  # half-width at half-maximum
                position_tol = float(np.median(fwhm)) if np.any(np.isfinite(fwhm)) else 0.0
            else:
                position_tol = 0.0
            if position_tol <= 0:
                position_tol = 0.1  # conservative fallback

        if detection_limit is None:
            mean_I = float(np.mean(I_obs)) if n > 0 else 0.0
            std_I = float(np.std(I_obs)) if n > 0 else 0.0
            n_sigma_use = float(n_sigma_val) if n_sigma_val is not None else 0.0
            detection_limit = mean_I + n_sigma_use * std_I
        else:
            detection_limit = float(detection_limit) / I_scale

        # Precompute range for edge tapering of reference oscillator strengths
        if n > 0:
            loc_min, loc_max = float(np.min(p_obs)), float(np.max(p_obs))
        else:
            loc_min, loc_max = 0.0, 1.0

        def tukey_weight(loc, edge_frac=0.1):
            """Raised-cosine taper: middle 80% at 1.0, outer 10% fades to 0."""
            if loc_max <= loc_min:
                return 1.0
            rel = (loc - loc_min) / (loc_max - loc_min)
            if rel < 0 or rel > 1:
                return 0.0
            if rel < edge_frac:
                return 0.5 * (1 - np.cos(np.pi * rel / edge_frac))
            if rel > 1 - edge_frac:
                return 0.5 * (1 - np.cos(np.pi * (1 - rel) / edge_frac))
            return 1.0

        def _normalize_refs(ref_list):
            norm = []
            for ref in ref_list:
                ref_copy = dict(ref)
                inten = np.asarray(ref_copy["intensity"], dtype=float)
                inten = np.clip(inten, 0, None)
                max_inten = float(np.max(inten)) if inten.size else 0.0
                if np.isfinite(max_inten) and max_inten > 0:
                    inten = inten / max_inten
                ref_copy["intensity"] = inten
                norm.append(ref_copy)
            return norm

        if refs is None:
            kwargs = (interference_kwargs or {}).copy()
            ground_state = kwargs.pop("ground_state", False)
            element_list = kwargs.pop("element_list", None)
            wid = kwargs.pop("wid", 1)
            rang = kwargs.pop("rang", 1)
            log_amp_limit = kwargs.pop("log_amp_limit", 6)

            # Iteratively build and prune references starting from the strongest peaks
            refs_map: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: {"pos": [], "intensity": []})
            bad_pairs: Set[tuple] = set()
            refs = []

            for center in p_obs:
                hits = self.peak_interference(
                    center,
                    wid=wid,
                    rang=rang,
                    log_amp_limit=log_amp_limit,
                    ground_state=ground_state,
                    element_list=element_list,
                )
                if len(hits) == 1 and hits[0] == []:
                    continue
                for el, ion, location, *_rest, A in hits:
                    pair = (el, ion)
                    if pair in bad_pairs:
                        continue
                    ref = refs_map[pair]
                    ref["pos"].append(float(location))
                    if np.isfinite(A):
                        weight = tukey_weight(location)
                        ref["intensity"].append(float((10 ** A) * weight))
                    else:
                        ref["intensity"].append(0.0)

                # Build refs for current state and prune obvious mismatches
                current_refs = []
                for (el, ion), data in refs_map.items():
                    if data["pos"]:
                        current_refs.append(
                            {
                                "element": el,
                                "ion": ion,
                                "pos": np.asarray(data["pos"], dtype=float),
                                "intensity": np.asarray(data["intensity"], dtype=float),
                            }
                        )
                current_refs = _normalize_refs(current_refs)
                if current_refs:
                    pruned = self._prune_references_for_spectrum(
                        p_obs,
                        I_obs,
                        current_refs,
                        detection_limit,
                        position_tol,
                        **(prune_kwargs or {}),
                    )
                    removed = {(r["element"], r["ion"]) for r in current_refs} - {(r["element"], r["ion"]) for r in pruned}
                    bad_pairs.update(removed)
                    refs_map = defaultdict(lambda: {"pos": [], "intensity": []})
                    for ref in pruned:
                        key = (ref.get("element"), ref.get("ion"))
                        refs_map[key]["pos"].extend(np.asarray(ref["pos"], dtype=float).tolist())
                        refs_map[key]["intensity"].extend(np.asarray(ref["intensity"], dtype=float).tolist())
                    refs = pruned
        else:
            # Normalize provided refs intensities to [0, 1]
            refs = _normalize_refs(refs)

        # Normalize any built refs
        if refs:
            refs = _normalize_refs(refs)
        prune_args = prune_kwargs or {}
        refs = self._prune_references_for_spectrum(
            p_obs,
            I_obs,
            refs,
            detection_limit,
            position_tol,
            **prune_args,
        )
        M = len(refs)

        # --- Flatten reference peaks into a single index space f = 0..P-1 ---
        q_list = []
        R_list = []
        ref_id_list = []   # which reference each flattened peak belongs to
        peak_id_list = []  # local index within that reference
        ref_elements = []
        ref_ions = []
        for m, ref in enumerate(refs):
            pos = np.asarray(ref["pos"], dtype=float)
            inten = np.asarray(ref["intensity"], dtype=float)
            if pos.shape != inten.shape:
                raise ValueError(f"ref[{m}]['pos'] and ref[{m}]['intensity'] must have the same shape")
            q_list.append(pos)
            R_list.append(inten)
            ref_id_list.append(np.full(pos.shape, m, dtype=int))
            peak_id_list.append(np.arange(pos.shape[0], dtype=int))
            ref_elements.append(ref.get("element"))
            ref_ions.append(ref.get("ion"))

        if not q_list:
            raise ValueError("refs must contain at least one reference spectrum")

        q = np.concatenate(q_list)            # shape (P,)
        R = np.concatenate(R_list)            # shape (P,)
        ref_id = np.concatenate(ref_id_list)  # shape (P,)
        peak_id = np.concatenate(peak_id_list)  # shape (P,)
        P = q.shape[0]

        # --- Vectorized construction of candidate matches based on position tolerance ---

        # delta_pos[i, f] = p_obs[i] - q[f]
        delta_pos = p_obs[:, None] - q[None, :]
        abs_delta = np.abs(delta_pos)
        candidate_mask = abs_delta <= position_tol

        # Indices of candidate edges e: (i_idx[e], f_idx[e])
        i_idx, f_idx = np.nonzero(candidate_mask)
        E = i_idx.shape[0]

        # Precompute position costs for each candidate edge (vectorized)
        pos_cost_e = (delta_pos[i_idx, f_idx] ** 2)

        # --- Heuristic big-M if not given: based on data scale ---
        if M_big is None:
            max_I = float(np.max(I_obs)) if n > 0 else 1.0
            max_R = float(np.max(R)) if P > 0 else 1.0
            M_big = 10.0 * (max_I + max_R)

        # --- Build adjacency lists for constraints (small Python loops; main math is already vectorized) ---
        edges_by_obs = [[] for _ in range(n)]      # for each i: list of e
        for e, i in enumerate(i_idx):
            edges_by_obs[i].append(e)

        edges_by_ref_peak = [[] for _ in range(P)]  # for each f: list of e
        for e, f in enumerate(f_idx):
            edges_by_ref_peak[f].append(e)

        # --- MILP model ---
        prob = pulp.LpProblem("spectrum_reference_matching", pulp.LpMinimize)

        # Decision variables
        # x_e: binary assignment for candidate edge e
        x_vars = [pulp.LpVariable(f"x_{e}", lowBound=0, upBound=1, cat="Binary") for e in range(E)]
        # s_e: nonnegative residual for intensity mismatch
        s_vars = [pulp.LpVariable(f"s_{e}", lowBound=0, cat="Continuous") for e in range(E)]
        # u_i: binary noise indicator for observed peaks
        u_vars = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]
        # v_f: binary hidden indicator for reference peaks
        v_vars = [pulp.LpVariable(f"v_{f}", lowBound=0, upBound=1, cat="Binary") for f in range(P)]
        # c_m: nonnegative component fractions
        c_vars = [pulp.LpVariable(f"c_{m}", lowBound=0, cat="Continuous") for m in range(M)]

        # --- Constraints ---

        # 1) Each observed peak is either assigned to one reference peak or noise
        for i in range(n):
            prob += (
                pulp.lpSum(x_vars[e] for e in edges_by_obs[i]) + u_vars[i] == 1,
                f"obs_assign_{i}",
            )

        # 2) Each reference peak can be matched to at most one observed peak
        for f in range(P):
            prob += (
                pulp.lpSum(x_vars[e] for e in edges_by_ref_peak[f]) <= 1,
                f"ref_peak_once_{f}",
            )

        # 3) Absolute intensity residual constraints for each candidate edge
        for e in range(E):
            i = int(i_idx[e])
            f = int(f_idx[e])
            m = int(ref_id[f])
            I_i = float(I_obs[i])
            R_f = float(R[f])

            # s_e >= I_i - c_m * R_f - M*(1 - x_e)
            prob += (
                s_vars[e]
                >= I_i - c_vars[m] * R_f - M_big * (1 - x_vars[e]),
                f"s_pos_{e}",
            )
            # s_e >= c_m * R_f - I_i - M*(1 - x_e)
            prob += (
                s_vars[e]
                >= c_vars[m] * R_f - I_i - M_big * (1 - x_vars[e]),
                f"s_neg_{e}",
            )

        # 4) Detection limit / missing expected peaks constraints for each reference peak f
        for f in range(P):
            m = int(ref_id[f])
            R_f = float(R[f])
            edges_f = edges_by_ref_peak[f]

            # c_m * R_f <= L + M * (v_f + sum_e x_e)
            prob += (
                c_vars[m] * R_f
                <= detection_limit
                + M_big * (v_vars[f] + pulp.lpSum(x_vars[e] for e in edges_f)),
                f"det_limit_{f}",
            )

            # Optional: don't allow a peak to be both matched and hidden
            prob += (
                v_vars[f] + pulp.lpSum(x_vars[e] for e in edges_f) <= 1,
                f"hidden_or_matched_{f}",
            )

        # --- Objective: position + intensity costs + noise + hidden peaks ---
        objective_terms = []

        # Per-edge terms
        for e in range(E):
            objective_terms.append(w_pos * float(pos_cost_e[e]) * x_vars[e])
            objective_terms.append(w_int * s_vars[e])

        # Noise penalties
        for i in range(n):
            objective_terms.append(alpha * u_vars[i])

        # Hidden reference peak penalties
        for f in range(P):
            objective_terms.append(beta * v_vars[f])

        prob += pulp.lpSum(objective_terms)

        # --- Solve ---
        if solver is None:
            solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]

        # --- Extract solution ---
        def _safe_val(var):
            val = pulp.value(var)
            return float(val) if val is not None and np.isfinite(val) else 0.0

        c_est = np.array([_safe_val(c_vars[m]) for m in range(M)], dtype=float)
        noise_mask = np.array([_safe_val(u_vars[i]) > 0.5 for i in range(n)], dtype=bool)
        hidden_ref_mask = np.array([_safe_val(v_vars[f]) > 0.5 for f in range(P)], dtype=bool)

        # For each observed peak, find its assigned reference (if any)
        assignments = [None] * n
        element_matches: Dict[str, Element] = {}
        for e in range(E):
            if _safe_val(x_vars[e]) > 0.5:
                i = int(i_idx[e])
                f = int(f_idx[e])
                m = int(ref_id[f])
                j = int(peak_id[f])
                el = ref_elements[m]
                ion = ref_ions[m]
                if el is not None and ion is not None:
                    loc_ref = q[f]
                    element_obj = element_matches.setdefault(el, Element(el))
                    element_obj.update([el, ion, loc_ref, p_obs[i]], i)
                assignments[i] = {
                    "ref_index": m,
                    "ref_peak_index": j,
                    "ref_peak_global_index": f,
                }

        result = {
            "status": status,
            "c": c_est,
            "assignments": assignments,
            "noise_mask": noise_mask,
            "hidden_ref_mask": hidden_ref_mask,
            "objective_value": float(pulp.value(prob.objective)),
            "flattened_reference": {
                "pos": q,
                "intensity": R,
                "ref_index": ref_id,
                "ref_peak_index": peak_id,
                "ref_element": np.asarray(ref_elements, dtype=object),
                "ref_ion": np.asarray(ref_ions, dtype=object),
            },
            "elements": element_matches,
        }

        return result


class Element:
    """Container for matched element peaks."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Nested mapping ion -> {peak_idx: np.array([ref_location, obs_location])}
        self.ions: Dict[float, Dict[int, np.ndarray]] = {}


    def update(self, data, idx: int) -> None:
        """Record a matched peak for an ion."""
        el, ion, location, x, *_ = data
        if el != self.name:
            raise ValueError(f"Element mismatch: container {self.name}, data {el}")

        ion_dict = self.ions.setdefault(ion, {})
        if idx not in ion_dict:
            ion_dict[idx] = np.array([location, x])
