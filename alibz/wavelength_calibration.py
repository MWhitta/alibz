"""Apply independent gas calibrations over their measured wavelength support.

Gas offsets are absolute residuals of the incoming instrument wavelength axis,
not increments to an Fe calibration or the mixed-element residual estimator.
The instrument calibration and raw spectrum are never modified here.
"""
from __future__ import annotations

import numpy as np

from alibz.utils.wavelength import shift_at


class RegionalShift:
    """A baseline shift with independently calibrated regional replacements.

    ``at`` takes observed wavelengths. ``at_in_frame`` also accepts database
    coordinates, so anchors at interval boundaries use the correct offset.
    Legacy scalar/segment attributes describe the baseline; ``regions`` is the
    authoritative description of applied gas corrections.
    """

    def __init__(self, baseline, regions):
        self.baseline = baseline
        self.regions = tuple(dict(region) for region in regions)

    def __float__(self):
        return float(self.baseline)

    def __getattr__(self, name):
        # Avoid recursion during pickle/deepcopy reconstruction.
        if name == "baseline":
            raise AttributeError(name)
        return getattr(self.baseline, name)

    def at(self, wl):
        return self.at_in_frame(wl, frame="observed")

    def _database_offsets(self, values):
        """Continuous forward correction, constant gas offset in each core."""
        result = np.broadcast_to(np.asarray(shift_at(self.baseline, values),
                                             dtype=float), values.shape).copy()
        for region in self.regions:
            lo, hi = region["supported_range_nm"]
            width = region["transition_width_nm"]
            weight = np.clip(np.minimum((values - lo) / width,
                                       (hi - values) / width), 0., 1.)
            use = (values >= lo) & (values <= hi)
            result = np.where(use, (1 - weight) * result +
                              weight * region["offset_nm"], result)
        return result

    def at_in_frame(self, wl, *, frame="observed"):
        if frame not in {"observed", "database"}:
            raise ValueError("wavelength frame must be observed or database")
        values = np.asarray(wl, dtype=float)
        if frame == "database":
            result = self._database_offsets(values)
            return float(result) if values.ndim == 0 else result
        result = np.broadcast_to(np.asarray(shift_at(self.baseline, values),
                                             dtype=float), values.shape).copy()
        for region in self.regions:
            lo, hi = region["observed_range_nm"]
            use = (values >= lo) & (values <= hi)
            # |d shift / d database wavelength| <= .25 in the taper,
            # so this inversion is a contraction. Only invert supported images;
            # baseline behavior outside gas regions stays exactly unchanged.
            guess = values - result
            for _ in range(24):
                guess = np.where(use, values - self._database_offsets(guess), guess)
            result = np.where(use, values - guess, result)
        return float(result) if values.ndim == 0 else result


def _regions(calibrations):
    """Only independent, finite, bracketed segment estimates may be applied."""
    regions = []
    for element in ("Ar", "O"):
        record = calibrations.get(element, {})
        if record.get("status") != "calibrated":
            continue
        minimum = 3 if element == "Ar" else 2
        if int(record.get("n_inliers", 0)) < minimum:
            continue
        for part in record.get("segments", [record]):
            if part.get("status") != "calibrated" or int(part.get("n_inliers", 0)) < 2:
                continue
            try:
                lo, hi = map(float, part["supported_range_nm"])
                offset = float(part["offset_nm"])
                uncertainty = float(part["uncertainty_nm"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in (lo, hi, offset, uncertainty)):
                continue
            if lo >= hi or uncertainty <= 0:
                continue
            regions.append(dict(source=element, supported_range_nm=[lo, hi],
                                offset_nm=offset, uncertainty_nm=uncertainty,
                                n_inliers=int(part["n_inliers"])))
    return regions


def apply_gas_calibrations(baseline, calibrations, *, mode="apply"):
    """Return effective shift and JSON application evidence.

    Overlapping Ar/O estimates must agree within three combined standard
    uncertainties. If they agree, use the more precise estimate without
    averaging correlated measurements from the same detector. Conflicts retain
    the baseline in the overlap. No correction is extrapolated beyond inlier
    anchor ranges or across detector segments.
    """
    if mode not in {"off", "report", "apply"}:
        raise ValueError("gas_wavelength_calibration must be off, report, or apply")
    candidates = _regions(calibrations) if mode != "off" else []
    bounds = sorted({v for region in candidates for v in region["supported_range_nm"]})
    selected, conflicts = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        middle = (lo + hi) / 2
        covering = [r for r in candidates if r["supported_range_nm"][0] <= middle
                    <= r["supported_range_nm"][1]]
        if not covering:
            continue
        inconsistent = any(
            abs(a["offset_nm"] - b["offset_nm"]) >
            3 * np.hypot(a["uncertainty_nm"], b["uncertainty_nm"])
            for i, a in enumerate(covering) for b in covering[i + 1:])
        if inconsistent:
            conflicts.append(dict(supported_range_nm=[lo, hi],
                                  reason="independent_gas_offsets_disagree",
                                  estimates=covering))
            continue
        best = min(covering, key=lambda r: (r["uncertainty_nm"], r["source"]))
        selected.append(dict(best, supported_range_nm=[lo, hi],
                             corroborating_sources=sorted({r["source"] for r in covering})))
    # Merge neighboring pieces with the same selected reference so agreement
    # boundaries do not create unnecessary transitions back to the baseline.
    merged = []
    for region in selected:
        if (merged and merged[-1]["supported_range_nm"][1] == region["supported_range_nm"][0]
                and all(merged[-1][key] == region[key]
                        for key in ("source", "offset_nm", "uncertainty_nm"))):
            merged[-1]["supported_range_nm"][1] = region["supported_range_nm"][1]
            merged[-1]["corroborating_sources"] = sorted(
                set(merged[-1]["corroborating_sources"]) & set(region["corroborating_sources"]))
        else:
            merged.append(dict(region, supported_range_nm=list(region["supported_range_nm"])))
    applicable, unavailable = [], []
    for region in merged:
        lo, hi = region["supported_range_nm"]
        if any(lo < edge < hi for edge in getattr(baseline, "edges", ())):
            unavailable.append(dict(region, reason="region_crosses_detector_segment"))
            continue
        base_lo, base_hi = float(shift_at(baseline, lo)), float(shift_at(baseline, hi))
        width = max(.5, 4 * max(abs(region["offset_nm"] - base_lo),
                               abs(region["offset_nm"] - base_hi)))
        if hi - lo <= 2 * width or base_lo != base_hi:
            unavailable.append(dict(region, reason="insufficient_span_for_continuous_transition"))
            continue
        applicable.append(dict(region, transition_width_nm=width,
                               full_offset_range_nm=[lo + width, hi - width],
                               observed_range_nm=[lo + base_lo, hi + base_hi]))
    applied = applicable if mode == "apply" else []
    baseline_record = dict(
        source="mixed_element_residual_estimator",
        global_offset_nm=float(baseline),
        segment_edges_nm=list(getattr(baseline, "edges", ())),
        segment_offsets_nm=[float(v) for v in getattr(baseline, "shifts", (float(baseline),))],
        anchor_counts=list(getattr(baseline, "n_matches", ())),
    )
    report = dict(mode=mode, offset_convention="observed_minus_database_nm",
                  input_instrument_calibration_modified=False,
                  baseline=baseline_record, gases=calibrations,
                  proposed_regions=selected, applied_regions=applied,
                  unavailable_regions=unavailable,
                  conflicts=conflicts,
                  applied=bool(applied),
                  outside_support="baseline", conflict_policy="baseline_in_overlap",
                  boundary_policy="linear_transition_inside_support; full gas offset in interior")
    return (RegionalShift(baseline, applied) if applied else baseline), report
