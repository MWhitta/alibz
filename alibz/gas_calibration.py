"""Independent wavelength calibration from neutral Ar I and O I lines.

The offsets returned here are measured in the incoming instrument frame:

    observed wavelength = database wavelength + offset

No Fe or mixed-element shift is accepted by this API.  Multiplets are one
independent anchor.  A multiplet contributes a precise offset only when at
least two of its components are resolved at a common displacement; otherwise
its centroid is explicitly reported as ambiguous and is not used.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from alibz.gas_detection import (
    _AR_GROUPS,
    _O_GROUPS,
    _database_groups,
    _externally_supported,
    _strong_line_catalog,
)


_DEFAULTS = {
    "max_abs_offset_nm": 0.50,
    "min_snr": 6.0,
    "noise_floor": 1.0e-12,
    "consistency_nm": 0.08,
    "component_consistency_nm": 0.055,
    "component_match_nm": 0.075,
    "min_center_uncertainty_nm": 0.004,
    "max_gap_nm": 0.30,
    "segment_edges_nm": (365.0, 620.0),
    "saturation_ceiling": None,
    "competitor_match_nm": 0.10,
    "competitor_min_groups": 3,
    "competitor_min_coverage": 0.0,
    "competitor_strong_fraction": 0.10,
    "competitor_band_nm": 100.0,
    "competitor_temperature_K": 10_000.0,
    "competitor_elements": (),
    "check_competitors": True,
}


def _config(config: Mapping | None) -> dict:
    out = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        unknown = set(config) - set(out)
        if unknown:
            raise ValueError(f"unknown gas-calibration config keys: {sorted(unknown)}")
        out.update(config)
    out["segment_edges_nm"] = tuple(
        float(v) for v in np.sort(np.asarray(out["segment_edges_nm"], dtype=float))
    )
    if float(out["max_abs_offset_nm"]) <= 0:
        raise ValueError("max_abs_offset_nm must be positive")
    return out


def _validate_spectrum(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 5:
        raise ValueError("x and y must be equal-length one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


def _local_pitch(x: np.ndarray, wavelength: float) -> float:
    use = (x >= wavelength - 2.0) & (x <= wavelength + 2.0)
    dx = np.diff(x[use])
    if not dx.size:
        dx = np.diff(x)
    return float(np.median(dx))


def _robust_noise(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return float(floor)
    mad = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    diff = np.abs(np.diff(values))
    diff_sigma = (float(np.percentile(diff, 25)) / 0.4506241100
                  if diff.size else 0.0)
    good = [v for v in (mad, diff_sigma) if np.isfinite(v) and v > 0]
    return max(float(np.median(good)) if good else 0.0, float(floor))


def _discover_peaks(x: np.ndarray, y: np.ndarray, cfg: dict) -> np.ndarray:
    """Conservative native-grid maxima when a fitted peak table is absent."""
    maxima = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    records = []
    for i in maxima:
        pitch = _local_pitch(x, float(x[i]))
        side = ((x >= x[i] - max(0.60, 5 * pitch)) &
                (x <= x[i] + max(0.60, 5 * pitch)) &
                (np.abs(x - x[i]) >= max(0.18, 2 * pitch)))
        if np.sum(side) < 5:
            continue
        base = float(np.median(y[side]))
        noise = _robust_noise(y[side], float(cfg["noise_floor"]))
        height = float(y[i] - base)
        if height / noise < float(cfg["min_snr"]):
            continue
        # Three-cell parabolic vertex supplies a sub-cell center.  The native
        # cell still remains as an uncertainty floor below.
        xx = x[i - 1:i + 2]
        yy = y[i - 1:i + 2]
        coeff = np.polyfit(xx - x[i], yy, 2)
        delta = (-coeff[1] / (2 * coeff[0])
                 if coeff[0] < 0 else 0.0)
        if abs(delta) > pitch:
            delta = 0.0
        records.append((height, float(x[i] + delta), pitch, 0.0))
    return np.asarray(records, dtype=float).reshape((-1, 4))


def _validate_peaks(peak_array, x, y, cfg) -> tuple[np.ndarray, str]:
    if peak_array is None:
        return _discover_peaks(x, y, cfg), "native_local_maxima"
    peaks = np.asarray(peak_array, dtype=float)
    if peaks.size == 0:
        return np.empty((0, 4), dtype=float), "supplied_observed_frame"
    if peaks.ndim != 2 or peaks.shape[1] < 2 or not np.all(np.isfinite(peaks[:, :2])):
        raise ValueError("peak_array must be finite 2-D rows [amplitude, observed_center, ...]")
    if peaks.shape[1] < 4:
        peaks = np.column_stack((peaks[:, :2], np.zeros((len(peaks), 2))))
    return np.asarray(peaks[:, :4], dtype=float), "supplied_observed_frame"


def _window_evidence(x, y, observed_positions, cfg) -> dict:
    positions = np.asarray(observed_positions, dtype=float)
    center = float(np.mean(positions))
    pitch = _local_pitch(x, center)
    margin = max(0.10, 1.25 * pitch)
    lo, hi = float(np.min(positions) - margin), float(np.max(positions) + margin)
    inner = max(0.16, 1.2 * pitch)
    outer = inner + max(0.40, 6.0 * pitch)
    core = (x >= lo) & (x <= hi)
    side = ((x >= lo - outer) & (x <= lo - inner)) | \
           ((x >= hi + inner) & (x <= hi + outer))
    base = {
        "covered": False,
        "snr": None,
        "local_pitch_nm": float(pitch),
        "native_cell_floor_nm": float(pitch / np.sqrt(12.0)),
        "flat_top": False,
        "measurement_window_nm": [lo, hi],
        "reasons": [],
    }
    if np.sum(core) < 2 or np.sum(side) < 5:
        base["reasons"].append("insufficient_local_coverage")
        return base
    gap_limit = max(float(cfg["max_gap_nm"]), 2.5 * pitch)
    if np.any(np.diff(x[(x >= lo - outer) & (x <= hi + outer)]) > gap_limit):
        base["reasons"].append("coverage_gap")
        return base
    base["covered"] = True
    baseline = float(np.median(y[side]))
    noise = _robust_noise(y[side], float(cfg["noise_floor"]))
    height = float(np.max(y[core]) - baseline)
    base["snr"] = float(height / noise)
    repeated = int(np.sum(np.isclose(y[core], np.max(y[core]), rtol=0, atol=1e-12)))
    ceiling = cfg["saturation_ceiling"]
    base["flat_top"] = bool(
        repeated >= 3 or
        (ceiling is not None and float(np.max(y[core])) >= float(ceiling))
    )
    if base["snr"] < float(cfg["min_snr"]):
        base["reasons"].append("below_snr_threshold")
    if base["flat_top"]:
        base["reasons"].append("flat_top_or_saturation")
    return base


def _native_maximum_matches(x, y, positions):
    """Match proposed fitted centers to distinct maxima on the native grid.

    This prevents a fitted/deblended center beside a real peak from borrowing
    that peak's SNR.  On a coarse detector grid the allowed distance is still
    less than one cell and is carried into the center uncertainty.
    """
    positions = np.asarray(positions, dtype=float)
    maxima = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    if not maxima.size:
        return None
    matched, used = [], set()
    for position in positions:
        pitch = _local_pitch(x, float(position))
        tolerance = max(0.015, 0.75 * pitch)
        order = np.argsort(np.abs(x[maxima] - position))
        choice = None
        for k in order:
            native_i = int(maxima[k])
            if native_i in used:
                continue
            distance = abs(float(x[native_i] - position))
            if distance <= tolerance:
                choice = (native_i, distance)
                break
        if choice is None:
            return None
        used.add(choice[0])
        matched.append(choice)
    return matched


def _assign_components(db_wl, peak_centers, seed, cfg):
    """Greedy one-to-one component assignment at a proposed common offset."""
    choices = []
    for line_i, wl in enumerate(db_wl):
        distance = np.abs(peak_centers - (wl + seed))
        for peak_i in np.argsort(distance):
            if distance[peak_i] <= float(cfg["component_match_nm"]):
                choices.append((float(distance[peak_i]), line_i, int(peak_i)))
            else:
                break
    assigned_lines, assigned_peaks, pairs = set(), set(), []
    for _, line_i, peak_i in sorted(choices):
        if line_i in assigned_lines or peak_i in assigned_peaks:
            continue
        assigned_lines.add(line_i)
        assigned_peaks.add(peak_i)
        pairs.append((line_i, peak_i))
    return pairs


def _candidate_records(group, peaks, x, y, cfg) -> tuple[list[dict], dict]:
    db_wl = np.asarray(group["wavelengths"], dtype=float)
    peak_centers = peaks[:, 1] if peaks.size else np.empty(0)
    max_shift = float(cfg["max_abs_offset_nm"])
    search_lo = float(db_wl.min() - max_shift)
    search_hi = float(db_wl.max() + max_shift)
    local_peak_idx = np.flatnonzero((peak_centers >= search_lo) & (peak_centers <= search_hi))
    pitch = _local_pitch(x, float(np.mean(db_wl)))
    base = {
        "group": str(group["name"]),
        "database_wavelengths_nm": [float(v) for v in db_wl],
        "database_anchor_nm": float(np.mean(db_wl)),
        "database_line_count": int(db_wl.size),
        "covered": bool(x[0] <= search_lo and x[-1] >= search_hi),
        "status": "no_candidate",
        "candidate_count": 0,
        "selected": False,
        "observed_positions_nm": [],
        "offset_nm": None,
        "uncertainty_nm": None,
        "snr": None,
        "blend_elements": [],
        "flags": [],
        "reasons": [],
    }
    if not base["covered"]:
        base["reasons"].append("search_window_out_of_range")
        return [], base
    if local_peak_idx.size == 0:
        base["reasons"].append("no_observed_peak_in_search_window")
        return [], base

    candidates = []
    if db_wl.size == 1:
        for peak_i in local_peak_idx:
            delta = float(peak_centers[peak_i] - db_wl[0])
            evidence = _window_evidence(x, y, [peak_centers[peak_i]], cfg)
            if (not evidence["covered"] or evidence["flat_top"] or
                    evidence["snr"] < float(cfg["min_snr"])):
                continue
            native_matches = _native_maximum_matches(
                x, y, [peak_centers[peak_i]])
            if native_matches is None:
                continue
            width = abs(float(peaks[peak_i, 2])) + abs(float(peaks[peak_i, 3]))
            uncertainty = max(
                evidence["native_cell_floor_nm"],
                float(cfg["min_center_uncertainty_nm"]),
                width / max(float(evidence["snr"]), 1.0),
                float(native_matches[0][1]),
            )
            candidates.append({
                "group": group["name"], "database_anchor_nm": float(db_wl[0]),
                "database_wavelengths_nm": [float(db_wl[0])],
                "observed_positions_nm": [float(peak_centers[peak_i])],
                "offset_nm": delta, "uncertainty_nm": float(uncertainty),
                "snr": float(evidence["snr"]), "peak_indices": [int(peak_i)],
                "flags": [], "blend_elements": [],
            })
    else:
        # A precise multiplet offset must come from at least two distinct
        # observed components at one common displacement.  A single unresolved
        # centroid has an unknown intensity weighting and is never assigned to
        # the strongest component or a fixed-ratio centroid.
        seeds = []
        for peak_i in local_peak_idx:
            for wl in db_wl:
                delta = float(peak_centers[peak_i] - wl)
                if abs(delta) <= max_shift:
                    seeds.append(delta)
        seen = set()
        for seed in seeds:
            pairs = _assign_components(db_wl, peak_centers, seed, cfg)
            if len(pairs) < 2:
                continue
            deltas = np.asarray([peak_centers[p] - db_wl[l] for l, p in pairs])
            offset = float(np.median(deltas))
            if abs(offset) > max_shift:
                continue
            if (np.max(np.abs(deltas - offset)) >
                    float(cfg["component_consistency_nm"])):
                continue
            key = tuple(sorted(p for _, p in pairs))
            if key in seen:
                continue
            seen.add(key)
            observed = [float(peak_centers[p]) for _, p in pairs]
            native_matches = _native_maximum_matches(x, y, observed)
            if native_matches is None:
                # Multiple fitted centers mapped to one unresolved native
                # maximum are not independent component measurements.
                continue
            evidence = _window_evidence(x, y, observed, cfg)
            if (not evidence["covered"] or evidence["flat_top"] or
                    evidence["snr"] < float(cfg["min_snr"])):
                continue
            scatter = (1.4826 * float(np.median(np.abs(deltas - offset)))
                       if len(deltas) > 2 else float(np.ptp(deltas)) / 2.0)
            uncertainty = max(
                evidence["native_cell_floor_nm"] / np.sqrt(len(pairs)),
                float(cfg["min_center_uncertainty_nm"]), scatter,
                max(distance for _, distance in native_matches),
            )
            candidates.append({
                "group": group["name"],
                "database_anchor_nm": float(np.mean(db_wl[[l for l, _ in pairs]])),
                "database_wavelengths_nm": [float(db_wl[l]) for l, _ in pairs],
                "observed_positions_nm": observed,
                "offset_nm": offset, "uncertainty_nm": float(uncertainty),
                "snr": float(evidence["snr"]),
                "peak_indices": [int(p) for _, p in pairs],
                "flags": ["resolved_multiplet"], "blend_elements": [],
            })
        if not candidates:
            base["status"] = "ambiguous_unresolved_multiplet"
            base["flags"].append("unresolved_centroid_not_used")
            base["reasons"].append(
                "multiplet_requires_two_resolved_components_at_common_offset")
            base["uncertainty_nm"] = float(
                max(np.ptp(db_wl) / 2.0, pitch / np.sqrt(12.0)))

    base["candidate_count"] = int(len(candidates))
    if candidates:
        base["status"] = "candidate"
    elif base["status"] == "no_candidate":
        base["reasons"].append("no_clean_high_snr_candidate")
    return candidates, base


def _consensus(candidates_by_group, required, cfg):
    all_candidates = [c for cs in candidates_by_group.values() for c in cs]
    if not all_candidates:
        return None, []
    tol = float(cfg["consistency_nm"])
    solutions = []
    for seed in [c["offset_nm"] for c in all_candidates]:
        chosen = []
        for candidates in candidates_by_group.values():
            near = [c for c in candidates if abs(c["offset_nm"] - seed) <= tol]
            if near:
                chosen.append(min(near, key=lambda c: abs(c["offset_nm"] - seed)))
        if not chosen:
            continue
        median = float(np.median([c["offset_nm"] for c in chosen]))
        chosen = [c for c in chosen if abs(c["offset_nm"] - median) <= tol]
        if not chosen:
            continue
        offsets = np.asarray([c["offset_nm"] for c in chosen])
        score = (len(chosen), float(np.median([c["snr"] for c in chosen])),
                 -float(np.ptp(offsets)) if len(offsets) > 1 else 0.0)
        solutions.append((score, median, chosen))
    solutions.sort(key=lambda item: item[0], reverse=True)
    best = solutions[0]
    # Distinct equally populated solutions indicate a line-assignment alias.
    aliases = [s for s in solutions[1:]
               if s[0][0] == best[0][0] and
               abs(s[1] - best[1]) > tol and
               s[0][1] >= 0.8 * best[0][1]]
    if aliases:
        return None, ["ambiguous_common_offset_solutions"]
    chosen = best[2]
    if len(chosen) < required:
        return None, [f"requires_{required}_independent_groups"]
    offsets = np.asarray([c["offset_nm"] for c in chosen])
    median = float(np.median(offsets))
    if abs(median) > float(cfg["max_abs_offset_nm"]):
        return None, ["common_offset_outside_configured_bound"]
    if len(offsets) >= 3:
        loo = [float(np.median(np.delete(offsets, i))) for i in range(len(offsets))]
        if max(abs(v - median) for v in loo) > tol / 2.0:
            return None, ["leave_one_out_offset_instability"]
    scatter = (1.4826 * float(np.median(np.abs(offsets - median)))
               if len(offsets) >= 3 else float(np.ptp(offsets)) / 2.0)
    floors = [float(c["uncertainty_nm"]) for c in chosen]
    uncertainty = max(float(1.2533 * np.sqrt(np.sum(np.square(floors))) /
                            len(floors)),
                      1.2533 * scatter / np.sqrt(len(offsets)),
                      float(cfg["min_center_uncertainty_nm"]))
    return {"offset_nm": median, "uncertainty_nm": uncertainty,
            "inliers": chosen}, []


def _summarize_inliers(inliers, cfg):
    """Recompute a solution after an interference check removes anchors."""
    offsets = np.asarray([c["offset_nm"] for c in inliers], dtype=float)
    median = float(np.median(offsets))
    scatter = (1.4826 * float(np.median(np.abs(offsets - median)))
               if len(offsets) >= 3 else float(np.ptp(offsets)) / 2.0)
    floors = np.asarray([c["uncertainty_nm"] for c in inliers], dtype=float)
    uncertainty = max(float(1.2533 * np.sqrt(np.sum(floors ** 2)) /
                            len(floors)),
                      1.2533 * scatter / np.sqrt(len(offsets)),
                      float(cfg["min_center_uncertainty_nm"]))
    return {"offset_nm": median, "uncertainty_nm": uncertainty,
            "inliers": list(inliers)}


def _mark_competitors(solution, db, peaks, x_range, cfg, element):
    if solution is None or not bool(cfg["check_competitors"]):
        return solution
    catalog = _strong_line_catalog(db, x_range, cfg)
    centers_db = peaks[:, 1] - float(solution["offset_nm"])
    explicit = {str(v) for v in cfg["competitor_elements"]}
    retained, rejected = [], []
    for candidate in solution["inliers"]:
        group_wl = np.asarray(candidate["database_wavelengths_nm"])
        lo = float(np.min(group_wl) - 0.12)
        hi = float(np.max(group_wl) + 0.12)
        blends = []
        for other, lines in catalog.items():
            if other == element or not np.any((lines[:, 0] >= lo) & (lines[:, 0] <= hi)):
                continue
            if other in explicit or _externally_supported(lines, centers_db, lo, hi, cfg):
                blends.append(other)
        candidate["blend_elements"] = sorted(blends)
        if not blends:
            retained.append(candidate)
        else:
            rejected.append(candidate)
    solution["inliers"] = retained
    solution["rejected_blends"] = rejected
    return solution


def _calibrate_element(element, definitions, x, y, db, peaks, peak_source, cfg):
    groups = _database_groups(db, element, definitions)
    candidates_by_group, group_records = {}, []
    for group in groups:
        candidates, record = _candidate_records(group, peaks, x, y, cfg)
        candidates_by_group[group["name"]] = candidates
        group_records.append(record)
    covered = [g for g in group_records if g["covered"]]
    candidates = [g for g in group_records if g["candidate_count"]]
    required_global = 3 if element == "Ar" else 2
    edges = cfg["segment_edges_nm"]
    segment_ids = sorted(set(
        int(np.digitize(float(g["database_anchor_nm"]), edges)) for g in group_records
    ))
    segments, selected_names, diagnostic_reasons = [], set(), []
    for segment_id in segment_ids:
        names = {
            g["group"] for g in group_records
            if int(np.digitize(float(g["database_anchor_nm"]), edges)) == segment_id
        }
        subset = {name: records for name, records in candidates_by_group.items()
                  if name in names}
        # A segment is bracketed by at least two independent anchors.  The
        # element-level Ar gate is stricter (three inliers total), evaluated
        # after all independently calibrated detector segments are collected.
        minimum = 2
        solution, reasons = _consensus(subset, minimum, cfg)
        provisional = list(solution["inliers"]) if solution is not None else []
        solution = _mark_competitors(
            solution, db, peaks, (float(x[0]), float(x[-1])), cfg, element)
        if solution is not None:
            for rejected in solution.get("rejected_blends", []):
                record = next(g for g in group_records
                              if g["group"] == rejected["group"])
                record["status"] = "rejected_blend"
                record["blend_elements"] = list(rejected["blend_elements"])
                record["reasons"].append("supported_competitor_in_window")
        if solution is not None and len(solution["inliers"]) < minimum:
            reasons = ["supported_competitor_removed_required_anchor"]
            solution = None
        elif solution is not None:
            solution = _summarize_inliers(solution["inliers"], cfg)
        if solution is None:
            diagnostic_reasons.extend(reasons)
            continue
        anchors = sorted(solution["inliers"], key=lambda c: c["database_anchor_nm"])
        # Two anchors are the absolute minimum for a bounded interval.  No
        # offset is licensed beyond the first/last inlier in this segment.
        if len(anchors) < 2:
            continue
        supported = [float(anchors[0]["database_anchor_nm"]),
                     float(anchors[-1]["database_anchor_nm"])]
        segments.append({
            "segment_index": int(segment_id),
            "status": "calibrated",
            "offset_nm": float(solution["offset_nm"]),
            "uncertainty_nm": float(solution["uncertainty_nm"]),
            "supported_range_nm": supported,
            "n_inliers": int(len(anchors)),
            "anchor_groups": [str(a["group"]) for a in anchors],
            "database_anchor_positions_nm":
                [float(a["database_anchor_nm"]) for a in anchors],
            "observed_anchor_positions_nm":
                [[float(v) for v in a["observed_positions_nm"]] for a in anchors],
        })
        for anchor in anchors:
            selected_names.add(str(anchor["group"]))
            record = next(g for g in group_records if g["group"] == anchor["group"])
            record.update({
                "selected": True, "status": "inlier",
                "observed_positions_nm": [float(v) for v in anchor["observed_positions_nm"]],
                "offset_nm": float(anchor["offset_nm"]),
                "uncertainty_nm": float(anchor["uncertainty_nm"]),
                "snr": float(anchor["snr"]),
                "blend_elements": list(anchor["blend_elements"]),
                "flags": list(anchor["flags"]),
            })

    n_inliers = int(sum(s["n_inliers"] for s in segments))
    calibrated = bool(segments and n_inliers >= required_global)
    if not calibrated:
        # Preserve bracket evidence for diagnostics while making it impossible
        # for an integration consumer to mistake a two-line Ar hint for an
        # applicable calibration.
        for segment in segments:
            segment["status"] = "tentative"
    if calibrated:
        weights = np.asarray([1.0 / max(s["uncertainty_nm"], 1e-9) ** 2
                              for s in segments])
        offsets = np.asarray([s["offset_nm"] for s in segments])
        top_offset = float(np.sum(weights * offsets) / np.sum(weights))
        top_uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        supported_range = [float(min(s["supported_range_nm"][0] for s in segments)),
                           float(max(s["supported_range_nm"][1] for s in segments))]
        status, reasons = "calibrated", []
    elif not covered:
        top_offset = top_uncertainty = supported_range = None
        status, reasons = "out_of_range", ["no_complete_anchor_search_window"]
    elif not candidates:
        top_offset = top_uncertainty = supported_range = None
        status, reasons = "no_evidence", ["no_clean_resolved_anchor_candidates"]
    else:
        # Candidate offsets are diagnostic only and are never applied.
        raw_candidates = [c for values in candidates_by_group.values() for c in values]
        top_offset = top_uncertainty = supported_range = None
        status = "tentative"
        reasons = diagnostic_reasons or [f"requires_{required_global}_consistent_groups"]

    candidate_offsets = [c["offset_nm"] for values in candidates_by_group.values()
                         for c in values]
    candidate_offset = (float(np.median(candidate_offsets))
                        if candidate_offsets else None)
    return {
        "element": element,
        "stage": 1,
        "status": status,
        "offset_convention": "observed_minus_database_nm",
        "offset_nm": top_offset,
        "candidate_offset_nm": candidate_offset if not calibrated else top_offset,
        "uncertainty_nm": top_uncertainty,
        "supported_range_nm": supported_range,
        "n_database_groups": int(len(groups)),
        "n_covered_groups": int(len(covered)),
        "n_candidate_groups": int(len(candidates)),
        "n_inliers": n_inliers,
        "peak_source": peak_source,
        "independent_of_internal_calibration": True,
        "database_wavelength_convention": "repository_air_above_200_nm",
        "segments": segments,
        "groups": group_records,
        "reasons": list(dict.fromkeys(reasons)),
    }


def calibrate_background_gases(x, y, db, *, peak_array=None, config=None) -> dict:
    """Estimate independent Ar I and O I offsets from the raw observed frame.

    ``peak_array`` centers, when supplied, must be in the original observed
    instrument frame.  This function deliberately has no shift/prior argument.
    Returned offsets are absolute residuals relative to the repository's air
    wavelengths and must replace, rather than be added to, another calibration.
    """
    x, y = _validate_spectrum(x, y)
    cfg = _config(config)
    peaks, peak_source = _validate_peaks(peak_array, x, y, cfg)
    return {
        "Ar": _calibrate_element(
            "Ar", _AR_GROUPS, x, y, db, peaks, peak_source, cfg),
        "O": _calibrate_element(
            "O", _O_GROUPS, x, y, db, peaks, peak_source, cfg),
    }


__all__ = ["calibrate_background_gases"]
