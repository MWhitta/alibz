"""Conservative early detection of common Ar I and O I background features.

The detector works on raw native-grid spectra before whole-pattern indexing.
It groups unresolved multiplets into independent observable features, measures
each feature against local sidebands, and uses the atomic database only to
select quantitative stage-I anchors.  Results are evidence records, not gas
concentrations or statements about the source of atomic oxygen.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from alibz.utils.wavelength import shift_at


# Air-wavelength search intervals.  Exact expected wavelengths and strengths
# are always taken from Database.lines(), never from these rounded labels.
_AR_GROUPS = (
    ("ArI_696", 696.45, 696.63), ("ArI_706", 706.63, 706.81),
    ("ArI_738", 738.31, 738.49), ("ArI_750", 750.30, 750.48),
    ("ArI_751", 751.37, 751.56), ("ArI_763", 763.42, 763.60),
    ("ArI_772", 772.28, 772.52), ("ArI_794", 794.73, 794.91),
    ("ArI_800", 800.52, 800.71), ("ArI_801", 801.39, 801.57),
    ("ArI_810", 810.28, 810.46), ("ArI_811", 811.44, 811.62),
    ("ArI_826", 826.36, 826.55), ("ArI_840", 840.73, 840.91),
    ("ArI_842", 842.37, 842.56), ("ArI_852", 852.05, 852.24),
    ("ArI_866", 866.70, 866.89), ("ArI_912", 912.20, 912.39),
    ("ArI_922", 922.35, 922.55),
)

_O_GROUPS = (
    ("OI_615", 615.50, 615.90),
    ("OI_645", 645.28, 645.68),
    ("OI_777", 777.08, 777.66),
    ("OI_844", 844.52, 844.78),
    ("OI_926", 925.98, 926.70),
)

_DEFAULTS = {
    "min_snr": 5.0,
    "noise_floor": 1.0,
    "feature_margin_nm": 0.10,
    "side_inner_nm": 0.16,
    "side_outer_nm": 0.45,
    "match_tolerance_nm": 0.12,
    "max_gap_nm": 0.15,
    "competitor_match_nm": 0.12,
    "competitor_min_groups": 3,
    "competitor_min_coverage": 0.0,
    "competitor_strong_fraction": 0.10,
    "competitor_band_nm": 100.0,
    "competitor_temperature_K": 10_000.0,
    "saturation_ceiling": None,
    "competitor_elements": (),
}


def _config(config: Mapping | None) -> dict:
    out = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        out.update(config)
    return out


def _validate_spectrum(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("x and y must be equal-length one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


def _database_groups(db, element: str, definitions) -> list[dict]:
    rows = np.asarray(db.lines(element))
    if rows.size == 0:
        return []
    ion = rows[:, 0].astype(float)
    wl = rows[:, 1].astype(float)
    ga = rows[:, 3].astype(float)
    ek = rows[:, 5].astype(float)
    out = []
    for name, lo, hi in definitions:
        use = (ion == 1) & (wl >= lo) & (wl <= hi) & np.isfinite(ga) & (ga > 0)
        if not np.any(use):
            continue
        order = np.argsort(wl[use])
        out.append({
            "name": name,
            "wavelengths": wl[use][order],
            "gA": ga[use][order],
            "Ek": ek[use][order],
        })
    return out


def _measure_group(x, y, group, shift_nm, cfg) -> dict:
    db_wl = np.asarray(group["wavelengths"], dtype=float)
    obs_wl = db_wl + np.asarray(shift_at(shift_nm, db_wl, frame="database"), dtype=float)
    # Native grids can be much coarser than Profile Builder exports (the Z300
    # NIR pitch is about 0.18 nm).  Size every window from the LOCAL pitch so a
    # narrow Ar line still receives at least 3--5 real detector cells.
    center = float(np.mean(obs_wl))
    local = (x >= center - 2.0) & (x <= center + 2.0)
    local_diffs = np.diff(x[local])
    local_pitch = (float(np.median(local_diffs)) if local_diffs.size
                   else float(np.median(np.diff(x))))
    margin = max(float(cfg["feature_margin_nm"]), 1.6 * local_pitch)
    inner = max(float(cfg["side_inner_nm"]), 1.1 * local_pitch)
    outer = inner + max(float(cfg["side_outer_nm"]) - float(cfg["side_inner_nm"]),
                        7.0 * local_pitch)
    core_lo, core_hi = float(obs_wl.min() - margin), float(obs_wl.max() + margin)
    left = (x >= core_lo - outer) & (x <= core_lo - inner)
    right = (x >= core_hi + inner) & (x <= core_hi + outer)
    core = (x >= core_lo) & (x <= core_hi)
    base = {
        "group": group["name"],
        "expected_wavelengths_nm": [float(v) for v in db_wl],
        "observed_frame_wavelengths_nm": [float(v) for v in obs_wl],
        "database_line_count": int(db_wl.size),
        "local_pitch_nm": float(local_pitch),
        "measurement_window_nm": [core_lo, core_hi],
        "database_measurement_window_nm": [float(db_wl.min() - margin),
                                             float(db_wl.max() + margin)],
        "n_native_samples": int(np.sum(core)),
        "covered": False,
        "supported": False,
        "clean": False,
        "matched_wavelength_nm": None,
        "area": None,
        "area_sigma": None,
        "noise_per_native_sample": None,
        "snr": None,
        "blend_elements": [],
        "flags": [],
        "reasons": [],
    }
    if np.sum(core) < 3 or np.sum(left) < 3 or np.sum(right) < 3:
        base["reasons"].append("insufficient_window_samples")
        return base
    dx = np.diff(x[core])
    gap_limit = max(float(cfg["max_gap_nm"]), 2.5 * local_pitch)
    if dx.size and float(np.max(dx)) > gap_limit:
        base["reasons"].append("coverage_gap")
        return base
    base["covered"] = True

    x_core = x[core]
    y_core = y[core]
    xl, xr = float(np.median(x[left])), float(np.median(x[right]))
    yl, yr = float(np.median(y[left])), float(np.median(y[right]))
    slope = (yr - yl) / max(xr - xl, np.finfo(float).eps)
    baseline = yl + slope * (x_core - xl)
    residual = y_core - baseline
    peak_i = int(np.argmax(residual))
    peak_wl = float(x_core[peak_i])
    # Database/calibration tolerance plus half-cell quantisation.  Using two
    # full native cells admits unrelated peaks more than 0.25 nm away.
    align_tol = float(cfg["match_tolerance_nm"]) + 0.5 * local_pitch
    nearest = int(np.argmin(np.abs(obs_wl - peak_wl)))
    offset = peak_wl - float(obs_wl[nearest])
    aligned = abs(offset) <= align_tol

    def robust_mad(values):
        values = np.asarray(values, dtype=float)
        return 1.4826 * float(np.median(np.abs(values - np.median(values))))

    # A wide local sideband supplies enough native cells for a robust scale.
    # Combine independent left-MAD, right-MAD, and difference estimates.  For
    # Gaussian noise Q25(|y[i+1]-y[i]|) = sqrt(2)*Phi^-1(.625)*sigma.
    diff = np.concatenate((np.abs(np.diff(y[left])), np.abs(np.diff(y[right]))))
    diff_noise = (float(np.percentile(diff, 25)) / 0.4506241100
                  if diff.size else 0.0)
    estimates = [robust_mad(y[left]), robust_mad(y[right]), diff_noise]
    finite_positive = [v for v in estimates if np.isfinite(v) and v > 0]
    noise = max(float(np.median(finite_positive)) if finite_positive else 0.0,
                float(cfg["noise_floor"]))

    # Align the database multiplet by the measured common offset, then sum a
    # signed local residual around each component.  On a coarse native grid a
    # line may occupy one cell; on a fine export several cells contribute.
    half_support = max(0.60 * local_pitch, 0.07)
    support = np.zeros(x_core.size, dtype=bool)
    for expected in obs_wl + offset:
        support |= np.abs(x_core - expected) <= half_support
    area = float(np.sum(residual[support]) * local_pitch)
    area_sigma = float(noise * np.sqrt(max(int(np.sum(support)), 1)) * local_pitch)
    snr = area / max(area_sigma, np.finfo(float).eps)
    base.update({
        "matched_wavelength_nm": peak_wl,
        "area": area,
        "area_sigma": area_sigma,
        "noise_per_native_sample": float(noise),
        "snr": float(snr),
    })
    if not aligned:
        base["reasons"].append("local_peak_misaligned")
    if area <= 0:
        base["reasons"].append("nonpositive_area")
    if snr < float(cfg["min_snr"]):
        base["reasons"].append("below_snr_threshold")

    ceiling = cfg.get("saturation_ceiling")
    repeated_max = int(np.sum(np.isclose(y_core, np.max(y_core), rtol=0, atol=1e-12)))
    saturated = ((ceiling is not None and float(np.max(y_core)) >= float(ceiling))
                 or repeated_max >= 3)
    if saturated:
        base["flags"].append("possible_saturation_or_flat_top")
        base["reasons"].append("amplitude_not_reliable")
    base["supported"] = bool(aligned and area > 0 and snr >= float(cfg["min_snr"]))
    base["clean"] = bool(base["supported"] and not saturated)
    return base


def _strong_line_catalog(db, x_range, cfg) -> dict[str, np.ndarray]:
    cache_key = (
        round(float(x_range[0]), 3), round(float(x_range[1]), 3),
        float(cfg["competitor_temperature_K"]),
        float(cfg["competitor_strong_fraction"]),
        float(cfg["competitor_band_nm"]),
    )
    cache = getattr(db, "_gas_detection_catalog_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(db, "_gas_detection_catalog_cache", cache)
        except Exception:
            pass
    if cache_key in cache:
        return cache[cache_key]
    kT = 8.617333262e-5 * float(cfg["competitor_temperature_K"])
    fraction = float(cfg["competitor_strong_fraction"])
    catalog = {}
    excluded = set(getattr(db, "unsupported_elements", ()))
    for element in db.elements:
        if element in excluded or element in db.no_lines:
            continue
        rows = np.asarray(db.lines(element))
        if rows.size == 0:
            continue
        ion = rows[:, 0].astype(float)
        wl = rows[:, 1].astype(float)
        ga = rows[:, 3].astype(float)
        ek = rows[:, 5].astype(float)
        use = ((ion <= 2) & (wl >= x_range[0]) & (wl <= x_range[1])
               & np.isfinite(ga) & (ga > 0) & np.isfinite(ek))
        if not np.any(use):
            continue
        # Normalize within ion and local band: an intense UV line must not
        # hide a plausible NIR interferer.
        strength = ga[use] * np.exp(-ek[use] / kT) / wl[use]
        use_wl, use_ion = wl[use], ion[use]
        band = np.floor(use_wl / float(cfg["competitor_band_nm"])).astype(int)
        keep = np.zeros(use_wl.size, dtype=bool)
        for stage, segment in set(zip(use_ion, band)):
            part = (use_ion == stage) & (band == segment)
            keep[part] = strength[part] >= fraction * float(np.max(strength[part]))
        order = np.argsort(use_wl[keep])
        catalog[element] = np.column_stack(
            (use_wl[keep][order], use_ion[keep][order], band[keep][order]))
    cache[cache_key] = catalog
    return catalog


def _supported_competitors(db, peak_array, x_range, cfg):
    catalog = _strong_line_catalog(db, x_range, cfg)
    explicit = {str(v) for v in cfg.get("competitor_elements", ())}
    if peak_array is None:
        return explicit, catalog, None
    peaks = np.asarray(peak_array, dtype=float)
    if peaks.ndim != 2 or peaks.shape[1] < 2 or not np.all(np.isfinite(peaks[:, :2])):
        raise ValueError("peak_array must be a finite 2-D array with amplitude and center")
    return explicit, catalog, peaks[:, 1]


def _resolution_groups(values, width=0.15):
    groups = []
    for value in np.sort(np.asarray(values, dtype=float)):
        if not groups or value - groups[-1][-1] > width:
            groups.append([float(value)])
        else:
            groups[-1].append(float(value))
    return np.asarray([np.mean(group) for group in groups])


def _externally_supported(lines, centers, lo, hi, cfg):
    """Test an ion/band pattern without using the gas-window peak itself."""
    window = (lines[:, 0] >= lo) & (lines[:, 0] <= hi)
    for stage, segment in set(map(tuple, lines[window, 1:3])):
        part = lines[(lines[:, 1] == stage) & (lines[:, 2] == segment), 0]
        external = _resolution_groups(part[(part < lo) | (part > hi)])
        matched = set()
        for wl in external:
            if centers.size:
                i = int(np.argmin(np.abs(centers - wl)))
                if abs(float(centers[i] - wl)) <= float(cfg["competitor_match_nm"]):
                    matched.add(i)
        distinct = _resolution_groups(centers[sorted(matched)]) if matched else []
        coverage = len(distinct) / max(len(external), 1)
        if (len(distinct) >= int(cfg["competitor_min_groups"]) and
                coverage >= float(cfg["competitor_min_coverage"])):
            return True
    return False


def _detect(element, definitions, x, y, db, shift_nm, peak_array, config) -> dict:
    x, y = _validate_spectrum(x, y)
    cfg = _config(config)
    groups = _database_groups(db, element, definitions)
    measured = [_measure_group(x, y, group, shift_nm, cfg) for group in groups]
    explicit, catalog, centers = _supported_competitors(
        db, peak_array, (float(x[0]), float(x[-1])), cfg)
    for record in measured:
        if not record["covered"]:
            continue
        lo, hi = (float(v) for v in record["database_measurement_window_nm"])
        blends = []
        for el, lines in catalog.items():
            in_window = np.any((lines[:, 0] >= lo) & (lines[:, 0] <= hi))
            if el != element and in_window and (el in explicit or
                    (centers is not None and
                     _externally_supported(lines, centers, lo, hi, cfg))):
                blends.append(el)
        record["blend_elements"] = blends
        if blends:
            record["clean"] = False
            record["reasons"].append("supported_competitor_in_window")

    supported = [r for r in measured if r["supported"]]
    clean = [r for r in measured if r["clean"]]
    covered = [r for r in measured if r["covered"]]
    required = 3 if element == "Ar" else 2
    reasons = []
    if not covered:
        status = "out_of_range"
        reasons.append("no_anchor_group_has_complete_local_coverage")
    elif not supported:
        status = "no_evidence"
        reasons.append("no_group_clears_local_5sigma_and_alignment_gates")
    elif peak_array is None:
        status = "tentative"
        reasons.append("interference_check_requires_peak_array")
    elif len(clean) >= required:
        status = "detected"
    else:
        status = "tentative"
        reasons.append(f"requires_{required}_independent_clean_groups")
    if element == "O" and len(supported) == 1 and supported[0]["group"] == "OI_777":
        if "OI_777_is_one_unresolved_group" not in reasons:
            reasons.append("OI_777_is_one_unresolved_group")

    return {
        "element": element,
        "stage": 1,
        "status": status,
        "n_database_groups": int(len(groups)),
        "n_covered_groups": int(len(covered)),
        "n_supported_groups": int(len(supported)),
        "n_clean_groups": int(len(clean)),
        "threshold_snr": float(cfg["min_snr"]),
        "expected_background_context": True,
        "origin": "unknown_purge_ambient_or_sample" if element == "O"
                  else "unknown_purge_ambient_or_sample",
        "concentration_inference": False,
        "groups": measured,
        "reasons": reasons,
    }


def detect_argon(x, y, db, *, shift_nm=0.0, peak_array=None, config=None) -> dict:
    """Detect Ar I from at least three independent clean NIR groups."""
    return _detect("Ar", _AR_GROUPS, x, y, db, shift_nm, peak_array, config)


def detect_oxygen(x, y, db, *, shift_nm=0.0, peak_array=None, config=None) -> dict:
    """Detect atomic O I; the 777 nm triplet counts as one group."""
    return _detect("O", _O_GROUPS, x, y, db, shift_nm, peak_array, config)


def detect_background_gases(
    x, y, db, *, shift_nm=0.0, peak_array=None, config=None,
) -> dict:
    """Return JSON-safe early evidence records for Ar I and atomic O I."""
    return {
        "Ar": detect_argon(x, y, db, shift_nm=shift_nm,
                           peak_array=peak_array, config=config),
        "O": detect_oxygen(x, y, db, shift_nm=shift_nm,
                           peak_array=peak_array, config=config),
    }


__all__ = ["detect_argon", "detect_oxygen", "detect_background_gases"]
