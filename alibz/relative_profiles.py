"""Multi-line relative elemental profiles for single-shot LIBS scans.

The output is intentionally *within-element*: it answers whether Fe, Li, or
another element rises or falls along the profile without claiming that an Fe
score of one equals a Li score of one.  This avoids the unstable closed-sum
normalization exposed by the MW2-112 quantitative pilot.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from alibz.peaky_finder import PeakyFinder
from alibz.utils.database import Database


KB_EV_K = 8.617333262e-5
REFERENCE_TEMPERATURE_K = 9000.0
SEGMENT_EDGES_NM = (365.0, 620.0)
DEFAULT_PROFILE_FWHM_NM = (0.12, 0.14, 0.16)
DEFAULT_ELEMENTS = (
    "H", "Li", "B", "C", "O", "F", "Na", "Mg", "Al", "Si", "P",
    "S", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Rb", "Sr", "Zr", "Ba",
)


def _cluster_lines(wavelength, strength, merge_nm=0.06):
    order = np.argsort(wavelength)
    wavelength = np.asarray(wavelength, dtype=float)[order]
    strength = np.asarray(strength, dtype=float)[order]
    clusters = []
    start = 0
    for i in range(1, wavelength.size + 1):
        if i < wavelength.size and wavelength[i] - wavelength[i - 1] <= merge_nm:
            continue
        wl = wavelength[start:i]
        weights = strength[start:i]
        total = float(np.sum(weights))
        clusters.append({
            "wavelength_nm": float(np.sum(wl * weights) / total),
            "strength": total,
            "components_nm": wl.astype(float).tolist(),
            "component_weights": (weights / total).astype(float).tolist(),
        })
        start = i
    return clusters


def build_line_features(
    dbpath: str,
    elements: Sequence[str] = DEFAULT_ELEMENTS,
    temperature_k: float = REFERENCE_TEMPERATURE_K,
    features_per_stage: int = 5,
    isolation_nm: float = 0.12,
    required_wavelengths: dict[tuple[str, int], Sequence[float]] | None = None,
    exhaustive_competitors: bool = False,
    relative_strength_floor: float = 1e-5,
    required_tolerance_nm: float = 0.03,
    abundance_multipliers: dict[str, float] | None = None,
    minimum_competitor_prior_ratio: float = 0.03,
) -> list[dict]:
    """Select strong same-stage multiplets and label cross-element blends.

    ``required_wavelengths`` adds pre-registered diagnostic/resonance lines to
    the strength-ranked candidates.  With ``exhaustive_competitors=True``, a
    feature is marked contested by any physically material line in the
    expected-element catalog, rather than only by another selected feature.
    Materiality uses ``gA exp(-E/kT)`` times the natural crustal-abundance prior
    (and optional sample multiplier), with the explicit relative threshold
    ``minimum_competitor_prior_ratio``.  This stricter mode is intended for
    trace-element identification, where a weak candidate coincident with an
    Fe-rich matrix line is not independent evidence.
    """
    db = Database(dbpath)
    kT = KB_EV_K * float(temperature_k)
    catalog = []
    for element in elements:
        if (element in db.no_lines or element not in db.atom_dict
                or element in getattr(db, "analysis_excluded_elements", ())):
            continue
        lines = db.lines(element)
        if lines.size == 0:
            continue
        ion = lines[:, 0].astype(float)
        wavelength = lines[:, 1].astype(float)
        gA = lines[:, 3].astype(float)
        energy = lines[:, 5].astype(float)
        valid = ((ion >= 1) & (ion <= 2) & (wavelength >= 190.0)
                 & (wavelength <= 910.0) & (gA > 0))
        for stage in (1, 2):
            select = valid & (ion == stage)
            if not np.any(select):
                continue
            strength = gA[select] * np.exp(-energy[select] / kT)
            clusters = _cluster_lines(wavelength[select], strength)
            clusters.sort(key=lambda row: row["strength"], reverse=True)
            strongest = clusters[0]["strength"]
            for rank, feature in enumerate(clusters):
                if feature["strength"] < strongest * relative_strength_floor:
                    continue
                center = feature["wavelength_nm"]
                if any(abs(center - edge) < 1.0 for edge in SEGMENT_EDGES_NM):
                    continue
                catalog.append({
                    **feature,
                    "element": element,
                    "ion_stage": stage,
                    "rank_in_stage": rank + 1,
                    "relative_strength": feature["strength"] / strongest,
                    "crustal_abundance_prior": float(db.abundance(element)),
                    "abundance_multiplier": float(
                        (abundance_multipliers or {}).get(element, 1.0)),
                    "prior_line_strength": (
                        feature["strength"] * float(db.abundance(element))
                        * float((abundance_multipliers or {}).get(
                            element, 1.0))),
                    "is_canonical": 0,
                    "canonical_target_nm": np.nan,
                })

    by_stage = defaultdict(list)
    for feature in catalog:
        by_stage[(feature["element"], feature["ion_stage"])].append(feature)

    candidate_ids = set()
    for key, features in by_stage.items():
        features.sort(key=lambda row: row["rank_in_stage"])
        if exhaustive_competitors:
            initial = features[:features_per_stage]
        else:
            initial = features[:max(12, features_per_stage * 2)]
        candidate_ids.update(id(row) for row in initial)
        for target in (required_wavelengths or {}).get(key, ()):
            nearest = min(
                features,
                key=lambda row: abs(row["wavelength_nm"] - float(target)),
                default=None,
            )
            if (nearest is not None
                    and abs(nearest["wavelength_nm"] - float(target))
                    <= required_tolerance_nm):
                nearest["is_canonical"] = 1
                nearest["canonical_target_nm"] = float(target)
                candidate_ids.add(id(nearest))

    candidates = [row for row in catalog if id(row) in candidate_ids]
    competitor_catalog = catalog if exhaustive_competitors else candidates

    # In strict mode, inspect the full material catalog.  Record exact rival
    # transitions so an element label can be audited rather than relying on an
    # opaque boolean blend flag.
    for feature in candidates:
        rivals = []
        rival_lines = []
        all_rival_lines = []
        stage_rivals = []
        for other in competitor_catalog:
            if other is feature:
                continue
            separation = abs(other["wavelength_nm"] - feature["wavelength_nm"])
            if separation > isolation_nm:
                continue
            if other["element"] == feature["element"]:
                if other["ion_stage"] != feature["ion_stage"]:
                    stage_rivals.append(
                        f"{other['element']}_{other['ion_stage']}_"
                        f"{other['wavelength_nm']:.4f}")
                continue
            if (exhaustive_competitors
                    or other["rank_in_stage"] <= features_per_stage):
                components = "|".join(
                    f"{float(value):.4f}"
                    for value in other.get("components_nm", ()))
                rival_label = (
                    f"{other['element']}_{other['ion_stage']}_"
                    f"{other['wavelength_nm']:.4f}[{components}]")
                all_rival_lines.append(rival_label)
                if (exhaustive_competitors
                        and other["prior_line_strength"]
                        < (minimum_competitor_prior_ratio
                           * feature["prior_line_strength"])):
                    continue
                rivals.append(other["element"])
                rival_lines.append(rival_label)
        feature["contested_by"] = sorted(set(rivals))
        feature["competitor_lines"] = sorted(set(rival_lines))
        feature["all_competitor_lines"] = sorted(set(all_rival_lines))
        feature["stage_ambiguous_by"] = sorted(set(stage_rivals))

    selected = []
    grouped = defaultdict(list)
    for feature in candidates:
        grouped[(feature["element"], feature["ion_stage"])].append(feature)
    for (_element, _stage), features in grouped.items():
        if exhaustive_competitors:
            # Strength-ranked lines plus canonical additions were already
            # chosen above.  Do not replace strong contested lines with weak
            # clean ones; retain them for audit and reject them downstream.
            keep = sorted(features, key=lambda row: (
                not row["is_canonical"], row["rank_in_stage"]))
        else:
            clean = [row for row in features if not row["contested_by"]]
            contested = [row for row in features if row["contested_by"]]
            keep = (clean + contested)[:features_per_stage]
        for feature in keep:
            feature = dict(feature)
            feature["line_id"] = (
                f"{feature['element']}_{feature['ion_stage']}_"
                f"{feature['wavelength_nm']:.4f}")
            selected.append(feature)
    return sorted(selected, key=lambda row: (
        row["element"], row["ion_stage"], row["rank_in_stage"]))


def _matched_area(x, y, centers, component_weights, fwhm):
    sigma = max(float(fwhm) / 2.354820045, 0.015)
    lo = min(centers) - max(4.0 * sigma, 0.18)
    hi = max(centers) + max(4.0 * sigma, 0.18)
    mask = (x >= lo) & (x <= hi)
    if np.sum(mask) < 5:
        return np.nan, np.nan
    xw, yw = x[mask], y[mask]
    template = np.zeros_like(xw)
    norm = sigma * np.sqrt(2.0 * np.pi)
    for center, weight in zip(centers, component_weights):
        template += (weight / norm) * np.exp(
            -0.5 * ((xw - center) / sigma) ** 2)
    design = np.column_stack([template, np.ones_like(template)])
    try:
        coefficient, _residual, _rank, _singular = np.linalg.lstsq(
            design, yw, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    area = max(float(coefficient[0]), 0.0)
    residual = yw - design @ coefficient
    noise = 1.4826 * float(np.median(
        np.abs(residual - np.median(residual))))
    centered_template = template - np.mean(template)
    sigma_area = noise / max(np.sqrt(np.sum(centered_template ** 2)), 1e-12)
    return area, sigma_area


def measure_relative_features(
    path: str,
    test_id: int,
    height_um: float,
    calibration: dict,
    features: Sequence[dict],
    configured_response_fallback: float = 3.9,
) -> tuple[list[dict], dict]:
    """Matched-filter all selected line features in one spectrum."""
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    x, y = values[:, 0], values[:, 1]
    finder = PeakyFinder.__new__(PeakyFinder)
    background = finder.find_background(x, y)
    residual = y - background

    response = calibration.get("response_measured")
    response_source = "measured"
    if response is None:
        response = calibration.get("response_prior")
        response_source = "shared_profile"
    if response is None:
        response = configured_response_fallback
        response_source = "configured_fallback"
    response = float(response)
    corrected = residual.copy()
    corrected[x >= 620.0] /= response

    shift = [float(value) if value is not None else 0.0
             for value in calibration.get("shift_prior_nm", [None] * 3)]
    fwhm = [float(value) if value is not None else DEFAULT_PROFILE_FWHM_NM[i]
            for i, value in enumerate(
                calibration.get("profile_fwhm_nm", [None] * 3))]
    pitch = float(np.median(np.diff(x)))
    shot_line_power = float(np.sum(np.maximum(corrected, 0.0)) * pitch)

    records = []
    for feature in features:
        segment = int(np.searchsorted(SEGMENT_EDGES_NM,
                                      feature["wavelength_nm"]))
        centers = [float(value) + shift[segment]
                   for value in feature["components_nm"]]
        area, area_sigma = _matched_area(
            x, corrected, centers, feature["component_weights"],
            fwhm[segment])
        records.append({
            "test_id": int(test_id),
            "height_um": float(height_um),
            "line_id": feature["line_id"],
            "element": feature["element"],
            "ion_stage": feature["ion_stage"],
            "wavelength_nm": feature["wavelength_nm"],
            "contested": int(bool(feature["contested_by"])),
            "contested_by": ";".join(feature["contested_by"]),
            "area": area,
            "area_sigma": area_sigma,
            "snr": (area / area_sigma if area_sigma and area_sigma > 0
                    else np.nan),
            "area_fraction": (area / shot_line_power
                              if shot_line_power > 0 else np.nan),
            "response": response,
            "response_source": response_source,
            "shift_nm": shift[segment],
            "profile_fwhm_nm": fwhm[segment],
        })
    qc = {
        "test_id": int(test_id), "height_um": float(height_um),
        "shot_line_power": shot_line_power,
        "response": response, "response_source": response_source,
        "shift_uv_nm": shift[0], "shift_vis_nm": shift[1],
        "shift_nir_nm": shift[2],
        "fwhm_uv_nm": fwhm[0], "fwhm_vis_nm": fwhm[1],
        "fwhm_nir_nm": fwhm[2],
    }
    return records, qc


def combine_relative_profiles(line_records: Iterable[dict],
                              detection_snr: float = 3.0,
                              minimum_line_coherence: float = 0.15) -> list[dict]:
    """Robustly combine line profiles after within-line corpus scaling."""
    records = [dict(row) for row in line_records]
    by_line = defaultdict(list)
    for row in records:
        value = float(row["area_fraction"])
        if np.isfinite(value) and value > 0:
            by_line[row["line_id"]].append(value)
    scales = {line_id: float(np.quantile(values, 0.9))
              for line_id, values in by_line.items() if values}
    for row in records:
        scale = scales.get(row["line_id"], np.nan)
        value = float(row["area_fraction"])
        row["line_relative"] = (value / scale
                                if scale > 0 and np.isfinite(value) else np.nan)

    # A genuine stage must reproduce the same spatial pattern in multiple
    # independent lines.  Reject absent ion stages and chance coincidences by
    # a pre-registered, vendor-independent Spearman coherence threshold.
    from scipy.stats import spearmanr
    line_series = defaultdict(dict)
    all_ids = sorted({int(row["test_id"]) for row in records})
    for row in records:
        line_series[(row["element"], row["ion_stage"], row["line_id"])][
            int(row["test_id"])] = row["line_relative"]
    stage_lines = defaultdict(list)
    for element, stage, line_id in line_series:
        stage_lines[(element, stage)].append(line_id)
    coherent_lines = set()
    stage_coherence = {}
    minimum_observations = min(20, max(2, len(all_ids) // 2))
    for key, line_ids in stage_lines.items():
        coherence = {}
        for line_id in line_ids:
            first = np.array([line_series[(*key, line_id)].get(test_id, np.nan)
                              for test_id in all_ids], dtype=float)
            correlations = []
            for other_id in line_ids:
                if other_id == line_id:
                    continue
                second = np.array([
                    line_series[(*key, other_id)].get(test_id, np.nan)
                    for test_id in all_ids], dtype=float)
                valid = np.isfinite(first) & np.isfinite(second)
                if np.sum(valid) < minimum_observations:
                    continue
                if (np.ptp(first[valid]) == 0.0
                        or np.ptp(second[valid]) == 0.0):
                    continue
                rho = float(spearmanr(first[valid], second[valid]).statistic)
                if np.isfinite(rho):
                    correlations.append(rho)
            coherence[line_id] = (float(np.median(correlations))
                                  if correlations else np.nan)
        accepted = [line_id for line_id, value in coherence.items()
                    if np.isfinite(value)
                    and value >= minimum_line_coherence]
        if len(accepted) >= 2:
            coherent_lines.update(accepted)
        stage_coherence[key] = (float(np.median(
            [coherence[line_id] for line_id in accepted]))
            if len(accepted) >= 2 else None)

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["test_id"], row["height_um"], row["element"],
                 row["ion_stage"])].append(row)
    stage_rows = []
    for (test_id, height, element, stage), rows in grouped.items():
        supported_rows = [row for row in rows
                          if row["line_id"] in coherent_lines]
        detected = [row for row in supported_rows
                    if np.isfinite(row["line_relative"])
                    and float(row["snr"]) >= detection_snr]
        clean = [row for row in detected if not row["contested"]]
        use = clean if clean else detected
        score = float(np.median([row["line_relative"] for row in use])) \
            if use else 0.0
        stage_rows.append({
            "test_id": test_id, "height_um": height, "element": element,
            "ion_stage": stage, "stage_score": score,
            "stage_supported": int(len(supported_rows) >= 2),
            "stage_coherence": stage_coherence.get((element, stage)),
            "n_lines": len(rows), "n_coherent_lines": len(supported_rows),
            "n_detected": len(detected),
            "n_clean_detected": len(clean),
        })

    by_element = defaultdict(list)
    for row in stage_rows:
        by_element[(row["test_id"], row["height_um"], row["element"])].append(row)
    output = []
    for (test_id, height, element), stages in by_element.items():
        supported = [row for row in stages if row["n_detected"] > 0]
        supported_stages = [row for row in stages if row["stage_supported"]]
        score = (float(np.median([row["stage_score"] for row in supported]))
                 if supported else 0.0)
        n_detected = sum(row["n_detected"] for row in stages)
        n_clean = sum(row["n_clean_detected"] for row in stages)
        stage_scores = {int(row["ion_stage"]): row["stage_score"]
                        for row in stages}
        positive = [value for value in stage_scores.values() if value > 0]
        disagreement = (abs(np.log(positive[0] / positive[1]))
                        if len(positive) >= 2 else None)
        min_required = 2
        status = ("unsupported" if not supported_stages else
                  "detected" if n_clean >= min_required else
                  "contested" if n_detected >= min_required else
                  "single-line" if n_detected == 1 else "not-detected")
        output.append({
            "test_id": test_id, "height_um": height, "element": element,
            "relative_score": score, "status": status,
            "n_lines_detected": n_detected,
            "n_clean_lines_detected": n_clean,
            "n_coherent_lines": sum(row["n_coherent_lines"]
                                    for row in stages),
            "supported_stages": ";".join(
                str(row["ion_stage"]) for row in supported_stages),
            "stage_1_coherence": next((row["stage_coherence"] for row in stages
                                       if row["ion_stage"] == 1), None),
            "stage_2_coherence": next((row["stage_coherence"] for row in stages
                                       if row["ion_stage"] == 2), None),
            "stage_1_score": stage_scores.get(1),
            "stage_2_score": stage_scores.get(2),
            "stage_log_disagreement": disagreement,
        })
    return sorted(output, key=lambda row: (
        int(row["test_id"]), row["element"]))
