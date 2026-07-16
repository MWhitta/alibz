"""Peak-window PCA relative quantification for the 929-shot MW2-112 profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from alibz.mw2_112 import (
    PROFILE_FIRST_ID,
    PROFILE_STEP_UM,
    atomic_json,
    database_state,
    hash_file,
    load_frozen_calibration,
    software_state,
    utc_now,
    write_csv,
)
from alibz.peak_window_pca import (
    PHYSICAL_COORDINATES,
    assign_independent_window_clusters,
    basis_characteristics,
    load_basis,
    pairwise_spatial_coherence,
    project_window,
    reference_peak,
    resolve_basis_path,
    weighted_median,
)
from alibz.peaky_finder import PeakyFinder
from alibz.profile_pca import build_reference_stack
from alibz.relative_profiles import build_line_features


# Pre-registered from geological plausibility plus the vendor instrument's
# fixed reporting panel. Vendor values are not used for window selection.
ELEMENT_POLICY = tuple(
    "H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Rb Sr Y Zr Nb Mo Ag Cd In Sn Sb Cs Ba La Ce Hf "
    "Ta W Re Pb Bi Th U".split()
)
SEGMENT_EDGES_NM = (365.0, 620.0)
SEGMENT_LABELS = ("uv", "vis", "nir")

# Pre-registered, widely used LIBS diagnostics for the dominant mudrock and
# accessory-mineral elements.  These supplement rather than replace the
# Saha-Boltzmann strength ranking.  Wavelengths are in the instrument/database
# air convention above 200 nm.
CANONICAL_LINES = {
    ("H", 1): (486.135, 656.279),
    ("Li", 1): (610.3605, 670.7808, 812.6372),
    ("O", 1): (777.194, 777.417, 777.539),
    ("Na", 1): (588.9951, 589.5924, 818.3256, 819.4821),
    ("Mg", 1): (285.2124, 517.2684, 518.3604),
    ("Mg", 2): (292.8633, 293.6510),
    ("Al", 1): (237.3145, 308.2151, 394.4006, 396.1520),
    ("Si", 1): (212.4123, 250.6897, 252.8508, 288.1578),
    ("K", 1): (404.4142, 693.8764, 766.4899, 769.8964),
    ("Ca", 1): (422.6728, 445.4779, 643.9075, 646.2567),
    ("Ca", 2): (393.3663, 396.8468, 854.2093, 866.2141),
    ("Ti", 1): (453.3239, 498.1730, 499.1066),
    ("Ti", 2): (334.9293, 336.1212),
    ("Fe", 1): (248.3272, 302.0748, 358.1204),
    ("Fe", 2): (234.1070, 238.1786, 259.9395, 263.1195),
    ("Rb", 1): (780.0269, 794.7603),
    ("Sr", 1): (460.7333,),
    ("Sr", 2): (407.7709, 421.5519),
    ("Ba", 1): (553.5482,),
    ("Ba", 2): (455.4033, 493.4077),
}

# Emission can be real while the inferred sample abundance is not.  With no
# blanks, atmosphere control, or hydrogen standard, these elements remain QC
# diagnostics and are never promoted to chemical-anchor profiles.
QUANTIFICATION_EXCLUSIONS = {
    "H": "ambient/plasma hydrogen is not separable from sample hydrogen",
    "N": "ambient nitrogen is not separable from sample nitrogen",
    "O": "ambient oxygen is not separable from sample oxygen",
}

PHYSICAL_INTERPRETATION = {
    "shift": "residual line displacement after shared shift correction",
    "gaussian_broadening": "Gaussian-like widening relative to corpus mean",
    "lorentzian_wings": "enhanced broad wings/pressure-like broadening",
    "flattening": "flat-topped profile consistent with self-absorption",
    "splitting": "symmetric shoulder/doublet-like structure or unresolved blend",
}


def _optional_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _atomic_write_rows(path: Path, rows, header) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _load_vendor(vendor_dir: Path) -> dict[int, dict]:
    output = {}
    for path in vendor_dir.glob("*.csv"):
        with path.open() as fh:
            row = next(csv.DictReader(fh))
        output[int(row["Test #"])] = row
    return output


def _preprocess(raw_dir: Path, calibration: dict[int, dict], log):
    X, wavelength, ids = build_reference_stack(str(raw_dir))
    corrected = np.zeros_like(X, dtype=np.float32)
    aligned = np.zeros_like(X, dtype=np.float32)
    line_power = np.zeros(ids.size, dtype=float)
    shifts = np.zeros((ids.size, 3), dtype=float)
    finder = PeakyFinder.__new__(PeakyFinder)
    pitch = float(np.median(np.diff(wavelength)))
    for index, (test_id, spectrum) in enumerate(zip(ids, X)):
        if not np.any(spectrum):
            continue
        background = finder.find_background(wavelength, spectrum)
        residual = np.asarray(spectrum - background, dtype=float)
        record = calibration[int(test_id)]
        response = record.get("response_measured")
        if response is None:
            response = record.get("response_prior")
        response = float(response) if response else 3.9
        residual[wavelength >= 620.0] /= response
        corrected[index] = residual.astype(np.float32)
        line_power[index] = float(
            np.sum(np.maximum(residual, 0.0)) * pitch)
        shift_values = [float(value) if value is not None else 0.0
                        for value in record["shift_prior_nm"]]
        shifts[index] = shift_values
        for segment in range(3):
            low = -np.inf if segment == 0 else SEGMENT_EDGES_NM[segment - 1]
            high = (np.inf if segment == 2 else SEGMENT_EDGES_NM[segment])
            mask = (wavelength >= low) & (wavelength < high)
            aligned[index, mask] = np.interp(
                wavelength[mask] + shift_values[segment],
                wavelength, residual).astype(np.float32)
        if (index + 1) % 100 == 0 or index + 1 == ids.size:
            log(f"preprocess [{index + 1}/{ids.size}]")
    valid = (ids != 1632) & (line_power > 0)
    reference = np.mean(aligned[valid], axis=0, dtype=np.float64)
    return corrected, reference, wavelength, ids, line_power, shifts


def _aligned_reference(corrected, wavelength, ids, line_power, shifts):
    aligned = np.zeros_like(corrected, dtype=np.float32)
    for index, spectrum in enumerate(corrected):
        if line_power[index] <= 0:
            continue
        for segment in range(3):
            low = -np.inf if segment == 0 else SEGMENT_EDGES_NM[segment - 1]
            high = np.inf if segment == 2 else SEGMENT_EDGES_NM[segment]
            mask = (wavelength >= low) & (wavelength < high)
            aligned[index, mask] = np.interp(
                wavelength[mask] + shifts[index, segment],
                wavelength, spectrum).astype(np.float32)
    valid = (ids != 1632) & (line_power > 0)
    return np.mean(aligned[valid], axis=0, dtype=np.float64)


def _finite_column_median(arrays, n_positions):
    output = np.full(n_positions, np.nan)
    if not arrays:
        return output
    values = np.asarray(arrays, dtype=float)
    for index in range(n_positions):
        finite = values[:, index][np.isfinite(values[:, index])]
        if finite.size:
            output[index] = float(np.median(finite))
    return output


def _robust_spread(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


def _validate_shift_prior(candidates, measurements, shifts, args, log):
    """Keep per-shot shifts only when shared peak centroids become tighter."""
    applied = np.array(shifts, dtype=float, copy=True)
    policy = []
    for segment, label in enumerate(SEGMENT_LABELS):
        raw_arrays, corrected_arrays = [], []
        for feature in candidates:
            if (int(np.searchsorted(SEGMENT_EDGES_NM,
                                    feature["center_nm"])) != segment
                    or feature.get("contested_by")
                    or feature["reference_snr"] <
                    args.shift_validation_reference_snr):
                continue
            values = measurements[feature["line_id"]]
            use = ((values["peak_snr"] >= args.detection_snr)
                   & (values["reconstruction_r2"] >=
                      args.minimum_median_r2))
            if np.mean(use) < args.minimum_detection_rate:
                continue
            for key, destination in (
                    ("centroid_raw_nm", raw_arrays),
                    ("centroid_corrected_nm", corrected_arrays)):
                array = np.array(values[key], copy=True)
                center = np.nanmedian(array[use])
                array = array - center
                array[~use] = np.nan
                destination.append(array)
        raw_drift = _finite_column_median(raw_arrays, shifts.shape[0])
        corrected_drift = _finite_column_median(
            corrected_arrays, shifts.shape[0])
        raw_spread = _robust_spread(raw_drift)
        corrected_spread = _robust_spread(corrected_drift)
        keep_variable = (
            len(raw_arrays) >= args.minimum_shift_validation_windows
            and np.isfinite(raw_spread) and raw_spread > 0
            and np.isfinite(corrected_spread)
            and corrected_spread <= (
                args.maximum_corrected_shift_spread_ratio * raw_spread)
        )
        if not keep_variable:
            applied[:, segment] = np.nanmedian(shifts[:, segment])
        policy.append({
            "segment": label,
            "n_validation_windows": len(raw_arrays),
            "raw_common_centroid_spread_nm": raw_spread,
            "proposed_corrected_centroid_spread_nm": corrected_spread,
            "spread_ratio": (corrected_spread / raw_spread
                             if raw_spread and np.isfinite(raw_spread)
                             else np.nan),
            "proposed_shift_median_nm": float(np.nanmedian(
                shifts[:, segment])),
            "proposed_shift_robust_spread_nm": _robust_spread(
                shifts[:, segment]),
            "applied_mode": ("per-shot frozen prior" if keep_variable
                             else "constant segment median"),
            "accepted_variable_prior": int(keep_variable),
        })
        log(
            f"shift validation {label}: {len(raw_arrays)} windows, "
            f"raw={raw_spread * 1000:.2f} pm, "
            f"proposed={corrected_spread * 1000:.2f} pm, "
            f"mode={policy[-1]['applied_mode']}")
    return applied, policy


def _candidate_windows(dbpath: Path, reference, wavelength, basis, args, log):
    features = build_line_features(
        str(dbpath), elements=ELEMENT_POLICY,
        temperature_k=args.reference_temperature,
        features_per_stage=args.features_per_stage,
        isolation_nm=args.deconfliction_radius_nm,
        required_wavelengths=CANONICAL_LINES,
        exhaustive_competitors=True,
        minimum_competitor_prior_ratio=(
            args.minimum_competitor_prior_ratio))
    candidates = []
    audited = []
    for feature in features:
        refined = reference_peak(
            wavelength, reference, float(feature["wavelength_nm"]), basis,
            search_half_width_nm=args.center_search_nm,
            step_nm=args.center_step_nm,
            n_components=args.pca_components)
        record = {**feature, **refined}
        at_edge = (abs(refined["offset_nm"]) >=
                   args.center_search_nm - 0.5 * args.center_step_nm)
        screen_pass = (refined["reference_snr"] >= args.reference_screen_snr
                       and not at_edge)
        record["screen_pass"] = int(screen_pass)
        record["screen_rejection"] = (
            "search-edge optimum" if at_edge else
            "summed-spectrum SNR" if not screen_pass else "")
        record["contested_by"] = list(feature.get("contested_by", []))
        record["competitor_lines"] = list(
            feature.get("competitor_lines", []))
        record["all_competitor_lines"] = list(
            feature.get("all_competitor_lines", []))
        record["stage_ambiguous_by"] = list(
            feature.get("stage_ambiguous_by", []))
        audited.append(record)
        if screen_pass:
            candidates.append(record)
    log(f"reference screen: {len(candidates)}/{len(features)} windows")
    return candidates, audited


def _empty_measurement(n_positions, n_components):
    values = {
        "area_fraction": np.full(n_positions, np.nan),
        "peak_snr": np.zeros(n_positions),
        "reconstruction_r2": np.full(n_positions, np.nan),
        "reconstruction_rmse": np.full(n_positions, np.nan),
        "centroid_raw_nm": np.full(n_positions, np.nan),
        "centroid_corrected_nm": np.full(n_positions, np.nan),
    }
    for name in PHYSICAL_COORDINATES:
        values[f"{name}_score"] = np.full(n_positions, np.nan)
        values[f"{name}_z"] = np.full(n_positions, np.nan)
        if name == "shift":
            values["shift_raw_z"] = np.full(n_positions, np.nan)
    values["pc_scores"] = np.full((n_positions, n_components), np.nan)
    return values


def _measure_windows(candidates, corrected, wavelength, ids, line_power,
                     shifts, basis, args, log):
    measurements = {}
    for candidate_index, feature in enumerate(candidates):
        values = _empty_measurement(ids.size, args.pca_components)
        segment = int(np.searchsorted(SEGMENT_EDGES_NM,
                                      feature["center_nm"]))
        center = float(feature["center_nm"])
        for index in range(ids.size):
            if line_power[index] <= 0:
                continue
            raw_projection = project_window(
                wavelength, corrected[index], center, basis,
                n_components=args.pca_components)
            projection = project_window(
                wavelength, corrected[index],
                center + shifts[index, segment], basis,
                n_components=args.pca_components)
            if projection is None:
                continue
            values["area_fraction"][index] = (
                projection.area / line_power[index])
            values["peak_snr"][index] = projection.peak_snr
            values["reconstruction_r2"][index] = projection.reconstruction_r2
            values["reconstruction_rmse"][index] = (
                projection.reconstruction_rmse)
            values["centroid_corrected_nm"][index] = (
                projection.centroid_offset_nm)
            values["pc_scores"][index] = projection.scores[
                :args.pca_components]
            for name in PHYSICAL_COORDINATES:
                values[f"{name}_score"][index] = (
                    projection.physical_scores[name])
                values[f"{name}_z"][index] = projection.physical_z[name]
            if raw_projection is not None:
                values["centroid_raw_nm"][index] = (
                    raw_projection.centroid_offset_nm)
                values["shift_raw_z"][index] = (
                    raw_projection.physical_z["shift"])
        measurements[feature["line_id"]] = values
        if ((candidate_index + 1) % 25 == 0
                or candidate_index + 1 == len(candidates)):
            log(f"windows [{candidate_index + 1}/{len(candidates)}]")
    return measurements


def _window_selection(candidates, measurements, args):
    profiles, valid = {}, {}
    for feature in candidates:
        line_id = feature["line_id"]
        values = measurements[line_id]
        use = ((values["peak_snr"] >= args.detection_snr)
               & (values["reconstruction_r2"] >= args.minimum_r2)
               & np.isfinite(values["area_fraction"])
               & (values["area_fraction"] > 0))
        scale_values = values["area_fraction"][use]
        scale = float(np.quantile(scale_values, 0.90)) \
            if scale_values.size else np.nan
        profile = (values["area_fraction"] / scale
                   if np.isfinite(scale) and scale > 0
                   else np.zeros_like(values["area_fraction"]))
        profiles[line_id] = profile
        valid[line_id] = use
        feature["line_scale_q90"] = scale

    clusters = assign_independent_window_clusters(
        candidates, args.deconfliction_radius_nm)
    summary = []
    for feature in candidates:
        line_id = feature["line_id"]
        values = measurements[line_id]
        use = valid[line_id]
        detected = values["peak_snr"] >= args.detection_snr
        r2_values = values["reconstruction_r2"][detected]
        sa = np.maximum(values["flattening_z"], values["splitting_z"])
        sa_use = detected & np.isfinite(sa)
        record = {
            **feature,
            "contested_by": ";".join(feature.get("contested_by", [])),
            "competitor_lines": ";".join(
                feature.get("competitor_lines", [])),
            "all_competitor_lines": ";".join(
                feature.get("all_competitor_lines", [])),
            "stage_ambiguous_by": ";".join(
                feature.get("stage_ambiguous_by", [])),
            "components_nm": ";".join(
                f"{value:.6f}" for value in feature["components_nm"]),
            "detection_rate": float(np.mean(detected)),
            "usable_rate": float(np.mean(use)),
            "median_reconstruction_r2": (
                float(np.nanmedian(r2_values)) if r2_values.size else np.nan),
            "self_absorption_flag_rate": (
                float(np.mean(sa[sa_use] >= args.self_absorption_z))
                if np.any(sa_use) else np.nan),
            "median_flattening_z": float(np.nanmedian(
                values["flattening_z"][detected])) if np.any(detected) else np.nan,
            "median_splitting_z": float(np.nanmedian(
                values["splitting_z"][detected])) if np.any(detected) else np.nan,
            "line_scale_q90": feature["line_scale_q90"],
        }
        for name in PHYSICAL_COORDINATES:
            coordinate = values[f"{name}_z"][detected]
            record[f"median_{name}_z"] = (
                float(np.nanmedian(coordinate))
                if coordinate.size else np.nan)
            record[f"{name}_positive_flag_rate"] = (
                float(np.mean(coordinate >= args.shape_anomaly_z))
                if coordinate.size else np.nan)
        cluster_index, cluster_size = clusters[line_id]
        record["independent_cluster_id"] = (
            f"{feature['element']}_{int(feature['ion_stage'])}_"
            f"C{cluster_index:02d}")
        record["independent_cluster_size"] = cluster_size
        record["quality_accepted"] = int(
            record["reference_snr"] >= args.minimum_reference_snr
            and record["detection_rate"] >= args.minimum_detection_rate
            and record["median_reconstruction_r2"] >= args.minimum_median_r2
        )
        summary.append(record)

    # Count one window per same-element/stage resolution cluster.  Selection
    # favors a quality-passing uncontested feature, then a canonical feature,
    # then the strongest/reliably detected alternative.
    by_cluster = _group_records(
        summary, lambda row: row["independent_cluster_id"])
    for rows in by_cluster.values():
        representative = max(rows, key=lambda row: (
            bool(row["quality_accepted"]) and not bool(row["contested_by"]),
            bool(row["quality_accepted"]),
            not bool(row["contested_by"]),
            bool(row.get("is_canonical", 0)),
            float(row["reference_snr"]),
            float(row["detection_rate"]),
            -int(row["rank_in_stage"]),
        ))
        for row in rows:
            row["independent_representative"] = int(row is representative)

    # Spatial coherence is evaluated only between independent, uncontested,
    # quality-passing representatives.  Coincident Fe/Ag (or similar) windows
    # therefore cannot manufacture multi-line evidence for a trace element.
    coherence_pool = [
        row for row in summary
        if (row["quality_accepted"] and row["independent_representative"]
            and not row["contested_by"])
    ]
    pair_rows = []
    coherence = {}
    for (element, stage), rows in _group_records(
            coherence_pool,
            lambda row: (row["element"], row["ion_stage"])).items():
        line_ids = [row["line_id"] for row in rows]
        medians, pairs = pairwise_spatial_coherence(
            {line_id: profiles[line_id] for line_id in line_ids},
            {line_id: valid[line_id] for line_id in line_ids},
            minimum_observations=args.minimum_coherence_observations)
        coherence.update(medians)
        for (first, second), rho in pairs.items():
            pair_rows.append({
                "element": element, "ion_stage": stage,
                "line_id_1": first, "line_id_2": second,
                "cluster_id_1": next(row["independent_cluster_id"]
                                     for row in rows
                                     if row["line_id"] == first),
                "cluster_id_2": next(row["independent_cluster_id"]
                                     for row in rows
                                     if row["line_id"] == second),
                "spearman_rho": rho,
            })

    for record in summary:
        line_id = record["line_id"]
        record["median_line_coherence"] = coherence.get(line_id, np.nan)
        record["window_accepted"] = int(
            record["quality_accepted"]
            and record["independent_representative"]
            and not record["contested_by"]
            and np.isfinite(record["median_line_coherence"])
            and record["median_line_coherence"] >= args.minimum_line_coherence
        )
        if record["contested_by"]:
            record["rejection_reason"] = "cross-element unresolved blend"
        elif not record["independent_representative"]:
            record["rejection_reason"] = "duplicate spectral cluster"
        elif record["reference_snr"] < args.minimum_reference_snr:
            record["rejection_reason"] = "summed-spectrum SNR"
        elif record["detection_rate"] < args.minimum_detection_rate:
            record["rejection_reason"] = "single-shot detection rate"
        elif record["median_reconstruction_r2"] < args.minimum_median_r2:
            record["rejection_reason"] = "PCA reconstruction"
        elif not np.isfinite(record["median_line_coherence"]):
            record["rejection_reason"] = "no independent line pair"
        elif record["median_line_coherence"] < args.minimum_line_coherence:
            record["rejection_reason"] = "line-profile incoherence"
        else:
            record["rejection_reason"] = ""

    supported_stages = set()
    for key, rows in _group_records(
            summary, lambda row: (row["element"], row["ion_stage"])).items():
        accepted = [row for row in rows if row["window_accepted"]]
        if len(accepted) >= 2:
            supported_stages.add(key)
    for row in summary:
        row["stage_supported"] = int(
            (row["element"], row["ion_stage"]) in supported_stages)
        row["quantification_exclusion"] = QUANTIFICATION_EXCLUSIONS.get(
            row["element"], "")
        row["use_for_element"] = int(
            row["window_accepted"] and row["stage_supported"]
            and not row["quantification_exclusion"])
        sa_rate = row["self_absorption_flag_rate"]
        sa_rate = sa_rate if np.isfinite(sa_rate) else 1.0
        coherence_value = max(row["median_line_coherence"], 0.0) \
            if np.isfinite(row["median_line_coherence"]) else 0.0
        row["global_weight"] = (
            np.sqrt(row["usable_rate"]) * coherence_value
            * max(row["median_reconstruction_r2"], 0.0)
            / (1.0 + 2.0 * sa_rate)
        ) if row["use_for_element"] else 0.0
    return summary, profiles, valid, pair_rows, supported_stages


def _group_records(records, key):
    output = defaultdict(list)
    for record in records:
        output[key(record)].append(record)
    return output


def _combine_elements(ids, summary, measurements, profiles, valid,
                      supported_stages, args):
    by_element = _group_records(
        [row for row in summary if row["use_for_element"]],
        lambda row: row["element"])
    output, element_summary = [], []
    for element, windows in sorted(by_element.items()):
        stage_names = sorted({int(row["ion_stage"]) for row in windows})
        element_summary.append({
            "element": element,
            "n_windows": len(windows),
            "n_clean_windows": sum(not row["contested_by"] for row in windows),
            "supported_stages": ";".join(str(stage) for stage in stage_names),
            "line_ids": ";".join(row["line_id"] for row in windows),
            "median_line_coherence": float(np.median([
                row["median_line_coherence"] for row in windows])),
            "median_detection_rate": float(np.median([
                row["detection_rate"] for row in windows])),
            "median_self_absorption_flag_rate": float(np.nanmedian([
                row["self_absorption_flag_rate"] for row in windows])),
            **{
                f"median_{name}_z": float(np.nanmedian([
                    row[f"median_{name}_z"] for row in windows]))
                for name in PHYSICAL_COORDINATES
            },
        })
        for index, test_id in enumerate(ids):
            line_values, line_weights, stages = [], [], defaultdict(list)
            flatten_values, split_values = [], []
            for window in windows:
                line_id = window["line_id"]
                if not valid[line_id][index]:
                    continue
                measurement = measurements[line_id]
                value = profiles[line_id][index]
                sa_z = max(measurement["flattening_z"][index],
                           measurement["splitting_z"][index], 0.0)
                shift_z = abs(measurement["shift_z"][index])
                local_weight = window["global_weight"] / (
                    1.0 + 0.20 * max(sa_z - 1.0, 0.0) ** 2
                    + 0.10 * max(shift_z - 2.0, 0.0) ** 2)
                line_values.append(value)
                line_weights.append(local_weight)
                stages[int(window["ion_stage"])].append(
                    (value, local_weight))
                flatten_values.append(measurement["flattening_z"][index])
                split_values.append(measurement["splitting_z"][index])
            score = weighted_median(line_values, line_weights)
            uncertainty = (1.4826 * float(np.median(np.abs(
                np.asarray(line_values) - score)))
                           if len(line_values) >= 2 else np.nan)
            stage_scores = {
                stage: weighted_median(
                    [value for value, _weight in records],
                    [weight for _value, weight in records])
                for stage, records in stages.items()
            }
            positive = [value for value in stage_scores.values()
                        if np.isfinite(value) and value > 0]
            disagreement = (abs(float(np.log(positive[0] / positive[1])))
                            if len(positive) >= 2 else np.nan)
            output.append({
                "test_id": int(test_id),
                "height_um": (int(test_id) - PROFILE_FIRST_ID) * PROFILE_STEP_UM,
                "element": element,
                "relative_score": score if np.isfinite(score) else 0.0,
                "line_mad_uncertainty": uncertainty,
                "status": ("missing" if int(test_id) == 1632 else
                           "detected" if len(line_values) >= 2 else
                           "single-line" if len(line_values) == 1 else
                           "not-detected"),
                "n_windows_used": len(line_values),
                "n_windows_available": len(windows),
                "supported_stages": ";".join(
                    str(stage) for stage in stage_names),
                "stage_1_score": stage_scores.get(1),
                "stage_2_score": stage_scores.get(2),
                "stage_log_disagreement": disagreement,
                "median_flattening_z": (float(np.nanmedian(flatten_values))
                                        if flatten_values else np.nan),
                "median_splitting_z": (float(np.nanmedian(split_values))
                                       if split_values else np.nan),
            })
    return output, element_summary


def _peak_physics_summary(summary):
    rows = []
    for window in summary:
        if not window["window_accepted"]:
            continue
        for name in PHYSICAL_COORDINATES:
            rows.append({
                "element": window["element"],
                "ion_stage": window["ion_stage"],
                "line_id": window["line_id"],
                "center_nm": window["center_nm"],
                "independent_cluster_id": window["independent_cluster_id"],
                "used_for_quantification": window["use_for_element"],
                "physical_coordinate": name,
                "interpretation": PHYSICAL_INTERPRETATION[name],
                "median_z": window[f"median_{name}_z"],
                "positive_flag_rate": window[
                    f"{name}_positive_flag_rate"],
            })
    return rows


def _shift_profiles(ids, proposed_shifts, applied_shifts, summary, measurements):
    accepted = [row for row in summary if row["use_for_element"]]
    by_segment = _group_records(
        accepted, lambda row: int(np.searchsorted(
            SEGMENT_EDGES_NM, row["center_nm"])))
    output = []
    for segment in range(3):
        windows = by_segment.get(segment, [])
        raw_centered, corrected_centered = {}, {}
        for row in windows:
            values = measurements[row["line_id"]]
            for source, destination in (
                    ("centroid_raw_nm", raw_centered),
                    ("centroid_corrected_nm", corrected_centered)):
                array = values[source]
                destination[row["line_id"]] = array - np.nanmedian(array)
        proposed = (proposed_shifts[:, segment]
                    - np.nanmedian(proposed_shifts[:, segment]))
        applied = (applied_shifts[:, segment]
                   - np.nanmedian(applied_shifts[:, segment]))
        for index, test_id in enumerate(ids):
            raw = [array[index] for array in raw_centered.values()
                   if np.isfinite(array[index])]
            corrected = [array[index] for array in corrected_centered.values()
                         if np.isfinite(array[index])]
            output.append({
                "test_id": int(test_id),
                "height_um": (int(test_id) - PROFILE_FIRST_ID) * PROFILE_STEP_UM,
                "segment": SEGMENT_LABELS[segment],
                "proposed_shift_nm": proposed_shifts[index, segment],
                "proposed_shift_drift_nm": proposed[index],
                "applied_shift_nm": applied_shifts[index, segment],
                "applied_shift_drift_nm": applied[index],
                "pca_raw_centroid_drift_nm": (float(np.median(raw))
                                              if raw else np.nan),
                "pca_corrected_centroid_drift_nm": (
                    float(np.median(corrected)) if corrected else np.nan),
                "n_windows": len(raw),
            })
    return output


def _validate_profiles(element_profiles, vendor_dir: Path,
                       previous_run: Path | None):
    by_element = defaultdict(dict)
    for row in element_profiles:
        by_element[row["element"]][int(row["test_id"])] = row
    validation = []
    vendor = _load_vendor(vendor_dir)
    for element, records in sorted(by_element.items()):
        x, y = [], []
        for test_id, vendor_row in vendor.items():
            raw = vendor_row.get(f"{element} (%)", "")
            if not raw or raw.startswith("<") or test_id not in records:
                continue
            try:
                x.append(float(records[test_id]["relative_score"]))
                y.append(float(raw))
            except ValueError:
                pass
        if len(x) >= 20 and np.ptp(x) > 0 and np.ptp(y) > 0:
            rho, pvalue = spearmanr(x, y)
            validation.append({
                "comparison": "vendor_summary", "element": element,
                "n": len(x), "spearman_rho": float(rho),
                "p_value": float(pvalue),
            })
    if previous_run and (previous_run / "relative_element_profiles.csv").exists():
        previous = defaultdict(dict)
        with (previous_run / "relative_element_profiles.csv").open() as fh:
            for row in csv.DictReader(fh):
                previous[row["element"]][int(row["test_id"])] = float(
                    row["relative_score"])
        for element, records in sorted(by_element.items()):
            common = sorted(set(records) & set(previous.get(element, {})))
            if len(common) < 20:
                continue
            x = [float(records[test_id]["relative_score"]) for test_id in common]
            y = [previous[element][test_id] for test_id in common]
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                rho, pvalue = spearmanr(x, y)
                validation.append({
                    "comparison": "matched_filter_relative_profiles",
                    "element": element, "n": len(common),
                    "spearman_rho": float(rho), "p_value": float(pvalue),
                })
    return validation


def _write_window_profiles(path, candidates, measurements, ids):
    header = [
        "test_id", "height_um", "line_id", "element", "ion_stage",
        "center_nm", "area_fraction", "peak_snr", "reconstruction_r2",
        "reconstruction_rmse", "centroid_raw_nm", "centroid_corrected_nm",
        *[f"PC{index + 1}" for index in range(
            next(iter(measurements.values()))["pc_scores"].shape[1])],
        *[f"{name}_score" for name in PHYSICAL_COORDINATES],
        *[f"{name}_z" for name in PHYSICAL_COORDINATES], "shift_raw_z",
    ]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for candidate in candidates:
            values = measurements[candidate["line_id"]]
            for index, test_id in enumerate(ids):
                row = {
                    "test_id": int(test_id),
                    "height_um": (int(test_id) - PROFILE_FIRST_ID)
                    * PROFILE_STEP_UM,
                    "line_id": candidate["line_id"],
                    "element": candidate["element"],
                    "ion_stage": candidate["ion_stage"],
                    "center_nm": candidate["center_nm"],
                    **{key: values[key][index] for key in (
                        "area_fraction", "peak_snr", "reconstruction_r2",
                        "reconstruction_rmse", "centroid_raw_nm",
                        "centroid_corrected_nm", "shift_raw_z")},
                }
                for pc_index, score in enumerate(values["pc_scores"][index]):
                    row[f"PC{pc_index + 1}"] = score
                for name in PHYSICAL_COORDINATES:
                    row[f"{name}_score"] = values[f"{name}_score"][index]
                    row[f"{name}_z"] = values[f"{name}_z"][index]
                writer.writerow(row)
    os.replace(tmp, path)


def _feature_manifest_rows(features):
    rows = []
    for feature in features:
        row = dict(feature)
        for key in ("components_nm", "component_weights"):
            row[key] = ";".join(
                f"{float(value):.9g}" for value in feature.get(key, []))
        for key in ("contested_by", "competitor_lines",
                    "all_competitor_lines",
                    "stage_ambiguous_by"):
            row[key] = ";".join(feature.get(key, []))
        rows.append(row)
    return rows


def _upstream_provenance(previous_run: Path | None) -> dict:
    if previous_run is None:
        return {"status": "not provided"}
    path = previous_run / "provenance_manifest.json"
    if not path.exists():
        return {"status": "manifest unavailable", "path": str(path)}
    manifest = json.loads(path.read_text())
    return {
        "status": "referenced",
        "path": str(path),
        "sha256": hash_file(path),
        "raw_inputs": manifest.get("raw_inputs"),
        "vendor_validation_inputs": manifest.get("vendor_validation_inputs"),
    }


def _product_checksums(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if (not path.is_file()
                or path.name == "product_checksums.json"):
            continue
        rows.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        })
    return rows


def _figures(output_dir, wavelength, reference, summary, element_profiles,
             shift_profiles, basis, validation):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    characteristics = basis_characteristics(basis)
    x = np.linspace(-1, 1, basis.n_window_points)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(x, basis.mean_peak, color="black")
    axes[0].set_title("Corpus mean peak")
    for index, ax in enumerate(axes[1:6]):
        ax.plot(x, basis.components[index])
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(
            f"PC{index + 1} ({basis.explained_variance_ratio[index]:.1%})\n"
            f"{characteristics[index]['dominant_template']}")
    fig.savefig(figure_dir / "corpus_peak_shape_basis.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    for segment, ax in enumerate(axes):
        low = 190 if segment == 0 else SEGMENT_EDGES_NM[segment - 1]
        high = 910 if segment == 2 else SEGMENT_EDGES_NM[segment]
        mask = (wavelength >= low) & (wavelength < high)
        ax.plot(wavelength[mask], reference[mask], color="0.2", lw=0.5)
        for row in summary:
            if row["use_for_element"] and low <= row["center_nm"] < high:
                ax.axvline(row["center_nm"], lw=0.7, alpha=0.6)
                ax.text(row["center_nm"], ax.get_ylim()[1] * 0.9,
                        row["element"], rotation=90, fontsize=6, va="top")
        ax.set_ylabel(SEGMENT_LABELS[segment].upper())
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.savefig(figure_dir / "summed_spectrum_selected_windows.png", dpi=180)
    plt.close(fig)

    by_element = _group_records(element_profiles, lambda row: row["element"])
    elements = sorted(by_element)
    if elements:
        fig, axes = plt.subplots(
            len(elements), 1, figsize=(12, 1.45 * len(elements)), sharex=True,
            constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, element in zip(axes, elements):
            rows = sorted(by_element[element], key=lambda row: row["test_id"])
            height = np.array([row["height_um"] / 1000 for row in rows])
            score = np.array([row["relative_score"] for row in rows])
            ax.plot(height, score, lw=0.8)
            ax.set_ylabel(element, rotation=0, ha="right")
        axes[-1].set_xlabel("Height above sample bottom (mm)")
        fig.savefig(figure_dir / "element_profiles.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for accepted, marker, color in ((0, "x", "0.6"), (1, "o", "#1f77b4")):
        rows = [row for row in summary if row["use_for_element"] == accepted]
        ax.scatter([row["detection_rate"] for row in rows],
                   [row["median_line_coherence"] for row in rows],
                   marker=marker, color=color, alpha=0.7,
                   label="selected" if accepted else "rejected")
    ax.axhline(0.2, color="0.4", ls="--")
    ax.set_xlabel("Single-shot detection rate")
    ax.set_ylabel("Median same-stage line coherence")
    ax.legend()
    fig.savefig(figure_dir / "window_selection.png", dpi=180)
    plt.close(fig)

    by_segment = _group_records(shift_profiles, lambda row: row["segment"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    for ax, label in zip(axes, SEGMENT_LABELS):
        rows = sorted(by_segment[label], key=lambda row: row["test_id"])
        height = np.array([row["height_um"] / 1000 for row in rows])
        ax.plot(height, np.array([row["proposed_shift_drift_nm"]
                                  for row in rows]) * 1000,
                label="proposed frozen prior", lw=0.8, alpha=0.7)
        ax.plot(height, np.array([row["applied_shift_drift_nm"]
                                  for row in rows]) * 1000,
                label="validated applied correction", lw=1.0)
        ax.plot(height, np.array([row["pca_raw_centroid_drift_nm"]
                                  for row in rows]) * 1000,
                label="raw PCA-window centroid", lw=0.7, alpha=0.8)
        ax.plot(height, np.array([row["pca_corrected_centroid_drift_nm"]
                                  for row in rows]) * 1000,
                label="after shift correction", lw=0.7, alpha=0.8)
        ax.set_ylabel(f"{label.upper()} pm")
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Height above sample bottom (mm)")
    fig.savefig(figure_dir / "detector_shift_diagnostics.png", dpi=180)
    plt.close(fig)


def _report(output_dir, summary, element_summary, validation, shift_profiles,
            basis, args):
    selected = [row for row in summary if row["use_for_element"]]
    elements = [row["element"] for row in element_summary]
    validations = defaultdict(list)
    for row in validation:
        validations[row["comparison"]].append(row)
    vendor_text = ", ".join(
        f"{row['element']} ρ={row['spearman_rho']:.3f}"
        for row in sorted(validations["vendor_summary"],
                          key=lambda row: row["element"])) or "not available"
    previous_text = ", ".join(
        f"{row['element']} ρ={row['spearman_rho']:.3f}"
        for row in sorted(validations["matched_filter_relative_profiles"],
                          key=lambda row: row["element"])) or "not available"
    sa_lines = sorted(
        selected, key=lambda row: row["self_absorption_flag_rate"], reverse=True)
    sa_text = ", ".join(
        f"{row['line_id']} ({row['self_absorption_flag_rate']:.0%})"
        for row in sa_lines[:8] if np.isfinite(row["self_absorption_flag_rate"]))
    physics_text = []
    for name in PHYSICAL_COORDINATES:
        ranked = sorted(
            selected,
            key=lambda row: row[f"{name}_positive_flag_rate"],
            reverse=True)
        text = ", ".join(
            f"{row['line_id']} ({row[f'{name}_positive_flag_rate']:.0%})"
            for row in ranked[:4]
            if np.isfinite(row[f"{name}_positive_flag_rate"]))
        physics_text.append(f"- {name}: {text or 'none'}")
    excluded = sorted({
        row["element"] for row in summary
        if row["stage_supported"] and row["quantification_exclusion"]
    })
    shift_modes = []
    for label, rows in _group_records(
            shift_profiles, lambda row: row["segment"]).items():
        applied = np.array([row["applied_shift_drift_nm"] for row in rows])
        mode = ("constant median" if np.ptp(applied) < 1e-12
                else "per-shot frozen prior")
        shift_modes.append(f"{label.upper()}={mode}")
    report = f"""# MW2-112 peak-window PCA relative quantification

Generated: {utc_now()}

## Outcome

All 929 scan positions were projected at every summed-spectrum-screened known
line window. The conservative multi-line gate supports {len(elements)}
elements: {', '.join(elements) or 'none'}.

The primary output is a within-element relative spatial trend. PCA separates
normalized peak shape from peak range, reconstructs a shape-constrained window
area, and downweights positions/lines with shift, flattening, splitting, or
poor reconstruction. It does not provide absolute or cross-element
concentrations.

## Window evidence

- {len(summary)} database-prior windows passed the summed-spectrum screen.
- {len(selected)} windows passed single-shot detection, reconstruction, and
  same-stage spatial-coherence gates.
- A supported ion stage requires at least two spectrally independent,
  uncontested windows. Same-stage candidates within
  {args.deconfliction_radius_nm:.3f} nm count as one feature.
- Every candidate was checked against all material 9000 K transitions from
  the expected-element policy within the same radius. Every overlap is
  recorded; those with abundance-weighted prior strength at least
  {args.minimum_competitor_prior_ratio:g} times the candidate are material,
  audit-only, and cannot support quantification.
- ID 1632 remains an explicit all-zero/missing observation.

Highest self-absorption-like flag rates among selected windows: {sa_text or 'none'}.

Peak-shape anomaly rates (positive coordinate >= {args.shape_anomaly_z:g}
corpus standard deviations):

{chr(10).join(physics_text)}

Spectrally supported but excluded from sample-abundance quantification because
ambient/plasma contributions cannot be separated: {', '.join(excluded) or 'none'}.

## Corpus PCA prior

The archived basis contains {basis.components.shape[0]} components trained on
{basis.manifest['training']['peak_windows']:,} peak windows from
{basis.manifest['training']['spectra']} spectra. It explains
{np.sum(basis.explained_variance_ratio):.3%} of normalized peak-shape variance.
MW2-112 projection uses the first {args.pca_components} components, explaining
{np.sum(basis.explained_variance_ratio[:args.pca_components]):.3%}; the small
higher-order components are omitted because a native 0.311 nm window contains
only about ten detector samples and a ten-component interpolation would be
nearly saturated rather than independently informative.
Projection uses the training-consistent {basis.half_window_nm:.6f} nm
half-window. The 0.0841 nm value later calculated in `peaky_data.ipynb` was not
used because it does not match the saved `05x` basis training window.

PC/template coordinates diagnose red/blue shift, Gaussian-like broadening,
Lorentzian wings, flat-top self-absorption, and symmetric splitting. These
coordinates are correlated diagnostics rather than unique causal labels.
Common-mode wavelength drift may be consistent with detector-temperature
change, but no detector-temperature telemetry exists, so temperature is not
claimed as the demonstrated cause.

The frozen per-shot wavelength correction was retained only where it reduced
the dispersion of common peak centroids; otherwise the segment median was
used. Applied modes: {', '.join(shift_modes)}. The full proposed-versus-applied
diagnostic is retained in `shift_correction_policy.csv` and
`detector_shift_profiles.csv`.

## Independent comparisons

Vendor summaries were not used for window selection or profile fitting:
{vendor_text}.

Comparison with the independent matched-filter relative-profile method:
{previous_text}.

## Fixed selection thresholds

- summed reference screen SNR: {args.reference_screen_snr:g}; final SNR:
  {args.minimum_reference_snr:g}
- per-shot detection SNR: {args.detection_snr:g}
- minimum detection rate: {args.minimum_detection_rate:g}
- minimum median reconstruction R²: {args.minimum_median_r2:g}
- minimum same-stage Spearman coherence: {args.minimum_line_coherence:g}
- self-absorption-like template threshold: {args.self_absorption_z:g} corpus σ
- exhaustive line deconfliction radius: {args.deconfliction_radius_nm:g} nm
- minimum material-competitor prior ratio:
  {args.minimum_competitor_prior_ratio:g}
- variable shift retained only if corrected common-centroid spread <=
  {args.maximum_corrected_shift_spread_ratio:g} times raw spread using at
  least {args.minimum_shift_validation_windows} clean windows

## Limitations

- The 0.0333 nm MW2-112 sampling is coarser than the 0.01 nm corpus PCA grid;
  interpolation enables projection but cannot create lost subpixel detail.
- The global corpus basis mixes detector segments; segment-specific PC bases
  were not preserved with per-peak scores in a compact production artifact.
- Strong resonance lines can remain nonlinear even after PCA shape
  downweighting. Multi-line agreement is the principal protection.
- The deconfliction catalog is conditional on expected elements, database
  completeness, oscillator strengths, and a 9000 K LTE ranking. Unknown or
  missing transitions can still blend an accepted window.
- No standards or replicate shots exist, so systematic and empirical
  concentration uncertainties remain unavailable.
"""
    (output_dir / "analysis_report.md").write_text(report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--db", type=Path, default=Path("db"))
    parser.add_argument("--basis", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--features-per-stage", type=int, default=8)
    parser.add_argument("--reference-temperature", type=float, default=9000.0)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--center-search-nm", type=float, default=0.10)
    parser.add_argument("--center-step-nm", type=float, default=0.01)
    parser.add_argument("--reference-screen-snr", type=float, default=1.0)
    parser.add_argument("--minimum-reference-snr", type=float, default=1.0)
    parser.add_argument("--detection-snr", type=float, default=3.0)
    parser.add_argument("--minimum-r2", type=float, default=0.0)
    parser.add_argument("--minimum-median-r2", type=float, default=0.35)
    parser.add_argument("--minimum-detection-rate", type=float, default=0.25)
    parser.add_argument("--minimum-line-coherence", type=float, default=0.25)
    parser.add_argument("--minimum-coherence-observations", type=int, default=100)
    parser.add_argument("--self-absorption-z", type=float, default=2.0)
    parser.add_argument("--shape-anomaly-z", type=float, default=2.0)
    parser.add_argument("--deconfliction-radius-nm", type=float, default=0.16)
    parser.add_argument("--minimum-competitor-prior-ratio", type=float,
                        default=0.25)
    parser.add_argument("--shift-validation-reference-snr", type=float,
                        default=10.0)
    parser.add_argument("--minimum-shift-validation-windows", type=int,
                        default=5)
    parser.add_argument("--maximum-corrected-shift-spread-ratio", type=float,
                        default=0.90)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dbpath = args.db.expanduser().resolve()
    basis_path = resolve_basis_path(args.basis)
    calibration_path = args.calibration.expanduser().resolve()
    vendor_dir = args.vendor_dir.expanduser().resolve()
    previous_run = (args.previous_run.expanduser().resolve()
                    if args.previous_run else None)
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("output directory must be outside the raw input tree")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message):
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    basis = load_basis(basis_path)
    if not 1 <= args.pca_components <= basis.components.shape[0]:
        raise SystemExit(
            f"--pca-components must be between 1 and "
            f"{basis.components.shape[0]}")
    if args.deconfliction_radius_nm <= 0:
        raise SystemExit("--deconfliction-radius-nm must be positive")
    if args.minimum_competitor_prior_ratio <= 0:
        raise SystemExit("--minimum-competitor-prior-ratio must be positive")
    if args.minimum_shift_validation_windows < 2:
        raise SystemExit("--minimum-shift-validation-windows must be >= 2")
    if not 0 < args.maximum_corrected_shift_spread_ratio <= 1:
        raise SystemExit(
            "--maximum-corrected-shift-spread-ratio must be in (0, 1]")
    calibration = load_frozen_calibration(calibration_path)
    log("loading, background-correcting, response-correcting, and aligning spectra")
    corrected, reference, wavelength, ids, line_power, proposed_shifts = _preprocess(
        input_dir, calibration, log)
    candidates, all_features = _candidate_windows(
        dbpath, reference, wavelength, basis, args, log)
    measurements = _measure_windows(
        candidates, corrected, wavelength, ids, line_power, proposed_shifts,
        basis, args, log)
    applied_shifts, shift_policy = _validate_shift_prior(
        candidates, measurements, proposed_shifts, args, log)
    if not np.array_equal(applied_shifts, proposed_shifts):
        log("rebuilding reference and window projections with validated shifts")
        reference = _aligned_reference(
            corrected, wavelength, ids, line_power, applied_shifts)
        candidates, all_features = _candidate_windows(
            dbpath, reference, wavelength, basis, args, log)
        measurements = _measure_windows(
            candidates, corrected, wavelength, ids, line_power,
            applied_shifts, basis, args, log)
    summary, profiles, valid, pair_rows, supported_stages = _window_selection(
        candidates, measurements, args)
    element_profiles, element_summary = _combine_elements(
        ids, summary, measurements, profiles, valid, supported_stages, args)
    shift_profiles = _shift_profiles(
        ids, proposed_shifts, applied_shifts, summary, measurements)
    validation = _validate_profiles(element_profiles, vendor_dir, previous_run)

    write_csv(output_dir / "basis_characteristics.csv",
              basis_characteristics(basis))
    write_csv(output_dir / "candidate_prior_manifest.csv",
              _feature_manifest_rows(all_features))
    write_csv(output_dir / "window_manifest.csv", summary)
    write_csv(output_dir / "line_pair_coherence.csv", pair_rows)
    write_csv(output_dir / "element_profiles.csv", element_profiles)
    write_csv(output_dir / "element_summary.csv", element_summary)
    write_csv(output_dir / "peak_physics_summary.csv",
              _peak_physics_summary(summary))
    write_csv(output_dir / "shift_correction_policy.csv", shift_policy)
    write_csv(output_dir / "detector_shift_profiles.csv", shift_profiles)
    write_csv(output_dir / "validation.csv", validation)
    _write_window_profiles(
        output_dir / "window_profiles.csv", candidates, measurements, ids)
    write_csv(output_dir / "summed_spectrum.csv", [
        {"wavelength_nm": float(x), "mean_corrected_intensity": float(y)}
        for x, y in zip(wavelength, reference)])
    _figures(output_dir, wavelength, reference, summary, element_profiles,
             shift_profiles, basis, validation)
    _report(output_dir, summary, element_summary, validation, shift_profiles,
            basis, args)

    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "created_utc": utc_now(),
        "sample": "MW2-112",
        "method": "corpus-prior peak-window PCA multi-line relative profiles",
        "input_dir": str(input_dir),
        "upstream_provenance": _upstream_provenance(previous_run),
        "n_positions": int(ids.size),
        "zero_test_id": 1632,
        "basis": {
            "path": str(basis_path), "sha256": hash_file(basis_path),
            "manifest_sha256": hash_file(basis_path.with_suffix(".json")),
            "archived_components": int(basis.components.shape[0]),
            "active_components": int(args.pca_components),
            "active_explained_variance_fraction": float(np.sum(
                basis.explained_variance_ratio[:args.pca_components])),
            "training_half_window_nm": basis.half_window_nm,
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": hash_file(calibration_path),
        },
        "shift_correction_policy": shift_policy,
        "database": database_state(dbpath),
        "software": software_state(repo),
        "element_policy": ELEMENT_POLICY,
        "canonical_lines": {
            f"{element}_{stage}": wavelengths
            for (element, stage), wavelengths in CANONICAL_LINES.items()
        },
        "quantification_exclusions": QUANTIFICATION_EXCLUSIONS,
        "parameters": vars(args),
        "products": {
            "prior_windows": len(all_features),
            "candidate_windows": len(candidates),
            "spectrally_accepted_windows": sum(
                row["window_accepted"] for row in summary),
            "selected_windows": sum(
                row["use_for_element"] for row in summary),
            "supported_elements": [row["element"] for row in element_summary],
        },
        "claim_level": "within-element relative spatial trends only",
    }
    manifest["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["parameters"].items()
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    recipe = {
        "schema_version": 1,
        "purpose": "exact-parameter regeneration from raw MW2-112 spectra",
        "requirements": [
            "the raw directory must match the upstream portable input hash",
            "the output directory must not exist or must be empty",
            "the basis, frozen calibration, database, and source revision "
            "must match run_manifest.json",
        ],
        "argv_template": [
            "mw2-112", "peak-pca", "RAW_DIR", "NEW_OUTPUT",
            "--db", str(dbpath),
            "--basis", str(basis_path),
            "--calibration", str(calibration_path),
            "--vendor-dir", str(vendor_dir),
            "--previous-run", str(previous_run) if previous_run else "",
            "--features-per-stage", str(args.features_per_stage),
            "--reference-temperature", str(args.reference_temperature),
            "--pca-components", str(args.pca_components),
            "--center-search-nm", str(args.center_search_nm),
            "--center-step-nm", str(args.center_step_nm),
            "--reference-screen-snr", str(args.reference_screen_snr),
            "--minimum-reference-snr", str(args.minimum_reference_snr),
            "--detection-snr", str(args.detection_snr),
            "--minimum-r2", str(args.minimum_r2),
            "--minimum-median-r2", str(args.minimum_median_r2),
            "--minimum-detection-rate", str(args.minimum_detection_rate),
            "--minimum-line-coherence", str(args.minimum_line_coherence),
            "--minimum-coherence-observations",
            str(args.minimum_coherence_observations),
            "--self-absorption-z", str(args.self_absorption_z),
            "--shape-anomaly-z", str(args.shape_anomaly_z),
            "--deconfliction-radius-nm",
            str(args.deconfliction_radius_nm),
            "--minimum-competitor-prior-ratio",
            str(args.minimum_competitor_prior_ratio),
            "--shift-validation-reference-snr",
            str(args.shift_validation_reference_snr),
            "--minimum-shift-validation-windows",
            str(args.minimum_shift_validation_windows),
            "--maximum-corrected-shift-spread-ratio",
            str(args.maximum_corrected_shift_spread_ratio),
        ],
        "verification": (
            "compare scientific CSV tables and product_checksums.json; "
            "generated timestamps and image metadata may differ"),
    }
    if previous_run is None:
        index = recipe["argv_template"].index("--previous-run")
        del recipe["argv_template"][index:index + 2]
    atomic_json(output_dir / "reproduction_recipe.json", recipe)
    log(f"complete: {len(element_summary)} elements, "
        f"{manifest['products']['selected_windows']} windows")
    atomic_json(output_dir / "product_checksums.json", {
        "schema_version": 1,
        "created_utc": utc_now(),
        "excludes": ["product_checksums.json"],
        "products": _product_checksums(output_dir),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
