"""Diagnose and quantify optically thick K I resonance lines in MW2-112."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr

from alibz.mw2_112 import (
    PROFILE_FIRST_ID,
    PROFILE_STEP_UM,
    atomic_json,
    hash_file,
    load_frozen_calibration,
    software_state,
    utc_now,
    write_csv,
)
from alibz.peak_window_pca import measure_broad_line
from scripts.run_mw2_112_peak_window_pca import _preprocess


K_LINES = (
    {"line_id": "K_I_693.8764", "center_nm": 693.876396,
     "relative_strength": 0.00655092165605007, "role": "weak_anchor"},
    {"line_id": "K_I_766.4899", "center_nm": 766.489910,
     "relative_strength": 1.0, "role": "strong_resonance"},
    {"line_id": "K_I_769.8964", "center_nm": 769.896450,
     "relative_strength": 0.4984931955549362, "role": "weak_resonance"},
)
EXPECTED_DOUBLET_RATIO = (
    K_LINES[1]["relative_strength"] / K_LINES[2]["relative_strength"])


def _optical_depth_from_ratio(ratio: float) -> tuple[float, str]:
    """Infer tau of 769.9 nm from a homogeneous-slab doublet peak ratio."""
    q = EXPECTED_DOUBLET_RATIO
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan, "invalid"
    if ratio >= q:
        return 0.0, "optically_thin_or_noise"
    if ratio <= 1.0:
        return np.inf, "saturated_lower_bound"

    def model(tau):
        return ((1.0 - np.exp(-q * tau))
                / (1.0 - np.exp(-tau))) - ratio

    return float(brentq(model, 1e-7, 50.0)), "finite"


def _load_previous_k(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    profile = path / "relative_element_profiles.csv"
    if not profile.exists():
        return {}
    output = {}
    with profile.open() as handle:
        for row in csv.DictReader(handle):
            if row["element"] == "K" and row["status"] == "detected":
                output[int(row["test_id"])] = float(row["relative_score"])
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--base-element-profiles", type=Path)
    parser.add_argument("--detection-snr", type=float, default=5.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    previous_run = (args.previous_run.expanduser().resolve()
                    if args.previous_run else None)
    base_element_profiles = (
        args.base_element_profiles.expanduser().resolve()
        if args.base_element_profiles else None)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message):
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    calibration = load_frozen_calibration(calibration_path)
    corrected, reference, wavelength, ids, line_power, shifts = _preprocess(
        input_dir, calibration, log)
    records = []
    for index, test_id in enumerate(ids):
        for line in K_LINES:
            base = {
                "test_id": int(test_id),
                "height_um": (int(test_id) - PROFILE_FIRST_ID)
                * PROFILE_STEP_UM,
                **line,
            }
            if line_power[index] <= 0:
                records.append({**base, "status": "missing"})
                continue
            center = line["center_nm"] + shifts[index, 2]
            measurement = measure_broad_line(
                wavelength, corrected[index], center)
            if measurement is None:
                records.append({**base, "status": "measurement_failed"})
                continue
            records.append({
                **base,
                "status": ("detected" if measurement.snr >= args.detection_snr
                           else "low_snr"),
                "applied_center_nm": center,
                "area": measurement.area,
                "area_sigma": measurement.area_sigma,
                "snr": measurement.snr,
                "area_fraction": measurement.area / line_power[index],
                "height": measurement.height,
                "center_height": measurement.center_height,
                "peak_offset_nm": measurement.peak_offset_nm,
                "centroid_offset_nm": measurement.centroid_offset_nm,
                "fwhm_nm": measurement.fwhm_nm,
                "top_flatness": measurement.top_flatness,
                "noise": measurement.noise,
            })
        if (index + 1) % 100 == 0 or index + 1 == ids.size:
            log(f"wide K windows [{index + 1}/{ids.size}]")

    by_line = {line["line_id"]: [] for line in K_LINES}
    for row in records:
        by_line[row["line_id"]].append(row)
    scales = {}
    for line_id, rows in by_line.items():
        values = np.array([
            float(row.get("area_fraction", np.nan)) for row in rows])
        finite = values[np.isfinite(values) & (values > 0)]
        scales[line_id] = float(np.quantile(finite, 0.90))
        for row, value in zip(rows, values):
            row["line_relative"] = (
                value / scales[line_id] if np.isfinite(value) else np.nan)

    by_test = {}
    for row in records:
        by_test.setdefault(int(row["test_id"]), {})[row["line_id"]] = row
    optical_depth_rows = []
    for test_id in ids:
        lines = by_test[int(test_id)]
        strong = lines["K_I_766.4899"]
        weak = lines["K_I_769.8964"]
        if strong.get("status") == "missing":
            optical_depth_rows.append({
                "test_id": int(test_id),
                "height_um": (int(test_id) - PROFILE_FIRST_ID)
                * PROFILE_STEP_UM,
                "status": "missing",
            })
            continue
        ratio = (float(strong.get("height", np.nan))
                 / float(weak.get("height", np.nan)))
        tau, tau_status = _optical_depth_from_ratio(ratio)
        optical_depth_rows.append({
            "test_id": int(test_id),
            "height_um": (int(test_id) - PROFILE_FIRST_ID) * PROFILE_STEP_UM,
            "status": ("detected" if strong.get("status") == "detected"
                       and weak.get("status") == "detected" else "low_snr"),
            "peak_height_ratio_766_769": ratio,
            "optically_thin_expected_ratio": EXPECTED_DOUBLET_RATIO,
            "tau_769_peak_model": tau,
            "tau_766_peak_model": EXPECTED_DOUBLET_RATIO * tau,
            "tau_status": tau_status,
            "fwhm_ratio_766_769": (
                float(strong.get("fwhm_nm", np.nan))
                / float(weak.get("fwhm_nm", np.nan))),
        })

    optical_depth_by_id = {
        int(row["test_id"]): row for row in optical_depth_rows}
    k_profile = []
    for test_id in ids:
        line_rows = by_test[int(test_id)]
        detected = [
            row for row in line_rows.values()
            if row.get("status") == "detected"
            and np.isfinite(row.get("line_relative", np.nan))]
        values = np.array([
            float(row["line_relative"]) for row in detected], dtype=float)
        score = float(np.median(values)) if values.size else np.nan
        uncertainty = (1.4826 * float(np.median(np.abs(values - score)))
                       if values.size >= 2 else np.nan)
        depth = optical_depth_by_id[int(test_id)]
        k_profile.append({
            "test_id": int(test_id),
            "height_um": (int(test_id) - PROFILE_FIRST_ID) * PROFILE_STEP_UM,
            "element": "K",
            "relative_score": score if np.isfinite(score) else 0.0,
            "line_mad_uncertainty": uncertainty,
            "status": ("missing" if int(test_id) == 1632 else
                       "detected" if values.size >= 2 else
                       "single-line" if values.size == 1 else
                       "not-detected"),
            "n_windows_used": int(values.size),
            "n_windows_available": len(K_LINES),
            "supported_stages": "1",
            "stage_1_score": score,
            "stage_2_score": np.nan,
            "stage_log_disagreement": np.nan,
            "median_flattening_z": np.nan,
            "median_splitting_z": np.nan,
            "peak_height_ratio_766_769": depth.get(
                "peak_height_ratio_766_769", np.nan),
            "tau_769_peak_model": depth.get("tau_769_peak_model", np.nan),
            "profile_claim": (
                "relative broad-resonance trend; optical-depth compressed"),
        })

    pair_rows = []
    line_ids = [line["line_id"] for line in K_LINES]
    for first_index, first in enumerate(line_ids):
        for second in line_ids[first_index + 1:]:
            a = {int(row["test_id"]): row for row in by_line[first]
                 if row.get("status") == "detected"}
            b = {int(row["test_id"]): row for row in by_line[second]
                 if row.get("status") == "detected"}
            common = sorted(set(a) & set(b))
            rho = (float(spearmanr(
                [a[key]["line_relative"] for key in common],
                [b[key]["line_relative"] for key in common]).statistic)
                if len(common) >= 3 else np.nan)
            pair_rows.append({
                "line_id_1": first, "line_id_2": second,
                "n_common_detections": len(common), "spearman_rho": rho,
            })

    previous = _load_previous_k(previous_run)
    validation = []
    for line_id, rows in by_line.items():
        common = [row for row in rows
                  if int(row["test_id"]) in previous
                  and row.get("status") == "detected"]
        if len(common) >= 3:
            validation.append({
                "comparison": "matched_filter_K",
                "line_id": line_id,
                "n": len(common),
                "spearman_rho": float(spearmanr(
                    [row["line_relative"] for row in common],
                    [previous[int(row["test_id"])] for row in common]
                ).statistic),
            })
    common_profile = [
        row for row in k_profile
        if int(row["test_id"]) in previous and row["status"] == "detected"]
    if common_profile:
        validation.append({
            "comparison": "matched_filter_K",
            "line_id": "combined_broad_K",
            "n": len(common_profile),
            "spearman_rho": float(spearmanr(
                [row["relative_score"] for row in common_profile],
                [previous[int(row["test_id"])] for row in common_profile]
            ).statistic),
        })

    line_summary = []
    for line in K_LINES:
        rows = by_line[line["line_id"]]
        detected = [row for row in rows if row.get("status") == "detected"]
        line_summary.append({
            **line,
            "n_detected": len(detected),
            "detection_rate": len(detected) / len(rows),
            "area_fraction_q90": scales[line["line_id"]],
            "median_snr": float(np.median([
                row["snr"] for row in detected])) if detected else np.nan,
            "median_fwhm_nm": float(np.median([
                row["fwhm_nm"] for row in detected])) if detected else np.nan,
            "median_top_flatness": float(np.median([
                row["top_flatness"] for row in detected]))
                if detected else np.nan,
        })

    valid_tau = np.array([
        row.get("tau_769_peak_model", np.nan) for row in optical_depth_rows
        if row.get("tau_status") == "finite"], dtype=float)
    finite_ratios = np.array([
        row.get("peak_height_ratio_766_769", np.nan)
        for row in optical_depth_rows if row.get("status") == "detected"],
        dtype=float)
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "sample": "MW2-112",
        "method": "broad-window K I equivalent area and doublet optical depth",
        "expected_optically_thin_peak_ratio_766_769": EXPECTED_DOUBLET_RATIO,
        "median_observed_peak_ratio_766_769": float(np.nanmedian(finite_ratios)),
        "median_tau_769_finite": (float(np.median(valid_tau))
                                  if valid_tau.size else np.nan),
        "fraction_peak_ratio_below_1p8": float(np.mean(finite_ratios < 1.8)),
        "lines": line_summary,
        "pairwise_coherence": pair_rows,
        "validation": validation,
        "claim": (
            "relative wide-line intensity and optical-depth diagnostic; "
            "not absolute K concentration"),
        "calibration_sha256": hash_file(calibration_path),
        "software": software_state(Path(__file__).resolve().parents[1]),
    }

    write_csv(output_dir / "k_wide_line_profiles.csv", records)
    write_csv(output_dir / "k_doublet_optical_depth.csv", optical_depth_rows)
    write_csv(output_dir / "k_relative_profile.csv", k_profile)
    write_csv(output_dir / "k_line_summary.csv", line_summary)
    write_csv(output_dir / "k_line_pair_coherence.csv", pair_rows)
    write_csv(output_dir / "validation.csv", validation)
    if base_element_profiles is not None:
        with base_element_profiles.open() as handle:
            combined = [row for row in csv.DictReader(handle)
                        if row["element"] != "K"]
        combined.extend(k_profile)
        combined.sort(key=lambda row: (str(row["element"]),
                                       int(row["test_id"])))
        write_csv(output_dir / "element_profiles_with_k.csv", combined)
    atomic_json(output_dir / "run_manifest.json", summary)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    for line, color in zip(K_LINES, ("#7b4ea3", "#4c1d78", "#9b72b0")):
        rows = by_line[line["line_id"]]
        height = np.array([row["height_um"] for row in rows]) / 1000
        value = np.array([row.get("line_relative", np.nan) for row in rows])
        axes[0].plot(height, value, lw=0.7, label=line["line_id"], color=color)
        fwhm = np.array([row.get("fwhm_nm", np.nan) for row in rows])
        axes[1].plot(height, fwhm, lw=0.6, label=line["line_id"], color=color)
    axes[0].set_ylabel("relative wide area")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("FWHM (nm)")
    height = np.array([row["height_um"] for row in optical_depth_rows]) / 1000
    ratio = np.array([row.get("peak_height_ratio_766_769", np.nan)
                      for row in optical_depth_rows])
    tau = np.array([row.get("tau_769_peak_model", np.nan)
                    for row in optical_depth_rows])
    axes[2].plot(height, ratio, color="#6a3d9a", lw=0.7)
    axes[2].axhline(EXPECTED_DOUBLET_RATIO, color="black", ls="--", lw=0.8)
    axes[2].set_ylabel("766/769 peak ratio")
    axes[3].plot(height, np.minimum(tau, 5), color="#6a3d9a", lw=0.7)
    axes[3].set_ylabel("tau 769 (cap 5)")
    axes[3].set_xlabel("height above sample bottom (mm)")
    fig.savefig(output_dir / "k_resonance_diagnostics.png", dpi=180)
    plt.close(fig)
    log("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
