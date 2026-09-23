#!/usr/bin/env python3
"""Validate independent Ar I/O I wavelength calibration on pinned spectra.

The real-data portion is an availability and repeatability audit: the archive
does not contain independently labelled argon/oxygen wavelength truth.  A
separate semi-synthetic check adds known, independently shifted Ar I and O I
lines to one measured background.  Gas calibration always receives the raw
instrument wavelength axis and observed-frame peaks.  The mixed-element
calibration is computed afterwards and is retained only as a comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/alibz-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alibz.gas_calibration import calibrate_background_gases
from alibz.utils.database import Database
from alibz.wavelength_calibration import apply_gas_calibrations
from scripts.benchmark_physical_triage import (
    blind_shift,
    crop_to_coverage,
    load_replay_archive,
    production_peaks,
    quick_peaks,
)


DEFAULT_MANIFEST = Path("provenance/physical-triage-real-data-20260922.json")
DEFAULT_ARCHIVE = Path("provenance/physical-triage-real-data-20260922.npz")
DEFAULT_OUT = Path("reports/gas-calibration-real-20260923.json")

# Exact air wavelengths from the pinned repository database.  Ar uses clean
# singleton groups.  O uses every resolved component of two NIR multiplets;
# the benchmark never substitutes one strongest wavelength for a blend.
INJECTION_ANCHORS_NM = {
    "Ar": (
        696.542978, 706.721736, 738.397998,
        750.386765, 763.510524, 794.817572,
    ),
    "O": (
        645.360246, 645.444423, 645.597682,
        777.194430, 777.416570, 777.538737,
    ),
}
INJECTION_CASES_NM = (
    {"Ar": 0.070, "O": -0.060},
    {"Ar": -0.070, "O": 0.060},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_inputs(manifest_path: Path, archive_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "physical-triage-real-data-v1":
        raise ValueError("unsupported or missing real-data manifest schema")
    datasets = manifest.get("datasets", [])
    if len(datasets) != 1:
        raise ValueError("gas benchmark expects exactly one pinned dataset")
    dataset = datasets[0]
    archive_meta = dataset.get("repo_local_archive", {})
    expected = archive_meta.get("sha256")
    if not expected:
        raise ValueError("manifest does not pin the replay archive hash")
    records, actual = load_replay_archive(archive_path, expected)
    if len(records) != archive_meta.get("run_means") or len(records) != 26:
        raise ValueError("expected all 26 archived real run means")
    intervals = dataset.get("grid", {}).get("coverage_intervals_nm", [])
    if not intervals:
        raise ValueError("manifest has no audited coverage")
    # The manifest explicitly flags its dense post-gap tail as invalid for
    # peak work.  Only the first interval is an analysis interval here.
    coverage = [tuple(map(float, intervals[0]))]
    return manifest, dataset, records, coverage, actual


def _mixed_baseline(peaks_observed: np.ndarray, db: Database):
    """Compute the legacy mixed-element estimate only for later comparison."""
    shift, anchors, _ = blind_shift(peaks_observed, db)
    return shift, {
        "method": "estimate_wavelength_shift_segments over all database elements",
        "role": "comparison_only_never_gas_seed",
        "offset_nm": float(shift),
        "anchor_matches": int(anchors),
        "representation": repr(shift),
    }


def calibrate_one(x_raw: np.ndarray, y_raw: np.ndarray,
                  peaks_observed: np.ndarray, db: Database,
                  calibrator=calibrate_background_gases) -> tuple[dict, dict, dict]:
    """Preserve the critical ordering and coordinate-frame separation."""
    gas = calibrator(x_raw, y_raw, db, peak_array=peaks_observed)
    mixed_shift, mixed = _mixed_baseline(peaks_observed, db)
    _, application = apply_gas_calibrations(mixed_shift, gas, mode="apply")
    return gas, mixed, application


def _summary(samples: list[dict]) -> dict:
    result = {}
    for species in ("Ar", "O"):
        statuses = {name: 0 for name in
                    ("calibrated", "tentative", "no_evidence", "out_of_range")}
        offsets, uncertainties = [], []
        for sample in samples:
            record = sample["gas_calibration"][species]
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1
            if record["status"] in {"calibrated", "tentative"}:
                if record.get("offset_nm") is not None:
                    offsets.append(float(record["offset_nm"]))
                if record.get("uncertainty_nm") is not None:
                    uncertainties.append(float(record["uncertainty_nm"]))
        result[species] = {
            "status_counts": statuses,
            "offset_nm_median": float(np.median(offsets)) if offsets else None,
            "offset_nm_range": [min(offsets), max(offsets)] if offsets else None,
            "uncertainty_nm_median": (
                float(np.median(uncertainties)) if uncertainties else None),
            "truth_available": False,
            "accuracy_scored": False,
        }
    mixed = [float(row["mixed_element_baseline"]["offset_nm"])
             for row in samples]
    result["mixed_element_baseline"] = {
        "offset_nm_median": float(np.median(mixed)),
        "offset_nm_range": [min(mixed), max(mixed)],
        "role": "comparison_only_never_gas_seed",
    }
    return result


def _inject_on_measured_background(x: np.ndarray, y: np.ndarray,
                                   offsets_nm: dict[str, float]):
    """Add controlled lines without manufacturing fitted peak centers."""
    out = np.asarray(y, dtype=float).copy()
    p50, p99 = np.percentile(out, [50.0, 99.0])
    height = max(5000.0, 2.5 * max(p99 - p50, 1.0))
    injected_truth = []
    local_pitches = []
    sigmas = []
    for species, wavelengths in INJECTION_ANCHORS_NM.items():
        offset = float(offsets_nm[species])
        for wavelength in wavelengths:
            center = wavelength + offset
            nearest = int(np.argmin(np.abs(x - center)))
            lo, hi = max(0, nearest - 3), min(len(x), nearest + 4)
            pitch = float(np.median(np.diff(x[lo:hi])))
            sigma_nm = max(0.035, 0.55 * pitch)
            out += height * np.exp(-0.5 * ((x - center) / sigma_nm) ** 2)
            injected_truth.append({"species": species,
                                   "database_wavelength_nm": wavelength,
                                   "injected_center_nm": center})
            local_pitches.append(pitch)
            sigmas.append(sigma_nm)
    return out, injected_truth, {
        "line_height_counts": height,
        "gaussian_sigma_nm_range": [min(sigmas), max(sigmas)],
        "local_native_pitch_nm_range": [min(local_pitches), max(local_pitches)],
        "peak_center_origin": "blind production refit from injected raw spectrum",
        "truth_centers_passed_to_calibrator": False,
    }


def evaluate_semi_synthetic_case(x_raw: np.ndarray, y_raw: np.ndarray,
                                 offsets: dict[str, float], db: Database,
                                 *, peak_extractor=production_peaks,
                                 calibrator=calibrate_background_gases) -> dict:
    """Blindly re-extract a controlled injection on the native measured grid."""
    y_injected, injected_truth, injection = _inject_on_measured_background(
        x_raw, y_raw, offsets)
    extracted = peak_extractor(x_raw, y_injected)
    observed_peaks = extracted[0] if isinstance(extracted, tuple) else extracted
    gas, mixed, application = calibrate_one(
        x_raw, y_injected, observed_peaks, db, calibrator=calibrator)
    recovery = {}
    for species in ("Ar", "O"):
        estimate = gas[species].get("offset_nm")
        recovery[species] = {
            "expected_offset_nm": offsets[species],
            "estimated_offset_nm": estimate,
            "error_nm": (None if estimate is None else
                         float(estimate) - offsets[species]),
            "status": gas[species]["status"],
            "accuracy_scored": gas[species]["status"] == "calibrated",
        }
    return {
        "classification": "semi_synthetic_end_to_end_not_real_calibration_truth",
        "independent_injected_offsets_nm": dict(offsets),
        "common_offset_forced": False,
        "anchor_wavelengths_air_nm": INJECTION_ANCHORS_NM,
        "injected_truth": injected_truth,
        "injection": injection,
        "blind_observed_peak_count": int(len(observed_peaks)),
        "gas_calibration": gas,
        "recovery": recovery,
        "mixed_element_baseline": mixed,
        "automatic_regional_application": application,
    }


def run_benchmark(manifest_path: Path, archive_path: Path,
                  limit: int | None = None,
                  production_checks: int = 3) -> dict:
    manifest, dataset, records, coverage, archive_hash = load_inputs(
        manifest_path, archive_path)
    db = Database("db")
    real_samples = []
    first_background = None
    for case_index, (metadata, x_native, y_native) in enumerate(records[:limit]):
        x_raw, y_raw = crop_to_coverage(x_native, y_native, coverage)
        if first_background is None:
            first_background = (metadata, x_raw, y_raw)
        peaks_observed, _, _ = quick_peaks(x_raw, y_raw, coverage)
        gas, mixed, application = calibrate_one(
            x_raw, y_raw, peaks_observed, db)
        item = {
            "run_id": metadata["run_id"],
            "ledger_index": metadata.get("ledger_index"),
            "shots_averaged": metadata.get("shots_averaged"),
            "native_rows": int(len(x_native)),
            "analysis_rows": int(len(x_raw)),
            "analysis_range_nm": [float(x_raw[0]), float(x_raw[-1])],
            "observed_peak_count": int(len(peaks_observed)),
            "gas_input_frame": "raw instrument axis and observed-frame peaks",
            "gas_calibration": gas,
            "mixed_element_baseline": mixed,
            "automatic_regional_application": application,
        }
        if case_index < max(0, production_checks):
            production_observed, _, _ = production_peaks(x_raw, y_raw)
            production_gas, production_mixed, production_application = calibrate_one(
                x_raw, y_raw, production_observed, db)
            item["production_peak_check"] = {
                "observed_peak_count": int(len(production_observed)),
                "gas_calibration": production_gas,
                "mixed_element_baseline": production_mixed,
                "automatic_regional_application": production_application,
            }
        real_samples.append(item)

    semi_synthetic = []
    if first_background is not None:
        metadata, x_raw, y_raw = first_background
        for offsets in INJECTION_CASES_NM:
            item = evaluate_semi_synthetic_case(x_raw, y_raw, offsets, db)
            item["background_run_id"] = metadata["run_id"]
            semi_synthetic.append(item)

    engine_path = REPO / "alibz/gas_calibration.py"
    application_path = REPO / "alibz/wavelength_calibration.py"
    db_path = REPO / "db/el_lines92.pickle"
    script_path = Path(__file__).resolve()
    return {
        "schema_version": "gas-calibration-real-benchmark-v1",
        "truth_policy": {
            "real_spectra": "No independently labelled Ar I/O I wavelength truth is available; statuses, offsets, uncertainty, and repeatability are reported without accuracy claims.",
            "semi_synthetic": "Known injected offsets test recovery on a measured background but are not real calibration truth.",
        },
        "configuration": {
            "peak_method": "quick native-grid extractor for all run means",
            "production_peak_checks": min(max(0, production_checks), len(real_samples)),
            "gas_coordinate_frame": "raw instrument wavelength axis",
            "gas_peak_frame": "observed instrument frame; never mixed/Fe shifted",
            "coverage_intervals_nm": coverage,
            "manifest_known_gap_nm": dataset["grid"].get("known_gap_nm"),
            "excluded_after_valid_coverage_nm": [
                coverage[0][1], float(records[0][1][-1])],
            "invalid_tail_excluded": True,
            "requested_limit": limit,
        },
        "provenance": {
            "git_head": _git_head(),
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "archive": {"path": str(archive_path), "sha256": archive_hash,
                        "run_means": len(records)},
            "benchmark_script": {"path": str(script_path.relative_to(REPO)),
                                 "sha256": sha256(script_path)},
            "gas_calibration_source": {
                "path": str(engine_path.relative_to(REPO)),
                "sha256": sha256(engine_path),
            },
            "wavelength_application_source": {
                "path": str(application_path.relative_to(REPO)),
                "sha256": sha256(application_path),
            },
            "atomic_database": {"path": str(db_path.relative_to(REPO)),
                                "sha256": sha256(db_path)},
        },
        "real_data": {
            "dataset_id": dataset["id"],
            "run_means_available": len(records),
            "run_means_evaluated": len(real_samples),
            "independent_gas_truth_available": False,
            "accuracy_scored": False,
            "summary": _summary(real_samples),
            "samples": real_samples,
        },
        "semi_synthetic_recovery": semi_synthetic,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--production-checks", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest, dataset, records, coverage, archive_hash = load_inputs(
        args.manifest, args.replay_archive)
    if args.dry_run:
        print(json.dumps({
            "status": "ok",
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "archive": str(args.replay_archive),
            "archive_sha256": archive_hash,
            "dataset_id": dataset["id"],
            "run_means": len(records),
            "coverage_intervals_nm": coverage,
            "invalid_tail_excluded": True,
            "would_write": str(args.out),
        }, indent=2))
        return 0
    result = run_benchmark(args.manifest, args.replay_archive, args.limit,
                           max(0, args.production_checks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "ok", "out": str(args.out),
        "real_run_means": result["real_data"]["run_means_evaluated"],
        "real_summary": result["real_data"]["summary"],
        "semi_synthetic_recovery": [row["recovery"] for row in
                                    result["semi_synthetic_recovery"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
