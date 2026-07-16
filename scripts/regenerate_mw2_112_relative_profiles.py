"""Regenerate MW2-112 products from raw spectra and frozen run priors."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alibz.mw2_112 import (
    load_frozen_calibration,
    relative_measurement_job,
    write_csv,
    write_relative_tables,
)
from scripts.mw2_112_provenance import (
    hash_file,
    utc_now,
    verify_input_inventory,
)
from scripts.reassemble_relative_profiles import main as reassemble_main
from scripts.report_mw2_112_relative_profiles import main as report_main


DETERMINISTIC_TABLES = (
    "line_profiles.csv",
    "relative_profile_qc.csv",
    "relative_element_profiles.csv",
    "relative_contrast_profiles.csv",
    "layer_boundaries.csv",
    "mineral_association_indices.csv",
    "scattering_reference.csv",
    "vendor_validation.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-measure MW2-112 with frozen line/calibration priors")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vendor-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    vendor_dir = args.vendor_dir.expanduser().resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("output directory must be outside the raw input tree")

    failures = verify_input_inventory(bundle / "input_inventory.csv", input_dir)
    if failures:
        raise SystemExit("raw-input verification failed:\n" + "\n".join(failures))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input_inventory.csv", "session_calibration.csv",
                 "line_feature_manifest.json", "run_manifest.json"):
        shutil.copy2(bundle / name, output_dir / name)

    with (bundle / "input_inventory.csv").open() as fh:
        inventory = list(csv.DictReader(fh))
    calibration = load_frozen_calibration(bundle / "session_calibration.csv")
    features = json.loads(
        (bundle / "line_feature_manifest.json").read_text())["features"]
    jobs = []
    for row in inventory:
        test_id = int(row["test_id"])
        jobs.append((
            test_id,
            str(input_dir / row["file"]),
            float(row["height_um"]),
            calibration[test_id],
            features,
        ))
    if args.workers == 1:
        results = [relative_measurement_job(job) for job in jobs]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(relative_measurement_job, jobs))
    results.sort(key=lambda item: item[0])
    errors = [(test_id, error) for test_id, _records, _qc, status,
              error, _elapsed in results if status != "ok"]
    if errors:
        raise SystemExit(f"measurement failures: {errors}")
    line_records = [record for _test_id, records, _qc, _status, _error,
                    _elapsed in results for record in records]
    qc_records = [qc for _test_id, _records, qc, _status, _error,
                  _elapsed in results]
    write_csv(output_dir / "line_profiles.csv", line_records)
    write_csv(output_dir / "relative_profile_qc.csv", qc_records)
    write_relative_tables(output_dir, line_records)

    reassemble_main([str(output_dir)])
    report_main([str(output_dir), "--vendor-dir", str(vendor_dir)])

    comparisons = {}
    mismatches = []
    for name in DETERMINISTIC_TABLES:
        expected = hash_file(bundle / name)
        actual = hash_file(output_dir / name)
        match = actual == expected
        comparisons[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "byte_identical": match,
        }
        if not match:
            mismatches.append(name)
    (output_dir / "regeneration_comparison.json").write_text(
        json.dumps({
            "completed_utc": utc_now(),
            "reference_bundle": str(bundle),
            "raw_input_dir": str(input_dir),
            "vendor_input_dir": str(vendor_dir),
            "python": sys.version,
            "frozen_inputs": {
                "line_feature_manifest_sha256": hash_file(
                    bundle / "line_feature_manifest.json"),
                "session_calibration_sha256": hash_file(
                    bundle / "session_calibration.csv"),
                "input_inventory_sha256": hash_file(
                    bundle / "input_inventory.csv"),
            },
            "tables": comparisons,
            "all_byte_identical": not mismatches,
        }, indent=2, sort_keys=True) + "\n")
    if mismatches:
        print("regeneration completed, but tables differ: "
              + ", ".join(mismatches))
        return 1
    print(f"regeneration passed: {len(DETERMINISTIC_TABLES)} tables are "
          "byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
