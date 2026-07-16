"""Build full-corpus, multi-line relative element profiles for MW2-112."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from alibz.pipeline import resolve_dbpath
from alibz.mw2_112 import (
    FEATURES_PER_STAGE,
    REFERENCE_TEMPERATURE_K,
    atomic_json,
    build_inventory,
    calibration_job,
    database_state,
    discover_spectra,
    jsonable,
    relative_measurement_job,
    sha256_bytes,
    software_state,
    utc_now,
    write_csv,
    write_inventory,
    write_relative_tables,
    write_session_calibration,
)
from alibz.relative_profiles import (
    DEFAULT_ELEMENTS,
    build_line_features,
)
from alibz.session_calibration import build_shared_calibration


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--db", default=None)
    parser.add_argument("--workers", type=_positive_int, default=6)
    parser.add_argument("--features-per-stage", type=_positive_int,
                        default=FEATURES_PER_STAGE)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("output directory must be outside the raw input tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message):
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    dbpath = Path(resolve_dbpath(args.db))
    paths = discover_spectra(input_dir)
    inventory = build_inventory(paths, progress=log)
    write_inventory(inventory, output_dir / "input_inventory.csv")
    by_id = {int(row["test_id"]): row for row in inventory}

    log("measuring session response, shifts, and shared peak widths")
    cal_jobs = [(int(row["test_id"]), row["path"], str(dbpath))
                for row in inventory]
    measured = {}
    if args.workers == 1:
        for i, job in enumerate(cal_jobs, 1):
            test_id, result = calibration_job(job)
            measured[test_id] = result
            if i % 100 == 0 or i == len(cal_jobs):
                log(f"calibration [{i}/{len(cal_jobs)}]")
    else:
        import multiprocessing as mp
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=context) as pool:
            futures = [pool.submit(calibration_job, job) for job in cal_jobs]
            for i, future in enumerate(as_completed(futures), 1):
                test_id, result = future.result()
                measured[test_id] = result
                if i % 100 == 0 or i == len(cal_jobs):
                    log(f"calibration [{i}/{len(cal_jobs)}]")
    calibration = build_shared_calibration(
        inventory, [measured[int(row["test_id"])] for row in inventory])
    write_session_calibration(calibration, output_dir / "session_calibration.csv")
    calibration_by_id = {int(row["test_id"]): row for row in calibration}

    features = build_line_features(
        str(dbpath), elements=DEFAULT_ELEMENTS,
        temperature_k=REFERENCE_TEMPERATURE_K,
        features_per_stage=args.features_per_stage)
    atomic_json(output_dir / "line_feature_manifest.json", {
        "reference_temperature_k": REFERENCE_TEMPERATURE_K,
        "elements": DEFAULT_ELEMENTS,
        "features": features,
    })
    calibration_hash = sha256_bytes(json.dumps(
        jsonable(calibration), sort_keys=True,
        separators=(",", ":")).encode())
    feature_hash = sha256_bytes(json.dumps(
        jsonable(features), sort_keys=True,
        separators=(",", ":")).encode())
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "running",
        "method": "within-element multi-line relative profiles",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_inputs": len(inventory),
        "calibration_sha256": calibration_hash,
        "feature_sha256": feature_hash,
        "software": software_state(repo),
        "database": database_state(dbpath),
    }
    manifest["config_hash"] = sha256_bytes(json.dumps(
        jsonable({key: manifest[key] for key in (
            "schema_version", "method", "n_inputs", "calibration_sha256",
            "feature_sha256", "software", "database")}),
        sort_keys=True, separators=(",", ":")).encode())
    atomic_json(output_dir / "run_manifest.json", manifest)

    jobs = []
    completed = {}
    for test_id in sorted(by_id):
        checkpoint = checkpoints / f"{test_id:04d}.json"
        if checkpoint.exists():
            try:
                saved = json.loads(checkpoint.read_text())
            except json.JSONDecodeError:
                saved = None
            if saved and saved.get("config_hash") == manifest["config_hash"]:
                completed[test_id] = saved
                continue
        row = by_id[test_id]
        jobs.append((test_id, row["path"], row["height_um"],
                     calibration_by_id[test_id], features))

    def record(result):
        test_id, records, qc, status, error, elapsed = result
        checkpoint = {
            "config_hash": manifest["config_hash"], "test_id": test_id,
            "status": status, "error": error, "elapsed_s": elapsed,
            "records": records, "qc": qc,
        }
        atomic_json(checkpoints / f"{test_id:04d}.json", checkpoint)
        completed[test_id] = checkpoint
        if len(completed) % 50 == 0 or status != "ok":
            log(f"profiles [{len(completed)}/{len(inventory)}] {test_id} {status}")

    if args.workers == 1:
        for job in jobs:
            record(relative_measurement_job(job))
    elif jobs:
        import multiprocessing as mp
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=context) as pool:
            for result in pool.map(relative_measurement_job, jobs):
                record(result)

    line_records = [record for test_id in sorted(completed)
                    for record in completed[test_id]["records"]]
    qc_records = [completed[test_id]["qc"] for test_id in sorted(completed)]
    write_csv(output_dir / "line_profiles.csv", line_records)
    write_csv(output_dir / "relative_profile_qc.csv", qc_records)
    write_relative_tables(output_dir, line_records)
    errors = sum(record["status"] != "ok" for record in completed.values())
    manifest.update({
        "status": "complete" if errors == 0 else "completed_with_errors",
        "completed_utc": utc_now(), "completed_checkpoints": len(completed),
        "errors": errors, "line_features": len(features),
    })
    atomic_json(output_dir / "run_manifest.json", manifest)
    log(f"status={manifest['status']} features={len(features)} errors={errors}")
    return 0 if len(completed) == len(inventory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
