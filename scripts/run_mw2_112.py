"""Immutable, resumable MW2-112 LIBS profile analysis.

Unlike the legacy directory CLI, this driver never writes into the input
directory.  It inventories all raw files, writes one atomic checkpoint per
completed spectrum, and rebuilds tables from those checkpoints on every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np

from alibz.pipeline import (
    DEFAULT_DRAWS,
    DEFAULT_N_CALLS,
    DEFAULT_TIMEOUT_S,
    _analyze_file,
    _error_row,
    resolve_dbpath,
    write_detections_csv,
    write_summary_csv,
)
from alibz.mw2_112 import (
    PROFILE_BEAM_DIAMETER_UM_APPROX,
    PROFILE_FIRST_ID,
    PROFILE_LAST_ID,
    PROFILE_STEP_UM,
    atomic_json as _atomic_json,
    build_inventory,
    calibration_job as _calibration_job,
    database_state,
    discover_spectra,
    jsonable as _jsonable,
    sha256_bytes as _sha256_bytes,
    software_state,
    utc_now as _utc_now,
    write_inventory,
    write_session_calibration,
)
from alibz.scattering import (
    DEFAULT_XRAY_Q_INV_A,
    REFERENCE_NEUTRON_WAVELENGTH_A,
    scattering_contributions,
    write_scattering_csv,
)
from alibz.session_calibration import (
    build_shared_calibration,
)


SCHEMA_VERSION = 1
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def parse_ids(spec: str) -> list[int]:
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending ID range: {part}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def choose_pilot_ids(entries: Sequence[dict], count: int = 20) -> list[int]:
    """Deterministic profile coverage plus observed detector/QC extremes."""
    available = {int(row["test_id"]): row for row in entries}
    fixed = [989, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1632,
             1700, 1800, 1874, 1875, 1900, 1915, 1916, 1917]
    chosen = [test_id for test_id in fixed if test_id in available]
    candidates = [row for row in entries if not row["all_zero"]]
    extrema = []
    for key, reverse in (("negative_fraction", True),
                         ("intensity_p99", True),
                         ("intensity_p99", False)):
        valid = [row for row in candidates if row[key] is not None]
        if valid:
            extrema.append(sorted(valid, key=lambda row: row[key],
                                  reverse=reverse)[0]["test_id"])
    for test_id in extrema:
        if test_id not in chosen:
            chosen.append(int(test_id))
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        quantile_ids = np.linspace(PROFILE_FIRST_ID, PROFILE_LAST_ID,
                                   count, dtype=int)
        for test_id in quantile_ids:
            nearest = min(available, key=lambda value: abs(value - test_id))
            if nearest not in chosen:
                chosen.append(nearest)
            if len(chosen) >= count:
                break
    return sorted(chosen[:count])


def _config_hash(manifest: dict) -> str:
    immutable = {
        "schema_version": manifest["schema_version"],
        "profile": manifest["profile"],
        "analysis": manifest["analysis"],
        "software": manifest["software"],
        "database": manifest["database"],
        "selected_inputs": manifest["selected_inputs"],
    }
    payload = json.dumps(_jsonable(immutable), sort_keys=True,
                         separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _checkpoint_valid(checkpoint: dict, entry: dict,
                      config_hash: str) -> bool:
    source = checkpoint.get("input") or {}
    return (
        checkpoint.get("schema_version") == SCHEMA_VERSION
        and checkpoint.get("config_hash") == config_hash
        and source.get("path") == entry["path"]
        and source.get("size_bytes") == entry["size_bytes"]
        and source.get("sha256") == entry["sha256"]
        and isinstance(checkpoint.get("row"), dict)
    )


def _analysis_job(job: tuple) -> tuple[int, dict, float]:
    test_id, args = job
    start = time.monotonic()
    return test_id, _analyze_file(args), time.monotonic() - start


def _write_tables(rows: Sequence[dict], output_dir: Path,
                  q_values: Sequence[float], neutron_wavelength_a: float) -> None:
    writers = (
        ("summary_raw.csv", lambda path: write_summary_csv(rows, str(path))),
        ("detections.csv", lambda path: write_detections_csv(rows, str(path))),
    )
    for name, writer in writers:
        final = output_dir / name
        tmp = final.with_name(f".{name}.{os.getpid()}.tmp")
        writer(tmp)
        os.replace(tmp, final)
    contrast = scattering_contributions(
        rows, q_values=q_values, neutron_wavelength_a=neutron_wavelength_a)
    final = output_dir / "scattering_proxies.csv"
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    write_scattering_csv(contrast, str(tmp))
    os.replace(tmp, final)

    raw_records = []
    resolved_records = []
    point_records = []
    for row in rows:
        common = {key: row.get(key, "") for key in
                  ("test_id", "height_um", "file", "sample")}
        for element, fraction in (row.get("fractions") or {}).items():
            raw_records.append({
                **common, "element": element, "fraction": fraction,
                "concentration": (row.get("concentrations") or {}).get(element),
                "unc": (row.get("uncertainties") or {}).get(element),
            })
        for det in row.get("detections") or []:
            resolved_records.append({
                **common,
                "element": det.get("element", ""),
                "status": det.get("status", ""),
                "fraction": det.get("fraction"),
                "fraction_resolved": det.get("fraction_resolved"),
                "fraction_hi": det.get("fraction_hi"),
                "unc": det.get("unc"),
                "upper_limit": det.get("upper_limit"),
                "z": det.get("z"),
                "n_lines": det.get("n_lines"),
                "clear_lines": det.get("clear_lines"),
                "confounder": det.get("confounder"),
            })
        point_records.append({
            **common,
            **{key: row.get(key, "") for key in (
                "status", "n_peaks", "shift_pm", "shift_segments_pm",
                "shift_anchor_counts", "shift_segment_applied",
                "shift_prior_applied",
                "response_620", "response_620_unc", "response_620_source",
                "T_K", "log_ne", "r_squared", "sa_converged",
                "qc_status", "qc_reasons", "flags")},
        })

    def write_records(name: str, records: list[dict], header: list[str]) -> None:
        final_path = output_dir / name
        tmp_path = final_path.with_name(f".{name}.{os.getpid()}.tmp")
        with tmp_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        os.replace(tmp_path, final_path)

    write_records(
        "composition_long.csv", raw_records,
        ["test_id", "height_um", "file", "sample", "element", "fraction",
         "concentration", "unc"],
    )
    write_records(
        "composition_resolved.csv", resolved_records,
        ["test_id", "height_um", "file", "sample", "element", "status",
         "fraction", "fraction_resolved", "fraction_hi", "unc",
         "upper_limit", "z", "n_lines", "clear_lines", "confounder"],
    )
    point_header = list(point_records[0]) if point_records else [
        "test_id", "height_um", "file", "sample", "status"]
    write_records("point_qc.csv", point_records, point_header)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _three_floats(value: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected three comma-separated numbers") from exc
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(
            "expected three comma-separated numbers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable MW2-112 LIBS analysis with immutable inputs")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--db", default=None)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--pilot", action="store_true",
                        help="run the deterministic 20-spectrum pilot")
    select.add_argument("--ids", help="comma-separated IDs/ranges")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--n-calls", type=_positive_int,
                        default=DEFAULT_N_CALLS)
    parser.add_argument("--draws", type=_positive_int, default=DEFAULT_DRAWS)
    parser.add_argument("--timeout", type=_positive_int,
                        default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--stimulated-emission",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--session-calibration", action=argparse.BooleanOptionalAction,
        default=True,
        help="pool neighboring shots for response and segment-shift priors")
    parser.add_argument("--response-fallback", type=float, default=None,
                        help="override weak-continuum 620 nm response")
    parser.add_argument(
        "--shift-offsets-pm", type=_three_floats, default=None,
        help="shared UV,VIS,NIR offsets relative to each shot's pooled "
             "shift, used only where a segment lacks an accepted shift")
    parser.add_argument("--xray-q", type=float, action="append",
                        help="X-ray Q [1/Angstrom], repeatable")
    parser.add_argument("--neutron-wavelength", type=float,
                        default=REFERENCE_NEUTRON_WAVELENGTH_A)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"not a directory: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("output_dir must be separate from the raw input tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    paths = discover_spectra(input_dir, args.pattern)
    log(f"discovered {len(paths)} spectra")
    inventory = build_inventory(paths, progress=log)
    write_inventory(inventory, output_dir / "input_inventory.csv")
    by_id = {int(row["test_id"]): row for row in inventory}

    if args.pilot:
        selected_ids = choose_pilot_ids(inventory)
    elif args.ids:
        try:
            selected_ids = parse_ids(args.ids)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        selected_ids = sorted(by_id)
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise SystemExit(f"selected IDs absent from input: {missing}")

    repo = Path(__file__).resolve().parents[1]
    dbpath = Path(resolve_dbpath(args.db))
    q_values = tuple(args.xray_q or DEFAULT_XRAY_Q_INV_A)
    shift_offsets_nm = (None if args.shift_offsets_pm is None else
                        tuple(value / 1000.0 for value in
                              args.shift_offsets_pm))
    calibration_by_id = {}
    calibration_hash = None
    if args.session_calibration and not args.inventory_only:
        log("building shared session calibration")
        jobs = [(int(entry["test_id"]), entry["path"], str(dbpath))
                for entry in inventory]
        measured_by_id = {}
        saved_threads_cal = {
            name: os.environ.get(name) for name in _BLAS_THREAD_VARS}
        saved_mpl = os.environ.get("MPLCONFIGDIR")
        mpl_dir = output_dir / ".mplconfig"
        mpl_dir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
        for name in _BLAS_THREAD_VARS:
            os.environ[name] = "1"
        try:
            if args.workers == 1:
                for i, job in enumerate(jobs, 1):
                    test_id, measurement = _calibration_job(job)
                    measured_by_id[test_id] = measurement
                    if i % 100 == 0 or i == len(jobs):
                        log(f"calibration [{i}/{len(jobs)}]")
            else:
                import multiprocessing as mp
                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=args.workers,
                                         mp_context=context) as pool:
                    futures = [pool.submit(_calibration_job, job)
                               for job in jobs]
                    for i, future in enumerate(as_completed(futures), 1):
                        test_id, measurement = future.result()
                        measured_by_id[test_id] = measurement
                        if i % 100 == 0 or i == len(jobs):
                            log(f"calibration [{i}/{len(jobs)}]")
        finally:
            for name, value in saved_threads_cal.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            if saved_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = saved_mpl
        measurements = [measured_by_id[int(entry["test_id"])]
                        for entry in inventory]
        calibration_records = build_shared_calibration(
            inventory, measurements)
        write_session_calibration(
            calibration_records, output_dir / "session_calibration.csv")
        calibration_by_id = {
            int(record["test_id"]): record for record in calibration_records}
        calibration_payload = json.dumps(
            _jsonable(calibration_records), sort_keys=True,
            separators=(",", ":")).encode()
        calibration_hash = _sha256_bytes(calibration_payload)
        n_response = sum(record["response_prior"] is not None
                         for record in calibration_records)
        n_shift = [sum(record["shift_prior_nm"][segment] is not None
                       for record in calibration_records)
                   for segment in range(3)]
        log(f"session calibration response={n_response}/{len(inventory)} "
            f"shift_uv_vis_nir={n_shift}")
    selected_inputs = [
        {key: by_id[test_id][key]
         for key in ("test_id", "path", "size_bytes", "sha256")}
        for test_id in selected_ids
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "inventory_only" if args.inventory_only else "running",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "profile": {
            "sample": "MW2-112",
            "first_test_id": PROFILE_FIRST_ID,
            "last_test_id": PROFILE_LAST_ID,
            "step_um": PROFILE_STEP_UM,
            "origin": "sample bottom",
            "positive_direction": "upward along sedimentation/gravity",
            "single_pulse_per_csv": True,
            "beam_diameter_um_approx": PROFILE_BEAM_DIAMETER_UM_APPROX,
        },
        "analysis": {
            "n_calls": args.n_calls,
            "draws": args.draws,
            "timeout_s": args.timeout,
            "workers": args.workers,
            "stimulated_emission": args.stimulated_emission,
            "segment_response_fallback_ratio": args.response_fallback,
            "segment_shift_offsets_nm": shift_offsets_nm,
            "shared_session_calibration": args.session_calibration,
            "shared_session_calibration_sha256": calibration_hash,
            "natural_isotopic_abundance": True,
            "xray_q_inv_a": q_values,
            "neutron_wavelength_a": args.neutron_wavelength,
            "claim_level": "relative emitter fractions and contrast proxies",
        },
        "software": software_state(repo),
        "database": database_state(dbpath),
        "selected_inputs": selected_inputs,
        "selected_ids": selected_ids,
        "inventory_count": len(inventory),
    }
    manifest["config_hash"] = _config_hash(manifest)
    manifest_path = output_dir / "run_manifest.json"
    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("config_hash") != manifest["config_hash"]:
            raise SystemExit(
                "existing run manifest is incompatible; choose a new output "
                "directory or restore the exact code/database/config/input")
        manifest["created_utc"] = existing.get("created_utc",
                                                manifest["created_utc"])
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_dir / "config.json", {
        "config_hash": manifest["config_hash"],
        "profile": manifest["profile"],
        "analysis": manifest["analysis"],
        "database_tree_sha256": manifest["database"]["tree_sha256"],
        "source_tree_sha256": manifest["software"]["source_tree_sha256"],
    })
    log(f"config_hash={manifest['config_hash']} selected={selected_ids}")
    if args.inventory_only:
        return 0

    rows_by_id = {}
    pending = []
    for test_id in selected_ids:
        entry = by_id[test_id]
        checkpoint_path = checkpoints_dir / f"{test_id:04d}.json"
        checkpoint = None
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
            except (OSError, json.JSONDecodeError):
                checkpoint = None
        if checkpoint and _checkpoint_valid(checkpoint, entry,
                                            manifest["config_hash"]):
            rows_by_id[test_id] = checkpoint["row"]
            log(f"resume {test_id}: {checkpoint['row'].get('status')}")
        else:
            calibration = calibration_by_id.get(test_id) or {}
            if args.response_fallback is not None:
                response_fallback = args.response_fallback
                response_source = "fallback_override"
                response_uncertainty = None
            else:
                response_fallback = calibration.get("response_prior")
                response_source = ("shared_profile"
                                   if response_fallback is not None else None)
                response_uncertainty = calibration.get(
                    "response_prior_uncertainty")
            shift_prior = None
            if shift_offsets_nm is None and calibration.get("shift_prior_nm"):
                shift_prior = tuple(
                    float(value) if value is not None else float("nan")
                    for value in calibration["shift_prior_nm"])
            job_args = (
                entry["path"], str(dbpath), args.n_calls, args.draws,
                args.timeout, args.stimulated_emission,
                response_fallback, shift_offsets_nm, shift_prior,
                response_source, response_uncertainty,
            )
            pending.append((test_id, job_args))

    def record(test_id: int, row: dict, elapsed_s: float) -> None:
        entry = by_id[test_id]
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": manifest["config_hash"],
            "completed_utc": _utc_now(),
            "elapsed_s": elapsed_s,
            "input": {key: entry[key] for key in
                      ("test_id", "path", "size_bytes", "sha256")},
            "row": row,
        }
        _atomic_json(checkpoints_dir / f"{test_id:04d}.json", checkpoint)
        rows_by_id[test_id] = _jsonable(row)
        log(f"complete {test_id} ({elapsed_s:.1f}s): {row.get('status')}")

    interrupted = False
    saved_threads = {name: os.environ.get(name) for name in _BLAS_THREAD_VARS}
    for name in _BLAS_THREAD_VARS:
        os.environ[name] = "1"
    try:
        if args.workers == 1:
            for job in pending:
                record(*_analysis_job(job))
        elif pending:
            import multiprocessing as mp
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=context) as pool:
                futures = {pool.submit(_analysis_job, job): job[0]
                           for job in pending}
                for future in as_completed(futures):
                    record(*future.result())
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted; completed checkpoints are preserved")
    finally:
        for name, value in saved_threads.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assembled = []
    for test_id in selected_ids:
        row = rows_by_id.get(test_id)
        if row is None:
            row = _error_row(by_id[test_id]["path"], "pending")
            row["status"] = "pending"
        row = dict(row)
        row["test_id"] = test_id
        row["height_um"] = by_id[test_id]["height_um"]
        assembled.append(row)
    _write_tables(assembled, output_dir, q_values,
                  args.neutron_wavelength)

    errors = sum(row.get("status") != "ok" for row in assembled)
    completed = len(rows_by_id)
    manifest["completed_utc"] = _utc_now()
    manifest["completed_checkpoints"] = completed
    manifest["error_or_pending_rows"] = errors
    manifest["status"] = ("interrupted" if interrupted else
                          "complete" if completed == len(selected_ids)
                          and errors == 0 else
                          "completed_with_errors" if completed == len(selected_ids)
                          else "partial")
    _atomic_json(manifest_path, manifest)
    log(f"status={manifest['status']} checkpoints={completed}/"
        f"{len(selected_ids)} errors_or_pending={errors}")
    if interrupted:
        return 130
    return 0 if completed == len(selected_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
