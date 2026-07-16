"""Freeze and verify provenance for the MW2-112 relative-profile bundle.

The original run manifest predates a portable provenance bundle: several
analysis sources were untracked and its tree hashes included absolute paths.
This utility captures a restorable source overlay, portable file manifests,
the exact Python package set, and checksums for every shipped artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from alibz.mw2_112 import hash_file, utc_now


SCHEMA_VERSION = 1
RUNTIME_SOURCE_FILES = (
    "alibz/detector.py",
    "alibz/mw2_112.py",
    "alibz/pipeline.py",
    "alibz/relative_profiles.py",
    "alibz/scattering.py",
    "alibz/session_calibration.py",
    "corrections/detector.json",
    "pyproject.toml",
    "scripts/build_mw2_112_relative_profiles.py",
    "scripts/mw2_112.py",
    "scripts/mw2_112_provenance.py",
    "scripts/reassemble_relative_profiles.py",
    "scripts/regenerate_mw2_112_relative_profiles.py",
    "scripts/report_mw2_112_relative_profiles.py",
    "scripts/run_mw2_112.py",
)
ATOMIC_PROVENANCE_FILES = (
    "db/atomic_line_sources.json",
    "db/observed_lines_nist.tsv",
    "docs/atomic_data.md",
)
CORE_ARTIFACTS = (
    "README.md",
    "PROVENANCE.md",
    "analysis_plan.md",
    "analysis_report.md",
    "input_inventory.csv",
    "layer_boundaries.csv",
    "line_feature_manifest.json",
    "line_profiles.csv",
    "mineral_association_indices.csv",
    "relative_contrast_profiles.csv",
    "relative_element_profiles.csv",
    "relative_profile_qc.csv",
    "run.log",
    "run_manifest.json",
    "scattering_reference.csv",
    "session_calibration.csv",
    "vendor_validation.csv",
)


def aggregate_manifest_hash(rows: Iterable[dict]) -> str:
    """Hash file identity independent of the filesystem's absolute path."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        digest.update(str(row["path"]).encode())
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def file_manifest(root: Path, paths: Iterable[Path] | None = None) -> list[dict]:
    root = root.resolve()
    candidates = paths if paths is not None else root.rglob("*")
    rows = []
    for path in sorted((Path(value).resolve() for value in candidates),
                       key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        })
    return rows


def write_csv(path: Path, rows: Sequence[dict], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def vendor_manifest(vendor_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(vendor_dir.glob("*.csv")):
        rows.append({
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        })
    return rows


def _copy_relative(source_root: Path, destination_root: Path,
                   relatives: Iterable[str]) -> list[Path]:
    copied = []
    for relative in relatives:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def freeze_bundle(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    vendor_dir = args.vendor_dir.expanduser().resolve()
    db_dir = args.db.expanduser().resolve()
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_root / "docs/mw2_112/analysis_plan.md",
                 run_dir / "analysis_plan.md")
    shutil.copy2(source_root / "docs/mw2_112/provenance.md",
                 run_dir / "PROVENANCE.md")

    original = json.loads((run_dir / "run_manifest.json").read_text())
    base_commit = original["software"]["git_commit"]
    source_archive = provenance_dir / f"alibz_base_{base_commit}.tar.gz"
    subprocess.run([
        "git", "archive", "--format=tar.gz", f"--output={source_archive}",
        base_commit,
    ], cwd=source_root, check=True)

    overlay = provenance_dir / "source_overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    _copy_relative(source_root, overlay, RUNTIME_SOURCE_FILES)
    _copy_relative(source_root, provenance_dir / "atomic_provenance",
                   ATOMIC_PROVENANCE_FILES)

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    freeze = [line for line in freeze if not line.lower().startswith(
        ("-e git+", "alibz=="))]
    lock_path = provenance_dir / "environment-py3.13-macos-arm64.lock.txt"
    lock_path.write_text(
        "# Captured from the production virtual environment.\n"
        "# Install this lock, then install the restored alibz tree with "
        "--no-deps.\n" + "\n".join(freeze) + "\n")

    raw_rows = []
    with (run_dir / "input_inventory.csv").open() as fh:
        for row in csv.DictReader(fh):
            raw_rows.append({
                "path": row["file"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
            })
    vendor_rows = vendor_manifest(vendor_dir)
    write_csv(provenance_dir / "vendor_input_inventory.csv", vendor_rows,
              ("path", "size_bytes", "sha256"))

    db_rows = file_manifest(db_dir)
    write_csv(provenance_dir / "atomic_database_inventory.csv", db_rows,
              ("path", "size_bytes", "sha256"))

    source_rows = file_manifest(overlay)
    atomic_rows = file_manifest(provenance_dir / "atomic_provenance")
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "sample": "MW2-112",
        "method": "within-element multi-line relative profiles",
        "canonical_regeneration": {
            "description": (
                "Re-measure raw spectra using the shipped line-feature and "
                "session-calibration tables, then rebuild all tables and "
                "figures. This route does not re-estimate the frozen priors."
            ),
            "command": (
                "mw2-112 regenerate "
                "BUNDLE RAW_DIR NEW_OUTPUT --vendor-dir VENDOR_DIR"
            ),
            "features_per_stage": 5,
            "reference_temperature_k": 9000.0,
            "workers_do_not_affect_scientific_configuration": True,
        },
        "from_first_principles_audit": {
            "description": (
                "Re-estimate session priors and line selection from the "
                "inventoried atomic database. This is an audit rerun and may "
                "not be byte-identical to the frozen-prior product."
            ),
            "commands": [
                "mw2-112 relative RAW_DIR "
                "NEW_OUTPUT --db DB_DIR --workers 6 --features-per-stage 5",
                "mw2-112 reassemble NEW_OUTPUT",
                "mw2-112 report "
                "NEW_OUTPUT --vendor-dir VENDOR_DIR",
            ],
        },
        "profile_geometry": {
            "first_test_id": 989,
            "last_test_id": 1917,
            "step_um": 100.0,
            "beam_diameter_um_approx": 50.0,
            "origin": "sample bottom",
            "positive_direction": "upward along sedimentation/gravity",
            "single_pulse_per_csv": True,
            "mapping": "height_um = (test_id - 989) * 100",
        },
    }
    write_json(run_dir / "reproduction_recipe.json", recipe)

    database_portable_hash = aggregate_manifest_hash(db_rows)
    source_portable_hash = aggregate_manifest_hash(source_rows)
    raw_portable_hash = aggregate_manifest_hash(raw_rows)
    vendor_portable_hash = aggregate_manifest_hash(vendor_rows)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "sample": "MW2-112",
        "claim_level": (
            "within-element relative spatial profiles; not cross-element or "
            "absolute concentration"
        ),
        "original_run": {
            "created_utc": original.get("created_utc"),
            "completed_utc": original.get("completed_utc"),
            "git_commit": base_commit,
            "git_dirty": original["software"].get("git_dirty"),
            "git_diff_sha256_path_dependent": original["software"].get(
                "git_diff_sha256"),
            "source_tree_sha256_path_dependent": original["software"].get(
                "source_tree_sha256"),
            "postprocessing_source_tree_sha256_path_dependent": (
                original.get("postprocessing", {}).get("source_tree_sha256")
            ),
            "configuration_sha256": original.get("config_hash"),
            "calibration_sha256": original.get("calibration_sha256"),
            "feature_sha256": original.get("feature_sha256"),
        },
        "raw_inputs": {
            "count": len(raw_rows),
            "portable_manifest_sha256": raw_portable_hash,
            "inventory": "input_inventory.csv",
            "verified_source_at_freeze": str(input_dir),
        },
        "vendor_validation_inputs": {
            "count": len(vendor_rows),
            "portable_manifest_sha256": vendor_portable_hash,
            "inventory": "provenance/vendor_input_inventory.csv",
            "role": "independent validation only; never a fitting target",
        },
        "atomic_database": {
            "count": len(db_rows),
            "portable_manifest_sha256": database_portable_hash,
            "original_path_dependent_sha256": original["database"].get(
                "tree_sha256"),
            "inventory": "provenance/atomic_database_inventory.csv",
            "required_for_canonical_regeneration": False,
            "reason": (
                "The selected component wavelengths, weights, stages, and "
                "strengths are frozen in line_feature_manifest.json; the "
                "per-position detector/shift/width priors are frozen in "
                "session_calibration.csv."
            ),
        },
        "source_capture": {
            "captured_utc": utc_now(),
            "base_archive": source_archive.relative_to(run_dir).as_posix(),
            "base_archive_sha256": hash_file(source_archive),
            "overlay": "provenance/source_overlay",
            "overlay_file_count": len(source_rows),
            "overlay_portable_sha256": source_portable_hash,
            "environment_lock": lock_path.relative_to(run_dir).as_posix(),
            "python": sys.version,
            "platform": platform.platform(),
            "limitation": (
                "The production worktree was dirty and its full patch was not "
                "saved at run time. The later source capture is therefore not "
                "claimed to reconstruct the original dirty tree byte for "
                "byte. Canonical frozen-prior regeneration is used to verify "
                "the scientific tables directly."
            ),
        },
        "recipe": "reproduction_recipe.json",
        "artifact_inventory": "provenance/artifact_checksums.csv",
    }
    validation_path = provenance_dir / "regeneration_validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text())
        provenance["regeneration_validation"] = {
            "record": validation_path.relative_to(run_dir).as_posix(),
            "completed_utc": validation.get("completed_utc"),
            "all_byte_identical": validation.get("all_byte_identical"),
            "deterministic_table_count": len(validation.get("tables", {})),
        }
    write_json(run_dir / "provenance_manifest.json", provenance)

    artifact_paths = [run_dir / name for name in CORE_ARTIFACTS]
    artifact_paths.extend((run_dir / "figures").glob("*"))
    artifact_paths.extend((run_dir / "checkpoints").glob("*.json"))
    artifact_paths.extend([
        run_dir / "provenance_manifest.json",
        run_dir / "reproduction_recipe.json",
    ])
    artifact_paths.extend(path for path in provenance_dir.rglob("*")
                          if path.is_file()
                          and path.name != "artifact_checksums.csv")
    missing = [path for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bundle artifacts: {missing}")
    artifact_rows = file_manifest(run_dir, artifact_paths)
    write_csv(provenance_dir / "artifact_checksums.csv", artifact_rows,
              ("path", "size_bytes", "sha256"))

    failures = verify_input_inventory(run_dir / "input_inventory.csv", input_dir)
    failures += verify_simple_inventory(
        provenance_dir / "vendor_input_inventory.csv", vendor_dir)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(
        f"frozen: {len(artifact_rows)} artifacts, {len(raw_rows)} raw inputs, "
        f"{len(vendor_rows)} vendor inputs, {len(db_rows)} database files")
    return 0


def verify_input_inventory(inventory_path: Path, input_dir: Path) -> list[str]:
    failures = []
    with inventory_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        path = input_dir / row["file"]
        failures.extend(_verify_one(
            path, int(row["size_bytes"]), row["sha256"]))
    return failures


def verify_simple_inventory(inventory_path: Path, root: Path) -> list[str]:
    failures = []
    with inventory_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        path = root / row["path"]
        failures.extend(_verify_one(
            path, int(row["size_bytes"]), row["sha256"]))
    return failures


def _verify_one(path: Path, expected_size: int, expected_hash: str) -> list[str]:
    if not path.is_file():
        return [f"missing {path}"]
    if path.stat().st_size != expected_size:
        return [f"size {path}: {path.stat().st_size} != {expected_size}"]
    actual_hash = hash_file(path)
    if actual_hash != expected_hash:
        return [f"sha256 {path}: {actual_hash} != {expected_hash}"]
    return []


def verify_bundle(args: argparse.Namespace) -> int:
    bundle = args.bundle.expanduser().resolve()
    failures = verify_simple_inventory(
        bundle / "provenance/artifact_checksums.csv", bundle)
    if args.input_dir:
        failures += verify_input_inventory(
            bundle / "input_inventory.csv", args.input_dir.expanduser().resolve())
    if args.vendor_dir:
        failures += verify_simple_inventory(
            bundle / "provenance/vendor_input_inventory.csv",
            args.vendor_dir.expanduser().resolve())
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"verification failed: {len(failures)} problem(s)")
        return 1
    print("verification passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("run_dir", type=Path)
    freeze.add_argument("--input-dir", required=True, type=Path)
    freeze.add_argument("--vendor-dir", required=True, type=Path)
    freeze.add_argument("--db", required=True, type=Path)
    freeze.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    freeze.set_defaults(func=freeze_bundle)

    verify = subparsers.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--input-dir", type=Path)
    verify.add_argument("--vendor-dir", type=Path)
    verify.set_defaults(func=verify_bundle)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
