"""Shared I/O and deterministic assembly for the MW2-112 workflows.

Sample-specific command drivers live in :mod:`scripts`; this module holds the
small, testable operations they share so stage scripts do not import private
helpers from one another.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROFILE_FIRST_ID = 989
PROFILE_LAST_ID = 1917
PROFILE_STEP_UM = 100.0
PROFILE_BEAM_DIAMETER_UM_APPROX = 50.0
REFERENCE_TEMPERATURE_K = 9000.0
FEATURES_PER_STAGE = 5
DETECTION_SNR = 3.0
MINIMUM_LINE_COHERENCE_SPEARMAN = 0.15
MINIMUM_COHERENT_LINES_PER_STAGE = 2
XRAY_REFERENCE_Q_INV_A = 4.0
ACTIVE_WAVELENGTH_NM = (190.0, 910.0)
_ID_RE = re.compile(r"^(\d+)-")
_ACQUISITION_RE = re.compile(
    r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})(?:\.csv)?$")
_LINE_NUMERIC_COLUMNS = (
    "test_id",
    "height_um",
    "ion_stage",
    "wavelength_nm",
    "contested",
    "area",
    "area_sigma",
    "snr",
    "area_fraction",
    "response",
    "shift_nm",
    "profile_fwhm_nm",
)


# Provenance primitives were promoted to alibz.provenance so every
# analyze_directory run records them; re-exported here for compatibility
# with the frozen MW2-112 workflow and its tests.
from alibz.provenance import (  # noqa: F401
    _git_value,
    _tree_hash,
    atomic_json,
    database_state,
    hash_file,
    jsonable,
    sha256_bytes,
    software_state,
    utc_now,
)


def write_csv(path: Path, records: Iterable[dict],
              header: Sequence[str] | None = None) -> None:
    """Atomically write dictionaries with stable first-record column order."""
    records = list(records)
    if header is None:
        header = list(records[0]) if records else []
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, path)


def parse_test_id(path: Path) -> int:
    match = _ID_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse test ID from {path.name!r}")
    return int(match.group(1))


def parse_acquisition_time(path: Path) -> datetime | None:
    match = _ACQUISITION_RE.search(path.name)
    if not match:
        return None
    return datetime(*(int(value) for value in match.groups()))


def discover_spectra(input_dir: Path, pattern: str = "*.csv") -> list[Path]:
    pairs = []
    for path in input_dir.glob(pattern):
        try:
            pairs.append((parse_test_id(path), path.resolve()))
        except ValueError:
            continue
    pairs.sort()
    ids = [test_id for test_id, _path in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate test IDs in input directory")
    if not pairs:
        raise FileNotFoundError(f"no test-ID spectra in {input_dir}")
    return [path for _test_id, path in pairs]


def _load_numeric_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("not a two-column table")
        return values[:, 0], values[:, 1]
    except ValueError:
        from alibz.pipeline import load_spectrum_csv
        return load_spectrum_csv(str(path))


def inventory_spectrum(path: Path) -> dict:
    test_id = parse_test_id(path)
    stat = path.stat()
    x, y = _load_numeric_spectrum(path)
    active = (x >= ACTIVE_WAVELENGTH_NM[0]) & (x <= ACTIVE_WAVELENGTH_NM[1])
    active_values = y[active]
    finite = np.isfinite(active_values)
    values = active_values[finite]
    pitch = float(np.median(np.diff(x))) if x.size > 1 else None
    acquired = parse_acquisition_time(path)
    return {
        "test_id": test_id,
        "height_um": (test_id - PROFILE_FIRST_ID) * PROFILE_STEP_UM,
        "file": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "acquisition_time": acquired.isoformat() if acquired else None,
        "sha256": hash_file(path),
        "n_rows": int(x.size),
        "wavelength_min_nm": float(np.min(x)) if x.size else None,
        "wavelength_max_nm": float(np.max(x)) if x.size else None,
        "wavelength_pitch_nm": pitch,
        "active_channels": int(np.sum(active)),
        "finite_active_channels": int(np.sum(finite)),
        "negative_fraction": (float(np.mean(values < 0))
                              if values.size else None),
        "intensity_p50": (float(np.median(values)) if values.size else None),
        "intensity_p95": (float(np.quantile(values, 0.95))
                          if values.size else None),
        "intensity_p99": (float(np.quantile(values, 0.99))
                          if values.size else None),
        "intensity_max": (float(np.max(values)) if values.size else None),
        "all_zero": bool(values.size and np.all(values == 0.0)),
        "grid_ok": bool(x.size == 23431
                        and np.isclose(x[0], 180.0, atol=0.05)
                        and np.isclose(x[-1], 961.0, atol=0.05)
                        and pitch is not None
                        and np.isclose(pitch, 1.0 / 30.0, atol=1e-5)),
    }


def build_inventory(paths: Sequence[Path], progress=print) -> list[dict]:
    entries = []
    previous_time = None
    for index, path in enumerate(paths, 1):
        entry = inventory_spectrum(path)
        acquisition_time = parse_acquisition_time(path)
        if previous_time is None or acquisition_time is None:
            entry["acquisition_gap_s"] = None
        else:
            entry["acquisition_gap_s"] = max(
                0.0, (acquisition_time - previous_time).total_seconds())
        previous_time = acquisition_time
        entries.append(entry)
        if index == 1 or index % 100 == 0 or index == len(paths):
            progress(f"inventory [{index}/{len(paths)}] {path.name}")
    return entries


def write_inventory(entries: Sequence[dict], path: Path) -> None:
    header = list(entries[0]) if entries else []
    write_csv(path, entries, header=header)


def calibration_job(job: tuple[int, str, str]) -> tuple[int, dict]:
    from alibz.session_calibration import measure_session_characteristics

    test_id, path, dbpath = job
    try:
        result = measure_session_characteristics(path, dbpath)
    except Exception as exc:
        result = {
            "response_source": "invalid", "response_ratio": None,
            "response_uncertainty": None,
            "shift_deltas_nm": [[], [], []],
            "peak_fwhm_nm": [[], [], []], "n_quick_peaks": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return test_id, result


def write_session_calibration(records: Sequence[dict], path: Path) -> None:
    rows = []
    labels = ("uv", "vis", "nir")
    for record in records:
        row = {key: value for key, value in record.items()
               if not isinstance(value, list)}
        for label, value, uncertainty, count in zip(
                labels, record["shift_prior_nm"],
                record["shift_prior_uncertainty_nm"],
                record["shift_prior_n"]):
            row[f"shift_prior_{label}_nm"] = value
            row[f"shift_prior_{label}_uncertainty_nm"] = uncertainty
            row[f"shift_prior_{label}_n"] = count
        for label, value, count in zip(
                labels, record.get("profile_fwhm_nm", [None] * 3),
                record.get("profile_fwhm_n", [0] * 3)):
            row[f"profile_fwhm_{label}_nm"] = value
            row[f"profile_fwhm_{label}_n"] = count
        rows.append(row)
    write_csv(path, rows)


def relative_measurement_job(job: tuple) -> tuple:
    """Measure one shot for process-pool execution."""
    from alibz.relative_profiles import measure_relative_features

    test_id, path, height_um, calibration, features = job
    start = time.monotonic()
    try:
        records, qc = measure_relative_features(
            path, test_id, height_um, calibration, features)
        status, error = "ok", None
    except Exception as exc:
        records, qc = [], {"test_id": test_id, "height_um": height_um}
        status, error = "error", f"{type(exc).__name__}: {exc}"
    return test_id, records, qc, status, error, time.monotonic() - start


def read_line_profiles(path: Path) -> list[dict]:
    """Read the lossless line table and restore numeric fields."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in _LINE_NUMERIC_COLUMNS:
            value = row.get(key, "")
            if value != "":
                row[key] = float(value)
        row["test_id"] = int(row["test_id"])
        row["ion_stage"] = int(row["ion_stage"])
        row["contested"] = int(row["contested"])
    return rows


def load_line_records(run_dir: Path) -> list[dict]:
    """Load line records, recovering the table from checkpoints if needed."""
    line_path = run_dir / "line_profiles.csv"
    if line_path.exists():
        return read_line_profiles(line_path)
    checkpoints = sorted((run_dir / "checkpoints").glob("*.json"))
    if not checkpoints:
        raise FileNotFoundError(
            f"missing {line_path} and checkpoint JSON files")
    records = []
    for checkpoint in checkpoints:
        saved = json.loads(checkpoint.read_text())
        if saved.get("status") != "ok":
            raise ValueError(f"cannot reassemble failed checkpoint {checkpoint}")
        records.extend(saved.get("records") or [])
    write_csv(line_path, records)
    return records


def contrast_profiles(relative: Iterable[dict]) -> list[dict]:
    """Attach natural-isotope contrast factors to relative element scores."""
    from alibz.scattering import natural_element_properties

    output = []
    for row in relative:
        prop = natural_element_properties(
            row["element"], q_values=(XRAY_REFERENCE_Q_INV_A,))
        xray = prop["xray_f0"][XRAY_REFERENCE_Q_INV_A]
        output.append({
            **row,
            "xray_f0_q4": xray,
            "xray_relative_contrast_q4": row["relative_score"] * xray ** 2,
            "neutron_b_c_fm": prop["neutron_b_c_fm"],
            "neutron_relative_coherent": (
                row["relative_score"] * prop["neutron_coherent_b"]
                if prop["neutron_coherent_b"] is not None else None),
            "neutron_relative_incoherent": (
                row["relative_score"] * prop["neutron_incoherent_b"]
                if prop["neutron_incoherent_b"] is not None else None),
            "neutron_relative_absorption": (
                row["relative_score"] * prop["neutron_absorption_b"]
                if prop["neutron_absorption_b"] is not None else None),
            "cross_element_ranking_valid": 0,
        })
    return output


def write_relative_tables(run_dir: Path, line_records: Iterable[dict]) \
        -> tuple[list[dict], list[dict]]:
    """Build and atomically write element and contrast tables."""
    from alibz.relative_profiles import combine_relative_profiles

    relative = combine_relative_profiles(line_records)
    contrast = contrast_profiles(relative)
    write_csv(run_dir / "relative_element_profiles.csv", relative)
    write_csv(run_dir / "relative_contrast_profiles.csv", contrast)
    return relative, contrast


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def load_frozen_calibration(path: Path) -> dict[int, dict]:
    """Restore the vector fields consumed by frozen-prior regeneration."""
    output = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            test_id = int(row["test_id"])
            output[test_id] = {
                "test_id": test_id,
                "response_measured": _optional_float(row["response_measured"]),
                "response_prior": _optional_float(row["response_prior"]),
                "shift_prior_nm": [
                    _optional_float(row[f"shift_prior_{label}_nm"])
                    for label in ("uv", "vis", "nir")
                ],
                "profile_fwhm_nm": [
                    _optional_float(row[f"profile_fwhm_{label}_nm"])
                    for label in ("uv", "vis", "nir")
                ],
            }
    return output
