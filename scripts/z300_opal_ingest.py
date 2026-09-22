#!/usr/bin/env python3
"""Read one exact Z300 test on Opal and emit its archived native-grid ZIP.

Only SELECTs and file reads are requested from the instrument. The old
instrument SQLite shell has no readonly flag and can itself recover a hot
journal; no application-level writes or temporary files are requested. A SQLite read
transaction selects the exact current, nondeleted Couchbase Lite revision;
two identical full blob reads and a repeated revision query establish a
bounded, stable acquisition observation. No database copy, app control,
export, acquisition, instrument-side temporary file, or grid interpolation
is involved. The instrument can still change after that observation.

Successful stdout is binary ZIP; diagnostics and --dry-run plans use stderr.
An acquisition storing fewer than --min-shots exits 3 (PendingData) with empty
stdout and may be retried later; storing more than --expected-shots is a hard
error, never published.
"""
from __future__ import annotations

import argparse
import gzip
import io
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile

import numpy as np

try:
    from . import z300_fb_decode as decoder
except ImportError:  # Direct script execution on Opal.
    import z300_fb_decode as decoder


SCHEMA = "alibz.opal-native.v1"
DB_PATH = "/sdcard/libzdata/libzdb.cblite"
SPECTRA_PATH = "/sdcard/libzdata/spectra"
LEGACY_SPECTRA_PATH = "/sdcard/libzdata/spectrum"
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 128 * 1024 * 1024
MAX_CSV_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_SHOTS = 1000
MIN_STORED_SHOTS = 6
PLACEHOLDER_CALIBRATION = np.array([961.0, -0.0004, 1e-12, 1e-12])
LEGACY_REFERENCE_SHA256 = "8a6135d058f09c8de62542695c7a1ac923d577906fdc88b8db2e8ded89bd156d"
CALIBRATION_NOTE = (
    "Native stored detector samples; each shot's four cubic wavelength "
    "polynomials and segment edges are read from the bundle. Pixel offset "
    "-18 was empirically inferred against vendor exports, not supplied by "
    "a vendor schema. Historical validation is approximate (correlation "
    "0.94-0.98, RMS about 1.2% full scale); absolute wavelength accuracy "
    "and equivalence to vendor export processing are not certified. No "
    "export-grid inversion, uniform resampling or empirical fallback is used. "
    "The known inactive fourth-channel placeholder is excluded by its exact "
    "calibration and seam signature; all four raw channels remain archived."
)
LEGACY_CALIBRATION_NOTE = (
    "Native stored detector samples from the exact legacy ZIP/gzip JSON bundle. "
    "Cubic wavelength coefficients use raw zero-based pixel index (offset 0), "
    "evaluated and clipped to closed knot intervals as pantheum/alibz/"
    "z300_calibration.py pixels_to_wavelength (2026-09-12). Historical validation "
    "checked overall knot-range endpoints, not certified absolute per-line "
    "wavelengths or same-acquisition intensity agreement. The known inactive "
    "fourth-channel placeholder is excluded by exact calibration/seam signature; "
    "all source channels remain archived. No interpolation is used."
)


class IngestError(ValueError):
    """The acquisition cannot safely be published."""


class PendingData(IngestError):
    """The exact requested acquisition is not complete and stable yet."""


@dataclass(frozen=True)
class Config:
    run_id: str
    test_id: str
    expected_shots: int
    adb: str
    serial: str
    adb_port: int
    archive_root: Path
    source_revision: str | None = None
    timeout: float = 45.0
    min_shots: int | None = None


def effective_min_shots(config: Config) -> int:
    """min_shots, or min(MIN_STORED_SHOTS, expected_shots) when unset."""
    if config.min_shots is not None:
        return config.min_shots
    return min(MIN_STORED_SHOTS, config.expected_shots)


def validate_config(config: Config) -> None:
    for label, value in (("run_id", config.run_id), ("test_id", config.test_id),
                         ("serial", config.serial)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
            raise IngestError(f"unsafe {label}")
    if not 1 <= config.expected_shots <= MAX_SHOTS:
        raise IngestError(f"expected_shots must be 1..{MAX_SHOTS}")
    if not 1 <= config.adb_port <= 65535 or not 0 < config.timeout <= 300:
        raise IngestError("invalid ADB port or timeout")
    if config.min_shots is not None and not 1 <= config.min_shots <= config.expected_shots:
        raise IngestError(f"min_shots must be 1..{config.expected_shots}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Adb:
    """Legacy Android shell for text; ADB pull for exact binary bytes."""

    def __init__(self, config: Config):
        self.config = config

    def read(self, command: str, limit: int) -> bytes:
        # Keep the read interface convenient for deterministic SQLite fixtures;
        # never send binary cat through Android's CRLF-transforming shell.
        tokens = shlex.split(command)
        if tokens and tokens[0] == "cat":
            fb_path = (len(tokens) == 2 and tokens[1].startswith(SPECTRA_PATH + "/")
                       and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                                        tokens[1][len(SPECTRA_PATH) + 1:]))
            legacy_path = (len(tokens) == 2 and re.fullmatch(
                re.escape(LEGACY_SPECTRA_PATH) + r"/([0-9a-f]{2})/\1[0-9a-f]{38}", tokens[1]))
            if not (fb_path or legacy_path):
                raise IngestError("binary read requires a safe spectra filename")
            return self.pull(tokens[1], limit)
        args = [self.config.adb, "-P", str(self.config.adb_port), "-s",
                self.config.serial, "shell", command]
        with tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors)
            except OSError as exc:
                raise IngestError("could not execute configured ADB") from exc
            timer = threading.Timer(self.config.timeout, process.kill)
            timer.daemon = True
            timer.start()
            chunks, size = [], 0
            try:
                assert process.stdout is not None
                while True:
                    chunk = process.stdout.read(min(65536, limit + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > limit:
                        raise IngestError("instrument response exceeds size limit")
                result = process.wait()
                if result:
                    raise PendingData("instrument read unavailable or timed out")
                # sqlite metadata is hex encoded; only shell line endings vary.
                return b"".join(chunks).replace(b"\r\n", b"\n")
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()


    def pull(self, remote_path: str, limit: int) -> bytes:
        with tempfile.TemporaryDirectory(prefix="z300-binary-") as directory:
            target = Path(directory) / "all.fb"
            args = [self.config.adb, "-P", str(self.config.adb_port), "-s",
                    self.config.serial, "pull", remote_path, str(target)]
            try:
                process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                raise IngestError("could not execute configured ADB") from exc
            stop, oversized = threading.Event(), threading.Event()

            def monitor_size():
                # Observe only our local temporary file, never poll the device.
                while not stop.wait(0.02):
                    try:
                        if target.stat().st_size > limit:
                            oversized.set()
                            process.kill()
                            return
                    except FileNotFoundError:
                        pass

            monitor = threading.Thread(target=monitor_size, daemon=True)
            monitor.start()
            try:
                try:
                    result = process.wait(timeout=self.config.timeout)
                except subprocess.TimeoutExpired as exc:
                    raise PendingData("instrument binary pull timed out") from exc
                if oversized.is_set():
                    raise IngestError("instrument response exceeds size limit")
                if result or not target.is_file():
                    raise PendingData("instrument binary pull unavailable")
                if target.stat().st_size > limit:
                    raise IngestError("instrument response exceeds size limit")
                return target.read_bytes()
            finally:
                stop.set()
                monitor.join()
                if process.poll() is None:
                    process.kill()
                process.wait()


def revision_query(test_id: str) -> str:
    # Config validation makes SQL literals safe; never select by time/latest/name.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", test_id):
        raise IngestError("unsafe test_id")
    sql = (
        "BEGIN; SELECT CAST(r.sequence AS TEXT)||'|'||hex(r.revid)||'|'||"
        "hex(d.docid)||'|'||hex(r.json) FROM docs AS d JOIN revs AS r "
        "ON r.doc_id=d.doc_id WHERE d.docid='" + test_id + "' "
        "AND r.current=1 AND r.deleted=0 LIMIT 2; COMMIT;"
    )
    # Old sqlite3 opens nonexistent paths by creating a DB; guard missing mounts.
    return ("test -f " + shlex.quote(DB_PATH) + " && sqlite3 -batch -noheader "
            + shlex.quote(DB_PATH) + " " + shlex.quote(sql))


def parse_revision(raw: bytes, test_id: str) -> tuple[dict, dict, bytes]:
    lines = raw.strip().splitlines()
    if not lines:
        raise PendingData("exact current nondeleted test revision is not available")
    if len(lines) != 1:
        raise IngestError("test has conflicting current revisions")
    try:
        sequence, revision, document_id, document = lines[0].decode("ascii").split("|")
        metadata = {"sequence": int(sequence),
                    "revision_id": bytes.fromhex(revision).decode("utf-8"),
                    "document_id": bytes.fromhex(document_id).decode("utf-8")}
        document_bytes = bytes.fromhex(document)
        if len(document_bytes) > MAX_DOCUMENT_BYTES:
            raise IngestError("test document exceeds size limit")
        doc = json.loads(document_bytes)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise IngestError("invalid SQLite revision response") from exc
    if metadata["document_id"] != test_id or metadata["sequence"] < 1:
        raise IngestError("revision does not identify the exact requested test")
    if not isinstance(doc, dict):
        raise IngestError("test document must be an object")
    return metadata, doc, document_bytes


def bundle_filename(doc: dict) -> str:
    table = doc.get("shotTable")
    if not isinstance(table, dict) or not table.get("all_fb"):
        raise PendingData("exact test has no all_fb bundle yet")
    name = table["all_fb"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise IngestError("all_fb must be a safe plain filename")
    return name


def bundle_source(doc: dict) -> tuple[str, str, str]:
    """Resolve only an exact stored source pointer; never guess between tests."""
    table = doc.get("shotTable")
    if not isinstance(table, dict):
        raise PendingData("exact test has no stored spectrum bundle yet")
    if table.get("all_fb"):
        return "flatbuffer", SPECTRA_PATH + "/" + bundle_filename(doc), "raw/all.fb"
    digest = table.get("all")
    if not digest:
        raise PendingData("exact test has no all_fb or legacy all bundle yet")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
        raise IngestError("legacy all must be a lowercase SHA1 filename")
    return "legacy-zip-gzip-json", LEGACY_SPECTRA_PATH + "/" + digest[:2] + "/" + digest, "raw/all.zip"


def stored_configuration(doc: dict) -> dict:
    """Guard the documented CBL test fields without inventing absent values."""
    observed = {}
    if "onlyAvgSaved" in doc:
        observed["onlyAvgSaved"] = doc["onlyAvgSaved"]
        if doc["onlyAvgSaved"] not in (False, 0, "false", "False", "0"):
            raise IngestError("onlyAvgSaved test cannot prove individual stored shots")
    config = doc.get("config", {})
    if not isinstance(config, dict):
        raise IngestError("stored test config must be an object")
    fields = ("numShotsToAvg", "numShotsPerLocation", "numCleaningShotsPerLocation", "rasterNumLocations")
    observed["config"] = {key: config[key] for key in fields if key in config}
    if "numShotsToAvg" in config and (isinstance(config["numShotsToAvg"], bool)
                                      or config["numShotsToAvg"] not in (1, "1")):
        raise IngestError("numShotsToAvg must be 1; averaged pulses cannot prove individual stored shots")
    return observed


class StrictReader(decoder._Reader):
    """Add bounds checks without changing the established decoder's math."""

    def check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > len(self.b):
            raise IngestError("FlatBuffer offset or length is out of bounds")

    def u16(self, off: int) -> int:
        self.check(off, 2)
        return super().u16(off)

    def u32(self, off: int) -> int:
        self.check(off, 4)
        return super().u32(off)

    def i32(self, off: int) -> int:
        self.check(off, 4)
        return super().i32(off)

    def vtable(self, table: int) -> list[int]:
        vt = table - self.i32(table)
        self.check(vt, 4)
        length, object_size = self.u16(vt), self.u16(vt + 2)
        if length < 4 or length % 2 or object_size < 4:
            raise IngestError("invalid FlatBuffer vtable")
        self.check(vt, length)
        self.check(table, object_size)
        entries = [self.u16(vt + 4 + 2 * i) for i in range((length - 4) // 2)]
        if any(entry and not 4 <= entry < object_size for entry in entries):
            raise IngestError("FlatBuffer field lies outside its table")
        return entries

    def field(self, table: int, index: int) -> int | None:
        result = super().field(table, index)
        if result is not None:
            vt = table - self.i32(table)
            if result + 4 > table + self.u16(vt + 2):
                raise IngestError("FlatBuffer field overflows its table")
        return result

    def follow(self, pos: int) -> int:
        delta = self.u32(pos)
        if delta < 4:
            raise IngestError("invalid FlatBuffer forward offset")
        result = pos + delta
        self.check(result, 4)
        return result

    def doubles(self, pos: int) -> np.ndarray:
        length, data = self.vector(pos)
        self.check(data, 8 * length)
        values = np.frombuffer(self.b, dtype="<f8", count=length, offset=data)
        if not np.all(np.isfinite(values)):
            raise IngestError("FlatBuffer contains nonfinite values")
        return values


def decode_native(blob: bytes, expected_shots: int, min_shots: int) -> list[tuple[np.ndarray, np.ndarray]]:
    reader = StrictReader(blob)
    try:
        root = reader.follow(0)
        shot_pos = reader.field(root, 3)
        if shot_pos is None:
            raise IngestError("FlatBuffer root lacks a shot vector")
        count, shot_data = reader.vector(shot_pos)
        if not 0 < count <= MAX_SHOTS:
            raise IngestError("FlatBuffer shot count is outside supported bounds")
        if count > expected_shots:
            raise IngestError(f"stored shot count {count} exceeds expected {expected_shots}")
        if count < min_shots:
            raise PendingData(
                f"stored shot count {count} is below min_shots {min_shots} of expected {expected_shots}")
        reader.check(shot_data, 4 * count)
        tables = [reader.follow(shot_data + 4 * i) for i in range(count)]
        shots = []
        for table in tables:
            edge_pos, segment_pos = reader.field(table, 0), reader.field(table, 1)
            if edge_pos is None or segment_pos is None:
                raise IngestError("shot lacks segment edges or data")
            edges = reader.doubles(edge_pos)
            if edges.size != decoder.N_SEGMENTS + 1 or not np.all(np.diff(edges) > 0):
                raise IngestError("segment edges must be five finite increasing values")
            n_segments, data = reader.vector(segment_pos)
            if n_segments != decoder.N_SEGMENTS:
                raise IngestError("shot does not have four segments")
            for index in range(n_segments):
                segment = reader.follow(data + 4 * index)
                calibration_pos, intensity_pos = reader.field(segment, 0), reader.field(segment, 1)
                if calibration_pos is None or intensity_pos is None:
                    raise IngestError("segment lacks calibration or data")
                calibration, intensity = reader.doubles(calibration_pos), reader.doubles(intensity_pos)
                if calibration.size != 4 or intensity.size != decoder.PIXELS_PER_SEGMENT:
                    raise IngestError("segment requires four cubic coefficients and 2066 samples")
                if index == 3 and not (np.array_equal(calibration, PLACEHOLDER_CALIBRATION)
                                       and np.array_equal(edges[3:], [960.0, 961.0])):
                    raise IngestError("unrecognized fourth channel; native scientific interpretation requires review")
            edges, segments = decoder.read_shot(reader, table)
            for wavelength, _ in segments:
                differences = np.diff(wavelength)
                if not np.all(np.isfinite(wavelength)) or not (
                        np.all(differences > 0) or np.all(differences < 0)):
                    raise IngestError("calibrated segment axis is not finite and strictly monotonic")
            # The exact fourth-channel signature is an inactive placeholder on
            # this Z300, not scientific detector coverage. Preserve it only raw.
            wavelength, intensity = decoder.stitch(edges[:4], segments[:3])
            if wavelength.size < 2 or not np.all(np.diff(wavelength) > 0):
                raise IngestError("stitched native wavelength axis is not strictly increasing")
            if shots and not np.array_equal(wavelength, shots[0][0]):
                raise IngestError("shot wavelength axes differ; no shared native grid for averaging")
            shots.append((wavelength, intensity))
        return shots
    except decoder.FlatBufferError as exc:
        raise IngestError(str(exc)) from exc


def legacy_spectrum(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """The known JSON calibration math, with strict production shape checks."""
    try:
        edges = np.asarray(record["knots"], dtype=float)
        pixels = np.asarray(record["pixels"], dtype=float)
        calibrations = record["wlCalibrations"]
        if isinstance(calibrations, dict):
            calibrations = calibrations["coefficients"]
        coefficients = []
        for segment in calibrations:
            if isinstance(segment, dict):
                segment = segment.get("pixToNm", segment)["coefficients"]
            coefficients.append(segment)
        coefficients = np.asarray(coefficients, dtype=float)
    except (KeyError, ValueError, TypeError) as exc:
        raise IngestError("legacy spectrum has invalid calibration/data shape") from exc
    if edges.shape != (5,) or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise IngestError("legacy knots require five finite increasing values")
    if pixels.shape != (4, decoder.PIXELS_PER_SEGMENT) or coefficients.shape != (4, 4):
        raise IngestError("legacy spectrum requires four cubic calibrations and four 2066-sample segments")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(coefficients)):
        raise IngestError("legacy spectrum contains nonfinite values")
    if not (np.array_equal(coefficients[3], PLACEHOLDER_CALIBRATION)
            and np.array_equal(edges[3:], [960.0, 961.0])):
        raise IngestError("unrecognized fourth channel; native scientific interpretation requires review")
    index = np.arange(decoder.PIXELS_PER_SEGMENT, dtype=float)
    waves, amplitudes = [], []
    for segment, calibration in enumerate(coefficients):
        # Match z300_calibration._polyval's ascending power accumulation at p=0,
        # including its closed knot intervals. The FlatBuffer's -18 is not used.
        wavelength, power = np.zeros_like(index), np.ones_like(index)
        for coefficient in calibration:
            wavelength += coefficient * power
            power *= index
        differences = np.diff(wavelength)
        if not np.all(np.isfinite(wavelength)) or not (
                np.all(differences > 0) or np.all(differences < 0)):
            raise IngestError("legacy calibrated segment axis is not finite and strictly monotonic")
        if segment < 3:
            keep = (wavelength >= edges[segment]) & (wavelength <= edges[segment + 1])
            if not np.any(keep):
                raise IngestError("legacy active segment has no points within its knots")
            waves.append(wavelength[keep])
            amplitudes.append(pixels[segment][keep])
    wavelength, intensity = np.concatenate(waves), np.concatenate(amplitudes)
    order = np.argsort(wavelength)
    wavelength, intensity = wavelength[order], intensity[order]
    if wavelength.size < 2 or not np.all(np.diff(wavelength) > 0):
        raise IngestError("legacy native wavelength axis is not strictly increasing")
    return wavelength, intensity


def decode_legacy_native(blob: bytes, expected_shots: int, min_shots: int) -> tuple[
        list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """Bound ZIP/gzip expansion; require the vendor average plus 0..n-1 shots."""
    full_expected = {"-1", *(str(i) for i in range(expected_shots))}
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if len(names) != len(entries) or names - full_expected:
                raise IngestError("legacy ZIP has duplicate or unexpected shot entries")
            if "-1" not in names:
                raise PendingData("legacy ZIP does not contain the average yet")
            stored = sorted(int(name) for name in names if name != "-1")
            count = len(stored)
            if stored != list(range(count)):
                raise IngestError("legacy ZIP shots are not contiguous starting at 0")
            if count < min_shots:
                raise PendingData(
                    f"legacy ZIP has {count} shots below min_shots {min_shots} of expected {expected_shots}")
            if (any(entry.file_size > MAX_DOCUMENT_BYTES or entry.flag_bits & 1 for entry in entries)
                    or sum(entry.file_size for entry in entries) > MAX_PAYLOAD_BYTES):
                raise IngestError("legacy ZIP expanded members exceed size limit or are encrypted")
            decoded, total = {}, 0
            for name in ["-1", *(str(i) for i in range(count))]:
                with archive.open(name) as stream:
                    raw = stream.read(MAX_DOCUMENT_BYTES + 1)
                if len(raw) > MAX_DOCUMENT_BYTES:
                    raise IngestError("legacy ZIP member exceeds size limit")
                if raw.startswith(b"\x1f\x8b"):
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                        raw = stream.read(MAX_DOCUMENT_BYTES + 1)
                total += len(raw)
                if len(raw) > MAX_DOCUMENT_BYTES or total > MAX_PAYLOAD_BYTES:
                    raise IngestError("legacy gzip JSON expansion exceeds size limit")
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise IngestError("legacy shot must be a JSON object")
                decoded[name] = legacy_spectrum(record)
            reference = decoded["0"][0]
            if any(not np.array_equal(reference, spectrum[0]) for spectrum in decoded.values()):
                raise IngestError("legacy shot/average wavelength axes differ; no shared native grid")
            return [decoded[str(i)] for i in range(count)], decoded["-1"]
    except (zipfile.BadZipFile, gzip.BadGzipFile, EOFError, UnicodeError,
            json.JSONDecodeError, RecursionError, RuntimeError) as exc:
        raise IngestError("invalid legacy ZIP/gzip JSON bundle") from exc


@contextmanager
def run_lock(root: Path, run_id: str):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ("." + run_id + ".lock")
    # Keep the lock inode: unlinking it could create two independent locks.
    # Both operating systems release the advisory lock if the owner crashes.
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if lock.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise PendingData("another ingestion owns this run") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PendingData("another ingestion owns this run") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_native_csv(path: Path, wavelength: np.ndarray, intensity: np.ndarray) -> None:
    # 17 digits preserve binary64 native axes on a round trip through CSV.
    with path.open("w", encoding="ascii", newline="") as stream:
        stream.write("wavelength,intensity\n")
        for x, y in zip(wavelength, intensity):
            stream.write(f"{x:.17g},{y:.17g}\n")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise IngestError("native CSV exceeds size limit")


def verify_archive(directory: Path, config: Config) -> Path:
    try:
        manifest_raw = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        if (manifest["schema"], manifest["run_id"], manifest["test_id"], manifest["grid"]) != (
                SCHEMA, config.run_id, config.test_id, "native"):
            raise IngestError("existing archive belongs to a different acquisition")
        min_shots = effective_min_shots(config)
        stored_shots = manifest["shots"]
        if (not isinstance(stored_shots, int) or isinstance(stored_shots, bool)
                or not min_shots <= stored_shots <= config.expected_shots
                or manifest["provenance"].get("expected_shots") != config.expected_shots):
            raise IngestError("existing archive belongs to a different acquisition")
        source_format = manifest["provenance"].get("source_format", "flatbuffer")
        raw_name = {"flatbuffer": "raw/all.fb", "legacy-zip-gzip-json": "raw/all.zip"}.get(source_format)
        if raw_name is None:
            raise IngestError("existing archive has an unknown source format")
        required = {"average.csv", "raw/test.json", raw_name, "raw/revision.json"}
        required.update(f"shots/shot-{i}.csv" for i in range(stored_shots))
        if set(manifest["files"]) != required:
            raise IngestError("existing archive has unexpected files")
        for name, digest in manifest["files"].items():
            if file_hash(directory / name) != digest:
                raise IngestError("existing archive file hash mismatch")
        payload = directory / "native.zip"
        receipt = json.loads((directory / "archive-receipt.json").read_bytes())
        if payload.stat().st_size > MAX_PAYLOAD_BYTES or file_hash(payload) != receipt["zip_sha256"]:
            raise IngestError("existing archive ZIP hash mismatch")
        with zipfile.ZipFile(payload) as archive:
            if set(archive.namelist()) != required | {"manifest.json"}:
                raise IngestError("existing archive ZIP membership mismatch")
            if archive.read("manifest.json") != manifest_raw:
                raise IngestError("existing archive ZIP manifest mismatch")
        return payload
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise IngestError(f"cannot verify immutable archive: {exc}") from exc


def ingest(config: Config, adb: Adb | None = None) -> Path:
    validate_config(config)
    adb = adb or Adb(config)
    with run_lock(config.archive_root, config.run_id):
        final = config.archive_root / config.run_id
        if final.exists():
            return verify_archive(final, config)
        stage = Path(tempfile.mkdtemp(prefix="." + config.run_id + ".", dir=config.archive_root))
        try:
            started = utc_now()
            query = revision_query(config.test_id)
            response_limit = 2 * MAX_DOCUMENT_BYTES + 16384
            revision_raw = adb.read(query, response_limit)
            revision, document, document_bytes = parse_revision(revision_raw, config.test_id)
            observed_config = stored_configuration(document)
            source_format, source_path, raw_name = bundle_source(document)
            command = "cat " + shlex.quote(source_path)
            blob = adb.read(command, MAX_BLOB_BYTES)
            repeated_blob = adb.read(command, MAX_BLOB_BYTES)
            if len(blob) != len(repeated_blob) or sha256(blob) != sha256(repeated_blob):
                raise PendingData("bundle changed between complete reads")
            del repeated_blob
            if adb.read(query, response_limit) != revision_raw:
                raise PendingData("test revision changed during bundle read")
            observed = utc_now()
            min_shots = effective_min_shots(config)
            if source_format == "flatbuffer":
                shots = decode_native(blob, config.expected_shots, min_shots)
                average = np.zeros_like(shots[0][1])
                for _, intensity in shots:
                    average += intensity / len(shots)
            else:
                shots, (_, average) = decode_legacy_native(blob, config.expected_shots, min_shots)
            (stage / "raw").mkdir()
            (stage / "shots").mkdir()
            (stage / "raw" / "test.json").write_bytes(document_bytes)
            (stage / raw_name).write_bytes(blob)
            (stage / "raw" / "revision.json").write_bytes(json_bytes(revision))
            for i, (wavelength, intensity) in enumerate(shots):
                write_native_csv(stage / "shots" / f"shot-{i}.csv", wavelength, intensity)
            if not np.all(np.isfinite(average)):
                raise IngestError("native-grid mean is not finite")
            write_native_csv(stage / "average.csv", shots[0][0], average)
            files = {str(path.relative_to(stage)).replace(os.sep, "/"): file_hash(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()}
            if sum((stage / name).stat().st_size for name in files) > MAX_PAYLOAD_BYTES:
                raise IngestError("uncompressed payload exceeds size limit")
            producer_hash = file_hash(Path(__file__))
            manifest = {
                "schema": SCHEMA, "run_id": config.run_id, "test_id": config.test_id,
                "shots": len(shots), "grid": "native", "files": files,
                "provenance": {
                    "source": "z300-couchbase-lite-exact-current-revision",
                    "source_database": DB_PATH, "source_blob": source_path, "source_format": source_format,
                    "source_revision": config.source_revision or "sha256:" + producer_hash,
                    "document_revision": revision, "revision_query_sha256": sha256(query.encode()),
                    "instrument_serial": config.serial, "read_started_utc": started,
                    "read_completed_utc": observed, "blob_bytes": len(blob),
                    "blob_sha256": sha256(blob), "expected_shots": config.expected_shots,
                    "min_shots": min_shots, "dropped_frames": config.expected_shots - len(shots),
                    "stored_configuration": observed_config,
                    "consistency": "transactional SELECT; identical full blob reads; identical revision recheck",
                    "database_archive": None,
                    "database_archive_note": "Exact revision and blob preserved; whole live DB not copied.",
                    "decoder_sha256": (file_hash(Path(decoder.__file__)) if source_format == "flatbuffer"
                                       else producer_hash),
                    "producer_sha256": producer_hash,
                    "pixel_offset": decoder.PIXEL_OFFSET if source_format == "flatbuffer" else 0.0,
                    "calibration_note": CALIBRATION_NOTE if source_format == "flatbuffer" else LEGACY_CALIBRATION_NOTE,
                    "legacy_reference_sha256": LEGACY_REFERENCE_SHA256 if source_format != "flatbuffer" else None,
                    "included_segments": [0, 1, 2],
                    "excluded_segments": [{"index": 3, "reason": "known inactive channel placeholder",
                                           "calibration": PLACEHOLDER_CALIBRATION.tolist(),
                                           "seam_nm": [960.0, 961.0], "raw_preserved": True}],
                    "average": ("arithmetic mean on exactly equal per-shot native wavelength axes"
                                if source_format == "flatbuffer" else "instrument-provided average on native axis"),
                    "native_points_per_shot": int(shots[0][0].size),
                },
            }
            (stage / "manifest.json").write_bytes(json_bytes(manifest))
            payload = stage / "native.zip"
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name in ["manifest.json", *files]:
                    archive.write(stage / name, name)
            if payload.stat().st_size > MAX_PAYLOAD_BYTES:
                raise IngestError("compressed payload exceeds size limit")
            (stage / "archive-receipt.json").write_bytes(json_bytes({"zip_sha256": file_hash(payload)}))
            verify_archive(stage, config)
            for path in stage.rglob("*"):
                if path.is_file():
                    # Windows FlushFileBuffers requires a write-capable handle.
                    with path.open("r+b") as stream:
                        os.fsync(stream.fileno())
            os.rename(stage, final)
            return final / "native.zip"
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--expected-shots", required=True, type=int)
    parser.add_argument("--adb", default=r"C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe")
    parser.add_argument("--serial", default="0123456789ABCDEF")
    parser.add_argument("--adb-port", type=int, default=5038)
    parser.add_argument("--archive-root", type=Path, default=Path(r"C:\LIBS-Staging\z300-native"))
    parser.add_argument("--source-revision")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--min-shots", type=int, default=None,
                         help="minimum stored shots to publish, 1..expected-shots "
                              "(default: min(6, expected-shots))")
    parser.add_argument("--output", type=Path, help="atomically write ZIP here instead of stdout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = Config(**{key: value for key, value in vars(args).items() if key not in {"dry_run", "output"}})
    try:
        validate_config(config)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "run_id": config.run_id, "test_id": config.test_id,
                              "expected_shots": config.expected_shots, "source_database": DB_PATH,
                              "archive": str(config.archive_root / config.run_id),
                              "output": str(args.output) if args.output else "stdout ZIP",
                              "actions": ["select exact current nondeleted revision", "read blob twice",
                                          "recheck revision", "validate and decode native grid",
                                          "archive atomically", "emit ZIP"]}, sort_keys=True), file=sys.stderr)
            return 0
        payload = ingest(config)
        if args.output:
            if args.output.resolve() != payload.resolve():
                args.output.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(prefix="." + args.output.name + ".", dir=args.output.parent)
                try:
                    with os.fdopen(descriptor, "wb") as target, payload.open("rb") as source:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, args.output)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            with zipfile.ZipFile(payload) as archive:
                stored_shots = json.loads(archive.read("manifest.json"))["shots"]
            print(f"native ZIP archived: {stored_shots} of {config.expected_shots} expected shots",
                  file=sys.stderr)
            return 0
        # Windows stdout otherwise translates newline bytes inside a binary ZIP.
        if os.name == "nt":
            import msvcrt
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        with payload.open("rb") as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer)
        sys.stdout.buffer.flush()
        return 0
    except (IngestError, OSError) as exc:
        print(f"z300 ingestion: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, PendingData) else 2


if __name__ == "__main__":
    raise SystemExit(main())
