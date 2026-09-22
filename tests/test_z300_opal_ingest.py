"""Opal producer tests: synthetic SQLite/FlatBuffers, no instrument access."""
from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sqlite3
import struct
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts import z300_fb_decode as decoder
from scripts import z300_opal_ingest as producer


class Builder:
    def __init__(self):
        self.data = bytearray(4)

    def pointer(self, field, target):
        struct.pack_into("<I", self.data, field, target - field)

    def table(self, fields):
        vt = len(self.data)
        self.data.extend(struct.pack("<HH", 4 + 2 * fields, 4 + 4 * fields))
        self.data.extend(struct.pack("<" + "H" * fields, *[4 + i * 4 for i in range(fields)]))
        table = len(self.data)
        self.data.extend(struct.pack("<i", table - vt) + bytes(4 * fields))
        return table, [table + 4 + 4 * i for i in range(fields)]

    def doubles(self, values):
        offset = len(self.data)
        self.data.extend(struct.pack("<I", len(values)))
        self.data.extend(np.asarray(values, dtype="<f8").tobytes())
        return offset

    def tables(self, count):
        offset = len(self.data)
        self.data.extend(struct.pack("<I", count) + bytes(4 * count))
        return offset, [offset + 4 + 4 * i for i in range(count)]


def bundle(shots=2, *, calibration_size=4, intensity_size=2066,
           bad_value=None, constant_axis=False, shifted=False,
           reversed_edges=False, unknown_fourth=False):
    builder = Builder()
    root, fields = builder.table(4)
    builder.pointer(0, root)
    vector, shot_pointers = builder.tables(shots)
    builder.pointer(fields[3], vector)
    for shot_index, shot_pointer in enumerate(shot_pointers):
        shot_table, shot_fields = builder.table(2)
        builder.pointer(shot_pointer, shot_table)
        edges = [180.0, 365.0, 620.0, 960.0, 961.0]
        if reversed_edges:
            edges[1] = 170.0
        builder.pointer(shot_fields[0], builder.doubles(edges))
        segments, segment_pointers = builder.tables(4)
        builder.pointer(shot_fields[1], segments)
        for i, segment_pointer in enumerate(segment_pointers):
            segment, segment_fields = builder.table(2)
            builder.pointer(segment_pointer, segment)
            if i == 3:
                coefficients = producer.PLACEHOLDER_CALIBRATION.copy()
                if unknown_fourth:
                    coefficients[1] = -0.001
            else:
                coefficients = np.array([edges[i + 1] + 2, -(edges[i + 1] - edges[i] + 4) / 2065, 0, 0])
                if constant_axis:
                    coefficients[1] = 0
                if shifted and shot_index:
                    coefficients[0] += 0.001
            coefficients = coefficients[:calibration_size]
            values = np.arange(intensity_size, dtype=float) + shot_index * 100
            if bad_value is not None:
                values[100] = bad_value
            builder.pointer(segment_fields[0], builder.doubles(coefficients))
            builder.pointer(segment_fields[1], builder.doubles(values))
    return bytes(builder.data)


class MockAdb:
    """Execute the producer's actual SQL against a synthetic CBL1 schema."""
    def __init__(self, test_id="test-id", document=None, blob=None):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE docs(doc_id INTEGER PRIMARY KEY, docid TEXT);
            CREATE TABLE revs(sequence INTEGER PRIMARY KEY, doc_id INTEGER,
                              revid TEXT, current INTEGER, deleted INTEGER, json BLOB);
        """)
        self.document = document if document is not None else {"shotTable": {"all_fb": "bundle-uuid"}}
        self.raw_document = json.dumps(self.document).encode()
        self.db.execute("INSERT INTO docs VALUES(1, ?)", (test_id,))
        self.db.execute("INSERT INTO revs VALUES(1,1,'1-exact',1,0,?)", (self.raw_document,))
        self.db.commit()
        self.blob = bundle() if blob is None else blob
        self.commands = []
        self.blob_reads = 0
        self.change_blob = False
        self.change_revision = False

    def read(self, command, limit):
        self.commands.append(command)
        args = shlex.split(command)
        if args[0] == "test":
            assert args[:4] == ["test", "-f", producer.DB_PATH, "&&"]
            args = args[4:]
            assert args[0] == "sqlite3"
            assert args[1:4] == ["-batch", "-noheader", producer.DB_PATH]
            statements = args[4].split(";")
            assert statements[0] == "BEGIN" and statements[2].strip() == "COMMIT"
            self.db.execute("BEGIN")
            rows = self.db.execute(statements[1]).fetchall()
            self.db.commit()
            result = b"".join((row[0] + "\n").encode() for row in rows)
        else:
            expected_path = (producer.SPECTRA_PATH + "/bundle-uuid" if self.document["shotTable"].get("all_fb")
                             else producer.LEGACY_SPECTRA_PATH + "/ab/" + "ab" * 20)
            assert args == ["cat", expected_path]
            self.blob_reads += 1
            result = self.blob
            if self.change_blob and self.blob_reads == 2:
                result += b"x"
            if self.change_revision and self.blob_reads == 2:
                self.db.execute("UPDATE revs SET revid='2-changed'")
                self.db.commit()
        assert len(result) <= limit
        return result


@pytest.fixture
def config(tmp_path):
    return producer.Config("run-id", "test-id", 2, "adb", "0123456789ABCDEF", 5038, tmp_path / "archive")


def test_native_payload_exact_revision_hashes_and_shared_mean(config):
    adb = MockAdb()
    # Later and obsolete records must not affect an exact current revision.
    adb.db.execute("INSERT INTO docs VALUES(2,'another-test')")
    adb.db.execute("INSERT INTO revs VALUES(2,2,'1-other',1,0,?)", (b"{}",))
    adb.db.execute("INSERT INTO revs VALUES(3,1,'0-obsolete',0,0,?)", (b"{}",))
    adb.db.commit()
    path = producer.ingest(config, adb)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert (manifest["schema"], manifest["run_id"], manifest["test_id"], manifest["shots"], manifest["grid"]) == (
            producer.SCHEMA, "run-id", "test-id", 2, "native")
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert archive.read("raw/test.json") == adb.raw_document
        assert archive.read("raw/all.fb") == adb.blob
        first = np.loadtxt(io.BytesIO(archive.read("shots/shot-0.csv")), delimiter=",", skiprows=1)
        second = np.loadtxt(io.BytesIO(archive.read("shots/shot-1.csv")), delimiter=",", skiprows=1)
        average = np.loadtxt(io.BytesIO(archive.read("average.csv")), delimiter=",", skiprows=1)
        np.testing.assert_array_equal(first[:, 0], second[:, 0])
        np.testing.assert_array_equal(average[:, 0], first[:, 0])
        np.testing.assert_array_equal(average[:, 1], (first[:, 1] + second[:, 1]) / 2)
        assert first[:, 0].max() < 960
        assert np.ptp(np.diff(first[:, 0])) > 0.001
        assert manifest["provenance"]["excluded_segments"][0]["index"] == 3
        assert manifest["provenance"]["pixel_offset"] == -18
        assert manifest["provenance"]["document_revision"]["revision_id"] == "1-exact"
    assert len(adb.commands) == 4
    assert not any("backup" in command or "root" in command for command in adb.commands)


def test_same_run_is_byte_identical_idempotent_without_instrument_read(config):
    adb = MockAdb()
    path = producer.ingest(config, adb)
    expected = path.read_bytes()
    adb.read = lambda *args: pytest.fail("idempotent retry contacted the instrument")
    assert producer.ingest(config, adb).read_bytes() == expected


def test_different_test_cannot_reuse_archive(config):
    producer.ingest(config, MockAdb())
    with pytest.raises(producer.IngestError, match="different acquisition"):
        producer.ingest(replace(config, test_id="different-id"), MockAdb())


def test_corrupted_archive_refused_without_overwrite(config):
    producer.ingest(config, MockAdb())
    target = config.archive_root / config.run_id / "average.csv"
    target.write_bytes(b"corruption")
    with pytest.raises(producer.IngestError, match="hash mismatch"):
        producer.ingest(config, MockAdb())
    assert target.read_bytes() == b"corruption"


@pytest.mark.parametrize("change", ["change_blob", "change_revision"])
def test_unstable_data_never_published(config, change):
    adb = MockAdb()
    setattr(adb, change, True)
    with pytest.raises(producer.PendingData, match="changed"):
        producer.ingest(config, adb)
    assert not (config.archive_root / config.run_id).exists()
    assert all(path.suffix == ".lock" for path in config.archive_root.iterdir())


@pytest.mark.parametrize("mode", ["missing", "deleted", "obsolete", "no_bundle", "shots"])
def test_incomplete_exact_test_is_pending(config, mode):
    adb = MockAdb(document={} if mode == "no_bundle" else None, blob=bundle(1) if mode == "shots" else None)
    if mode == "missing":
        adb.db.execute("DELETE FROM revs")
    elif mode == "deleted":
        adb.db.execute("UPDATE revs SET deleted=1")
    elif mode == "obsolete":
        adb.db.execute("UPDATE revs SET current=0")
    adb.db.commit()
    with pytest.raises(producer.PendingData):
        producer.ingest(config, adb)
    assert not (config.archive_root / config.run_id).exists()


def test_native_dropped_frame_archived_and_reverifies_without_instrument(config):
    # Only 1 of the 2 expected shots stored; explicit min_shots=1 tolerates it.
    cfg = replace(config, min_shots=1)
    adb = MockAdb(blob=bundle(1))
    payload = producer.ingest(cfg, adb)
    with zipfile.ZipFile(payload) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["shots"] == 1
        assert manifest["provenance"]["expected_shots"] == 2
        assert manifest["provenance"]["min_shots"] == 1
        assert manifest["provenance"]["dropped_frames"] == 1
        assert set(name for name in manifest["files"] if name.startswith("shots/")) == {"shots/shot-0.csv"}
    expected_bytes = payload.read_bytes()
    adb.read = lambda *args: pytest.fail("idempotent retry with fewer shots contacted the instrument")
    assert producer.ingest(cfg, adb).read_bytes() == expected_bytes


@pytest.mark.parametrize("min_shots", [0, 3, -1])
def test_min_shots_outside_expected_range_rejected(config, min_shots):
    with pytest.raises(producer.IngestError, match="min_shots"):
        producer.validate_config(replace(config, min_shots=min_shots))


def test_min_shots_defaults_to_min_of_six_and_expected_shots(config):
    assert producer.effective_min_shots(config) == 2
    assert producer.effective_min_shots(replace(config, expected_shots=10)) == 6
    assert producer.effective_min_shots(replace(config, expected_shots=10, min_shots=3)) == 3


def test_cli_min_shots_outside_range_is_rejected(config, capsys):
    assert producer.main(["--run-id", config.run_id, "--test-id", config.test_id,
                          "--expected-shots", "2", "--min-shots", "5",
                          "--archive-root", str(config.archive_root)]) == 2
    assert "min_shots" in capsys.readouterr().err


def test_conflicting_current_revisions_rejected(config):
    adb = MockAdb()
    adb.db.execute("INSERT INTO revs VALUES(2,1,'1-conflict',1,0,?)", (adb.raw_document,))
    adb.db.commit()
    with pytest.raises(producer.IngestError, match="conflicting"):
        producer.ingest(config, adb)
    assert not (config.archive_root / config.run_id).exists()


@pytest.mark.parametrize("document", [{"onlyAvgSaved": True}, {"onlyAvgSaved": "true"},
                                      {"config": {"numShotsToAvg": 2}},
                                      {"config": {"numShotsToAvg": True}}])
def test_explicit_averaged_storage_cannot_complete_individual_shots(config, document):
    document["shotTable"] = {"all_fb": "bundle-uuid"}
    adb = MockAdb(document=document)
    with pytest.raises(producer.IngestError, match="individual stored shots"):
        producer.ingest(config, adb)
    assert adb.blob_reads == 0


def test_storage_settings_record_observed_values_without_inventing_missing():
    assert producer.stored_configuration({}) == {"config": {}}
    assert producer.stored_configuration({"onlyAvgSaved": False,
                                           "config": {"numShotsToAvg": 1, "numShotsPerLocation": 5}}) == {
        "onlyAvgSaved": False, "config": {"numShotsToAvg": 1, "numShotsPerLocation": 5}}


@pytest.mark.parametrize("name", ["../other", "/absolute", "a/b", "a\\b", "x;cmd", ".", "..", ["x"]])
def test_all_fb_must_be_plain_filename(name):
    with pytest.raises(producer.IngestError):
        producer.bundle_filename({"shotTable": {"all_fb": name}})


@pytest.mark.parametrize("options, message", [
    ({"calibration_size": 3}, "four cubic"), ({"intensity_size": 2065}, "2066"),
    ({"bad_value": float("nan")}, "nonfinite"), ({"bad_value": float("inf")}, "nonfinite"),
    ({"constant_axis": True}, "strictly monotonic"), ({"reversed_edges": True}, "edges"),
    ({"shifted": True}, "axes differ"), ({"unknown_fourth": True}, "unrecognized fourth"),
])
def test_invalid_bundle_fails_closed(options, message):
    with pytest.raises(producer.IngestError, match=message):
        producer.decode_native(bundle(**options), 2, 2)


@pytest.mark.parametrize("malformed", [b"", b"\xff" * 32, bundle()[:25], bundle()[:-1]])
def test_flatbuffer_bounds_rejected(malformed):
    with pytest.raises(producer.IngestError):
        producer.decode_native(malformed, 2, 2)


def test_original_decoder_math_is_identical_for_real_channels(tmp_path):
    path = tmp_path / "all.fb"
    path.write_bytes(bundle())
    old, strict = decoder.decode(str(path)), producer.decode_native(path.read_bytes(), 2, 2)
    for (old_x, old_y), (new_x, new_y) in zip(old, strict):
        keep = old_x < 960
        np.testing.assert_array_equal(new_x, old_x[keep])
        np.testing.assert_array_equal(new_y, old_y[keep])


def test_flatbuffer_dropped_frame_tolerated_down_to_min_shots():
    # 2 of 3 expected shots stored (expected-1), min_shots allows as few as 2.
    shots = producer.decode_native(bundle(2), 3, 2)
    assert len(shots) == 2


def test_flatbuffer_below_min_shots_is_pending():
    with pytest.raises(producer.PendingData, match=r"below min_shots 2 of expected 3"):
        producer.decode_native(bundle(1), 3, 2)


def test_flatbuffer_more_shots_than_expected_is_rejected():
    with pytest.raises(producer.IngestError, match="exceeds expected"):
        producer.decode_native(bundle(3), 2, 2)


def test_dry_run_does_not_access_adb_or_create_files(config, monkeypatch, capsys):
    monkeypatch.setattr(producer, "Adb", lambda *args: pytest.fail("dry run constructed ADB"))
    assert producer.main(["--run-id", config.run_id, "--test-id", config.test_id,
                          "--expected-shots", "2", "--archive-root", str(config.archive_root),
                          "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["dry_run"] is True
    assert not config.archive_root.exists()


def test_cli_output_is_zip_or_empty_on_failure(config, monkeypatch, capsysbinary, tmp_path):
    monkeypatch.setattr(producer, "Adb", lambda *_: MockAdb())
    args = ["--run-id", config.run_id, "--test-id", config.test_id,
            "--expected-shots", "2", "--archive-root", str(config.archive_root)]
    assert producer.main(args) == 0
    captured = capsysbinary.readouterr()
    assert captured.out.startswith(b"PK") and captured.err == b""
    with zipfile.ZipFile(io.BytesIO(captured.out)) as archive:
        assert archive.testzip() is None
    destination = tmp_path / "out" / "native.zip"
    assert producer.main(args + ["--output", str(destination)]) == 0
    assert destination.read_bytes() == captured.out
    output_captured = capsysbinary.readouterr()
    assert output_captured.out == b""
    assert output_captured.err.decode().strip() == "native ZIP archived: 2 of 2 expected shots"
    assert producer.main([*args[:-1], str(tmp_path / "missing"), "--expected-shots", "3"]) == 3
    assert capsysbinary.readouterr().out == b""


def test_cli_output_reports_dropped_frames(config, monkeypatch, capsysbinary, tmp_path):
    monkeypatch.setattr(producer, "Adb", lambda *_: MockAdb(blob=bundle(1)))
    destination = tmp_path / "out" / "native.zip"
    args = ["--run-id", config.run_id, "--test-id", config.test_id,
            "--expected-shots", "2", "--min-shots", "1",
            "--archive-root", str(config.archive_root), "--output", str(destination)]
    assert producer.main(args) == 0
    err = capsysbinary.readouterr().err.decode().strip()
    assert err == "native ZIP archived: 1 of 2 expected shots"


def test_lock_excludes_concurrent_ingestion_and_is_reusable(config):
    with producer.run_lock(config.archive_root, config.run_id):
        with pytest.raises(producer.PendingData, match="owns"):
            producer.ingest(config, MockAdb())
    with producer.run_lock(config.archive_root, config.run_id):
        pass


def test_process_exit_releases_lock(config):
    code = ("import os\nfrom pathlib import Path\nfrom scripts.z300_opal_ingest import run_lock\n"
            f"with run_lock(Path({str(config.archive_root)!r}), {config.run_id!r}):\n    os._exit(0)\n")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    with producer.run_lock(config.archive_root, config.run_id):
        pass


def test_adb_output_limit_and_command_are_bounded(config, monkeypatch):
    class Process:
        stdout = io.BytesIO(b"123456")
        killed = False
        def wait(self):
            return 0
        def poll(self):
            return 0
        def kill(self):
            self.killed = True
    seen = []
    monkeypatch.setattr(producer.subprocess, "Popen", lambda args, **kw: seen.append(args) or Process())
    with pytest.raises(producer.IngestError, match="size limit"):
        producer.Adb(config).read("test -f /safe/file", 5)
    assert seen[0][:6] == ["adb", "-P", "5038", "-s", config.serial, "shell"]


def test_legacy_adb_shell_text_and_binary_pull_are_exact(config, tmp_path):
    fake = tmp_path / "adb"
    log = tmp_path / "adb-commands.jsonl"
    binary = b"\x00\r\n\x00\n\r\xffbinary\n\x00"
    fake.write_text(f"""#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
with open({str(log)!r}, 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[4] == 'shell':
    os.write(1, b'metadata\\r\\nnext\\r\\n')
elif args[4] == 'pull':
    pathlib.Path(args[6]).write_bytes({binary!r})
    if args[5].endswith('oversized'):
        time.sleep(2)
    elif args[5].endswith('timeout'):
        time.sleep(2)
else:
    sys.exit(97)
""")
    fake.chmod(0o755)
    adb = producer.Adb(replace(config, adb=str(fake)))
    assert adb.read("test -f /safe/db", 100) == b"metadata\nnext\n"
    assert adb.read("cat " + producer.SPECTRA_PATH + "/blob", 100) == binary
    with pytest.raises(producer.IngestError, match="size limit"):
        adb.read("cat " + producer.SPECTRA_PATH + "/oversized", 5)
    with pytest.raises(producer.IngestError, match="size limit"):
        adb.read("test -f /safe/db", 5)
    timeout_adb = producer.Adb(replace(config, adb=str(fake), timeout=0.1))
    with pytest.raises(producer.PendingData, match="timed out"):
        timeout_adb.read("cat " + producer.SPECTRA_PATH + "/timeout", 100)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [args[4] for args in calls] == ["shell", "pull", "pull", "shell", "pull"]
    assert all("exec-out" not in args for args in calls)
    assert all(not Path(args[6]).exists() for args in calls if args[4] == "pull")


@pytest.mark.parametrize("command", ["cat /private/data", "cat " + producer.SPECTRA_PATH + "/../secret"])
def test_binary_transport_rejects_non_spectrum_paths(config, command):
    with pytest.raises(producer.IngestError, match="safe spectra filename"):
        producer.Adb(config).read(command, 100)


def test_csv_and_total_payload_limits(config, monkeypatch):
    monkeypatch.setattr(producer, "MAX_CSV_BYTES", 20)
    with pytest.raises(producer.IngestError, match="CSV exceeds"):
        producer.ingest(config, MockAdb())
    monkeypatch.setattr(producer, "MAX_CSV_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(producer, "MAX_PAYLOAD_BYTES", 100)
    with pytest.raises(producer.IngestError, match="payload exceeds"):
        producer.ingest(config, MockAdb())
    assert not (config.archive_root / config.run_id).exists()


def legacy_record(offset=0.0):
    """Real calibration fixture plus synthetic full-length detector values."""
    fixture = Path(__file__).parent / "fixtures" / "z300_real_spectrum_trimmed.json"
    record = json.loads(fixture.read_bytes())
    record["pixels"] = [(np.arange(2066, dtype=float) + offset + segment * 10000).tolist()
                        for segment in range(4)]
    return record


def legacy_bundle(records=None, *, plain_json=False):
    records = records if records is not None else {"-1": legacy_record(999), "0": legacy_record(), "1": legacy_record(100)}
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, record in records.items():
            raw = json.dumps(record).encode()
            archive.writestr(name, raw if plain_json else gzip.compress(raw))
    return target.getvalue()


def test_legacy_exact_source_native_vendor_average_and_archive(config):
    blob = legacy_bundle()
    document = {"onlyAvgSaved": False, "config": {"numShotsToAvg": 1.0}, "shotTable": {"all": "ab" * 20}}
    adb = MockAdb(document=document, blob=blob)
    payload = producer.ingest(config, adb)
    with zipfile.ZipFile(payload) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read("raw/all.zip") == blob
        assert "raw/all.fb" not in archive.namelist()
        assert manifest["provenance"]["source_blob"] == "/sdcard/libzdata/spectrum/ab/" + "ab" * 20
        assert manifest["provenance"]["source_format"] == "legacy-zip-gzip-json"
        assert manifest["provenance"]["pixel_offset"] == 0
        assert manifest["provenance"]["average"] == "instrument-provided average on native axis"
        mean = np.loadtxt(io.BytesIO(archive.read("average.csv")), delimiter=",", skiprows=1)
        first = np.loadtxt(io.BytesIO(archive.read("shots/shot-0.csv")), delimiter=",", skiprows=1)
        assert len(mean) == 5849
        np.testing.assert_array_equal(mean[:, 0], first[:, 0])
        np.testing.assert_array_equal(mean[:, 1] - first[:, 1], np.full(len(mean), 999))
        assert mean[:, 0].max() < 960
    adb.read = lambda *args: pytest.fail("legacy idempotent archive contacted source")
    assert producer.ingest(config, adb) == payload


def test_legacy_calibration_matches_reference_scalar_pixel_zero_math():
    record = legacy_record()
    x, y = producer.legacy_spectrum(record)
    points = []
    # Independent scalar calculation from z300_calibration._polyval and its
    # closed clipping intervals, preserving the real fixture's coefficients.
    for segment in range(3):
        coefficients = record["wlCalibrations"][segment]["pixToNm"]["coefficients"]
        for pixel, intensity in enumerate(record["pixels"][segment]):
            total, power = 0.0, 1.0
            for coefficient in coefficients:
                total += coefficient * power
                power *= pixel
            if record["knots"][segment] <= total <= record["knots"][segment + 1]:
                points.append((total, intensity))
    points.sort()
    np.testing.assert_array_equal(x, np.asarray(points)[:, 0])
    np.testing.assert_array_equal(y, np.asarray(points)[:, 1])


def test_legacy_plain_json_members_supported():
    shots, average = producer.decode_legacy_native(legacy_bundle(plain_json=True), 2, 2)
    assert len(shots) == 2 and len(average[0]) == 5849


@pytest.mark.parametrize("digest", ["../a", "AB" * 20, "a" * 39, "a" * 41, ["a"]])
def test_legacy_source_requires_exact_safe_sha1(digest):
    with pytest.raises(producer.IngestError, match="SHA1"):
        producer.bundle_source({"shotTable": {"all": digest}})


def test_all_fb_precedence_is_preserved():
    assert producer.bundle_source({"shotTable": {"all_fb": "bundle-uuid", "all": "ab" * 20}}) == (
        "flatbuffer", "/sdcard/libzdata/spectra/bundle-uuid", "raw/all.fb")


def test_legacy_missing_average_is_pending():
    # "-1" (the vendor average) missing entirely: not ready yet, regardless of shots.
    records = {"0": legacy_record(), "1": legacy_record()}
    with pytest.raises(producer.PendingData, match="average"):
        producer.decode_legacy_native(legacy_bundle(records), 2, 2)


def test_legacy_trailing_shot_missing_is_pending_below_min_shots():
    # "1" missing but "0" present: a valid, still-filling-in contiguous prefix.
    records = {"-1": legacy_record(), "0": legacy_record()}
    with pytest.raises(producer.PendingData, match="below min_shots"):
        producer.decode_legacy_native(legacy_bundle(records), 2, 2)


def test_legacy_gap_in_shots_is_rejected():
    # "0" missing but "1" present: shots are not written out of order, so a
    # gap is a broken/inconsistent bundle rather than one still filling in.
    records = {"-1": legacy_record(), "1": legacy_record()}
    with pytest.raises(producer.IngestError, match="not contiguous"):
        producer.decode_legacy_native(legacy_bundle(records), 2, 1)


@pytest.mark.parametrize("extra", ["../0", "2", "0.json"])
def test_legacy_unexpected_entries_are_rejected(extra):
    records = {"-1": legacy_record(), "0": legacy_record(), "1": legacy_record(), extra: legacy_record()}
    with pytest.raises(producer.IngestError, match="unexpected"):
        producer.decode_legacy_native(legacy_bundle(records), 2, 2)


def test_legacy_more_members_than_expected_is_rejected():
    # A member at or beyond expected_shots is always an error: never publish
    # more shots than requested, however many the instrument has stored.
    records = {"-1": legacy_record(), "0": legacy_record(), "1": legacy_record(), "2": legacy_record()}
    with pytest.raises(producer.IngestError, match="unexpected"):
        producer.decode_legacy_native(legacy_bundle(records), 2, 1)


def test_legacy_dropped_frame_tolerated_down_to_min_shots():
    # 2 of 3 expected shots stored, min_shots allows as few as 2: succeeds.
    records = {"-1": legacy_record(999), "0": legacy_record(), "1": legacy_record(100)}
    shots, average = producer.decode_legacy_native(legacy_bundle(records), 3, 2)
    assert len(shots) == 2


def test_legacy_below_min_shots_is_pending():
    records = {"-1": legacy_record(999), "0": legacy_record()}
    with pytest.raises(producer.PendingData, match=r"below min_shots 2 of expected 3"):
        producer.decode_legacy_native(legacy_bundle(records), 3, 2)


@pytest.mark.parametrize("fault", ["pixels", "nonfinite", "knots", "constant", "dummy", "coefficients"])
def test_legacy_invalid_calibration_or_data_is_rejected(fault):
    record = legacy_record()
    if fault == "pixels":
        record["pixels"][0].pop()
    elif fault == "nonfinite":
        record["pixels"][0][0] = float("nan")
    elif fault == "knots":
        record["knots"][1] = 170
    elif fault == "constant":
        record["wlCalibrations"][0]["pixToNm"]["coefficients"] = [350, 0, 0, 0]
    elif fault == "dummy":
        record["wlCalibrations"][3]["pixToNm"]["coefficients"][1] = -0.001
    else:
        record["wlCalibrations"][0]["pixToNm"]["coefficients"].pop()
    with pytest.raises(producer.IngestError):
        producer.legacy_spectrum(record)


def test_legacy_differing_average_axis_rejected():
    average = legacy_record()
    average["wlCalibrations"][0]["pixToNm"]["coefficients"][0] += 0.001
    with pytest.raises(producer.IngestError, match="axes differ"):
        producer.decode_legacy_native(
            legacy_bundle({"-1": average, "0": legacy_record(), "1": legacy_record()}), 2, 2)


def test_legacy_gzip_and_zip_expansion_are_bounded(monkeypatch):
    data = legacy_bundle()
    monkeypatch.setattr(producer, "MAX_DOCUMENT_BYTES", 30000)
    with pytest.raises(producer.IngestError, match="expansion exceeds"):
        producer.decode_legacy_native(data, 2, 2)
    monkeypatch.setattr(producer, "MAX_DOCUMENT_BYTES", 100)
    with pytest.raises(producer.IngestError, match="members exceed"):
        producer.decode_legacy_native(data, 2, 2)


def test_legacy_sharded_binary_path_uses_pull(config, monkeypatch):
    calls = []
    adb = producer.Adb(config)
    monkeypatch.setattr(adb, "pull", lambda path, limit: calls.append((path, limit)) or b"ZIP")
    path = producer.LEGACY_SPECTRA_PATH + "/ab/" + "ab" * 20
    assert adb.read("cat " + path, 100) == b"ZIP"
    assert calls == [(path, 100)]
    with pytest.raises(producer.IngestError, match="safe spectra"):
        adb.read("cat " + path.replace("/ab/", "/cd/"), 100)
