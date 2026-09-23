import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.benchmark_gas_calibration as benchmark


def test_pinned_archive_has_all_26_runs_and_excludes_invalid_tail():
    manifest, dataset, records, coverage, digest = benchmark.load_inputs(
        benchmark.DEFAULT_MANIFEST, benchmark.DEFAULT_ARCHIVE)
    archive = dataset["repo_local_archive"]
    assert manifest["schema_version"] == "physical-triage-real-data-v1"
    assert len(records) == 26
    assert digest == archive["sha256"]
    assert coverage == [(186.0521, 947.9357)]
    assert coverage[0][1] == dataset["grid"]["known_gap_nm"][0]


def test_load_inputs_rejects_archive_not_matching_pinned_hash(tmp_path):
    manifest = json.loads(benchmark.DEFAULT_MANIFEST.read_text())
    bad_archive = tmp_path / "replay.npz"
    bad_archive.write_bytes(benchmark.DEFAULT_ARCHIVE.read_bytes() + b"changed")
    try:
        benchmark.load_inputs(benchmark.DEFAULT_MANIFEST, bad_archive)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("modified replay archive was accepted")


def test_gas_call_sees_raw_axis_and_observed_peaks_before_mixed_baseline(monkeypatch):
    x_raw = np.array([600.0, 600.2, 600.4])
    y_raw = np.array([1.0, 5.0, 1.0])
    observed = np.array([[4.0, 600.2, 0.05, 0.1]])
    events = []

    def fake_calibrator(x, y, db, *, peak_array):
        events.append(("gas", x.copy(), y.copy(), peak_array.copy()))
        return {
            "Ar": {"status": "no_evidence", "offset_nm": None,
                   "uncertainty_nm": None},
            "O": {"status": "no_evidence", "offset_nm": None,
                  "uncertainty_nm": None},
        }

    class Shift(float):
        def __repr__(self):
            return "dummy mixed shift"

    def fake_blind_shift(peaks, db):
        events.append(("mixed", peaks.copy()))
        return Shift(0.25), 4, peaks.copy()

    monkeypatch.setattr(benchmark, "blind_shift", fake_blind_shift)
    _, mixed, application = benchmark.calibrate_one(
        x_raw, y_raw, observed, object(), calibrator=fake_calibrator)
    assert [event[0] for event in events] == ["gas", "mixed"]
    np.testing.assert_array_equal(events[0][1], x_raw)
    np.testing.assert_array_equal(events[0][3], observed)
    np.testing.assert_array_equal(events[1][1], observed)
    assert mixed["offset_nm"] == 0.25
    assert mixed["role"] == "comparison_only_never_gas_seed"
    assert application["mode"] == "apply"
    assert application["applied"] is False


def test_controlled_injection_uses_independent_gas_origins_on_native_grid():
    x = np.linspace(600.0, 800.0, 4001)
    y = np.full_like(x, 100.0)
    offsets = {"Ar": 0.07, "O": -0.06}
    injected_y, truth, metadata = benchmark._inject_on_measured_background(
        x, y, offsets)
    expected = []
    for species in ("Ar", "O"):
        expected.extend(np.asarray(benchmark.INJECTION_ANCHORS_NM[species]) +
                        offsets[species])
    centers = [t["injected_center_nm"] for t in truth]
    np.testing.assert_allclose(np.sort(centers), np.sort(expected), atol=1e-12)
    assert {t["species"] for t in truth} == {"Ar", "O"}
    assert np.max(injected_y) > np.max(y)
    # The injection never hands truth centers to the calibrator: peaks are
    # re-extracted blindly from the injected spectrum by the benchmark.
    assert metadata["truth_centers_passed_to_calibrator"] is False
    assert metadata["peak_center_origin"].startswith("blind production refit")
    assert offsets["Ar"] != offsets["O"]


def test_dry_run_validates_without_writing_output(tmp_path, capsys):
    out = tmp_path / "must-not-exist.json"
    rc = benchmark.main([
        "--dry-run", "--replay-archive", str(benchmark.DEFAULT_ARCHIVE),
        "--out", str(out),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["run_means"] == 26
    assert payload["invalid_tail_excluded"] is True
    assert payload["archive_sha256"] == hashlib.sha256(
        benchmark.DEFAULT_ARCHIVE.read_bytes()).hexdigest()
    assert not out.exists()
