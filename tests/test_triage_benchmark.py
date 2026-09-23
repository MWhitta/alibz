import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_physical_triage import (
    crop_to_coverage,
    expand_samples,
    k_area_counterexample,
    load_manifest,
    load_replay_archive,
    quick_peaks,
)


def test_manifest_validates_hash_and_expands_all_run_means(tmp_path):
    spectrum = tmp_path / "average.csv"
    spectrum.write_text("wavelength,intensity\n200,1\n201,2\n202,1\n")
    digest = hashlib.sha256(spectrum.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "physical-triage-real-data-v1",
        "datasets": [{
            "id": "one", "truth": {"expected_positive_elements": ["Fe"]},
            "evidence": [],
            "samples": [{"path": str(spectrum), "sha256": digest,
                         "run_id": "run-one"}],
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = load_manifest(path)
    assert len(expand_samples(loaded["datasets"][0])) == 1
    spectrum.write_text(spectrum.read_text() + "203,0\n")
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("modified benchmark input was accepted")


def test_quick_extractor_uses_only_audited_coverage():
    x = np.arange(200.0, 212.0, 0.02)
    y = 2.0 + 0.05 * np.sin(x)
    y += 80.0 * np.exp(-0.5 * ((x - 205.0) / 0.04) ** 2)
    y += 200.0 * np.exp(-0.5 * ((x - 211.0) / 0.04) ** 2)
    xx, yy = crop_to_coverage(x, y, [(200.0, 210.0)])
    peaks, sigma, noise = quick_peaks(xx, yy, [(200.0, 210.0)])
    assert np.any(np.abs(peaks[:, 1] - 205.0) < 0.05)
    assert np.all(peaks[:, 1] <= 210.0)
    assert len(sigma) == len(peaks)
    assert noise[0].shape == noise[1].shape


def test_pinned_k_counterexample_recomputes_integrated_area_statistic():
    manifest = json.loads(Path(
        "provenance/physical-triage-real-data-20260922.json").read_text())
    result = k_area_counterexample(manifest)
    assert result["n"] == 928
    np.testing.assert_allclose(result["area_ratio_quantiles_5_50_95"],
                               [1.351270188, 1.440606688, 1.527383171],
                               rtol=0, atol=1e-9)
    assert result["outside_thin_plus_minus_20pct"] == 926


def test_repo_archive_replays_all_native_run_means():
    manifest = json.loads(Path(
        "provenance/physical-triage-real-data-20260922.json").read_text())
    archive = manifest["datasets"][0]["repo_local_archive"]
    records, digest = load_replay_archive(Path(archive["path"]), archive["sha256"])
    assert digest == archive["sha256"]
    assert len(records) == 26
    assert all(x.shape == (7914,) and y.shape == (7914,)
               for _, x, y in records)
