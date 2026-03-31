"""Tests for alibz.detector — three-segment detector model."""

import json
import numpy as np
import pickle
import tempfile
import os
import shutil
import unittest

from alibz.detector import (
    DetectorModel,
    Junction,
    Segment,
    _estimate_segment_background,
    _blend_junction,
)


def _make_synthetic_pca(tmp_dir):
    """Build a fake background PCA pickle and detector config."""
    wl = np.arange(190.0, 910.01, 0.1)
    n = len(wl)
    n_components = 5
    j1, j2 = 365.0, 620.0
    j1_idx = np.searchsorted(wl, j1)
    j2_idx = np.searchsorted(wl, j2)

    # Mean spectrum with step discontinuities at junctions
    mean = 100 + 20 * np.sin(2 * np.pi * wl / 400)
    mean[j1_idx:] += 30
    mean[j2_idx:] -= 15

    # Components: 2 and 3 have loading concentrated at junctions
    rng = np.random.RandomState(42)
    components = rng.randn(n_components, n) * 0.01

    # Component 2: concentrated near junction 1
    spike1 = np.exp(-0.5 * ((wl - j1) / 3.0) ** 2)
    components[2] = spike1 / np.linalg.norm(spike1)

    # Component 3: concentrated near junction 2
    spike2 = np.exp(-0.5 * ((wl - j2) / 3.0) ** 2)
    components[3] = spike2 / np.linalg.norm(spike2)

    evr = np.array([0.4, 0.2, 0.15, 0.1, 0.05])

    pca_path = os.path.join(tmp_dir, "bg_pca.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump({
            "wavelength": wl, "components": components, "mean": mean,
            "explained_variance_ratio": evr,
            "singular_values": np.sqrt(evr * n),
            "scores": rng.randn(10, n_components),
            "csv_files": [], "n_spectra": 10, "n_channels": n,
        }, f)

    # Write detector config
    corrections_dir = os.path.join(tmp_dir, "corrections")
    os.makedirs(corrections_dir)
    config_path = os.path.join(corrections_dir, "detector.json")
    with open(config_path, "w") as f:
        json.dump({
            "instrument": "test",
            "detector_segments": 3,
            "junctions_nm": [365.0, 620.0],
            "junction_zone_half_width_nm": 5.0,
            "blend_width_nm": 2.0,
            "segments": [
                {"index": 0, "label": "UV"},
                {"index": 1, "label": "VIS"},
                {"index": 2, "label": "NIR"},
            ],
        }, f)

    return pca_path, config_path, wl, mean


class TestDetectorModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pca_path, self.config_path, self.wl, self.mean = \
            _make_synthetic_pca(self.tmp)
        self.model = DetectorModel.from_pca(
            self.pca_path, config_path=self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_junctions_at_configured_positions(self):
        self.assertEqual(len(self.model.junctions), 2)
        self.assertEqual(self.model.junctions[0].wavelength, 365.0)
        self.assertEqual(self.model.junctions[1].wavelength, 620.0)

    def test_three_segments(self):
        self.assertEqual(len(self.model.segments), 3)
        self.assertEqual(self.model.segments[0].label, "UV")
        self.assertEqual(self.model.segments[1].label, "VIS")
        self.assertEqual(self.model.segments[2].label, "NIR")
        self.assertAlmostEqual(self.model.segments[0].wl_hi, 365.0)
        self.assertAlmostEqual(self.model.segments[1].wl_lo, 365.0)
        self.assertAlmostEqual(self.model.segments[1].wl_hi, 620.0)
        self.assertAlmostEqual(self.model.segments[2].wl_lo, 620.0)

    def test_artifact_components_detected(self):
        # Components 2 and 3 have spikes at the junctions
        self.assertIn(2, self.model.artifact_components)
        self.assertIn(3, self.model.artifact_components)
        self.assertNotIn(0, self.model.artifact_components)
        self.assertNotIn(1, self.model.artifact_components)

    def test_correct_returns_same_length(self):
        y = self.mean + np.random.RandomState(0).randn(len(self.wl)) * 5
        corrected = self.model.correct(self.wl, y)
        self.assertEqual(corrected.shape, y.shape)

    def test_correct_reduces_junction_discontinuity(self):
        rng = np.random.RandomState(0)
        y = self.mean.copy() + rng.randn(len(self.wl)) * 2

        corrected = self.model.correct(self.wl, y)

        for junc in self.model.junctions:
            idx = junc.index
            left = slice(max(0, idx - 50), idx)
            right = slice(idx, min(len(y), idx + 50))
            step_before = abs(np.mean(y[left]) - np.mean(y[right]))
            step_after = abs(np.mean(corrected[left]) - np.mean(corrected[right]))
            self.assertLess(step_after, step_before,
                            f"Junction at {junc.wavelength:.1f} nm: "
                            f"step grew from {step_before:.1f} to {step_after:.1f}")

    def test_correct_preserves_peaks(self):
        y = self.mean.copy()
        peak_idx = np.argmin(np.abs(self.wl - 500.0))
        y[peak_idx] += 1000.0

        corrected = self.model.correct(self.wl, y)
        local = slice(peak_idx - 20, peak_idx + 20)
        self.assertEqual(np.argmax(corrected[local]), 20)

    def test_correct_non_negative(self):
        y = self.mean.copy() + 50
        corrected = self.model.correct(self.wl, y)
        self.assertGreater(np.min(corrected), -50)

    def test_summary(self):
        text = self.model.summary()
        self.assertIn("DetectorModel", text)
        self.assertIn("365.0", text)
        self.assertIn("620.0", text)
        self.assertIn("UV", text)
        self.assertIn("VIS", text)
        self.assertIn("NIR", text)


class TestSegmentBackground(unittest.TestCase):
    def test_flat_baseline(self):
        x = np.linspace(0, 100, 1000)
        y = np.ones_like(x) * 50 + np.random.RandomState(0).randn(1000) * 0.1
        bg = _estimate_segment_background(x, y)
        self.assertAlmostEqual(np.std(bg), 0.0, delta=2.0)

    def test_peaks_above_background(self):
        x = np.linspace(0, 100, 1000)
        y = np.ones_like(x) * 10
        for center in [20, 50, 80]:
            y += 100 * np.exp(-0.5 * ((x - center) / 1.0) ** 2)
        bg = _estimate_segment_background(x, y)
        self.assertLess(np.max(bg), 80)


if __name__ == "__main__":
    unittest.main()
