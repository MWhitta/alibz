import json
import unittest

import numpy as np

from alibz.gas_detection import (
    detect_argon,
    detect_background_gases,
    detect_oxygen,
)
from alibz.utils.database import Database


class TestBackgroundGasDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    @staticmethod
    def _spectrum(centers, lo=600.0, hi=930.0, shift=0.0, amplitude=80.0):
        x = np.arange(lo, hi + 0.01, 0.02)
        # Deterministic, low-amplitude non-flat structure exercises the robust
        # noise path without making the fixtures probabilistic.
        y = 0.15 * np.sin(x * 3.1) + 0.10 * np.cos(x * 7.3)
        for center in centers:
            y += amplitude * np.exp(-0.5 * ((x - (center + shift)) / 0.045) ** 2)
        peaks = np.array([[amplitude, center, 0.045, 0.01]
                          for center in centers], dtype=float)
        if peaks.size == 0:
            peaks = np.empty((0, 4), dtype=float)
        return x, y, peaks

    def test_blank_has_no_evidence_and_is_json_safe(self):
        x, y, peaks = self._spectrum([])
        result = detect_background_gases(x, y, self.db, peak_array=peaks)
        self.assertEqual(result["Ar"]["status"], "no_evidence")
        self.assertEqual(result["O"]["status"], "no_evidence")
        json.dumps(result, allow_nan=False)

    def test_argon_requires_three_independent_groups(self):
        centers = [696.542978, 706.721736, 738.397998]
        x, y, peaks = self._spectrum(centers)
        detected = detect_argon(x, y, self.db, peak_array=peaks)
        self.assertEqual(detected["status"], "detected")
        self.assertGreaterEqual(detected["n_clean_groups"], 3)

        x, y, peaks = self._spectrum(centers[:1])
        one = detect_argon(x, y, self.db, peak_array=peaks)
        self.assertEqual(one["status"], "tentative")
        self.assertEqual(one["n_supported_groups"], 1)

    def test_oxygen_777_triplet_counts_once(self):
        x, y, peaks = self._spectrum([777.19443])
        one = detect_oxygen(x, y, self.db, peak_array=peaks)
        self.assertEqual(one["status"], "tentative")
        self.assertEqual(one["n_supported_groups"], 1)
        self.assertIn("OI_777_is_one_unresolved_group", one["reasons"])

        x, y, peaks = self._spectrum([777.19443, 844.635909])
        two = detect_oxygen(x, y, self.db, peak_array=peaks)
        self.assertEqual(two["status"], "detected")
        self.assertEqual(two["n_clean_groups"], 2)

    def test_no_peak_table_keeps_signal_tentative(self):
        x, y, _ = self._spectrum([696.542978, 706.721736, 738.397998])
        result = detect_argon(x, y, self.db)
        self.assertEqual(result["status"], "tentative")
        self.assertIn("interference_check_requires_peak_array", result["reasons"])

    def test_out_of_range_and_internal_hole(self):
        x = np.arange(200.0, 300.0, 0.02)
        y = np.zeros_like(x)
        result = detect_background_gases(
            x, y, self.db, peak_array=np.empty((0, 4)))
        self.assertEqual(result["Ar"]["status"], "out_of_range")
        self.assertEqual(result["O"]["status"], "out_of_range")

        x, y, peaks = self._spectrum([777.19443], lo=776.5, hi=778.2)
        keep = ~((x > 777.27) & (x < 777.47))
        result = detect_oxygen(x[keep], y[keep], self.db, peak_array=peaks)
        group = next(g for g in result["groups"] if g["group"] == "OI_777")
        self.assertFalse(group["covered"])
        self.assertIn("coverage_gap", group["reasons"])

    def test_shift_is_applied_to_raw_windows(self):
        centers = [696.542978, 706.721736, 738.397998]
        x, y, peaks = self._spectrum(centers, shift=0.25)
        shifted = detect_argon(x, y, self.db, shift_nm=0.25,
                               peak_array=peaks)
        self.assertEqual(shifted["status"], "detected")
        self.assertGreaterEqual(shifted["n_clean_groups"], 3)

    def test_native_nir_pitch_still_has_measurable_windows(self):
        centers = [696.542978, 763.510524, 801.478572]
        x = np.arange(620.0, 930.0, 0.18)
        y = 0.1 * np.sin(3.0 * x)
        for center in centers:
            y += 300.0 * np.exp(-0.5 * ((x - center) / 0.09) ** 2)
        peaks = np.array([[300.0, center, 0.08, 0.01]
                          for center in centers])
        result = detect_argon(x, y, self.db, peak_array=peaks)
        self.assertEqual(result["status"], "detected")
        self.assertGreaterEqual(result["n_clean_groups"], 3)
        self.assertTrue(all(group["n_native_samples"] >= 3
                            for group in result["groups"]
                            if group["supported"]))

    def test_native_gaussian_noise_is_calibrated_and_nulls_do_not_detect(self):
        x = np.arange(600.0, 930.0, 0.18)
        peaks = np.empty((0, 4), dtype=float)
        estimates = []
        for seed in range(40):
            y = np.random.default_rng(seed).normal(0.0, 5.0, x.size)
            result = detect_background_gases(
                x, y, self.db, peak_array=peaks,
                config={"noise_floor": 0.01},
            )
            self.assertNotEqual(result["Ar"]["status"], "detected")
            self.assertNotEqual(result["O"]["status"], "detected")
            estimates.extend(g["noise_per_native_sample"]
                             for record in result.values()
                             for g in record["groups"] if g["covered"])
        # Small sidebands make individual robust scales noisy; their pooled
        # median should still recover the injected point-noise sigma.
        self.assertAlmostEqual(float(np.median(estimates)), 5.0, delta=0.75)

    def test_supported_competitor_marks_group_blended(self):
        centers = [696.542978, 706.721736, 738.397998]
        x, y, peaks = self._spectrum(centers)
        result = detect_argon(
            x, y, self.db, peak_array=peaks,
            config={"competitor_elements": ("Rh",)},
        )
        group = next(g for g in result["groups"] if g["group"] == "ArI_696")
        self.assertIn("Rh", group["blend_elements"])
        self.assertFalse(group["clean"])
        self.assertEqual(result["status"], "tentative")

    def test_competitor_needs_three_external_same_ion_band_matches(self):
        # Fe II 645.638 nm can interfere with the O I 645 nm group.  Its own
        # target-window peak is excluded; three locally strong Fe II anchors
        # elsewhere in the 600--700 nm band establish the ambiguity.
        centers = [645.444423, 624.755699, 651.607687, 692.202656]
        x, y, peaks = self._spectrum(centers)
        result = detect_oxygen(x, y, self.db, peak_array=peaks)
        group = next(g for g in result["groups"] if g["group"] == "OI_645")
        self.assertIn("Fe", group["blend_elements"])
        self.assertFalse(group["clean"])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            detect_argon([1, 2], [1], self.db)
        with self.assertRaises(ValueError):
            detect_oxygen([2, 1, 3], [0, 0, 0], self.db)
        x, y, _ = self._spectrum([777.19443])
        with self.assertRaises(ValueError):
            detect_oxygen(x, y, self.db, peak_array=np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
