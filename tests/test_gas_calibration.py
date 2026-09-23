import inspect
import json
import unittest

import numpy as np

from alibz.gas_calibration import calibrate_background_gases
from alibz.utils.database import Database


AR = [696.542978, 706.721736, 738.397998, 750.386765, 763.510524]
O645 = [645.360246, 645.444423, 645.597682]
O777 = [777.194430, 777.416570, 777.538737]


class TestGasWavelengthCalibration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    @staticmethod
    def _spectrum(lines, offsets, *, pitch=0.02, sigma=0.035,
                  lo=600.0, hi=930.0, seed=None):
        x = np.arange(lo, hi + pitch / 2, pitch)
        if seed is None:
            y = 0.08 * np.sin(2.7 * x) + 0.05 * np.cos(6.1 * x)
        else:
            y = np.random.default_rng(seed).normal(0.0, 1.0, x.size)
        peaks = []
        for wavelength, offset, amplitude in zip(lines, offsets,
                                                  [120.0] * len(lines)):
            observed = wavelength + offset
            y += amplitude * np.exp(-0.5 * ((x - observed) / sigma) ** 2)
            peaks.append([amplitude, observed, sigma, 0.005])
        return x, y, np.asarray(peaks, dtype=float).reshape((-1, 4))

    @staticmethod
    def _cfg(**extra):
        return {"check_competitors": False, **extra}

    def test_argon_recovers_positive_absolute_offset(self):
        x, y, peaks = self._spectrum(AR[:3], [0.073] * 3)
        result = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())
        ar = result["Ar"]
        self.assertEqual(ar["status"], "calibrated")
        self.assertAlmostEqual(ar["offset_nm"], 0.073, delta=0.004)
        self.assertEqual(ar["n_inliers"], 3)
        self.assertEqual(ar["offset_convention"], "observed_minus_database_nm")
        self.assertTrue(ar["independent_of_internal_calibration"])

    def test_oxygen_recovers_negative_offset_from_resolved_multiplets(self):
        lines = O645 + O777
        x, y, peaks = self._spectrum(lines, [-0.061] * len(lines), sigma=0.025)
        oxygen = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["O"]
        self.assertEqual(oxygen["status"], "calibrated")
        self.assertAlmostEqual(oxygen["offset_nm"], -0.061, delta=0.004)
        self.assertEqual(oxygen["n_inliers"], 2)
        self.assertGreaterEqual(oxygen["supported_range_nm"][0], 645.0)
        self.assertLessEqual(oxygen["supported_range_nm"][1], 778.0)

    def test_argon_and_oxygen_offsets_are_independent(self):
        lines = AR[:3] + O645 + O777
        offsets = [0.071] * 3 + [-0.058] * (len(O645) + len(O777))
        x, y, peaks = self._spectrum(lines, offsets, sigma=0.025)
        result = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())
        self.assertAlmostEqual(result["Ar"]["offset_nm"], 0.071, delta=0.004)
        self.assertAlmostEqual(result["O"]["offset_nm"], -0.058, delta=0.004)

    def test_unrelated_fe_frame_cannot_seed_gas_offset(self):
        # Strong unrelated peaks at a different displacement are simply
        # observed-frame distractors.  The API has no existing-shift input.
        fe_like = [640.00, 660.00, 680.00, 720.00]
        lines = AR[:3] + fe_like
        offsets = [0.045] * 3 + [0.280] * len(fe_like)
        x, y, peaks = self._spectrum(lines, offsets)
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        self.assertEqual(ar["status"], "calibrated")
        self.assertAlmostEqual(ar["offset_nm"], 0.045, delta=0.004)
        self.assertNotIn("shift_nm", inspect.signature(
            calibrate_background_gases).parameters)

    def test_missing_argon_group_abstains(self):
        x, y, peaks = self._spectrum(AR[:2], [0.04, 0.04])
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        self.assertEqual(ar["status"], "tentative")
        self.assertIsNone(ar["offset_nm"])
        # Two bracketing lines are kept as diagnostic evidence but can never
        # be mistaken for an applicable calibration: every segment record is
        # demoted to "tentative" and the element-level offset stays None.
        self.assertTrue(all(seg["status"] == "tentative" for seg in ar["segments"]))
        self.assertTrue(all(seg["n_inliers"] < 3 for seg in ar["segments"]))

    def test_inconsistent_groups_fail_consensus(self):
        x, y, peaks = self._spectrum(AR[:3], [0.04, 0.04, 0.25])
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        self.assertEqual(ar["status"], "tentative")
        self.assertIsNone(ar["offset_nm"])
        self.assertTrue(any("requires_3" in reason or "ambiguous" in reason
                            for reason in ar["reasons"]))

    def test_unresolved_oxygen_centroid_is_not_a_precise_anchor(self):
        # One fitted center for the O I 777 multiplet cannot be assigned to a
        # component or fixed-ratio centroid.
        center = float(np.mean(O777)) + 0.05
        x, y, peaks = self._spectrum([float(np.mean(O777))], [0.05], sigma=0.16)
        peaks[0, 1] = center
        oxygen = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["O"]
        group = next(g for g in oxygen["groups"] if g["group"] == "OI_777")
        self.assertFalse(group["selected"])
        self.assertIn("unresolved_centroid_not_used", group["flags"])
        self.assertGreaterEqual(group["uncertainty_nm"], np.ptp(O777) / 2)
        self.assertNotEqual(oxygen["status"], "calibrated")

    def test_native_grid_has_cell_uncertainty_floor(self):
        x, y, peaks = self._spectrum(
            AR[:3], [0.07] * 3, pitch=0.18, sigma=0.11)
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        self.assertEqual(ar["status"], "calibrated")
        floor = 0.18 / np.sqrt(12.0)
        used = [g for g in ar["groups"] if g["selected"]]
        self.assertTrue(all(g["uncertainty_nm"] >= floor - 1e-6 for g in used))
        self.assertAlmostEqual(ar["offset_nm"], 0.07, delta=0.01)

    def test_blank_and_noise_do_not_false_calibrate(self):
        for seed in range(12):
            x = np.arange(600.0, 930.0, 0.05)
            y = np.random.default_rng(seed).normal(0.0, 1.0, x.size)
            result = calibrate_background_gases(
                x, y, self.db, peak_array=None,
                config=self._cfg(noise_floor=0.01))
            self.assertNotEqual(result["Ar"]["status"], "calibrated")
            self.assertNotEqual(result["O"]["status"], "calibrated")

    def test_out_of_range_and_json_safe(self):
        x = np.arange(200.0, 300.0, 0.05)
        y = np.zeros_like(x)
        result = calibrate_background_gases(
            x, y, self.db, peak_array=np.empty((0, 4)), config=self._cfg())
        self.assertEqual(result["Ar"]["status"], "out_of_range")
        self.assertEqual(result["O"]["status"], "out_of_range")
        json.dumps(result, allow_nan=False)

    def test_supported_range_is_database_bracket_and_segment_is_complete(self):
        x, y, peaks = self._spectrum(AR[:4], [0.09] * 4)
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        segment = ar["segments"][0]
        self.assertEqual(segment["status"], "calibrated")
        self.assertEqual(segment["n_inliers"], 4)
        self.assertAlmostEqual(segment["supported_range_nm"][0], AR[0], places=5)
        self.assertAlmostEqual(segment["supported_range_nm"][1], AR[3], places=5)
        np.testing.assert_allclose(
            np.ravel(segment["observed_anchor_positions_nm"]),
            np.asarray(AR[:4]) + 0.09, atol=2e-6)

    def test_flat_top_rejected(self):
        x, y, peaks = self._spectrum(AR[:3], [0.04] * 3)
        for center in np.asarray(AR[:3]) + 0.04:
            nearest = int(np.argmin(abs(x - center)))
            y[nearest - 1:nearest + 2] = 500.0
        ar = calibrate_background_gases(
            x, y, self.db, peak_array=peaks, config=self._cfg())["Ar"]
        self.assertNotEqual(ar["status"], "calibrated")

    def test_invalid_inputs_and_unknown_config_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_background_gases([1, 2], [1], self.db)
        x = np.arange(600.0, 930.0, 0.1)
        y = np.zeros_like(x)
        with self.assertRaises(ValueError):
            calibrate_background_gases(x, y, self.db, config={"shift_nm": 0.2})
        with self.assertRaises(ValueError):
            calibrate_background_gases(x, y, self.db,
                                       peak_array=np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
