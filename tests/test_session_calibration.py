import unittest

from alibz.session_calibration import build_shared_calibration


class TestSharedSessionCalibration(unittest.TestCase):
    def test_pools_neighbors_but_not_across_acquisition_break(self):
        inventory = [
            {"test_id": i, "acquisition_gap_s": gap}
            for i, gap in zip(range(6), (None, 1, 1, 600, 1, 1))
        ]
        measurements = []
        for i in range(6):
            group_shift = 0.02 if i < 3 else -0.08
            measurements.append({
                "response_ratio": 3.0 if i < 3 else 7.0,
                "response_uncertainty": 0.1,
                "shift_deltas_nm": [
                    [group_shift] * 6,
                    [group_shift + 0.01] * 6,
                    [group_shift - 0.01] * 6,
                ],
                "n_quick_peaks": 10,
            })
        got = build_shared_calibration(
            inventory, measurements, response_window=10, shift_window=10,
            minimum_response_neighbors=2, minimum_shift_anchors=10)
        self.assertAlmostEqual(got[1]["response_prior"], 3.0)
        self.assertAlmostEqual(got[4]["response_prior"], 7.0)
        self.assertAlmostEqual(got[1]["shift_prior_nm"][0], 0.02)
        self.assertAlmostEqual(got[4]["shift_prior_nm"][0], -0.08)

    def test_insufficient_neighbors_remains_missing(self):
        inventory = [{"test_id": 1, "acquisition_gap_s": None}]
        measurements = [{
            "response_ratio": None, "response_uncertainty": None,
            "shift_deltas_nm": [[], [], []], "n_quick_peaks": 0,
        }]
        got = build_shared_calibration(inventory, measurements)
        self.assertIsNone(got[0]["response_prior"])
        self.assertEqual(got[0]["shift_prior_nm"], [None, None, None])


if __name__ == "__main__":
    unittest.main()
