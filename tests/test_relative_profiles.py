import unittest

import numpy as np

from alibz.relative_profiles import (
    _matched_area,
    build_line_features,
    combine_relative_profiles,
)


class TestRelativeProfiles(unittest.TestCase):
    def test_matched_area_recovers_unit_area_gaussian_with_pedestal(self):
        x = np.linspace(500.0, 501.0, 1001)
        sigma = 0.05
        area = 12.0
        profile = np.exp(-0.5 * ((x - 500.5) / sigma) ** 2) \
            / (sigma * np.sqrt(2 * np.pi))
        got, uncertainty = _matched_area(
            x, 3.0 + area * profile, [500.5], [1.0],
            2.354820045 * sigma)
        self.assertAlmostEqual(got, area, places=4)
        self.assertLess(uncertainty, 1e-8)

    def test_database_features_include_expected_li_multiplet(self):
        features = build_line_features("db", elements=("Li",),
                                       features_per_stage=3)
        centers = [row["wavelength_nm"] for row in features]
        self.assertTrue(any(abs(center - 670.78) < 0.03
                            for center in centers))
        feature = min(features, key=lambda row: abs(
            row["wavelength_nm"] - 670.78))
        self.assertGreaterEqual(len(feature["components_nm"]), 2)

    def test_exhaustive_competitors_expose_ag_fe_coincidence(self):
        features = build_line_features(
            "db", elements=("Ag", "Fe"), features_per_stage=3,
            isolation_nm=0.16,
            required_wavelengths={("Ag", 2): (218.6768,)},
            exhaustive_competitors=True)
        feature = min(
            (row for row in features
             if row["element"] == "Ag" and row["ion_stage"] == 2),
            key=lambda row: abs(row["wavelength_nm"] - 218.6768))
        self.assertIn("Fe", feature["contested_by"])
        self.assertTrue(any(line.startswith("Fe_2_")
                            and "218.676" in line
                            for line in feature["competitor_lines"]))

    def test_combination_requires_multiple_clean_lines(self):
        records = []
        for test_id, factor in ((1, 1.0), (2, 2.0)):
            for line_id in ("Fe_1_a", "Fe_1_b"):
                records.append({
                    "test_id": test_id, "height_um": 100 * test_id,
                    "element": "Fe", "ion_stage": 1, "line_id": line_id,
                    "area_fraction": factor, "snr": 10.0,
                    "contested": 0,
                })
        got = combine_relative_profiles(records)
        self.assertTrue(all(row["status"] == "detected" for row in got))
        self.assertGreater(got[1]["relative_score"],
                           got[0]["relative_score"])


if __name__ == "__main__":
    unittest.main()
