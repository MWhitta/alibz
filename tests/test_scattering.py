import unittest

from alibz.scattering import (
    natural_element_properties,
    scattering_contributions,
)


class TestNaturalScattering(unittest.TestCase):
    def test_known_h_and_li_mechanisms_remain_separate(self):
        h = natural_element_properties("H")
        li = natural_element_properties("Li")
        self.assertLess(h["neutron_b_c_fm"], 0.0)
        self.assertGreater(h["neutron_incoherent_b"], 70.0)
        self.assertGreater(li["neutron_absorption_b"], 60.0)
        self.assertNotEqual(h["neutron_incoherent_b"],
                            h["neutron_absorption_b"])

    def test_absorption_scales_with_wavelength(self):
        base = natural_element_properties("Li", neutron_wavelength_a=1.798)
        double = natural_element_properties("Li", neutron_wavelength_a=3.596)
        self.assertAlmostEqual(double["neutron_absorption_b"],
                               2 * base["neutron_absorption_b"])

    def test_contributions_use_resolved_fraction(self):
        rows = [{
            "file": "a.csv", "sample": "a", "status": "ok",
            "fractions": {"Fe": 0.4, "Li": 0.1},
            "detections": [{"element": "Fe", "status": "detected",
                            "fraction_resolved": 0.25}],
        }]
        got = scattering_contributions(rows, q_values=(0.0,))
        fe = next(row for row in got if row["element"] == "Fe")
        self.assertEqual(fe["fraction_resolved"], 0.25)
        self.assertEqual(fe["transition_metal"], 1)
        self.assertGreater(fe["xray_power_q0"], 0.0)


if __name__ == "__main__":
    unittest.main()
