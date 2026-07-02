"""Vacuum-air conversion and database wavelength convention.

The pickled line lists store Ritz vacuum wavelengths; observed spectra are
air-calibrated.  These tests pin the Edlen conversion against NIST ASD
air/vacuum reference pairs and assert that Database serves AIR wavelengths.
"""
import unittest

import numpy as np

from alibz.utils.database import Database
from alibz.utils.wavelength import vacuum_to_air

# NIST ASD reference pairs: (vacuum_nm, air_nm)
REFERENCE_PAIRS = [
    (393.4777, 393.3663),   # Ca II K
    (396.9591, 396.8469),   # Ca II H
    (589.1583, 588.9950),   # Na D2
    (589.7558, 589.5924),   # Na D1
    (656.4610, 656.2790),   # H-alpha
    (670.9610, 670.7760),   # Li I resonance
    (766.7009, 766.4899),   # K I
    (770.1084, 769.8965),   # K I
]


class TestVacuumToAir(unittest.TestCase):

    def test_reference_pairs(self):
        for vac, air in REFERENCE_PAIRS:
            self.assertAlmostEqual(
                float(vacuum_to_air(vac)), air, delta=0.002,
                msg=f"vacuum {vac} nm",
            )

    def test_below_200nm_unchanged(self):
        wl = np.array([4.2456, 121.567, 180.0, 199.99])
        np.testing.assert_array_equal(vacuum_to_air(wl), wl)

    def test_monotone_and_vectorised(self):
        wl = np.linspace(200.0, 960.0, 500)
        air = vacuum_to_air(wl)
        self.assertTrue(np.all(np.diff(air) > 0))
        # Edlen shift grows from ~0.06 nm at 200 nm to ~0.26 nm at 960 nm.
        shift = wl - air
        self.assertGreater(shift[0], 0.04)
        self.assertLess(shift[0], 0.08)
        self.assertGreater(shift[-1], 0.20)
        self.assertLess(shift[-1], 0.30)


class TestDatabaseServesAirWavelengths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    def test_known_lines_at_air_positions(self):
        expected = {
            "Ca": [393.3663, 396.8469],
            "Na": [588.9950, 589.5924],
            "K": [766.4899, 769.8965],
            "Li": [670.7760],
            "H": [656.2790],
        }
        for el, air_positions in expected.items():
            wl = self.db.lines(el)[:, 1].astype(float)
            for air in air_positions:
                nearest = wl[np.argmin(np.abs(wl - air))]
                self.assertAlmostEqual(
                    nearest, air, delta=0.003,
                    msg=f"{el} line expected at air {air} nm, nearest db {nearest}",
                )

    def test_vuv_lines_not_converted(self):
        # High-charge X-ray / VUV lines below 200 nm stay in vacuum.
        wl = self.db.lines("Na")[:, 1].astype(float)
        self.assertAlmostEqual(float(wl[np.argmin(np.abs(wl - 4.2456))]), 4.2456, places=4)


if __name__ == "__main__":
    unittest.main()
