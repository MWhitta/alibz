"""Vacuum-air conversion and database wavelength convention.

The pickled line lists store Ritz vacuum wavelengths; observed spectra are
air-calibrated.  These tests pin the Edlen conversion against NIST ASD
air/vacuum reference pairs and assert that Database serves AIR wavelengths.
"""
import unittest

import numpy as np

from alibz.utils.database import Database
from alibz.utils.wavelength import air_to_vacuum, vacuum_to_air

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

    def test_air_to_vacuum_round_trip(self):
        vacuum = np.array([190.0, 200.0, 393.4777, 589.1583,
                           656.4610, 960.0])
        recovered = air_to_vacuum(vacuum_to_air(vacuum))
        np.testing.assert_allclose(recovered, vacuum, atol=2e-10, rtol=0)


class TestElementSupport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    def test_h_to_u_positions_and_explicit_unsupported_mask(self):
        self.assertEqual(len(self.db.elements), 92)
        self.assertEqual(self.db.elements[0], "H")
        self.assertEqual(self.db.elements[-1], "U")
        self.assertEqual(
            self.db.unsupported_elements, {"Pm", "Po", "At", "Rn", "Pa"}
        )
        self.assertEqual(self.db.support_mask.shape, (92,))
        self.assertEqual(int(self.db.support_mask.sum()), 87)
        for element in ("Se", "Th", "U"):
            self.assertTrue(self.db.is_supported(element))

    def test_se_th_u_have_quantitative_low_ion_lines(self):
        minimum_counts = {"Se": 8, "Th": 2000, "U": 1100}
        expected_stages = {"Se": {1}, "Th": {1, 2}, "U": {1, 2}}
        for element, minimum in minimum_counts.items():
            lines = self.db.lines(element)
            ion = lines[:, 0].astype(float).astype(int)
            wavelength = lines[:, 1].astype(float)
            gA = lines[:, 3].astype(float)
            Ei = lines[:, 4].astype(float)
            Ek = lines[:, 5].astype(float)
            in_band = (wavelength >= 180.0) & (wavelength <= 962.0)
            self.assertGreaterEqual(int(in_band.sum()), minimum, element)
            self.assertEqual(set(ion), expected_stages[element])
            self.assertTrue(np.all(gA > 0), element)
            self.assertTrue(np.all(Ek > Ei), element)

    def test_observed_only_se_ii_lines_are_retained_but_not_quantitative(self):
        lines = self.db.observed_lines("Se", ion=2)
        self.assertGreaterEqual(len(lines), 10)
        self.assertTrue(all(not row["quantitative_ready"] for row in lines))
        self.assertTrue(all(180.0 <= row["wavelength_nm"] <= 962.0
                            for row in lines))


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


class TestWavelengthShiftEstimation(unittest.TestCase):
    """Instrument thermal drift produces a systematic line shift on top of
    the steel-standard calibration; the estimator recovers it from bright
    lines matched to a locally-dominant anchor catalog."""

    # air positions of lines that are locally dominant in the database
    REFS = [257.610, 421.552, 422.673, 455.404, 460.733, 588.995,
            610.364, 670.781, 766.490, 769.896, 780.027, 794.760]

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    def _peaks(self, shift_nm):
        n = len(self.REFS)
        return np.column_stack([
            np.linspace(1000.0, 300.0, n),
            np.asarray(self.REFS) + shift_nm,
            np.full(n, 0.08),
            np.full(n, 0.02),
        ])

    def test_recovers_applied_shift(self):
        from alibz.utils.wavelength import estimate_wavelength_shift

        shift, n = estimate_wavelength_shift(self._peaks(-0.030), self.db)
        self.assertGreaterEqual(n, 10)
        self.assertAlmostEqual(shift, -0.030, delta=0.008)

        shift0, n0 = estimate_wavelength_shift(self._peaks(0.0), self.db)
        self.assertGreaterEqual(n0, 10)
        self.assertLess(abs(shift0), 0.005)

    def test_returns_zero_without_enough_matches(self):
        from alibz.utils.wavelength import estimate_wavelength_shift

        peaks = np.array([[100.0, 500.123, 0.08, 0.02]])
        shift, n = estimate_wavelength_shift(peaks, self.db)
        self.assertEqual(shift, 0.0)
        self.assertLess(n, 5)


class TestSegmentShiftEstimation(unittest.TestCase):
    """The three detector segments are independently calibrated and drift
    independently (measured ~33 pm apart on MW2-112); the per-segment
    estimator recovers each segment's own shift with a global fallback for
    under-anchored segments."""

    EDGES = (365.0, 620.0)

    @classmethod
    def setUpClass(cls):
        from alibz.utils.wavelength import _anchor_catalog
        cls.db = Database("db")
        cls.anchors = _anchor_catalog(cls.db)

    def _refs(self, per_segment=(1, 8, 8)):
        """Well-separated anchor positions, ``per_segment[i]`` from segment i.

        Sampling the catalog itself keeps the synthetic peaks exactly on
        anchor positions — the hand-picked line list of the scalar test
        sits up to tens of pm off the strength-weighted cluster centroids,
        which the pooled median tolerates but a 5-anchor segment median
        does not (and need not: real spectra give 15+ anchors/segment).
        """
        seg = np.digitize(self.anchors, self.EDGES)
        refs = []
        for s, want in enumerate(per_segment):
            got, last = [], -np.inf
            for wl in self.anchors[seg == s]:
                if wl - last >= 2.0:
                    got.append(float(wl))
                    last = wl
                if len(got) == want:
                    break
            refs.extend(got)
        return np.asarray(refs)

    def _peaks(self, seg_shifts, per_segment=(1, 8, 8)):
        refs = self._refs(per_segment)
        seg = np.digitize(refs, self.EDGES)
        shifted = refs + np.asarray(seg_shifts)[seg]
        n = refs.size
        return np.column_stack([np.linspace(1000.0, 300.0, n), shifted,
                                np.full(n, 0.08), np.full(n, 0.02)])

    def test_recovers_per_segment_shifts_with_fallback(self):
        from alibz.utils.wavelength import estimate_wavelength_shift_segments

        # segment 0 has ONE anchor (below the match floor): falls back to
        # the pooled global; segments 1 and 2 recover their own shifts
        shift, n = estimate_wavelength_shift_segments(
            self._peaks([-0.030, -0.030, +0.010]), self.db)
        self.assertGreaterEqual(n, 10)
        self.assertAlmostEqual(shift.at(500.0), -0.030, delta=0.008)
        self.assertAlmostEqual(shift.at(700.0), +0.010, delta=0.008)
        self.assertAlmostEqual(shift.at(300.0), float(shift), delta=1e-12)
        # vectorised evaluation follows the segment boundaries
        vals = shift.at(np.array([500.0, 700.0]))
        self.assertAlmostEqual(vals[0], -0.030, delta=0.008)
        self.assertAlmostEqual(vals[1], +0.010, delta=0.008)

    def test_uniform_shift_matches_scalar_estimator(self):
        from alibz.utils.wavelength import (estimate_wavelength_shift,
                                            estimate_wavelength_shift_segments)

        pk = self._peaks([-0.020, -0.020, -0.020], per_segment=(5, 8, 8))
        seg_shift, _ = estimate_wavelength_shift_segments(pk, self.db)
        scalar, _ = estimate_wavelength_shift(pk, self.db, n_peaks=60)
        self.assertAlmostEqual(float(seg_shift), scalar, delta=0.002)
        for wl in (250.0, 500.0, 800.0):
            self.assertAlmostEqual(seg_shift.at(wl), -0.020, delta=0.008)

    def test_too_few_matches_returns_zeros(self):
        from alibz.utils.wavelength import estimate_wavelength_shift_segments

        shift, n = estimate_wavelength_shift_segments(
            np.array([[100.0, 500.123, 0.08, 0.02]]), self.db)
        self.assertLess(n, 5)
        self.assertEqual(float(shift), 0.0)
        self.assertEqual(shift.at(700.0), 0.0)

    def test_shift_at_scalar_passthrough(self):
        from alibz.utils.wavelength import shift_at

        self.assertEqual(shift_at(-0.015, 700.0), -0.015)
        np.testing.assert_allclose(
            shift_at(-0.015, np.array([200.0, 700.0])), -0.015)


if __name__ == "__main__":
    unittest.main()
