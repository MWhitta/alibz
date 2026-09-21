"""Tests for alibz.utils.native_grid.

Synthetic round trip: proves the identified kernel (not-a-knot cubic
spline) + lstsq inversion recover native samples to near machine precision
WHEN the forward model assumptions hold exactly -- this is the control that
shows the method (the linear forward/inverse machinery) is sound.

Real-sample tests: the default calibrated mode (fixed instrument
dispersion + per-spectrum polynomial wavelength correction; seconds per
spectrum) must be exact on every segment of every bundled sample, be
deterministic, and recover an imposed calibration shift.  The full
pitch/phase search (mode='search', minutes per segment) is opt-in via
ALIBZ_SLOW_TESTS=1.  Skipped when data/remote_samples is absent.
"""
import os
import unittest

import numpy as np

from alibz.utils.native_grid import (
    EXACT_RELRES_THRESHOLD,
    KERNEL_NAME,
    export_kernel_matrix,
    recover_native_grid,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "remote_samples")
_SAMPLE_FILES = ["REE_01.csv", "REE_44.csv", "argon_noAr.csv", "scan9x9.csv"]

def _load(filename):
    path = os.path.join(_DATA_DIR, filename)
    w, y = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    return w, y


class TestExportKernelMatrix(unittest.TestCase):

    def test_shape_and_sparsity(self):
        x_native = np.linspace(620.0, 700.0, 300)
        x_export = np.arange(621.0, 699.0, 1.0 / 30.0)
        S = export_kernel_matrix(x_native, x_export)
        self.assertEqual(S.shape, (x_export.size, x_native.size))
        # not-a-knot cubic spline weights decay away from each evaluation
        # point; thresholding should leave the matrix genuinely sparse.
        self.assertLess(S.nnz, 0.5 * S.shape[0] * S.shape[1])

    def test_rejects_non_increasing_native_grid(self):
        x_native = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            export_kernel_matrix(x_native, np.array([2.5]))

    def test_rejects_export_outside_native_domain(self):
        x_native = np.linspace(0.0, 10.0, 20)
        with self.assertRaises(ValueError):
            export_kernel_matrix(x_native, np.array([10.5]))

    def test_reproduces_native_values_at_native_points(self):
        # An interpolating spline is exact AT the knots themselves.
        x_native = np.linspace(0.0, 10.0, 15)
        S = export_kernel_matrix(x_native, x_native, threshold=0.0)
        np.testing.assert_allclose(S.toarray(), np.eye(15), atol=1e-10)


class TestSyntheticRoundTrip(unittest.TestCase):
    """Forward-transform a known native spectrum with the identified
    kernel, recover it blind, and check near-machine-precision
    reconstruction -- the control proving the linear forward/inverse
    machinery underlying recover_native_grid is sound.
    """

    def _round_trip(self, seed, n_native=400, n_peaks=6):
        rng = np.random.default_rng(seed)
        x_native = np.sort(rng.uniform(0.0, 50.0, n_native))
        # enforce strictly increasing (resolve any accidental ties)
        x_native = x_native + np.arange(n_native) * 1e-9

        y_native = rng.normal(0.0, 1.0, n_native)
        for center in rng.uniform(5.0, 45.0, n_peaks):
            y_native += 40.0 * np.exp(-0.5 * ((x_native - center) / 0.4) ** 2)

        # export grid: uniform, finer than the native grid, strictly inside
        # the native domain so the spline needs no extrapolation.
        x_export = np.arange(x_native[2], x_native[-3], 1.0 / 30.0)

        S_forward = export_kernel_matrix(x_native, x_export, threshold=0.0)
        y_export = S_forward.toarray() @ y_native

        # blind recovery: same native grid, same kernel, invert by lstsq.
        S_inverse = export_kernel_matrix(x_native, x_export)
        y_native_hat, *_ = np.linalg.lstsq(S_inverse.toarray(), y_export, rcond=None)

        y_export_hat = S_inverse.toarray() @ y_native_hat
        relative_error = np.max(np.abs(y_export_hat - y_export)) / np.max(np.abs(y_export))
        return relative_error

    def test_round_trip_reaches_near_machine_precision(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                relative_error = self._round_trip(seed)
                self.assertLess(relative_error, 1e-8)


class TestRecoverNativeGridValidation(unittest.TestCase):
    """Fast input-validation tests -- no real data or optimization needed."""

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            recover_native_grid(np.array([1.0, 2.0]), np.array([1.0]))

    def test_rejects_non_increasing_x(self):
        x = np.array([620.0, 621.0, 620.5])
        y = np.zeros_like(x)
        with self.assertRaises(ValueError):
            recover_native_grid(x, y)

    def test_rejects_uncalibrated_segment_edges(self):
        x = np.arange(180.0, 961.0, 1.0 / 30.0)
        y = np.zeros_like(x)
        with self.assertRaises(ValueError):
            recover_native_grid(x, y, segment_edges=(400.0, 700.0))

    def test_rejects_segment_too_short_for_one_window(self):
        # Shorter than 2x the 6 nm search window -- no segment can be
        # meaningfully covered.
        x = np.arange(700.0, 705.0, 1.0 / 30.0)
        y = np.zeros_like(x)
        with self.assertRaises(ValueError):
            recover_native_grid(x, y)


@unittest.skipUnless(os.path.isdir(_DATA_DIR), "data/remote_samples not present")
class TestCalibratedRecovery(unittest.TestCase):
    """Default (fast, deterministic) mode: the instrument dispersion is a
    fixed calibration and each spectrum only needs a low-order polynomial
    wavelength correction per segment (the vendor's per-acquisition
    calibration).  Every segment of every bundled sample must reach the
    strict exact bar in seconds, without the search fallback.
    """

    def test_all_files_all_segments_exact_and_fast(self):
        import time
        for filename in _SAMPLE_FILES:
            with self.subTest(filename=filename):
                w, y = _load(filename)
                t0 = time.time()
                x_native, y_native, info = recover_native_grid(w, y, fallback=False)
                elapsed = time.time() - t0
                self.assertEqual(info["mode"], "calibrated")
                self.assertEqual(set(info["segments"]), {"UV", "VIS", "NIR"})
                for name, seg in info["segments"].items():
                    self.assertEqual(seg["mode"], "calibrated", msg=f"{filename} {name}")
                    self.assertFalse(seg["used_fallback"])
                    self.assertLess(seg["relres"], EXACT_RELRES_THRESHOLD,
                                    msg=f"{filename} {name}: relres={seg['relres']!r}")
                self.assertTrue(info["exact"])
                self.assertTrue(np.all(np.diff(x_native) > 0))
                self.assertEqual(x_native.shape, y_native.shape)
                self.assertLess(elapsed, 120.0, msg=f"{filename}: {elapsed:.0f} s")

    def test_deterministic(self):
        w, y = _load("REE_44.csv")
        mask = w >= 620.0
        a = recover_native_grid(w[mask], y[mask], fallback=False)
        b = recover_native_grid(w[mask], y[mask], fallback=False)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2]["segments"]["NIR"]["correction_coef"],
                         b[2]["segments"]["NIR"]["correction_coef"])

    def test_recovers_an_imposed_calibration_shift(self):
        """Export a recovered native spectrum through knots shifted by a
        known amount and stretched; the calibrated mode must find that
        correction (to < 0.01 native px) and stay exact."""
        from alibz.utils.native_grid import INSTRUMENT_CALIBRATION, _calibrated_knots
        w, y = _load("REE_01.csv")
        mask = (w >= 365.0) & (w <= 620.0)
        x_native, y_native, info = recover_native_grid(w[mask], y[mask], fallback=False)
        knots0 = _calibrated_knots(INSTRUMENT_CALIBRATION["VIS"])
        pitch = float(np.median(np.diff(knots0)))
        shift_px, stretch_ppm = 0.37, 80.0
        x_shift = x_native + shift_px * pitch + stretch_ppm * 1e-6 * (x_native - x_native.mean())
        x_export = w[mask]
        inside = (x_export > x_shift[0]) & (x_export < x_shift[-1])
        y_export = export_kernel_matrix(x_shift, x_export[inside], threshold=0.0) @ y_native
        xr, yr, info2 = recover_native_grid(x_export[inside], y_export, fallback=False)
        seg = info2["segments"]["VIS"]
        self.assertTrue(seg["exact"], msg=f"relres={seg['relres']!r}")
        # recovered knots must coincide with the imposed ones (interior)
        common = (xr > x_shift[5]) & (xr < x_shift[-6])
        nearest = np.array([np.min(np.abs(x_shift - v)) for v in xr[common]])
        self.assertLess(np.max(nearest) / pitch, 0.01)


@unittest.skipUnless(os.path.isdir(_DATA_DIR) and os.environ.get("ALIBZ_SLOW_TESTS") == "1",
                     "slow search-mode test; set ALIBZ_SLOW_TESTS=1")
class TestSearchModeRecovery(unittest.TestCase):
    """The full per-window pitch/phase search (mode='search', minutes per
    segment) -- the method that produced INSTRUMENT_CALIBRATION.  Opt-in."""

    def test_reference_sample_nir_exact(self):
        w, y = _load("REE_01.csv")
        mask = w >= 620.0
        x_native, y_native, info = recover_native_grid(w[mask], y[mask], mode="search")
        seg = info["segments"]["NIR"]
        self.assertEqual(seg["mode"], "search")
        self.assertLess(seg["relres"], EXACT_RELRES_THRESHOLD, msg=f"relres={seg['relres']!r}")


if __name__ == "__main__":
    unittest.main()
