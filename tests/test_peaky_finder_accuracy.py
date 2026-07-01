"""Ground-truth accuracy regression tests for PeakyFinder.

Synthetic spectra are built from known Voigt components; fitted peak
tables must recover the true areas, widths, and centers.  These pin the
finder repairs: physical fast-path seeds (area = height / V(0), not
height/2), modelled-wing pedestal subtraction, strongest-first global
refits, width-aware shoulder exclusion, one-sided contamination guards,
and noise-referenced significance thresholds.
"""
import unittest

import numpy as np
from scipy.special import voigt_profile

from alibz import PeakyFinder


def _synth(lines, w_lo=390.0, w_hi=510.0, inc=0.01, noise=0.0, seed=7):
    x = np.arange(w_lo, w_hi + inc, inc)
    y = np.zeros_like(x)
    for area, mu, sig, gam in lines:
        y += area * voigt_profile(x - mu, sig, gam)
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.size)
    return x, y


def _fit(x, y, skip_profile):
    finder = PeakyFinder.__new__(PeakyFinder)
    fit = finder.fit_spectrum(
        x, y, subtract_background=False, plot=False, n_sigma=0,
        skip_profile=skip_profile,
    )
    return fit["sorted_parameter_array"]


def _match(peaks, mu, tol=0.15):
    if peaks.shape[0] == 0:
        return None
    j = int(np.argmin(np.abs(peaks[:, 1] - mu)))
    if abs(peaks[j, 1] - mu) > tol:
        return None
    return peaks[j]


ISOLATED = [
    (area, mu, 0.03, 0.02)
    for area, mu in zip(
        [1e2, 5e2, 1e3, 5e3, 1e4, 1e5, 5e5, 1e6],
        np.linspace(395.0, 505.0, 8),
    )
]

# One dominant line with weak satellites on its Lorentzian wings.
WINGS = [(1e6, 450.0, 0.03, 0.02)] + [
    (area, 450.0 + off, 0.03, 0.02)
    for off, area in [(0.5, 2e3), (1.0, 2e3), (2.0, 2e3), (-0.7, 1e4), (-1.5, 1e4)]
]


class TestFullPathAccuracy(unittest.TestCase):
    """The production (skip_profile=False) path: seeds -> shoulders -> fit_all."""

    def test_isolated_lines_recovered_across_four_decades(self):
        x, y = _synth(ISOLATED)
        peaks = _fit(x, y, skip_profile=False)
        self.assertEqual(peaks.shape[0], len(ISOLATED))
        for area, mu, sig, gam in ISOLATED:
            row = _match(peaks, mu)
            self.assertIsNotNone(row, f"line at {mu} nm not found")
            self.assertLess(abs(row[0] - area) / area, 0.02, f"area at {mu} nm")
            self.assertLess(abs(row[1] - mu), 0.003, f"center at {mu} nm")
            self.assertLess(abs(row[2] - sig), 0.002, f"sigma at {mu} nm")
            self.assertLess(abs(row[3] - gam), 0.002, f"gamma at {mu} nm")

    def test_weak_satellites_on_strong_line_wings(self):
        """Weak lines 0.5-2 nm from a 100-500x stronger line must not
        inherit its Lorentzian pedestal (was +10-50%), and the strong
        line must not be split into near-duplicate components by
        shoulder fitting (was -62% with 3 spurious peaks)."""
        x, y = _synth(WINGS)
        peaks = _fit(x, y, skip_profile=False)
        self.assertEqual(peaks.shape[0], len(WINGS))
        for area, mu, sig, gam in WINGS:
            row = _match(peaks, mu)
            self.assertIsNotNone(row, f"line at {mu} nm not found")
            self.assertLess(abs(row[0] - area) / area, 0.02, f"area at {mu} nm")
            self.assertLess(abs(row[2] - sig), 0.002, f"sigma at {mu} nm")


class TestFastPathAccuracy(unittest.TestCase):
    """The corpus (skip_profile=True) path ships pure seeds: a uniform
    ~+8% area bias from the equal-FWHM-partition V(0) remains (documented),
    against +318% before the repair."""

    def test_isolated_lines_seed_accuracy(self):
        x, y = _synth(ISOLATED)
        peaks = _fit(x, y, skip_profile=True)
        self.assertEqual(peaks.shape[0], len(ISOLATED))
        for area, mu, _sig, _gam in ISOLATED:
            row = _match(peaks, mu)
            self.assertIsNotNone(row, f"line at {mu} nm not found")
            self.assertLess(abs(row[0] - area) / area, 0.15, f"area at {mu} nm")

    def test_wing_satellite_seed_accuracy(self):
        """Pedestal-corrected seeds: satellites at >=1 nm from the strong
        line within 25%.  The 0.5 nm satellite sits inside the strong
        line's near-wing where seed-quality wing shapes over-subtract; it
        must still be detected with the right center and a sane width
        (the contamination guard: was sigma=0.36 nm from a half-max
        crossing running up the neighbour's flank)."""
        x, y = _synth(WINGS)
        peaks = _fit(x, y, skip_profile=True)
        self.assertEqual(peaks.shape[0], len(WINGS))
        for area, mu, _sig, _gam in WINGS:
            row = _match(peaks, mu)
            self.assertIsNotNone(row, f"line at {mu} nm not found")
            self.assertLess(abs(row[1] - mu), 0.01, f"center at {mu} nm")
            self.assertLess(row[2], 0.1, f"sigma sanity at {mu} nm")
            if abs(mu - 450.0) >= 1.0 or area == 1e6:
                self.assertLess(abs(row[0] - area) / area, 0.25, f"area at {mu} nm")
            else:
                # Near-wing satellites (<1 nm from the 500x line) are the
                # regime the pedestal repair mattered most for (+330-385%
                # before); seed-quality wing shapes still limit them, but a
                # regression back to flank-inherited amplitudes must fail.
                self.assertLess(abs(row[0] - area) / area, 1.0, f"area at {mu} nm")


class TestNoiseReferencedSignificance(unittest.TestCase):
    """The noise machinery must work OUTSIDE the zero-noise limit, and the
    noise scale must survive the corpus path's linear-interpolation
    upsampling (lag-1 differences deflate by the upsampling factor)."""

    def test_noise_scale_survives_interpolation_upsampling(self):
        rng = np.random.default_rng(3)
        sigma = 20.0
        x_native = np.arange(200.0, 400.0, 0.0333)
        y_native = rng.normal(0.0, sigma, size=x_native.size)
        x_up = np.arange(200.0, 399.9, 0.01)
        y_up = np.interp(x_up, x_native, y_native)

        finder = PeakyFinder.__new__(PeakyFinder)
        est_native = finder._noise_scale(y_native)
        est_up = finder._noise_scale(y_up)
        self.assertLess(abs(est_native - sigma) / sigma, 0.15)
        self.assertLess(abs(est_up - sigma) / sigma, 0.35)

    def test_noise_scale_nonzero_on_quantised_sparse_counts(self):
        rng = np.random.default_rng(4)
        y = np.zeros(4000)
        spikes = rng.choice(4000, size=600, replace=False)
        y[spikes] = rng.integers(1, 3, size=600).astype(float)
        finder = PeakyFinder.__new__(PeakyFinder)
        self.assertGreater(finder._noise_scale(y), 0.0)

    def test_detection_on_noisy_spectrum_keeps_lines_rejects_noise(self):
        lines = [(a, m, 0.03, 0.02) for a, m in zip([2e3, 8e3, 4e4, 2e5], np.linspace(400.0, 500.0, 4))]
        x, y = _synth(lines, noise=5.0)
        finder = PeakyFinder.__new__(PeakyFinder)
        fit = finder.fit_spectrum(x, y, subtract_background=False, plot=False,
                                  n_sigma=0, skip_profile=True)
        peaks = fit["sorted_parameter_array"]
        for area, mu, _s, _g in lines:
            self.assertIsNotNone(_match(peaks, mu, tol=0.05), f"line at {mu} nm lost in noise")
        # thousands of noise maxima exist; the significance gates must
        # reject nearly all of them
        self.assertLess(peaks.shape[0], 60)

    def test_n_sigma_is_monotone(self):
        lines = [(a, m, 0.03, 0.02) for a, m in zip([1e3, 5e3, 2e4, 1e5], np.linspace(400.0, 500.0, 4))]
        x, y = _synth(lines, noise=5.0)
        finder = PeakyFinder.__new__(PeakyFinder)
        kept = []
        for n_sigma in (0, 1, 5, 20):
            fit = finder.fit_spectrum(x, y, subtract_background=False, plot=False,
                                      n_sigma=n_sigma, skip_profile=True)
            kept.append(fit["sorted_parameter_array"].shape[0])
        for a, b in zip(kept, kept[1:]):
            self.assertGreaterEqual(a, b, f"n_sigma not monotone: {kept}")


if __name__ == "__main__":
    unittest.main()
