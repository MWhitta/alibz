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


class TestResidualBaselineHump(unittest.TestCase):
    """Real spectra leave broad residual humps after background subtraction
    (detector junctions, imperfect baseline).  The global refit must not
    convert them into ultra-broad pseudo-Voigt components: measured on real
    data, unbounded fit_all widths produced a sigma=289 nm component
    carrying 89% of the total fitted area and destroyed ~400 genuine
    components, while the width sanity gate was disarmed because its median
    was computed over the corrupted population."""

    def test_broad_hump_does_not_become_giant_component_and_lines_survive(self):
        # Noiseless fixture on a compact window keeps the runtime sane: the
        # point is the RESIDUAL hump surviving into the fit stage, exactly
        # what happens on real data after imperfect background subtraction.
        lines = [(a, m, 0.03, 0.02) for a, m in zip([5e3, 2e4, 1e5], [430.0, 450.0, 470.0])]
        x, y = synth(lines, w_lo=420.0, w_hi=480.0)
        # residual baseline hump ~200 counts, junction-like
        y = y + 200.0 * np.exp(-0.5 * ((x - 448.0) / 15.0) ** 2)

        finder = PeakyFinder.__new__(PeakyFinder)
        fit = finder.fit_spectrum(x, y, subtract_background=False, plot=False, n_sigma=0)
        peaks = fit["sorted_parameter_array"]

        widths = finder.voigt_width(
            np.maximum(peaks[:, 2], 1e-9), np.maximum(peaks[:, 3], 1e-9)
        )
        span = x[-1] - x[0]
        self.assertLess(np.max(widths), span / 2.0,
                        f"window-spanning pseudo-line kept: {np.max(widths):.1f} nm")
        for area, mu, _s, _g in lines:
            row = _match(peaks, mu, tol=0.05)
            self.assertIsNotNone(row, f"line at {mu} nm lost")
            # generous: the un-modelled hump biases areas, but the lines
            # must survive with sane parameters
            self.assertLess(abs(row[0] - area) / area, 0.5, f"area at {mu} nm")


def synth(lines, w_lo=390.0, w_hi=510.0, inc=0.01, noise=0.0, seed=7):
    return _synth(lines, w_lo=w_lo, w_hi=w_hi, inc=inc, noise=noise, seed=seed)


class TestArplsBackground(unittest.TestCase):
    """find_background is arPLS: emission lines must not attract the
    baseline (the retired anchor method put +74 counts of excess baseline
    under the Li analyte line), and per-segment estimation must keep a
    hardware step from bleeding across a junction."""

    def test_baseline_recovered_under_and_between_lines(self):
        rng = np.random.default_rng(2)
        x = np.arange(400.0, 500.0, 0.02)
        true_bg = 120.0 + 60.0 * np.sin(x / 18.0)
        y = true_bg + rng.normal(0.0, 3.0, x.size)
        for area, mu in [(5e3, 420.0), (5e4, 450.0), (2e3, 470.0)]:
            y += area * voigt_profile(x - mu, 0.03, 0.02)

        finder = PeakyFinder.__new__(PeakyFinder)
        bg = finder.find_background(x, y)

        line_free = np.ones_like(x, dtype=bool)
        for mu in (420.0, 450.0, 470.0):
            line_free &= np.abs(x - mu) > 1.0
        self.assertLess(np.median(np.abs((bg - true_bg)[line_free])), 3.0)
        # under the strongest line the baseline must not climb the wings
        j = int(np.argmin(np.abs(x - 450.0)))
        self.assertLess(abs(bg[j] - true_bg[j]), 15.0)
        # and must not dip below the baseline on the flanks
        flank = (np.abs(x - 450.0) > 0.3) & (np.abs(x - 450.0) < 2.0)
        self.assertGreater(np.min((bg - true_bg)[flank]), -15.0)

    def test_segment_edges_isolate_hardware_steps(self):
        rng = np.random.default_rng(3)
        x = np.arange(300.0, 500.0, 0.05)
        true_bg = np.where(x < 400.0, 30.0, 180.0)  # junction gain/dark step
        y = true_bg + rng.normal(0.0, 2.0, x.size)

        finder = PeakyFinder.__new__(PeakyFinder)
        bg_seg = finder.find_background(x, y, segment_edges=(400.0,))
        near = np.abs(x - 400.0) < 3.0
        self.assertLess(np.median(np.abs((bg_seg - true_bg)[near])), 4.0)

    def test_legacy_kwargs_accepted(self):
        finder = PeakyFinder.__new__(PeakyFinder)
        x = np.arange(400.0, 410.0, 0.02)
        y = np.full_like(x, 50.0)
        bg = finder.find_background(x, y, range=5, n_sigma=1)
        self.assertEqual(bg.shape, y.shape)


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

    def test_local_noise_scale_tracks_segment_steps(self):
        """Detector segments have different noise floors (measured 1.8x
        spread on real SciAps data); the local scale must track a step
        while a single global scalar would be wrong on both sides."""
        rng = np.random.default_rng(9)
        y = np.concatenate([
            rng.normal(0.0, 5.0, size=8000),
            rng.normal(0.0, 25.0, size=8000),
        ])
        finder = PeakyFinder.__new__(PeakyFinder)
        local = finder._noise_scale_local(y)
        self.assertEqual(local.size, y.size)
        self.assertLess(abs(np.median(local[:6000]) - 5.0) / 5.0, 0.3)
        self.assertLess(abs(np.median(local[10000:]) - 25.0) / 25.0, 0.3)

    def test_detection_adapts_to_local_noise(self):
        """A line significant in the quiet segment must be kept even though
        it would fail a global threshold inflated by the noisy segment."""
        rng = np.random.default_rng(10)
        inc = 0.01
        x = np.arange(400.0, 560.0, inc)
        y = np.where(x < 480.0,
                     rng.normal(0.0, 2.0, size=x.size),
                     rng.normal(0.0, 30.0, size=x.size))
        # height 25 = 12.5 sigma locally, but only ~0.8 sigma of the noisy
        # segment's scale (global scale ~21 -> 3*global = 64 would kill it)
        from scipy.special import voigt_profile as _vp
        area = 25.0 / float(_vp(0.0, 0.03, 0.02))
        y += area * _vp(x - 440.0, 0.03, 0.02)

        finder = PeakyFinder.__new__(PeakyFinder)
        fit = finder.fit_spectrum(x, y, subtract_background=False, plot=False,
                                  n_sigma=0, skip_profile=True)
        peaks = fit["sorted_parameter_array"]
        self.assertIsNotNone(_match(peaks, 440.0, tol=0.05), "quiet-segment line lost")

    def test_segment_response_estimation_and_correction(self):
        from alibz.detector import correct_segment_response, estimate_segment_response

        x = np.arange(300.0, 700.0, 0.05)
        continuum = 100.0 * np.exp(-0.5 * ((x - 500.0) / 400.0) ** 2)
        response_true = np.where(x < 620.0, 1.0, 5.0)
        bg = continuum * response_true

        response = estimate_segment_response(x, bg, edges=(620.0,))
        self.assertAlmostEqual(response[0], 1.0)
        self.assertLess(abs(response[1] - 5.0) / 5.0, 0.15)

        peaks = np.array([[100.0, 500.0, 0.03, 0.02], [500.0, 650.0, 0.03, 0.02]])
        corrected = correct_segment_response(peaks, response, edges=(620.0,))
        self.assertAlmostEqual(corrected[0, 0], 100.0)
        self.assertLess(abs(corrected[1, 0] - 100.0) / 100.0, 0.15)
        # original untouched
        self.assertEqual(peaks[1, 0], 500.0)

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
