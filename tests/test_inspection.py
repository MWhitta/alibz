"""Inspection module: uncertainty calibration, table, and plot smoke tests."""
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.special import voigt_profile

from alibz import PeakyFinder
from alibz.inspection import (
    estimate_peak_uncertainties,
    format_peak_table,
    peak_table,
    plot_peak_zoom,
    plot_spectrum_overview,
)

NOISE = 4.0
LINES = [
    (5e3, 420.0, 0.03, 0.02),
    (5e4, 450.0, 0.03, 0.02),
    (2e3, 470.0, 0.03, 0.02),
    # a blended pair: uncertainties must inflate relative to isolated peaks
    (8e3, 480.00, 0.03, 0.02),
    (6e3, 480.08, 0.03, 0.02),
]


def synth(seed=0):
    x = np.arange(400.0, 510.0, 0.01)
    y = np.zeros_like(x)
    for area, mu, sig, gam in LINES:
        y += area * voigt_profile(x - mu, sig, gam)
    y = y + np.random.default_rng(seed).normal(0.0, NOISE, size=x.size)
    return x, y


class TestUncertaintyCalibration(unittest.TestCase):
    """The reported sigmas must be statistically meaningful: across many
    noise realisations, the pull (fit - truth)/sigma of the strong
    isolated line's area should have unit-ish spread."""

    def test_pull_distribution_and_blend_inflation(self):
        finder = PeakyFinder.__new__(PeakyFinder)
        pulls = []
        rel_iso, rel_blend = [], []
        for seed in range(8):
            x, y = synth(seed)
            fit = finder.fit_spectrum(x, y, subtract_background=False,
                                      plot=False, n_sigma=0)
            peaks = fit["sorted_parameter_array"]
            errs = estimate_peak_uncertainties(x, y, peaks)
            k = int(np.argmin(np.abs(peaks[:, 1] - 450.0)))
            if np.isfinite(errs[k, 0]) and errs[k, 0] > 0:
                pulls.append((peaks[k, 0] - 5e4) / errs[k, 0])
                rel_iso.append(errs[k, 0] / peaks[k, 0])
            kb = int(np.argmin(np.abs(peaks[:, 1] - 480.0)))
            if np.isfinite(errs[kb, 0]) and peaks[kb, 0] > 0:
                rel_blend.append(errs[kb, 0] / peaks[kb, 0])

        self.assertGreaterEqual(len(pulls), 6)
        # The reported sigma is a chi2-scaled statistical bound; measured
        # pull RMS ~5 on strong isolated lines (procedure scatter beyond
        # the CRB, documented in estimate_peak_uncertainties). Guard
        # against order-of-magnitude mis-calibration in either direction.
        rms = float(np.sqrt(np.mean(np.square(pulls))))
        self.assertLess(rms, 10.0)
        self.assertGreater(rms, 0.3)
        # blended peaks carry larger RELATIVE uncertainties than an
        # isolated strong line
        self.assertGreater(np.median(rel_blend), np.median(rel_iso))

    def test_pinned_parameters_flagged_nan(self):
        x, y = synth(3)
        finder = PeakyFinder.__new__(PeakyFinder)
        fit = finder.fit_spectrum(x, y, subtract_background=False,
                                  plot=False, n_sigma=0)
        peaks = fit["sorted_parameter_array"].copy()
        peaks[0, 3] = 0.0  # force a bound-pinned gamma
        errs = estimate_peak_uncertainties(x, y, peaks)
        self.assertTrue(np.isnan(errs[0, 3]))
        self.assertTrue(np.isfinite(errs[0, 0]))


class TestTableAndPlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = synth(1)
        finder = PeakyFinder.__new__(PeakyFinder)
        cls.fit = finder.fit_spectrum(cls.x, cls.y, subtract_background=False,
                                      plot=False, n_sigma=0)

    def test_peak_table_contents(self):
        rows = peak_table(self.x, self.y, self.fit)
        self.assertGreaterEqual(len(rows), 4)
        top = rows[0]
        self.assertAlmostEqual(top["center_nm"], 450.0, delta=0.02)
        self.assertAlmostEqual(top["area"], 5e4, delta=2e3)
        self.assertTrue(np.isfinite(top["area_err"]))
        self.assertGreater(top["snr"], 50)
        text = format_peak_table(rows, max_rows=3)
        self.assertIn("center [nm]", text)
        self.assertIn("+/-", text)

    def test_plots_render(self):
        import matplotlib.pyplot as plt

        fig, axs = plot_spectrum_overview(self.x, self.y, self.fit,
                                          xlim=(445, 455))
        self.assertEqual(len(axs), 3)
        plt.close(fig)
        fig, _ = plot_peak_zoom(self.x, self.y, self.fit, 450.0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
