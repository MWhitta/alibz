"""The wavelength -> visible-colour mapping used across all spectral plots."""
import unittest

import numpy as np

from alibz.utils.colors import (
    spectral_background,
    spectral_colormap,
    spectral_line,
    wavelength_to_rgb,
)


class TestWavelengthToRGB(unittest.TestCase):

    def test_known_hues(self):
        # dominant channel per band
        self.assertEqual(int(np.argmax(wavelength_to_rgb(450))), 2)   # blue
        self.assertEqual(int(np.argmax(wavelength_to_rgb(530))), 1)   # green
        self.assertEqual(int(np.argmax(wavelength_to_rgb(670))), 0)   # red
        # yellow ~ 580: red and green both high, blue ~ 0
        r, g, b = wavelength_to_rgb(585)
        self.assertGreater(r, 0.5)
        self.assertGreater(g, 0.4)
        self.assertLess(b, 0.05)

    def test_red_channel_rises_through_green_to_red(self):
        rs = [wavelength_to_rgb(w)[0] for w in range(500, 650, 15)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(rs, rs[1:])))

    def test_uv_and_nir_are_floored_but_visible(self):
        for wl in (200, 250, 300, 850, 950):
            rgb = wavelength_to_rgb(wl)
            self.assertGreater(float(np.max(rgb)), 0.0)      # not black
            self.assertLessEqual(float(np.max(rgb)), 0.6)    # dimmed
        # UV keeps a violet cast (blue >= red), NIR a red cast (red >= blue)
        self.assertGreaterEqual(wavelength_to_rgb(230)[2],
                                wavelength_to_rgb(230)[0] - 1e-9)
        self.assertGreaterEqual(wavelength_to_rgb(880)[0],
                                wavelength_to_rgb(880)[2])

    def test_vectorised_shape_and_range(self):
        out = wavelength_to_rgb(np.linspace(180, 950, 64))
        self.assertEqual(out.shape, (64, 3))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))
        self.assertEqual(np.asarray(wavelength_to_rgb(500.0)).shape, (3,))

    def test_colormap_builds(self):
        cm = spectral_colormap(180, 950, n=256)
        self.assertEqual(cm.N, 256)

    def test_background_preserves_data_limits(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([200.0, 900.0], [-50.0, 8_000.0])
        before = ax.get_xlim(), ax.get_ylim()
        spectral_background(ax)
        after = ax.get_xlim(), ax.get_ylim()
        np.testing.assert_allclose(after[0], before[0])
        np.testing.assert_allclose(after[1], before[1])
        plt.close(fig)


class TestSpectralLine(unittest.TestCase):

    def test_matches_plain_plot_limits_on_log_axis(self):
        # colouring the line must leave the intensity axis identical to a
        # plain ax.plot (the LineCollection must not drag the limits)
        import matplotlib.pyplot as plt

        x = np.linspace(200.0, 900.0, 400)
        y = np.abs(np.sin(x / 20.0)) * 1000.0 + 0.01
        figp, axp = plt.subplots()
        axp.plot(x, y)
        axp.set_yscale("log")
        ref = axp.get_ylim()
        plt.close(figp)

        figs, axs = plt.subplots()
        axs.set_yscale("log")
        lc = spectral_line(axs, x, y)
        got = axs.get_ylim()
        plt.close(figs)

        self.assertIsNotNone(lc)
        self.assertEqual(len(lc.get_segments()), x.size - 1)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_segment_colours_track_wavelength(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x = np.array([450.0, 460.0, 660.0, 670.0])   # blue pair, red pair
        lc = spectral_line(ax, x, np.ones_like(x))
        cols = lc.get_colors()
        # first segment (~455 nm) blue-dominant; last (~665 nm) red-dominant
        self.assertEqual(int(np.argmax(cols[0][:3])), 2)
        self.assertEqual(int(np.argmax(cols[-1][:3])), 0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
