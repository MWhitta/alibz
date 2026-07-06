"""Tests for alibz.profiles — per-segment, per-peak shape physics."""

import unittest

import numpy as np
from scipy.special import voigt_profile

from alibz.profiles import (
    analyze_peak_profiles,
    element_shape_quality,
    profile_summary,
    segment_width_floor,
)


def _spectrum(peaks, x=None, noise=0.5, seed=0):
    """Synthesise a spectrum from (area, mu, sigma, gamma) rows."""
    if x is None:
        x = np.arange(200.0, 500.0, 0.02)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, noise, x.size)
    for a, mu, s, g in peaks:
        y += a * voigt_profile(x - mu, s, g)
    return x, y


def _fit_dict(rows, x):
    arr = np.asarray(rows, dtype=float)
    return {"sorted_parameter_array": arr,
            "background": np.zeros_like(x)}


# instrument-like narrow peaks spread over the range establish the floor
_FLOOR_PEAKS = [(200.0, 210.0 + 15 * k, 0.030, 0.010) for k in range(12)]


class TestSegmentFloor(unittest.TestCase):
    def test_floor_reflects_narrow_peaks(self):
        rows = np.asarray(_FLOOR_PEAKS, dtype=float)
        floors = segment_width_floor(rows, segment_edges=(340.0,))
        # all peaks share one shape: floor ~ their common Voigt FWHM
        from alibz.utils.voigt import voigt_width
        w = float(voigt_width(0.030, 0.010))
        for s, f in floors.items():
            self.assertAlmostEqual(f, w, delta=0.3 * w)

    def test_per_segment_floors_differ(self):
        # segment 0 narrow, segment 1 twice as wide (different LSF)
        rows = ([(100.0, 210 + 8 * k, 0.030, 0.010) for k in range(8)]
                + [(100.0, 360 + 8 * k, 0.060, 0.020) for k in range(8)])
        floors = segment_width_floor(np.asarray(rows), segment_edges=(340.0,))
        self.assertLess(floors[0], floors[1])


class TestClassification(unittest.TestCase):
    def _run(self, extra_true, fitted_rows):
        """Spectrum = floor peaks + extra_true; fit table = floor + fitted."""
        true = _FLOOR_PEAKS + extra_true
        x, y = _spectrum(true)
        rows = _FLOOR_PEAKS + fitted_rows
        recs = analyze_peak_profiles(x, y, _fit_dict(rows, x),
                                     segment_edges=(340.0,))
        return recs[len(_FLOOR_PEAKS):]

    def test_clean_voigt_is_instrumental(self):
        clean = (500.0, 400.0, 0.030, 0.010)
        (r,) = self._run([clean], [clean])
        self.assertEqual(r["classification"], "instrumental")
        self.assertAlmostEqual(r["width_ratio"], 1.0, delta=0.35)

    def test_broadened_symmetric_peak(self):
        wide = (800.0, 400.0, 0.075, 0.012)   # ~2.3x floor, Gaussian excess
        (r,) = self._run([wide], [wide])
        self.assertEqual(r["classification"], "broadened")
        self.assertGreater(r["width_ratio"], 1.8)
        self.assertGreater(r["gaussian_fraction"], 0.5)  # Doppler-like

    def test_shoulder_detected_from_unresolved_overlap(self):
        main = (900.0, 400.0, 0.030, 0.010)
        hidden = (140.0, 400.09, 0.030, 0.010)   # unresolved companion
        (r,) = self._run([main, hidden], [main])  # fit knows only the main
        self.assertEqual(r["classification"], "shoulder")
        self.assertEqual(r["shoulder_side"], "red")
        self.assertGreater(r["shoulder_sigma"], 3.0)

    def test_sa_flat_top_is_sa_like(self):
        # simulate saturation: clip the true line's core (flat top), then
        # REFIT with least-squares exactly as production does -- feeding the
        # truth parameters instead would hide the fact that a converged fit
        # balances residuals over the core (measured: the wide-band median
        # core defect flips NEGATIVE for moderate saturation while the
        # narrow centre overshoot survives)
        from scipy.optimize import least_squares
        line = (1200.0, 400.0, 0.030, 0.010)
        for clip in (0.6, 0.75):
            x, y = _spectrum(_FLOOR_PEAKS, seed=int(100 * clip))
            prof = line[0] * voigt_profile(x - line[1], line[2], line[3])
            y += np.minimum(prof, clip * prof.max())
            m = np.abs(x - line[1]) <= 0.5
            def resid(p):
                return y[m] - p[0] * voigt_profile(x[m] - p[1], p[2], p[3])
            fit = least_squares(resid, x0=np.array(line),
                                bounds=([0, 399.5, 1e-4, 1e-4],
                                        [np.inf, 400.5, 0.5, 0.5]))
            recs = analyze_peak_profiles(
                x, y, _fit_dict(_FLOOR_PEAKS + [tuple(fit.x)], x),
                segment_edges=(340.0,))
            r = recs[-1]
            self.assertEqual(
                r["classification"], "sa-like",
                msg=f"clip={clip}: got {r['classification']} "
                    f"(core={r['core_defect_sigma']:.1f}, "
                    f"shoulder={r['shoulder_sigma']:.1f})")

    def test_sa_not_misrouted_to_shoulder(self):
        # a symmetric flat-top leaves near-equal residual lobes on BOTH
        # flanks after an LSQ refit; the one-sidedness test must veto
        # 'shoulder' (splitting SA manufactures phantom lines)
        from scipy.optimize import least_squares
        line = (1200.0, 400.0, 0.030, 0.010)
        x, y = _spectrum(_FLOOR_PEAKS, seed=7)
        prof = line[0] * voigt_profile(x - line[1], line[2], line[3])
        y += np.minimum(prof, 0.75 * prof.max())
        m = np.abs(x - line[1]) <= 0.5
        def resid(p):
            return y[m] - p[0] * voigt_profile(x[m] - p[1], p[2], p[3])
        fit = least_squares(resid, x0=np.array(line),
                            bounds=([0, 399.5, 1e-4, 1e-4],
                                    [np.inf, 400.5, 0.5, 0.5]))
        recs = analyze_peak_profiles(
            x, y, _fit_dict(_FLOOR_PEAKS + [tuple(fit.x)], x),
            segment_edges=(340.0,))
        self.assertNotEqual(recs[-1]["classification"], "shoulder")

    def test_narrow_spike_flagged(self):
        spike = (60.0, 400.0, 0.008, 0.001)   # far below instrument floor
        (r,) = self._run([spike], [spike])
        self.assertEqual(r["classification"], "narrow")

    def test_summary_counts(self):
        clean = (500.0, 400.0, 0.030, 0.010)
        recs = self._run([clean], [clean])
        s = profile_summary(recs)
        self.assertEqual(sum(s.values()), len(recs))


class TestElementShapeQuality(unittest.TestCase):
    def test_saturated_support_share(self):
        recs = [
            dict(index=0, area=1000.0, classification="sa-like"),
            dict(index=1, area=100.0, classification="instrumental"),
            dict(index=2, area=50.0, classification="shoulder"),
        ]
        q = element_shape_quality({"Ca": [0, 1], "Si": [2]}, recs)
        self.assertAlmostEqual(q["Ca"]["sa_share"], 1000.0 / 1100.0, places=6)
        self.assertEqual(q["Ca"]["clean_anchors"], 1)
        self.assertAlmostEqual(q["Si"]["shoulder_share"], 1.0, places=6)

    def test_empty_support_skipped(self):
        self.assertEqual(element_shape_quality({"X": []}, []), {})


if __name__ == "__main__":
    unittest.main()
