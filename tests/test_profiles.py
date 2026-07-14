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


class TestDeblendShoulders(unittest.TestCase):
    def _shoulder_setup(self, seed=3):
        # the blind fit converged ONE component onto an unresolved pair --
        # reproduce that with an actual least-squares refit, exactly as
        # production produces it (hand-made "merged" parameters are not a
        # plausible converged state)
        from scipy.optimize import least_squares
        main_true = (900.0, 400.0, 0.030, 0.010)
        hidden = (140.0, 400.09, 0.030, 0.010)
        x, y = _spectrum(_FLOOR_PEAKS + [main_true, hidden], seed=seed)
        m = np.abs(x - 400.0) <= 0.5
        def resid(p):
            return y[m] - p[0] * voigt_profile(x[m] - p[1], p[2], p[3])
        merged = tuple(least_squares(
            resid, x0=np.array(main_true),
            bounds=([0, 399.5, 1e-4, 1e-4], [np.inf, 400.5, 0.5, 0.5])).x)
        fit = _fit_dict(_FLOOR_PEAKS + [merged], x)
        from alibz.profiles import analyze_peak_profiles
        recs = analyze_peak_profiles(x, y, fit, segment_edges=(340.0,))
        return x, y, fit, recs, hidden

    def test_deblend_splits_contaminated_peak(self):
        from alibz.profiles import deblend_shoulders
        x, y, fit, recs, hidden = self._shoulder_setup()
        self.assertEqual(recs[-1]["classification"], "shoulder")
        new_fit, out = deblend_shoulders(x, y, fit, recs,
                                         segment_edges=(340.0,))
        acc = [r for r in out if r["action"] == "deblended"]
        self.assertEqual(len(acc), 1)
        peaks = new_fit["sorted_parameter_array"]
        self.assertEqual(peaks.shape[0], len(_FLOOR_PEAKS) + 2)
        # the new component sits at the hidden line and carries roughly
        # its area; the main peak sheds the contamination
        r = acc[0]
        self.assertAlmostEqual(r["new_center_nm"], hidden[1], delta=0.03)
        self.assertAlmostEqual(r["area_new"], hidden[0],
                               delta=0.5 * hidden[0])

    def test_deblend_respects_exclusion_zones(self):
        from alibz.profiles import deblend_shoulders
        x, y, fit, recs, _ = self._shoulder_setup()
        _, out = deblend_shoulders(x, y, fit, recs,
                                   exclude=((400.0, 0.5),),
                                   segment_edges=(340.0,))
        self.assertTrue(all(r["action"] == "excluded" for r in out))


class _SAStubSpecies:
    def __init__(self, element, ion):
        self.element, self.ion = element, ion


class _SAStubIndexer:
    """Mimics the PeakyIndexerV3 surface recover_sa_areas touches."""

    def __init__(self, obs_wl, obs_amp, A, species, anchored=()):
        self._obs_wl = np.asarray(obs_wl, dtype=float)
        self._obs_amp = np.asarray(obs_amp, dtype=float)
        self._last_A = np.asarray(A, dtype=float)
        self._species = species
        self._sa_doublet_info = {k: {} for k in anchored}

    def _solve_concentrations(self, T, ne):
        # trivial diagonal design: c_j = amp_j
        return self._obs_amp.copy(), 0.0

    def _aggregate_elements(self, c, A, amp_sigma=None):
        tot = max(float(np.sum(c)), 1e-300)
        fr = {}
        for k, sp in enumerate(self._species):
            fr[sp.element] = fr.get(sp.element, 0.0) + float(c[k]) / tot
        return dict(fr), fr, {el: 0.0 for el in fr}


def _sa_result(indexer, fractions):
    import dataclasses
    from alibz.peaky_indexer_v3 import FitResult
    n = indexer._obs_amp.size
    return FitResult(
        temperature=8000.0, ne=17.0, sigma=0.03, gamma=0.01,
        species=indexer._species,
        concentrations=indexer._obs_amp.copy(),
        predicted=indexer._obs_amp.copy(),
        observed=indexer._obs_amp.copy(),
        residuals=np.zeros(n), cost=0.0, r_squared=1.0,
        peak_assignments=[], unexplained_peaks=[], convergence_info={},
        element_concentrations=dict(fractions),
        element_fractions=dict(fractions),
        stage_disagreement={el: 0.0 for el in fractions})


class TestRecoverSAAreas(unittest.TestCase):
    def _setup(self, clip=0.6, anchored=()):
        # spectrum: floor peaks + one flat-top (saturated) line, LSQ-refit
        from scipy.optimize import least_squares
        line = (1200.0, 400.0, 0.030, 0.010)
        x, y = _spectrum(_FLOOR_PEAKS, seed=11)
        prof = line[0] * voigt_profile(x - line[1], line[2], line[3])
        y += np.minimum(prof, clip * prof.max())
        m = np.abs(x - line[1]) <= 0.5
        def resid(p):
            return y[m] - p[0] * voigt_profile(x[m] - p[1], p[2], p[3])
        fitp = least_squares(resid, x0=np.array(line),
                             bounds=([0, 399.5, 1e-4, 1e-4],
                                     [np.inf, 400.5, 0.5, 0.5])).x
        rows = _FLOOR_PEAKS + [tuple(fitp)]
        fit = _fit_dict(rows, x)
        from alibz.profiles import analyze_peak_profiles
        recs = analyze_peak_profiles(x, y, fit, segment_edges=(340.0,))
        peaks = np.asarray(rows, dtype=float)
        n = peaks.shape[0]
        species = [_SAStubSpecies("Si", 1)] * (n - 1) + [_SAStubSpecies("Li", 1)]
        idx = _SAStubIndexer(peaks[:, 1], peaks[:, 0], np.eye(n), species,
                             anchored=anchored)
        fr0 = idx._aggregate_elements(idx._obs_amp, idx._last_A)[1]
        return x, y, fit, recs, idx, _sa_result(idx, fr0), float(fitp[0])

    def test_recovers_emission_area_and_resolves(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, obs_area = self._setup()
        new_res, out, used = recover_sa_areas(idx, res, x, y, fit, recs)
        acc = [r for r in out if r["action"] == "sa-recovered"]
        self.assertTrue(used)
        self.assertEqual(len(acc), 1)
        r = acc[0]
        # emission area exceeds the saturated observed area, within the cap
        self.assertGreater(r["factor"], 1.0)
        self.assertLessEqual(r["factor"], 5.0)
        self.assertGreater(r["emission_area"], obs_area)
        # the re-solved composition shifted toward the saturated element
        self.assertGreater(new_res.element_fractions["Li"],
                           res.element_fractions["Li"])
        # original indexer amplitudes restored
        self.assertAlmostEqual(float(idx._obs_amp[-1]), obs_area, places=6)

    def test_anchored_species_skipped(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, _ = self._setup(anchored=(("Li", 1),))
        new_res, out, used = recover_sa_areas(idx, res, x, y, fit, recs)
        self.assertFalse(used)
        self.assertTrue(any(r["action"] == "anchored" for r in out))
        self.assertIs(new_res, res)

    def test_exclusion_zone_skipped(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, _ = self._setup()
        new_res, out, used = recover_sa_areas(idx, res, x, y, fit, recs,
                                              exclude=((400.0, 0.5),))
        self.assertFalse(used)
        self.assertTrue(any(r["action"] == "excluded" for r in out))

    def _premeasured(self, fit, factor=1.5, tau=0.6):
        mu = float(fit["sorted_parameter_array"][-1, 1])
        obs = float(fit["sorted_parameter_array"][-1, 0])
        return dict(center_nm=mu, factor=factor, tau_a=tau,
                    observed_area=obs, emission_area=factor * obs)

    def test_premeasured_merge_applied_inside_exclusion(self):
        # the pipeline excludes merge zones from the growth-curve refit;
        # the pre-measured refinement factor must still correct the row
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, obs_area = self._setup()
        pm = self._premeasured(fit)
        new_res, out, used = recover_sa_areas(
            idx, res, x, y, fit, recs,
            exclude=((400.0, 0.5),), premeasured=(pm,))
        self.assertTrue(used)
        acc = [r for r in out if r["action"] == "sa-recovered"]
        self.assertEqual(len(acc), 1)
        self.assertEqual(acc[0]["source"], "refinement-merge")
        self.assertGreater(new_res.element_fractions["Li"],
                           res.element_fractions["Li"])
        # original indexer amplitudes restored after the re-solve
        self.assertAlmostEqual(float(idx._obs_amp[-1]), obs_area, places=6)

    def test_premeasured_anchored_species_skipped(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, _ = self._setup(anchored=(("Li", 1),))
        pm = self._premeasured(fit)
        new_res, out, used = recover_sa_areas(
            idx, res, x, y, fit, recs,
            exclude=((400.0, 0.5),), premeasured=(pm,))
        self.assertFalse(used)
        self.assertIn("anchored",
                      [r["action"] for r in out
                       if r.get("source") == "refinement-merge"])
        self.assertIs(new_res, res)

    def test_premeasured_amplification_cap_rejected(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, _ = self._setup()
        pm = self._premeasured(fit, factor=8.0)
        new_res, out, used = recover_sa_areas(
            idx, res, x, y, fit, recs,
            exclude=((400.0, 0.5),), premeasured=(pm,))
        self.assertFalse(used)
        self.assertIn("rejected",
                      [r["action"] for r in out
                       if r.get("source") == "refinement-merge"])

    def test_premeasured_unmatched_center(self):
        from alibz.profiles import recover_sa_areas
        x, y, fit, recs, idx, res, _ = self._setup()
        pm = self._premeasured(fit)
        pm["center_nm"] = 405.0
        new_res, out, used = recover_sa_areas(
            idx, res, x, y, fit, recs,
            exclude=((400.0, 0.5),), premeasured=(pm,))
        self.assertFalse(used)
        self.assertIn("unmatched",
                      [r["action"] for r in out
                       if r.get("source") == "refinement-merge"])
