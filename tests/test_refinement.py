"""Refinement: blends vs asymmetry, decided by statistics + physics.

Fixtures run the REAL pipeline (PeakyFinder.fit_spectrum -> refine_fit)
across multiple noise seeds; single-seed passes are not accepted for the
headline behaviors (review 2026-07: the original SA fixture passed only
on its hard-coded seed).
"""
import unittest

import numpy as np
from scipy.special import voigt_profile

from alibz import PeakyFinder
from alibz.refinement import (
    _feature_candidates,
    classify_feature,
    refine_fit,
    sa_voigt,
)
from alibz.utils.database import Database
from alibz.utils.voigt import multi_voigt

NOISE = 30.0
SEEDS = range(7)


def synth(components, w_lo=640.0, w_hi=700.0, inc=0.0333, seed=2, sa=None,
          noise=NOISE):
    """components: list of (area, mu, sig, gam); sa: dict mu -> (tau, delta)."""
    x = np.arange(w_lo, w_hi + inc, inc)
    y = np.zeros_like(x)
    for area, mu, sig, gam in components:
        if sa and mu in sa:
            tau, delta = sa[mu]
            y += sa_voigt(x, area, mu, sig, gam, tau, delta)
        else:
            y += area * voigt_profile(x - mu, sig, gam)
    y = y + np.random.default_rng(seed).normal(0.0, noise, size=x.size)
    return x, y


def run_pipeline(x, y):
    finder = PeakyFinder.__new__(PeakyFinder)
    return finder.fit_spectrum(x, y, subtract_background=False, plot=False,
                               n_sigma=0)


def decision_at(decisions, center, tol=0.5):
    near = [d for d in decisions if abs(d["center"] - center) < tol]
    if not near:
        return None
    return min(near, key=lambda d: abs(d["center"] - center))


class TestSelfAbsorption(unittest.TestCase):
    """Li I 670.78-like resonance line with a shifted cold absorber."""

    def test_sa_resonance_line_merged_all_seeds(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                x, y = synth([(3e4, 670.776, 0.05, 0.03)], seed=seed,
                             sa={670.776: (1.5, 0.02)})
                fit = run_pipeline(x, y)
                new_fit, decisions = refine_fit(x, y, fit, db=self.db)
                dec = decision_at(decisions, 670.78, tol=0.2)
                self.assertIsNotNone(dec, "no decision at the SA line")
                self.assertEqual(dec["verdict"], "asymmetric",
                                 f"bic {dec['bic']}, s2 {dec['noise_rescale']}")
                self.assertEqual(dec["action"], "merge")
                self.assertTrue(dec["resonance_primary"])
                self.assertAlmostEqual(dec["tau_a"], 1.5, delta=0.4)
                # emission (unattenuated) recovered; observed is smaller
                self.assertAlmostEqual(dec["emission_area"] / 3e4, 1.0,
                                       delta=0.1)
                self.assertLess(dec["observed_area"], dec["emission_area"])
                # exactly one row survives, carrying the OBSERVED area
                pk = new_fit["sorted_parameter_array"]
                near = pk[np.abs(pk[:, 1] - 670.78) < 0.15]
                self.assertEqual(near.shape[0], 1)
                self.assertAlmostEqual(float(near[0, 0]),
                                       dec["observed_area"], places=6)

    def test_flat_top_phantom_split_merged_all_seeds(self):
        """tau=2.5 flat top: the first pass splits it into two phantom
        components; refinement must merge and recover the emission."""
        x_ref = np.arange(640.0, 700.0333, 0.0333)
        truth = sa_voigt(x_ref, 5e4, 670.776, 0.05, 0.03, 2.5, 0.0)
        observed_truth = float(np.trapezoid(truth, x_ref))
        for seed in SEEDS:
            with self.subTest(seed=seed):
                x, y = synth([(5e4, 670.776, 0.05, 0.03)], seed=seed,
                             sa={670.776: (2.5, 0.0)})
                fit = run_pipeline(x, y)
                new_fit, decisions = refine_fit(x, y, fit, db=self.db)
                dec = decision_at(decisions, 670.78, tol=0.2)
                self.assertIsNotNone(dec)
                self.assertEqual(dec["verdict"], "asymmetric",
                                 f"bic {dec['bic']}, s2 {dec['noise_rescale']}")
                self.assertEqual(dec["action"], "merge")
                self.assertAlmostEqual(dec["emission_area"] / 5e4, 1.0,
                                       delta=0.1)
                self.assertAlmostEqual(
                    dec["observed_area"] / observed_truth, 1.0, delta=0.1)
                pk = new_fit["sorted_parameter_array"]
                self.assertEqual(
                    int(np.sum(np.abs(pk[:, 1] - 670.78) < 0.3)), 1)

    def test_nonresonant_asymmetry_recorded_not_applied(self):
        """The same SA profile at a db line with an EXCITED lower level
        (649.876, Ei=1.19) must be recorded as asymmetric-nonresonant and
        left alone: deep optical depth needs a resonance-capable level."""
        x, y = synth([(5e4, 649.8759, 0.05, 0.03)],
                     sa={649.8759: (2.5, 0.0)})
        fit = run_pipeline(x, y)
        pk0 = fit["sorted_parameter_array"]
        n0 = int(np.sum(np.abs(pk0[:, 1] - 649.876) < 0.3))
        new_fit, decisions = refine_fit(x, y, fit, db=self.db)
        dec = decision_at(decisions, 649.876, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertEqual(dec["verdict"], "asymmetric-nonresonant")
        self.assertEqual(dec["action"], "none")
        self.assertFalse(dec["resonance_primary"])
        pk = new_fit["sorted_parameter_array"]
        self.assertEqual(int(np.sum(np.abs(pk[:, 1] - 649.876) < 0.3)), n0)

    def test_db_free_statistics_still_merge_flat_top(self):
        """db=None: pure statistics may act (documented policy — the
        resonance gate needs a database to veto)."""
        x, y = synth([(5e4, 670.776, 0.05, 0.03)],
                     sa={670.776: (2.5, 0.0)})
        fit = run_pipeline(x, y)
        new_fit, decisions = refine_fit(x, y, fit, db=None)
        dec = decision_at(decisions, 670.78, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertEqual(dec["verdict"], "asymmetric")
        pk = new_fit["sorted_parameter_array"]
        self.assertEqual(int(np.sum(np.abs(pk[:, 1] - 670.78) < 0.3)), 1)

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")


class TestStagedRefinement(unittest.TestCase):
    """Stage 3a (data-only, asymmetric deferred) followed by stage 3b
    (physics adjudication with an element posterior)."""

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    def _sa_fit(self):
        x, y = synth([(5e4, 670.776, 0.05, 0.03)],
                     sa={670.776: (2.5, 0.0)})
        return x, y, run_pipeline(x, y)

    def test_defer_records_but_does_not_merge(self):
        x, y, fit = self._sa_fit()
        n0 = int(np.sum(np.abs(
            fit["sorted_parameter_array"][:, 1] - 670.78) < 0.3))
        new_fit, decisions = refine_fit(x, y, fit, db=self.db,
                                        asymmetric="defer")
        dec = decision_at(decisions, 670.78, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertEqual(dec["verdict"], "asymmetric")
        self.assertEqual(dec["action"], "deferred")
        # the model evidence is recorded for the adjudication pass
        self.assertIn("emission_area", dec)
        # ... but the feature is untouched
        pk = new_fit["sorted_parameter_array"]
        self.assertEqual(int(np.sum(np.abs(pk[:, 1] - 670.78) < 0.3)), n0)

    def test_adjudication_merges_with_posterior_element(self):
        x, y, fit = self._sa_fit()
        step_a, _ = refine_fit(x, y, fit, db=self.db, asymmetric="defer")
        new_fit, decisions = refine_fit(x, y, step_a, db=self.db,
                                        elements=["Li", "Na", "K"],
                                        asymmetric="only")
        dec = decision_at(decisions, 670.78, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertEqual(dec["verdict"], "asymmetric")
        self.assertEqual(dec["action"], "merge")
        pk = new_fit["sorted_parameter_array"]
        near = pk[np.abs(pk[:, 1] - 670.78) < 0.3]
        self.assertEqual(near.shape[0], 1)
        self.assertAlmostEqual(float(near[0, 0]), dec["observed_area"],
                               places=6)

    def test_adjudication_vetoes_without_posterior_element(self):
        # posterior lacking any element with a resonance line at 670.78:
        # the merge must be vetoed as nonresonant, feature untouched
        x, y, fit = self._sa_fit()
        step_a, _ = refine_fit(x, y, fit, db=self.db, asymmetric="defer")
        n0 = int(np.sum(np.abs(
            step_a["sorted_parameter_array"][:, 1] - 670.78) < 0.3))
        new_fit, decisions = refine_fit(x, y, step_a, db=self.db,
                                        elements=["Fe", "Mg", "Si"],
                                        asymmetric="only")
        dec = decision_at(decisions, 670.78, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertIn(dec["verdict"],
                      ("asymmetric-nonresonant", "ambiguous"))
        self.assertEqual(dec["action"], "none")
        pk = new_fit["sorted_parameter_array"]
        self.assertEqual(int(np.sum(np.abs(pk[:, 1] - 670.78) < 0.3)), n0)

    def test_only_mode_does_not_touch_blends(self):
        # a true db-supported blend: 3a splits it; the 3b pass must not
        # act on (or re-merge) blend/single verdicts
        wl_a, wl_b = 645.4506, 645.5986
        x, y = synth([(2.5e4, wl_a, 0.05, 0.03),
                      (1.8e4, wl_b, 0.05, 0.03)])
        fit = run_pipeline(x, y)
        step_a, dec_a = refine_fit(x, y, fit, db=self.db,
                                   asymmetric="defer")
        pk_a = step_a["sorted_parameter_array"]
        n_a = int(np.sum(np.abs(pk_a[:, 1] - 645.52) < 0.4))
        self.assertGreaterEqual(n_a, 2)
        new_fit, dec_b = refine_fit(x, y, step_a, db=self.db,
                                    elements=["Fe", "Si"],
                                    asymmetric="only")
        pk_b = new_fit["sorted_parameter_array"]
        self.assertEqual(int(np.sum(np.abs(pk_b[:, 1] - 645.52) < 0.4)), n_a)
        for d in dec_b:
            if abs(d["center"] - 645.52) < 0.4:
                self.assertEqual(d["action"], "none")


class TestBlends(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")

    def test_true_blend_with_db_support_is_split(self):
        """Two genuine lines ~1 FWHM apart at REAL db positions
        (645.4506 / 645.5986) must come out as TWO components at the
        true centers — never merged."""
        wl_a, wl_b = 645.4506, 645.5986
        mid = 0.5 * (wl_a + wl_b)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                x, y = synth([(2.5e4, wl_a, 0.05, 0.03),
                              (1.8e4, wl_b, 0.05, 0.03)], seed=seed)
                fit = run_pipeline(x, y)
                new_fit, decisions = refine_fit(x, y, fit, db=self.db)
                pk = new_fit["sorted_parameter_array"]
                near = pk[np.abs(pk[:, 1] - mid) < 0.4]
                dec = decision_at(decisions, mid, tol=0.3)
                if dec is not None:
                    self.assertIn(dec["verdict"],
                                  ("blend", "blend-unassigned", "ambiguous"),
                                  f"bic {dec['bic']}")
                self.assertGreaterEqual(near.shape[0], 2)
                near = near[np.argsort(near[:, 0])[::-1]]
                centers = np.sort(near[:2, 1])
                self.assertAlmostEqual(centers[0], wl_a, delta=0.04)
                self.assertAlmostEqual(centers[1], wl_b, delta=0.04)

    def test_db_supported_blend_never_merged_asymmetric(self):
        """Two REAL Fe I lines (248.8143 / 249.0644, both resonance-capable,
        250 pm ~ 1.2 FWHM apart) — the measured MW2-112 defect where
        near-degenerate S/A/B statistics merged a db-supported blend into
        a fictitious tau~2.7 self-absorbed line at 248.99, whose exclusion
        zone then blocked recovery of the real residuals."""
        wl_a, wl_b = 248.8143, 249.0644
        for seed in SEEDS:
            with self.subTest(seed=seed):
                x, y = synth([(4.0e3, wl_a, 0.07, 0.04),
                              (1.24e4, wl_b, 0.07, 0.04)],
                             w_lo=245.0, w_hi=253.0, seed=seed)
                fit = run_pipeline(x, y)
                new_fit, decisions = refine_fit(x, y, fit, db=self.db)
                dec = decision_at(decisions, 0.5 * (wl_a + wl_b), tol=0.4)
                if dec is not None:
                    self.assertNotEqual(dec["action"], "merge",
                                        f"verdict {dec['verdict']}")
                pk = new_fit["sorted_parameter_array"]
                near = pk[np.abs(pk[:, 1] - 0.5 * (wl_a + wl_b)) < 0.4]
                self.assertGreaterEqual(near.shape[0], 2)

    def test_shift_frame_conversion(self):
        """The same db pair displaced by +0.15 nm with shift_nm=+0.15
        must be db-SUPPORTED (verdict 'blend', not 'blend-unassigned'):
        a sign error in the frame conversion breaks this."""
        shift = 0.15
        wl_a, wl_b = 645.4506 + shift, 645.5986 + shift
        x, y = synth([(2.5e4, wl_a, 0.05, 0.03), (1.8e4, wl_b, 0.05, 0.03)])
        fit = run_pipeline(x, y)
        new_fit, decisions = refine_fit(x, y, fit, db=self.db,
                                        shift_nm=shift)
        dec = decision_at(decisions, 0.5 * (wl_a + wl_b), tol=0.3)
        pk = new_fit["sorted_parameter_array"]
        near = pk[np.abs(pk[:, 1] - 0.5 * (wl_a + wl_b)) < 0.4]
        self.assertGreaterEqual(near.shape[0], 2)
        if dec is not None and dec["kind"] == "pair":
            self.assertEqual(dec["verdict"], "blend", f"dec {dec}")
        centers = np.sort(near[np.argsort(near[:, 0])[::-1]][:2, 1])
        self.assertAlmostEqual(centers[0], wl_a, delta=0.04)
        self.assertAlmostEqual(centers[1], wl_b, delta=0.04)

    def test_clean_symmetric_line_untouched(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                x, y = synth([(2e4, 670.776, 0.05, 0.03)], seed=seed)
                fit = run_pipeline(x, y)
                pk0 = fit["sorted_parameter_array"]
                n_before = int(np.sum(np.abs(pk0[:, 1] - 670.78) < 0.15))
                new_fit, decisions = refine_fit(x, y, fit, db=self.db)
                pk = new_fit["sorted_parameter_array"]
                n_after = int(np.sum(np.abs(pk[:, 1] - 670.78) < 0.15))
                self.assertEqual(n_after, n_before)
                blends = [d for d in decisions
                          if d["verdict"].startswith("blend")]
                self.assertEqual(blends, [])


class TestGuards(unittest.TestCase):
    """Unit-level coverage of the candidate/classification guards."""

    def test_mop_component_neither_pairs_nor_seeds(self):
        """A component 10x broader than the (area-weighted) median width
        is a background mop: excluded from pairing and residual seeds."""
        x = np.arange(495.0, 505.0, 0.01)
        peaks = np.array([
            [1000.0, 500.00, 0.04, 0.03],
            [80.0, 500.05, 0.40, 0.40],   # mop riding on the line
            [900.0, 501.00, 0.04, 0.03],
        ])
        model = multi_voigt(x, np.ravel(peaks[:, :4]))
        noise = np.full_like(x, 5.0)
        cands = _feature_candidates(x, model, peaks, model, noise)
        for kind, idx in cands:
            self.assertNotIn(1, idx, f"mop in candidate {kind} {idx}")

    def test_interference_gate_defers_dominated_features(self):
        """A weak satellite inside a 10x-stronger neighbour's window
        cannot be classified robustly: classify_feature returns None."""
        x = np.arange(498.0, 502.0, 0.01)
        peaks = np.array([
            [5000.0, 500.00, 0.04, 0.03],
            [100.0, 500.30, 0.04, 0.03],
        ])
        y = multi_voigt(x, np.ravel(peaks[:, :4]))
        noise = np.full_like(x, 5.0)
        others = multi_voigt(x, np.ravel(peaks[0, :4]))
        dec = classify_feature(x, y, peaks, (1,), noise,
                               model_others=others)
        self.assertIsNone(dec)

    def test_wing_soaker_absorbed_and_dropped_on_merge(self):
        """Broad low-amplitude wing-soakers near a merged feature are
        absorbed into it (decision records them; rows dropped)."""
        x, y = synth([(3e4, 670.776, 0.05, 0.03)], seed=0,
                     sa={670.776: (1.5, 0.02)})
        fit = run_pipeline(x, y)
        db = Database("db")
        new_fit, decisions = refine_fit(x, y, fit, db=db)
        dec = decision_at(decisions, 670.78, tol=0.2)
        self.assertIsNotNone(dec)
        self.assertEqual(dec["action"], "merge")
        self.assertGreaterEqual(len(dec["absorbed"]), 1)
        pk0 = fit["sorted_parameter_array"]
        pk = new_fit["sorted_parameter_array"]
        # every absorbed row is gone from the refined table
        for k in dec["absorbed"]:
            mu_k = pk0[k, 1]
            self.assertFalse(np.any(np.isclose(pk[:, 1], mu_k, atol=1e-9)))


if __name__ == "__main__":
    unittest.main()
