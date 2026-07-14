"""Minor-line seeding: prior-driven recovery, deblending, and its gates."""
import unittest

import numpy as np
from scipy.special import voigt_profile

from alibz import PeakyFinder
from alibz.minor_lines import (
    _element_lines,
    _strength,
    match_and_scale,
    seed_minor_lines,
)
from alibz.utils.database import Database

KT = 0.76
NOISE = 20.0
SIG, GAM = 0.05, 0.02


def ca_truth(db, lo=605.0, hi=660.0, peak_area=2e4):
    """Real Ca I db lines in [lo, hi] scaled so the strongest has
    ``peak_area``; returns (wl, area) arrays."""
    ion, wl, gA, Ek = _element_lines(db, "Ca", lo, hi)
    s = _strength(gA, Ek, KT)
    sel = ion == 1.0
    wl1, s1 = wl[sel], s[sel]
    keep = s1 >= 1e-4 * s1.max()
    wl1, s1 = wl1[keep], s1[keep]
    return wl1, peak_area * s1 / s1.max()


def synth(x, lines, seed=1, extra=()):
    y = np.zeros_like(x)
    for wl_j, area_j in zip(*lines):
        y += area_j * voigt_profile(x - wl_j, SIG, GAM)
    for area_j, wl_j in extra:
        y += area_j * voigt_profile(x - wl_j, SIG, GAM)
    return y + np.random.default_rng(seed).normal(0.0, NOISE, x.size)


def run_pipeline(x, y):
    finder = PeakyFinder.__new__(PeakyFinder)
    return finder.fit_spectrum(x, y, subtract_background=False, plot=False,
                               n_sigma=0)


class TestMinorLines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database("db")
        cls.x = np.arange(605.0, 660.0 + 0.0333, 0.0333)
        cls.lines = ca_truth(cls.db)

    def test_scales_recovered_from_clean_references(self):
        y = synth(self.x, self.lines)
        fit = run_pipeline(self.x, y)
        scales, matched = match_and_scale(
            fit["sorted_parameter_array"], self.db, ["Ca"], kT_ev=KT,
            x_range=(float(self.x[0]), float(self.x[-1])))
        self.assertIn(("Ca", 1), scales)
        info = scales[("Ca", 1)]
        self.assertGreaterEqual(info["n_ref"], 3)
        # the synthetic uses the exact Boltzmann ratios, so the
        # log-ratios must agree tightly
        self.assertLess(info["spread"], 0.1)
        ion, wl, gA, Ek = _element_lines(self.db, "Ca", 605.0, 660.0)
        s = _strength(gA, Ek, KT)
        true_scale = 2e4 / s[ion == 1.0].max()
        self.assertAlmostEqual(info["scale"] / true_scale, 1.0, delta=0.1)

    def test_minor_lines_recovered_across_seeds(self):
        """Lines the blind pass missed are added at the right positions
        with areas within ~30 % (blends at 2-5 pixel separations)."""
        truth = {round(float(w), 3): float(a) for w, a in zip(*self.lines)}
        for seed in range(5):
            with self.subTest(seed=seed):
                y = synth(self.x, self.lines, seed=seed)
                fit = run_pipeline(self.x, y)
                new_fit, records = seed_minor_lines(
                    self.x, y, fit, self.db, ["Ca"], kT_ev=KT)
                added = [r for r in records if r["action"] == "added"]
                self.assertGreaterEqual(len(added), 2)
                pk = new_fit["sorted_parameter_array"]
                for r in added:
                    t = truth.get(round(r["wavelength_db"], 3))
                    self.assertIsNotNone(t, r["wavelength_db"])
                    k = int(np.argmin(np.abs(pk[:, 1] - r["wavelength_db"])))
                    self.assertLess(abs(pk[k, 1] - r["wavelength_db"]), 0.06)
                    self.assertLess(abs(pk[k, 0] / t - 1.0), 0.35,
                                    f"{r['wavelength_db']}: {pk[k, 0]} vs {t}")

    def test_interference_with_established_line_deblended(self):
        """A weak Ca line under a 10x stronger foreign line 120 pm away
        (unresolved by the blind pass) is pulled out, center pinned at
        the db position, and the foreign line's contamination reported.

        At this contrast/separation the flux split between the two is
        partly degenerate: the robust, physical invariants are that the
        PAIR flux is conserved, the strong neighbour is recovered
        accurately, and the minor line lands at the right place within a
        factor of two.  (Tighter blends -- 60x at <100 pm -- are
        correctly refused by the consistency gate, exercised below.)"""
        F, wl_f, ca_truth = 450.0, 615.602 + 0.12, 44.7
        y = synth(self.x, self.lines, seed=3, extra=[(F, wl_f)])
        fit = run_pipeline(self.x, y)
        pk0 = fit["sorted_parameter_array"]
        # the blind pass fitted the pair as ONE component
        blend0 = pk0[np.abs(pk0[:, 1] - 615.66) < 0.1]
        self.assertEqual(blend0.shape[0], 1, "blend should start unresolved")
        area0 = float(blend0[0, 0])

        new_fit, records = seed_minor_lines(
            self.x, y, fit, self.db, ["Ca"], kT_ev=KT)
        rec = [r for r in records
               if abs(r["wavelength_db"] - 615.602) < 0.01]
        self.assertTrue(rec, "no record for the hidden Ca line")
        rec = rec[0]
        self.assertEqual(rec["action"], "added", rec)
        self.assertTrue(rec["interference"], "no interference reported")

        pk = new_fit["sorted_parameter_array"]
        k_f = int(np.argmin(np.abs(pk[:, 1] - wl_f)))
        k_c = int(np.argmin(np.abs(pk[:, 1] - 615.602)))
        # the foreign line's area is corrected (the blend record fires)
        self.assertNotAlmostEqual(pk[k_f, 0], area0, delta=1.0)
        # strong neighbour recovered accurately
        self.assertAlmostEqual(pk[k_f, 0] / F, 1.0, delta=0.15)
        # minor line placed correctly, area within a factor of two
        self.assertLess(abs(pk[k_c, 1] - 615.602), 0.03)
        self.assertLess(0.5, pk[k_c, 0] / ca_truth)
        self.assertLess(pk[k_c, 0] / ca_truth, 2.0)
        # pair flux conserved
        self.assertAlmostEqual((pk[k_f, 0] + pk[k_c, 0]) / (F + ca_truth),
                               1.0, delta=0.15)

    def test_extreme_blend_refused_by_consistency_gate(self):
        """A 60x contrast at 80 pm is genuinely degenerate: the fit
        cannot be trusted, and the consistency gate must NOT add it as a
        confident detection."""
        y = synth(self.x, self.lines, seed=3, extra=[(3000.0, 615.602 + 0.08)])
        fit = run_pipeline(self.x, y)
        _, records = seed_minor_lines(self.x, y, fit, self.db, ["Ca"],
                                      kT_ev=KT)
        rec = [r for r in records
               if abs(r["wavelength_db"] - 615.602) < 0.01]
        self.assertTrue(rec)
        self.assertNotEqual(rec[0]["action"], "added")

    def test_unestablished_element_adds_nothing(self):
        """An element with no genuine lines in the spectrum must not
        pass the reference gates and must add nothing (measured on
        MW2-112: junk N I references injected a fake component inside
        Li I 670.8 before these gates existed)."""
        y = synth(self.x, self.lines, seed=2)
        fit = run_pipeline(self.x, y)
        n0 = fit["sorted_parameter_array"].shape[0]
        for junk in (["N"], ["Fe"], ["N", "Fe"]):
            with self.subTest(elements=junk):
                new_fit, records = seed_minor_lines(
                    self.x, y, fit, self.db, junk, kT_ev=KT)
                added = [r for r in records if r["action"] == "added"]
                self.assertEqual(added, [])
                self.assertEqual(new_fit["sorted_parameter_array"].shape[0],
                                 n0)

    def test_self_absorbed_strongest_line_does_not_falsify(self):
        """Regression: a genuinely-present element whose STRONGEST
        (resonance) line is self-absorbed must NOT have its whole stage
        rejected — the falsification gate keys on whether the strong line
        is *observed*, and a suppressed-but-present line is evidence FOR
        the element.  (Anchoring on the clean references' max area
        deleted Ca I / Sr I outright.)"""
        ion, wl, gA, Ek = _element_lines(self.db, "Ca", 400.0, 660.0)
        s = _strength(gA, Ek, KT)
        sel = ion == 1.0
        wl1, s1 = wl[sel], s[sel]
        keep = s1 >= 1e-4 * s1.max()
        wl1, s1 = wl1[keep], s1[keep]
        areas = 2e4 * s1 / s1.max()
        jmax = int(np.argmax(s1))  # 422.67 resonance line
        x = np.arange(400.0, 660.0 + 0.0333, 0.0333)
        for frac in (1.0, 0.35):
            with self.subTest(strong_line_fraction=frac):
                y = np.zeros_like(x)
                for i, (w, a) in enumerate(zip(wl1, areas)):
                    y += a * (frac if i == jmax else 1.0) * \
                        voigt_profile(x - w, SIG, GAM)
                y += np.random.default_rng(1).normal(0.0, NOISE, x.size)
                fit = run_pipeline(x, y)
                _, records = seed_minor_lines(x, y, fit, self.db, ["Ca"],
                                              kT_ev=KT)
                added = [r for r in records if r["action"] == "added"]
                self.assertGreaterEqual(len(added), 3,
                    f"stage wrongly falsified at frac={frac}")

    def test_no_lines_and_unsupported_elements_are_safe(self):
        """An established-element list containing a no-lines element, a
        explicitly unsupported element, or an unknown symbol must not
        crash and must add nothing for them."""
        x = np.arange(605.0, 660.0 + 0.0333, 0.0333)
        y = synth(x, self.lines, seed=1)
        fit = run_pipeline(x, y)
        n0 = fit["sorted_parameter_array"].shape[0]
        for junk in (["Pm"], ["Zz"], ["Ca", "Pm", "Zz"]):
            with self.subTest(elements=junk):
                new_fit, records = seed_minor_lines(
                    x, y, fit, self.db, junk, kT_ev=KT)
                added = [r for r in records
                         if r["action"] == "added" and r["element"] in
                         ("Pm", "Zz")]
                self.assertEqual(added, [])
                if junk == ["Pm"] or junk == ["Zz"]:
                    self.assertEqual(
                        new_fit["sorted_parameter_array"].shape[0], n0)

    def test_shift_nm_frame_is_applied(self):
        """With the whole spectrum displaced by +0.1 nm and shift_nm=0.1,
        the same minor lines are recovered at the shifted positions; a
        sign error would place components 0.2 nm off and find nothing."""
        shift = 0.1
        x = np.arange(605.0, 660.0 + 0.0333, 0.0333)
        wl1, a1 = self.lines
        y = synth(x, (wl1 + shift, a1), seed=1)
        fit = run_pipeline(x, y)
        new_fit, records = seed_minor_lines(x, y, fit, self.db, ["Ca"],
                                            kT_ev=KT, shift_nm=shift)
        added = [r for r in records if r["action"] == "added"]
        self.assertGreaterEqual(len(added), 2)
        pk = new_fit["sorted_parameter_array"]
        for r in added:
            k = int(np.argmin(np.abs(pk[:, 1] - (r["wavelength_db"] + shift))))
            self.assertLess(abs(pk[k, 1] - (r["wavelength_db"] + shift)), 0.05)

    def test_missing_action_flags_absent_confident_line(self):
        """A confidently-predicted line (expected SNR >= 10) whose flux
        is absent from the spectrum is recorded 'missing', not added."""
        wl1, a1 = self.lines
        # drop the 2nd-strongest Ca line entirely; its prediction from
        # the rest of the (thin) spectrum is confident but the data are
        # empty there
        order = np.argsort(a1)[::-1]
        drop = int(order[1])
        keepmask = np.ones(wl1.size, bool)
        keepmask[drop] = False
        x = np.arange(605.0, 660.0 + 0.0333, 0.0333)
        y = synth(x, (wl1[keepmask], a1[keepmask]), seed=1)
        fit = run_pipeline(x, y)
        _, records = seed_minor_lines(x, y, fit, self.db, ["Ca"], kT_ev=KT)
        rec = [r for r in records
               if abs(r["wavelength_db"] - wl1[drop]) < 0.02]
        self.assertTrue(rec, "dropped line was never a candidate")
        self.assertIn(rec[0]["action"], ("missing", "inconsistent"))
        self.assertNotEqual(rec[0]["action"], "added")

    def test_prediction_consistency_gate(self):
        """With a wildly wrong temperature the scale mis-predicts line
        ratios; additions contradicting their own prediction by more
        than the consistency factor must be rejected, and confident
        absent predictions must be flagged 'missing'."""
        y = synth(self.x, self.lines, seed=1)
        fit = run_pipeline(self.x, y)
        _, records = seed_minor_lines(
            self.x, y, fit, self.db, ["Ca"], kT_ev=3.0)
        for r in records:
            if r["action"] == "added":
                self.assertLessEqual(r["area"] / r["expected_area"], 5.0)
                self.assertGreaterEqual(r["area"] / r["expected_area"],
                                        1.0 / 5.0)


if __name__ == "__main__":
    unittest.main()
