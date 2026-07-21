"""End-to-end synthetic round-trip: the first test that runs the WHOLE
``analyze_spectrum`` chain on a spectrum with known ground truth.

Tolerances follow the measured-then-pinned protocol: they encode what the
pipeline ACTUALLY did on 2026-07-16 (plus margin), not what we wish it did
— the deliverable is a regression floor that the fix program's stages
ratchet down, stage by stage.  Measured at pinning time (gp search,
n_calls=8, draws=2, seed 11): Ca 0.764 / Mg 0.236 against 0.60 / 0.40
truth, T 7000 K against 10000 K, r-squared 0.509, ~2 min wall.

The fast variant runs in the default suite (~2 min) because it is the only
end-to-end gate; set ``ALIBZ_SKIP_ROUNDTRIP=1`` to skip it during quick
local iterations.  The nightly variant (``ALIBZ_NIGHTLY=1``) adds the
harder multi-element cases and a determinism check.
"""

import os
import unittest

import numpy as np

from bench.synth_cases import CASES, render_case, score_recovery
from alibz.pipeline import analyze_spectrum

FAST_KW = dict(n_calls=8, draws=2)


def _run_case(name, seed=11, **kw):
    spec = dict(CASES[name])
    comp = spec.pop("composition")
    x, y, scene = render_case(comp, seed=seed, **spec)
    analysis = analyze_spectrum(x, y, "db", **kw)
    res = analysis["result"]
    return comp, scene, res, score_recovery(comp, res.element_fractions)


@unittest.skipIf(os.environ.get("ALIBZ_SKIP_ROUNDTRIP"),
                 "ALIBZ_SKIP_ROUNDTRIP set")
class TestRoundtripFast(unittest.TestCase):
    """The CI gate: binary scene, loose pinned tolerances."""

    @classmethod
    def setUpClass(cls):
        cls.comp, cls.scene, cls.res, cls.score = _run_case(
            "binary_ca_mg", **FAST_KW)

    def test_completes_and_detects_both_majors(self):
        self.assertEqual(self.score["majors_missed"], [],
                         msg=f"score={self.score}")

    def test_dominant_element_correct(self):
        self.assertTrue(self.score["dominant_correct"],
                        msg=f"fractions={dict(self.res.element_fractions)}")

    def test_fraction_error_within_pinned_floor(self):
        # measured 2026-07-16: max abs error 0.164 (Ca 0.764 vs 0.60);
        # pinned floor 0.30 absolute — tighten as the fix stages land
        for el, truth in self.comp.items():
            rec = float(self.res.element_fractions.get(el, 0.0))
            self.assertLess(abs(rec - truth), 0.30,
                            msg=f"{el}: recovered {rec:.3f} truth {truth}")

    def test_spurious_mass_bounded(self):
        # measured 2026-07-16: ~0.0 on the binary case; floor 0.25
        self.assertLess(self.score["spurious_mass"], 0.25,
                        msg=f"spurious={self.score['spurious']}")

    def test_temperature_within_pinned_band(self):
        # measured 2026-07-16: 7000 K at 10000 K truth; pinned band
        # truth * [0.55, 1.45]
        truth_t = self.scene.metadata["truth_temperature_k"]
        self.assertTrue(0.55 * truth_t < self.res.temperature < 1.45 * truth_t,
                        msg=f"T={self.res.temperature:.0f} truth={truth_t}")


@unittest.skipUnless(os.environ.get("ALIBZ_NIGHTLY"), "ALIBZ_NIGHTLY not set")
class TestRoundtripNightly(unittest.TestCase):
    """Harder cases + determinism, run nightly (beryl) at default budget."""

    def test_silicate_4el(self):
        comp, scene, res, score = _run_case("silicate_4el", n_calls=40,
                                            draws=8)
        # baseline 2026-07-16 at n_calls=12: Si MISSED entirely, Be
        # hallucinated at 0.23, ne railed at 14.0.  Initial nightly floor
        # only demands that the dominant recovered element is a truth
        # element and spurious mass stays < 0.5; the WNNLS/grid-BIC
        # stages must then ratchet this to majors_missed == [].
        rec = {el: f for el, f in res.element_fractions.items() if f > 0}
        self.assertTrue(rec, msg="empty composition")
        self.assertIn(max(rec, key=rec.get), comp, msg=f"rec={rec}")
        self.assertLess(score["spurious_mass"], 0.5, msg=f"score={score}")

    def test_argon_ambient_binary(self):
        comp, scene, res, score = _run_case("binary_argon_ambient",
                                            n_calls=40, draws=8)
        self.assertEqual(score["majors_missed"], [], msg=f"score={score}")
        self.assertTrue(score["dominant_correct"])

    def test_determinism_same_seed_same_result(self):
        _, _, res_a, _ = _run_case("binary_ca_mg", **FAST_KW)
        _, _, res_b, _ = _run_case("binary_ca_mg", **FAST_KW)
        for el in set(res_a.element_fractions) | set(res_b.element_fractions):
            np.testing.assert_allclose(
                res_a.element_fractions.get(el, 0.0),
                res_b.element_fractions.get(el, 0.0), rtol=0, atol=1e-12,
                err_msg=f"nondeterministic fraction for {el}")


if __name__ == "__main__":
    unittest.main()
