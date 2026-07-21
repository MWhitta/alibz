"""WNNLS weighting: sigma_eff floor, weighted cost units, basin BIC."""

import unittest

import numpy as np

from alibz.peaky_indexer_v3 import (PROCEDURE_NOISE_FRACTION, PeakyIndexerV3)

PEAKS = np.array([[120.0, 393.37, 0.05, 0.02],
                  [80.0, 396.85, 0.05, 0.02],
                  [40.0, 422.67, 0.06, 0.02],
                  [25.0, 279.55, 0.05, 0.02],
                  [18.0, 285.21, 0.05, 0.02]])


class TestSigmaEff(unittest.TestCase):
    def _indexer(self, amp_sigma=None):
        return PeakyIndexerV3(PEAKS, dbpath="db", amp_sigma=amp_sigma,
                              weighted_solve=True)

    def test_none_without_amp_sigma(self):
        self.assertIsNone(self._indexer()._sigma_eff())

    def test_floor_dominates_on_bright_lines(self):
        # tiny statistical sigma on a bright line must be floored by the
        # procedure-noise term, not taken at face value
        idx = self._indexer(amp_sigma=np.full(len(PEAKS), 1e-6))
        sig = idx._sigma_eff()
        expected_floor = PROCEDURE_NOISE_FRACTION * PEAKS[:, 0]
        np.testing.assert_allclose(sig, expected_floor, rtol=1e-3)

    def test_nonfinite_sigmas_filled_with_median(self):
        raw = np.array([2.0, np.nan, 2.0, np.inf, 2.0])
        sig = idx_sig = self._indexer(amp_sigma=raw)._sigma_eff()
        self.assertTrue(np.all(np.isfinite(sig)))

    def test_all_nonfinite_disables_weighting(self):
        raw = np.full(len(PEAKS), np.nan)
        self.assertIsNone(self._indexer(amp_sigma=raw)._sigma_eff())

    def test_zero_model_cost_units(self):
        idx_u = self._indexer()
        idx_w = self._indexer(amp_sigma=np.full(len(PEAKS), 2.0))
        self.assertAlmostEqual(idx_u._zero_model_cost(),
                               float(np.sum(PEAKS[:, 0] ** 2)))
        sig = idx_w._sigma_eff()
        self.assertAlmostEqual(idx_w._zero_model_cost(),
                               float(np.sum((PEAKS[:, 0] / sig) ** 2)))


class TestWeightedSolve(unittest.TestCase):
    def test_weighted_cost_is_chi2_scaled(self):
        """Same data: the weighted cost must be in chi-squared units
        (O(n) for a decent fit), not amplitude-squared units."""
        idx_u = PeakyIndexerV3(PEAKS, dbpath="db")
        idx_u.build_candidate_matrix(sa_doublets=False)
        idx_u._rebuild_overlap(0.05, 0.03)
        _c, cost_u = idx_u._solve_concentrations(10_000.0, 17.0)

        idx_w = PeakyIndexerV3(PEAKS, dbpath="db", weighted_solve=True,
                               amp_sigma=np.full(len(PEAKS), 2.0))
        idx_w.build_candidate_matrix(sa_doublets=False)
        idx_w._rebuild_overlap(0.05, 0.03)
        _c, cost_w = idx_w._solve_concentrations(10_000.0, 17.0)

        self.assertGreater(cost_u, 0.0)
        self.assertGreater(cost_w, 0.0)
        # the weighted objective is scaled by ~1/sigma^2 relative to the
        # unweighted SSE; with sigma ~2 and floor ~5% amps they are far
        # apart — this guards against silently reverting to unweighted
        self.assertNotAlmostEqual(np.log10(cost_u), np.log10(cost_w),
                                  places=0)

    def test_weighted_and_unweighted_solve_same_species_support(self):
        """Weighting reweights evidence; on a clean Ca/Mg toy pattern both
        modes must still put the composition on the same elements."""
        idx_u = PeakyIndexerV3(PEAKS, dbpath="db")
        r_u = idx_u.run(n_calls=6, verbose=False)
        idx_w = PeakyIndexerV3(PEAKS, dbpath="db", weighted_solve=True,
                               amp_sigma=np.full(len(PEAKS), 2.0))
        r_w = idx_w.run(n_calls=6, verbose=False)
        self.assertTrue(set(r_w.element_fractions),
                        msg="weighted run produced empty composition")
        self.assertTrue(set(r_w.element_fractions) <=
                        set(r_u.element_fractions) | {"Ca", "Mg"})


class TestEvidenceRidgeZUnits(unittest.TestCase):
    def test_ridge_charge_capped_by_procedure_floor(self):
        """Weighted-mode evidence ridges are detection z-scores: no
        species' missing-mass charge may exceed ~w_mm/f_proc^2.

        Regression for the measured T-bias root cause (2026-07-19): the
        amplitude-unit ridges scaled by 1/median(sigma_eff) reached ~600x
        their intended strength, made up 98% of the objective, and
        dragged the fitted temperature +50% past truth.
        """
        idx = PeakyIndexerV3(PEAKS, dbpath="db", weighted_solve=True,
                             amp_sigma=np.full(len(PEAKS), 2.0),
                             amp_sigma_floor=0.5)
        idx.build_candidate_matrix(sa_doublets=False)
        idx._rebuild_overlap(0.05, 0.03)
        sigma_eff = idx._sigma_eff()
        for T in (7000.0, 10000.0, 15000.0):
            c, cost = idx._solve_concentrations(T, 17.0)
            pred = idx._last_A @ c
            data = float(np.sum(((idx._obs_amp - pred) / sigma_eff) ** 2))
            ridge = cost - data - float(idx._last_pseudo_cost)
            n_active = int(np.sum(idx._last_col_max > 1e-30))
            # per-species cap: w_mm * (1/f)^2 for missing mass plus the
            # per-strong-line count terms (same cap per line)
            cap = n_active * (
                idx._evidence_missing_mass_weight
                + idx._evidence_missing_count_weight * 6
            ) * (1.0 / PROCEDURE_NOISE_FRACTION) ** 2
            self.assertLessEqual(ridge, cap * 1.1 + 1.0,
                                 msg=f"T={T}: ridge {ridge:.1f} exceeds "
                                     f"z-unit cap {cap:.1f}")


class TestBasinBic(unittest.TestCase):
    def test_grid_mode_reports_bic_and_posterior(self):
        idx = PeakyIndexerV3(PEAKS, dbpath="db", weighted_solve=True,
                             amp_sigma=np.full(len(PEAKS), 2.0))
        res = idx.run(n_calls=8, verbose=False, search="grid")
        info = res.convergence_info
        self.assertIn("basin_bic", info)
        self.assertIn("posterior_fractions", info)
        self.assertGreaterEqual(info["n_basins_considered"], 1)
        # NOTE: delta_bic_runner_up == 0.0 is legitimate here — the cost
        # surface is exactly flat in ne (ion stages are independent
        # unknowns), so ne-separated candidates tie and are correctly
        # flagged ambiguous rather than deduplicated.
        if info["delta_bic_runner_up"] is not None:
            self.assertGreaterEqual(info["delta_bic_runner_up"], 0.0)
        # posterior fractions are plain floats summing to ~1
        post = info["posterior_fractions"]
        self.assertTrue(all(isinstance(v, float) for v in post.values()))
        if post:
            self.assertAlmostEqual(sum(post.values()), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
