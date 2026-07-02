"""Self-absorption (escape-factor) correction in the v3 indexer.

Measured on a Li-rich mineral, the K I resonance doublet area ratio was
1.30 against the T-independent optically-thin value of 2.02 and Na D was
1.51 vs 2.00 — a ~35% compression the linear (thin) design matrix cannot
absorb.  These tests pin the escape-factor physics and the damped
fixed-point solve, including recovery of the global optical-depth scale
as an outer parameter.
"""
import unittest

import numpy as np

from alibz import PeakyIndexer
from alibz.peaky_indexer_v3 import LineTable
from alibz.utils.absorption import doublet_ratio, escape_factor
from alibz.utils.sahaboltzmann import SahaBoltzmann

T_TRUE, NE_TRUE = 10_000.0, 17.0
S_TAU_TRUE, C_TRUE = 3.0, 1.0

# The false-positive machinery (evidence ridges, pseudo-observations) is
# deliberately OFF: it penalises this fixture's deliberately incomplete
# 12-line table and is tested elsewhere.
BUILD_KW = dict(
    shift_tolerance=0.05,
    max_ion_stage=2,
    pseudo_obs_weight=0.0,
    evidence_top_k=0,
    evidence_missing_mass_weight=0.0,
    evidence_missing_count_weight=0.0,
    min_init_relative_intensity=0.0,
)


class TestEscapeFactor(unittest.TestCase):

    def test_limits_and_monotonicity(self):
        self.assertAlmostEqual(float(escape_factor(0.0)), 1.0)
        self.assertAlmostEqual(float(escape_factor(1e-9)), 1.0, places=8)
        self.assertAlmostEqual(float(escape_factor(10.0)), 0.1, places=3)
        tau = np.linspace(0.0, 20.0, 200)
        self.assertTrue(np.all(np.diff(escape_factor(tau)) < 0))
        # negative tau clipped, no explosion
        self.assertAlmostEqual(float(escape_factor(-1.0)), 1.0)

    def test_doublet_ratio_inverts_measured_k_ratio(self):
        # thin limit
        self.assertAlmostEqual(float(doublet_ratio(1e-9)), 2.0, places=6)
        # analytic inversion of the measured K I ratio 1.30 at strength 2:
        # ratio = 2 * (1+e^-tau)/2 / ... -> tau_weak ~ 1.25
        from scipy.optimize import brentq
        tau = brentq(lambda t: float(doublet_ratio(t, 2.02)) - 1.30, 1e-6, 20.0)
        self.assertAlmostEqual(tau, 1.25, delta=0.15)
        # saturated limit -> 1
        self.assertLess(float(doublet_ratio(50.0)), 1.05)


class TestSelfAbsorbedRecovery(unittest.TestCase):
    """Amplitudes generated with the solver's own SA forward model at a
    known (concentration, tau scale) must be recovered: the thin solve is
    biased ~-55%, the forced-scale fixed point is exact, and the fitted
    scale recovers both quantities."""

    @classmethod
    def setUpClass(cls):
        sb = SahaBoltzmann("db")
        table = LineTable(sb.db, sb, wl_range=(390.0, 560.0), max_ion_stage=2)
        probe = PeakyIndexer(np.array([[1.0, 475.0, 0.03, 0.02]]))
        probe.line_table = table
        lw = probe._line_weights(T_TRUE, NE_TRUE)
        fe = next(i for i, s in enumerate(table.species)
                  if s.element == "Fe" and s.ion == 1)
        sp = table.species[fe]
        jj = np.arange(sp.line_start, sp.line_end)
        sel = []
        for j in jj[np.argsort(lw[jj])[::-1][:40]]:
            if any(abs(table.wavelengths[j] - table.wavelengths[k]) < 0.3
                   for k in sel):
                continue
            sel.append(int(j))
            if len(sel) >= 12:
                break
        cls.wl_sel = table.wavelengths[np.array(sel)]

        dummy = cls._peaks(np.ones(len(cls.wl_sel)))
        gen = cls._build(dummy)
        amps = (
            gen._build_design_matrix(
                gen._line_weights(T_TRUE, NE_TRUE)
                * escape_factor(
                    S_TAU_TRUE * C_TRUE
                    * gen._absorption_strengths(T_TRUE, NE_TRUE)
                )
            )
            @ np.array([C_TRUE])
        ).ravel()
        cls.peaks = cls._peaks(amps)

    @classmethod
    def _peaks(cls, amps):
        n = len(cls.wl_sel)
        return np.column_stack(
            [amps, cls.wl_sel, np.full(n, 0.03), np.full(n, 0.02)]
        )

    @staticmethod
    def _build(peaks, **sa):
        idx = PeakyIndexer(peaks, temperature_init=T_TRUE, ne_init=NE_TRUE)
        idx.build_candidate_matrix(**BUILD_KW, **sa)
        keep = np.array(
            [s.element == "Fe" and s.ion == 1 for s in idx.line_table.species]
        )
        idx.line_table.filter_species(keep)
        idx._rebuild_overlap(0.03, 0.02)
        return idx

    def test_thin_solve_is_biased_low(self):
        idx = self._build(self.peaks)
        c, _ = idx._solve_concentrations(T_TRUE, NE_TRUE)
        self.assertLess(c[0], 0.6 * C_TRUE)

    def test_fixed_point_exact_at_true_scale(self):
        idx = self._build(self.peaks, sa_tau_scale=S_TAU_TRUE)
        c, _ = idx._solve_concentrations(T_TRUE, NE_TRUE)
        self.assertAlmostEqual(c[0], C_TRUE, delta=0.01)

    def test_fitted_tau_scale_recovers_concentration_and_depth(self):
        idx = self._build(self.peaks, sa_fit=True)
        result = idx.fit(
            T_bounds=(T_TRUE - 1.0, T_TRUE + 1.0),
            ne_bounds=(NE_TRUE - 1e-3, NE_TRUE + 1e-3),
            sigma_bounds=(0.0299, 0.0301),
            gamma_bounds=(0.0199, 0.0201),
            n_calls=25,
            verbose=False,
        )
        self.assertAlmostEqual(result.concentrations[0], C_TRUE, delta=0.1)
        self.assertAlmostEqual(
            result.convergence_info["sa_tau_scale"], S_TAU_TRUE, delta=0.5
        )
        self.assertGreater(result.r_squared, 0.99)


if __name__ == "__main__":
    unittest.main()
