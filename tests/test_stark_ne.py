"""Stark-width electron-density coupling (Track C).

With ion stages as independent unknowns, the fitted amplitudes are flat in
n_e (each stage's Saha factor is absorbed into its own concentration), so
linewidths are the only in-spectrum n_e handle.  These tests pin the
per-line Stark shape factors, the width-cost machinery, and end-to-end n_e
recovery from widths alone.
"""
import unittest

import numpy as np

from alibz import PeakyIndexer
from alibz.peaky_indexer_v3 import LineTable
from alibz.utils.sahaboltzmann import SahaBoltzmann
from alibz.utils.stark import (
    RYDBERG_EV,
    effective_quantum_number_sq,
    stark_hwhm,
    stark_shape_factor,
)


class TestStarkUtils(unittest.TestCase):

    def test_effective_quantum_number(self):
        # Hydrogenic ground state: E_ion - E_k = Ry, z = 1 -> n_eff = 1.
        self.assertAlmostEqual(
            float(effective_quantum_number_sq(RYDBERG_EV, 0.0, 1)), 1.0
        )
        # Upper level bound by Ry/9 -> n_eff^2 = 9.
        self.assertAlmostEqual(
            float(effective_quantum_number_sq(RYDBERG_EV, RYDBERG_EV * 8 / 9, 1)),
            9.0,
        )

    def test_shape_factor_zero_for_unbound_levels(self):
        self.assertEqual(float(stark_shape_factor(5.0, 5.0, 1)), 0.0)
        self.assertEqual(float(stark_shape_factor(5.0, 6.0, 1)), 0.0)

    def test_shape_factor_grows_with_upper_level(self):
        low = float(stark_shape_factor(10.0, 2.0, 1))
        high = float(stark_shape_factor(10.0, 9.0, 1))
        self.assertGreater(high, low)

    def test_stark_hwhm_linear_in_ne(self):
        w17 = float(stark_hwhm(1.0, 17.0, c4=1e-3, log_ne_ref=17.0))
        w18 = float(stark_hwhm(1.0, 18.0, c4=1e-3, log_ne_ref=17.0))
        self.assertAlmostEqual(w17, 1e-3)
        self.assertAlmostEqual(w18 / w17, 10.0)


class TestLineTableStarkShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sb = SahaBoltzmann("db")
        cls.table = LineTable(
            cls.sb.db, cls.sb, wl_range=(390.0, 700.0), max_ion_stage=2
        )

    def test_shape_aligned_and_mostly_populated(self):
        table = self.table
        self.assertEqual(table.stark_shape.shape, (table.n_lines,))
        self.assertTrue(np.all(np.isfinite(table.stark_shape)))
        self.assertTrue(np.all(table.stark_shape >= 0.0))
        # The bulk of optical lines have bound upper levels below the stage
        # ionization energy, so most shapes must be positive.
        self.assertGreater(np.mean(table.stark_shape > 0), 0.5)

    def test_hydrogen_excluded(self):
        for sp in self.table.species:
            if sp.element == "H":
                span = slice(sp.line_start, sp.line_end)
                self.assertTrue(np.all(self.table.stark_shape[span] == 0.0))

    def test_filter_species_compacts_shape(self):
        sb = SahaBoltzmann("db")
        table = LineTable(sb.db, sb, wl_range=(390.0, 700.0), max_ion_stage=2)
        keep = np.zeros(table.n_species, dtype=bool)
        keep[:3] = True
        expected = np.concatenate(
            [
                table.stark_shape[table.species[i].line_start : table.species[i].line_end]
                for i in range(3)
            ]
        )
        table.filter_species(keep)
        np.testing.assert_array_equal(table.stark_shape, expected)
        self.assertEqual(table.stark_shape.shape, (table.n_lines,))


class TestNeRecoveryFromWidths(unittest.TestCase):
    """End-to-end machinery test: peaks whose Lorentzian widths follow the
    Stark model at a known n_e must pull the optimiser to that n_e even
    though the amplitudes carry no n_e information.  (The generator uses
    the model's own scaling — this validates the inference machinery, not
    the c4 calibration, which comes from reference lines.)"""

    def _make_indexer(self, gamma_obs_fn):
        sb = SahaBoltzmann("db")
        table = LineTable(sb.db, sb, wl_range=(390.0, 560.0), max_ion_stage=2)
        idx_probe = PeakyIndexer(np.array([[1.0, 475.0, 0.03, 0.02]]))
        idx_probe.line_table = table
        lw = idx_probe._line_weights(10_000.0, 17.0)

        # Strong lines with positive Stark shape, at least 0.3 nm apart,
        # deliberately drawn from low/mid/high shape-factor terciles so
        # (gamma_inst, n_e) are jointly well conditioned: width_i =
        # gamma_inst + c4 * s_i * 10^(ne-ref) is affine in s_i, and a
        # narrow s spread makes the two parameters nearly degenerate.
        # The very strongest lines are resonance lines with LOW upper
        # levels (small n_eff), so reach into the top ~2000 to include
        # the high-n_eff lines that are the sensitive n_e probes.
        strong = np.argsort(lw)[::-1][:2000]
        strong = [j for j in strong if table.stark_shape[j] > 0]
        strong.sort(key=lambda j: table.stark_shape[j])
        chosen = []
        for tercile in np.array_split(np.asarray(strong), 3):
            picked = 0
            for j in tercile[np.argsort(lw[tercile])[::-1]]:
                if any(
                    abs(table.wavelengths[j] - table.wavelengths[k]) < 0.3
                    for k in chosen
                ):
                    continue
                chosen.append(int(j))
                picked += 1
                if picked >= 5:
                    break
        shapes = table.stark_shape[chosen]
        self.assertGreater(np.max(shapes) / np.min(shapes), 3.0)

        peaks = np.column_stack(
            [
                lw[chosen] / np.max(lw[chosen]),
                table.wavelengths[chosen],
                np.full(len(chosen), 0.03),
                gamma_obs_fn(shapes),
            ]
        )
        return peaks, shapes

    def test_ne_recovered_from_widths_alone(self):
        ne_true, gamma_inst_true = 17.5, 0.02
        sb_probe = None  # noqa: F841

        def widths(shapes):
            # choose c4 so the median Stark HWHM at ne_true is comparable
            # to the instrumental width (a realistic, measurable effect)
            self.c4 = 0.03 / (np.median(shapes) * 10.0 ** (ne_true - 17.0))
            return gamma_inst_true + stark_hwhm(shapes, ne_true, self.c4)

        peaks, _ = self._make_indexer(widths)

        result = PeakyIndexer(
            peaks, temperature_init=10_000.0, ne_init=17.0
        ).run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            T_bounds=(9_999.0, 10_001.0),
            ne_bounds=(16.0, 19.0),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.005, 0.05),
            n_calls=25,
            stark_width_weight=0.5,
            stark_c4=self.c4,
            verbose=False,
        )
        self.assertAlmostEqual(result.ne, ne_true, delta=0.3)

    def test_ne_unconstrained_without_coupling(self):
        """Control: same fixture with the coupling off pins n_e to a search
        bound instead of the truth (documents WHY the coupling exists)."""
        ne_true, gamma_inst_true = 17.5, 0.02

        def widths(shapes):
            c4 = 0.03 / (np.median(shapes) * 10.0 ** (ne_true - 17.0))
            self.c4 = c4
            return gamma_inst_true + stark_hwhm(shapes, ne_true, c4)

        peaks, _ = self._make_indexer(widths)

        result = PeakyIndexer(
            peaks, temperature_init=10_000.0, ne_init=17.0
        ).run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            T_bounds=(9_999.0, 10_001.0),
            ne_bounds=(16.0, 19.0),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.005, 0.05),
            n_calls=25,
            stark_width_weight=0.0,
            verbose=False,
        )
        near_bound = min(abs(result.ne - 16.0), abs(result.ne - 19.0)) < 0.4
        recovered = abs(result.ne - ne_true) < 0.3
        self.assertTrue(near_bound or not recovered)


if __name__ == "__main__":
    unittest.main()
