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
    calibrate_c4,
    effective_quantum_number_sq,
    halpha_log_ne,
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

    def test_shape_factor_zero_beyond_validity_ceiling(self):
        """Near-threshold Rydberg levels (Si II 670.13 nm class: bound by
        ~7 meV, n_eff ~ 91, raw shape ~1.7e7) are outside the
        quadratic-impact validity range and merged into the continuum at
        LIBS densities — excluded, not clamped, or one line swamps the
        width objective by six decades and pins n_e to a bound."""
        self.assertEqual(float(stark_shape_factor(11.872, 11.865, 2)), 0.0)
        # A level bound by ~1 eV (n_eff ~ 3.7 at z=1) stays in.
        self.assertGreater(float(stark_shape_factor(5.0, 4.0, 1)), 0.0)

    def test_zero_width_peaks_are_censored_not_observations(self):
        """A Lorentzian width fitted at its zero bound is a censored
        measurement (sigma/gamma degeneracy at low SNR), not an
        observation of zero width; treating it as data biases n_e low."""
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.03, 0.0]]))
        idx._stark_width_weight = 0.5
        idx._stark_c4 = 1e-3
        idx._stark_peak_shape = np.array([50.0])
        self.assertEqual(idx._width_cost(17.0), 0.0)
        self.assertIsNone(idx._last_gamma_inst)

    def test_stark_hwhm_linear_in_ne(self):
        w17 = float(stark_hwhm(1.0, 17.0, c4=1e-3, log_ne_ref=17.0))
        w18 = float(stark_hwhm(1.0, 18.0, c4=1e-3, log_ne_ref=17.0))
        self.assertAlmostEqual(w17, 1e-3)
        self.assertAlmostEqual(w18 / w17, 10.0)

    def test_halpha_anchor(self):
        # GGC 2003 reference point: FWHM = 0.549 nm at exactly 1e17 cm^-3.
        self.assertAlmostEqual(float(halpha_log_ne(0.549)), 17.0)
        # ~ne^0.68 scaling: 10x density -> 10^0.68 wider.
        ratio = 10.0 ** 0.67965
        self.assertAlmostEqual(float(halpha_log_ne(0.549 * ratio)), 18.0, places=6)

    def test_calibrate_c4_recovers_truth(self):
        rng = np.random.default_rng(11)
        shapes = rng.uniform(5.0, 400.0, size=40)
        c4_true, gamma_inst_true, log_ne = 3e-4, 0.021, 17.4
        gamma_obs = gamma_inst_true + stark_hwhm(shapes, log_ne, c4_true)
        c4, gamma_inst = calibrate_c4(gamma_obs, shapes, log_ne)
        self.assertAlmostEqual(c4, c4_true, places=10)
        self.assertAlmostEqual(gamma_inst, gamma_inst_true, places=10)

    def test_calibrate_c4_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            calibrate_c4([0.02, 0.03], [10.0, 10.0], 17.0)
        with self.assertRaises(ValueError):
            calibrate_c4([0.02], [10.0], 17.0)

    def test_halpha_ne_bounds(self):
        from alibz.utils.stark import halpha_ne_bounds

        # A Stark-broadened H-alpha peak yields bounds centred on its
        # implied density.
        peaks = np.array(
            [
                [100.0, 500.0, 0.03, 0.02],
                [50.0, 656.30, 0.04, 0.25],
            ]
        )
        bounds = halpha_ne_bounds(peaks)
        self.assertIsNotNone(bounds)
        lo, hi = bounds
        centre = float(halpha_log_ne(0.5))
        self.assertAlmostEqual((lo + hi) / 2.0, centre, places=10)
        self.assertLess(lo, centre)
        self.assertGreater(hi, centre)

        # No peak near H-alpha -> None.
        self.assertIsNone(halpha_ne_bounds(peaks[:1]))
        # Narrow (instrumental) width -> None: no Stark information.
        narrow = np.array([[50.0, 656.30, 0.04, 0.02]])
        self.assertIsNone(halpha_ne_bounds(narrow))
        # Empty table -> None.
        self.assertIsNone(halpha_ne_bounds(np.empty((0, 4))))


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

    def test_hydrogenic_species_excluded(self):
        """One-electron systems (H I, He II, ...) are linear-Stark
        dominated and must carry no quadratic-Stark shape factor."""
        checked = 0
        for sp in self.table.species:
            if sp.ion >= sp.Z:
                span = slice(sp.line_start, sp.line_end)
                self.assertTrue(
                    np.all(self.table.stark_shape[span] == 0.0),
                    f"{sp.element} stage {sp.ion}",
                )
                checked += 1
        # He II has optical lines (e.g. 468.6 nm region tables) — make sure
        # the rule actually fired for at least one species when present.
        sb = SahaBoltzmann("db")
        wide = LineTable(sb.db, sb, wl_range=(180.0, 961.0), max_ion_stage=2)
        hydrogenic = [sp for sp in wide.species if sp.ion >= sp.Z]
        self.assertGreater(len(hydrogenic), 0)
        for sp in hydrogenic:
            span = slice(sp.line_start, sp.line_end)
            self.assertTrue(np.all(wide.stark_shape[span] == 0.0))

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
        # A reference density distinct from both the default (17.0) and
        # ne_init, so a mis-wired stark_log_ne_ref cannot cancel out.
        ne_true, gamma_inst_true, ne_ref = 17.5, 0.02, 16.5

        def widths(shapes):
            # choose c4 so the median Stark HWHM at ne_true is comparable
            # to the instrumental width (a realistic, measurable effect)
            self.c4 = 0.03 / (np.median(shapes) * 10.0 ** (ne_true - ne_ref))
            return gamma_inst_true + stark_hwhm(
                shapes, ne_true, self.c4, log_ne_ref=ne_ref
            )

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
            stark_log_ne_ref=ne_ref,
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
        # Without the width channel the amplitudes are flat in n_e, so the
        # optimiser must both FAIL to recover the truth and drift to a
        # search bound (asserting only "not recovered" would pass vacuously
        # on a broken fixture).
        self.assertGreater(abs(result.ne - ne_true), 0.3)
        self.assertLess(min(abs(result.ne - 16.0), abs(result.ne - 19.0)), 0.4)


if __name__ == "__main__":
    unittest.main()
