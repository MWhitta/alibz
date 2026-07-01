import unittest

import numpy as np
from scipy.signal import find_peaks

from alibz import PeakyFinder, PeakyIndexer, PeakyMaker
from alibz.peaky_indexer_v3 import LineTable
from alibz.utils.sahaboltzmann import SahaBoltzmann


class TestSyntheticPhysicsPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = 10_000.0
        self.ne = 17.0
        self.w_range = (392.0, 397.5)
        self.maker = PeakyMaker("db")
        self.finder = PeakyFinder.__new__(PeakyFinder)

    def _run_pipeline(self, composition):
        fracs = np.zeros(self.maker.max_z, dtype=float)
        for element, fraction in composition.items():
            fracs[self.maker.db.elements.index(element)] = fraction

        x, y, _ = self.maker.peak_maker(
            fracs,
            w_lo=self.w_range[0],
            w_hi=self.w_range[1],
            inc=0.01,
            voigt_sig=0.03,
            voigt_gam=0.02,
            temperature=self.temperature,
            ne=self.ne,
        )

        fit = self.finder.fit_spectrum(
            x,
            y,
            subtract_background=False,
            plot=False,
            n_sigma=0,
        )
        peaks = fit["sorted_parameter_array"]

        indexer = PeakyIndexer(
            peaks,
            temperature_init=self.temperature,
            ne_init=self.ne,
        )
        # T is searched over a deliberately wide range so the recovery
        # assertions are falsifiable.  nₑ, by contrast, is only a PRIOR
        # here: with ion stages as independent unknowns, each species'
        # Saha factor is absorbed into its own concentration, so nₑ has
        # no handle on the fitted amplitudes unless both stages of an
        # element carry weight in the window (Ca I does not at 392-397.5
        # nm) — the optimiser pins nₑ to a bound, and it must not be
        # asserted as "recovered".
        result = indexer.run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            T_bounds=(6_000.0, 16_000.0),
            ne_bounds=(16.7, 17.3),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.01, 0.04),
            n_calls=15,
            verbose=False,
        )

        ranked_species = []
        for idx in np.argsort(result.concentrations)[::-1]:
            if result.concentrations[idx] <= 0:
                continue
            sp = result.species[idx]
            ranked_species.append((sp.element, sp.ion))

        assigned_species = {
            (assignment["element"], assignment["ion"])
            for assignment in result.peak_assignments
            if assignment["element"] is not None
        }

        return peaks, result, ranked_species, assigned_species

    def test_single_element_spectrum_recovers_ca_and_plasma_state(self) -> None:
        peaks, result, ranked_species, assigned_species = self._run_pipeline({"Ca": 1.0})

        self.assertGreaterEqual(peaks.shape[0], 2)
        self.assertEqual(ranked_species[0], ("Ca", 2))
        self.assertEqual(assigned_species, {("Ca", 2)})
        self.assertGreater(result.r_squared, 0.99)
        # Falsifiable: the search spans 6-16 kK, so hitting 10 kK within
        # 600 K requires the Boltzmann line ratios to carry real signal.
        self.assertAlmostEqual(result.temperature, self.temperature, delta=600.0)

    def test_two_element_spectrum_recovers_ca_and_al_and_plasma_state(self) -> None:
        peaks, result, ranked_species, assigned_species = self._run_pipeline(
            {"Ca": 0.6, "Al": 0.4}
        )

        self.assertGreaterEqual(peaks.shape[0], 4)
        self.assertEqual(set(ranked_species[:2]), {("Ca", 2), ("Al", 1)})
        self.assertEqual(assigned_species, {("Ca", 2), ("Al", 1)})
        self.assertGreater(result.r_squared, 0.99)
        self.assertAlmostEqual(result.temperature, self.temperature, delta=800.0)
        # Composition is deliberately NOT asserted here: nₑ pins to the
        # edge of its prior bounds (unidentifiable, see _run_pipeline) and
        # Al's minority-stage Saha amplification of that nₑ error dominates
        # the fractions — any tolerance would either mask regressions or
        # assert the artifact.  Composition recovery is pinned by
        # test_composition_recovered_in_physical_units_at_true_plasma_state.
        self.assertIn("Ca", result.element_fractions)
        self.assertIn("Al", result.element_fractions)

    def test_composition_recovered_in_physical_units_at_true_plasma_state(self) -> None:
        """With (T, nₑ) pinned at the synthesis truth, physical
        concentrations must recover the generating composition.

        Ca II is measured essentially exactly.  Al I amplitudes carry the
        known peak-finder bias on weak lines riding the Ca II Lorentzian
        wings (~+20%), so its tolerance stays wider until the finder's
        amplitude estimation is repaired.
        """
        fracs = np.zeros(self.maker.max_z, dtype=float)
        for element, fraction in {"Ca": 0.6, "Al": 0.4}.items():
            fracs[self.maker.db.elements.index(element)] = fraction

        x, y, _ = self.maker.peak_maker(
            fracs,
            w_lo=self.w_range[0],
            w_hi=self.w_range[1],
            inc=0.01,
            voigt_sig=0.03,
            voigt_gam=0.02,
            temperature=self.temperature,
            ne=self.ne,
        )
        fit = self.finder.fit_spectrum(
            x, y, subtract_background=False, plot=False, n_sigma=0
        )
        result = PeakyIndexer(
            fit["sorted_parameter_array"],
            temperature_init=self.temperature,
            ne_init=self.ne,
        ).run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            T_bounds=(self.temperature - 1.0, self.temperature + 1.0),
            ne_bounds=(self.ne - 1e-3, self.ne + 1e-3),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.01, 0.04),
            n_calls=6,
            verbose=False,
        )

        by_species = {
            (sp.element, sp.ion): float(result.concentrations[s])
            for s, sp in enumerate(result.species)
            if result.concentrations[s] > 0
        }
        self.assertAlmostEqual(by_species[("Ca", 2)], 0.6, delta=0.02)
        self.assertAlmostEqual(by_species[("Al", 1)], 0.4, delta=0.1)
        self.assertAlmostEqual(sum(result.element_fractions.values()), 1.0, places=12)

    def test_broadband_multielement_run_completes_with_evidence_and_pseudo(self) -> None:
        """Regression: broadband multi-element fit must not crash under the
        DEFAULT evidence + pseudo-observation kwargs.

        When several species stay active, the evidence prefilter appends
        missing-mass / missing-count penalty rows to the augmented design
        matrix; the pseudo-observation branch previously rebuilt the target
        vector from ``_obs_amp`` alone, dropping those rows and handing
        ``nnls`` mismatched dimensions. Narrow-window tests never triggered it
        because too few species remained active. This synthesises a wide
        180-961 nm, six-element spectrum where the two branches co-fire.
        """
        fracs = np.zeros(self.maker.max_z, dtype=float)
        composition = {"Ca": 0.30, "Fe": 0.25, "Na": 0.15, "Mg": 0.15, "Ti": 0.10, "Si": 0.05}
        for element, fraction in composition.items():
            fracs[self.maker.db.elements.index(element)] = fraction

        wave, spec, _ = self.maker.peak_maker(
            fracs,
            temperature=9_000.0,
            ne=16.5,
            voigt_sig=0.09,
            voigt_gam=0.09,
        )
        wave = np.asarray(wave, dtype=float)
        pk, _ = find_peaks(spec, height=spec.max() * 1e-3, distance=4)
        self.assertGreater(pk.size, 50)  # need a rich multi-species peak table
        peaks = np.column_stack(
            [spec[pk], wave[pk], np.full(pk.size, 0.09), np.full(pk.size, 0.09)]
        )
        peaks = peaks[np.argsort(peaks[:, 0])[::-1]]

        # DEFAULT run() kwargs (pseudo_obs_weight=1.0, evidence penalties on).
        result = PeakyIndexer(peaks).run(n_calls=10, verbose=False)

        # Completed without ValueError and produced a sensible multi-species fit.
        self.assertGreater(result.r_squared, 0.9)
        self.assertGreaterEqual(int(np.sum(result.concentrations > 0)), 3)
        top = [
            result.species[idx].element
            for idx in np.argsort(result.concentrations)[::-1]
            if result.concentrations[idx] > 0
        ]
        self.assertIn("Ca", top[:6])

    def test_false_positive_suppression_reduces_active_species_on_vis_blend(self) -> None:
        fracs = np.zeros(self.maker.max_z, dtype=float)
        for element, fraction in {"Ca": 0.5, "Al": 0.3, "Na": 0.2}.items():
            fracs[self.maker.db.elements.index(element)] = fraction

        x, y, _ = self.maker.peak_maker(
            fracs,
            w_lo=392.0,
            w_hi=590.5,
            inc=0.02,
            voigt_sig=0.03,
            voigt_gam=0.02,
            temperature=self.temperature,
            ne=self.ne,
        )
        fit = self.finder.fit_spectrum(
            x,
            y,
            subtract_background=False,
            plot=False,
            n_sigma=0,
        )
        peaks = fit["sorted_parameter_array"]

        loose = PeakyIndexer(
            peaks,
            temperature_init=self.temperature,
            ne_init=self.ne,
        ).run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            min_init_relative_intensity=0.0,
            pseudo_obs_weight=0.0,
            pseudo_max_lines_per_species=0,
            evidence_top_k=0,
            evidence_missing_mass_weight=0.0,
            evidence_missing_count_weight=0.0,
            evidence_max_refits=0,
            T_bounds=(9_000.0, 11_000.0),
            ne_bounds=(16.7, 17.3),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.01, 0.04),
            n_calls=6,
            verbose=False,
        )
        suppressed = PeakyIndexer(
            peaks,
            temperature_init=self.temperature,
            ne_init=self.ne,
        ).run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            min_init_relative_intensity=1e-3,
            pseudo_obs_weight=1.0,
            pseudo_line_rel_threshold=0.25,
            pseudo_max_lines_per_species=2,
            T_bounds=(9_000.0, 11_000.0),
            ne_bounds=(16.7, 17.3),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.01, 0.04),
            n_calls=6,
            verbose=False,
        )

        suppressed_top = []
        for idx in np.argsort(suppressed.concentrations)[::-1]:
            if suppressed.concentrations[idx] <= 0:
                continue
            sp = suppressed.species[idx]
            item = (sp.element, sp.ion)
            if item not in suppressed_top:
                suppressed_top.append(item)

        loose_assigned = {
            (assignment["element"], assignment["ion"])
            for assignment in loose.peak_assignments
            if assignment["element"] is not None
        }
        suppressed_assigned = {
            (assignment["element"], assignment["ion"])
            for assignment in suppressed.peak_assignments
            if assignment["element"] is not None
        }

        self.assertTrue({("Ca", 2), ("Al", 1), ("Na", 1)}.issubset(set(suppressed_top[:6])))
        self.assertLess(int(np.sum(suppressed.concentrations > 0)), int(np.sum(loose.concentrations > 0)))
        self.assertLess(len(suppressed_assigned), len(loose_assigned))
        self.assertGreater(suppressed.r_squared, 0.99)


class TestLineWeightConventionConsistency(unittest.TestCase):
    """The inverse model must weight lines with exactly the same physics as
    the forward synthesis — one shared emissivity convention, including the
    hc/lambda photon-energy factor."""

    def test_indexer_line_weights_match_forward_emissivity(self) -> None:
        wl_lo, wl_hi, min_gA = 390.0, 800.0, 100.0
        temperature, log_ne = 11_000.0, 17.0

        sb = SahaBoltzmann("db")
        table = LineTable(
            sb.db,
            sb,
            wl_range=(wl_lo, wl_hi),
            max_ion_stage=2,
            min_gA=min_gA,
        )
        indexer = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        indexer.line_table = table
        weights = indexer._line_weights(temperature, log_ne)

        checked = 0
        for element in ("Ca", "Fe", "Na"):
            lines = sb.db.lines(element)
            ionization = lines[:, 0].astype(float)
            wavelength = lines[:, 1].astype(float)
            gA = lines[:, 3].astype(float)

            # calculate() concatenates stages in sorted order, each stage in
            # database row order — invert that to index intensities by db row.
            _, intensity = sb.calculate(element, np.array([temperature]), log_ne)
            order = np.concatenate(
                [
                    np.flatnonzero(ionization == stage)
                    for stage in np.sort(np.unique(ionization))
                ]
            )
            intensity_by_db_row = np.empty(wavelength.size, dtype=float)
            intensity_by_db_row[order] = intensity[0]

            for sp in table.species:
                if sp.element != element:
                    continue
                mask = (
                    (ionization == float(sp.ion))
                    & (wavelength >= wl_lo)
                    & (wavelength <= wl_hi)
                    & (gA >= min_gA)
                )
                np.testing.assert_allclose(
                    weights[sp.line_start : sp.line_end],
                    intensity_by_db_row[mask],
                    rtol=1e-9,
                    err_msg=f"{element} ion {sp.ion}",
                )
                checked += 1

        self.assertGreaterEqual(checked, 4)

        # Independent recomputation with literal constants — deliberately
        # NOT via line_emissivity, so a convention regression in the shared
        # function itself fails here (the cross-check above only proves the
        # two callers share a function, not what that function computes).
        h_ev_s = 4.135667696e-15
        c_m_s = 2.99792458e8
        kb_ev_k = 8.617333262e-5
        checked_formula = 0
        for sp in table.species:
            if sp.element != "Ca":
                continue
            span = slice(sp.line_start, sp.line_end)
            partition = float(
                sb.stage_partition("Ca", np.array([temperature]), ion=sp.ion)[0]
            )
            ci, _ = sb.ionization_distribution(
                "Ca", np.array([temperature]), log_ne
            )
            stage_fraction = float(ci[0, sp.ion - 1])
            expected = (
                (4.0 * np.pi) ** -1
                * h_ev_s * c_m_s * 1e9 / table.wavelengths[span]
                * table.gA[span]
                * np.exp(-table.Ek[span] / (kb_ev_k * temperature))
                / partition
                * stage_fraction
            )
            np.testing.assert_allclose(
                weights[span], expected, rtol=1e-9, err_msg=f"Ca ion {sp.ion}"
            )
            checked_formula += 1
        self.assertGreaterEqual(checked_formula, 2)


if __name__ == "__main__":
    unittest.main()
