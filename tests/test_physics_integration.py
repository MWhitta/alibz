import unittest

import numpy as np

from alibz import PeakyFinder, PeakyIndexer, PeakyMaker


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
        result = indexer.run(
            shift_tolerance=0.05,
            max_ion_stage=2,
            T_bounds=(9_000.0, 11_000.0),
            ne_bounds=(16.7, 17.3),
            sigma_bounds=(0.02, 0.05),
            gamma_bounds=(0.01, 0.04),
            n_calls=6,
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
        self.assertAlmostEqual(result.temperature, self.temperature, delta=1_000.0)
        self.assertAlmostEqual(result.ne, self.ne, delta=0.35)

    def test_two_element_spectrum_recovers_ca_and_al_and_plasma_state(self) -> None:
        peaks, result, ranked_species, assigned_species = self._run_pipeline(
            {"Ca": 0.6, "Al": 0.4}
        )

        self.assertGreaterEqual(peaks.shape[0], 4)
        self.assertEqual(ranked_species[:2], [("Ca", 2), ("Al", 1)])
        self.assertEqual(assigned_species, {("Ca", 2), ("Al", 1)})
        self.assertGreater(result.r_squared, 0.99)
        self.assertAlmostEqual(result.temperature, self.temperature, delta=1_200.0)
        self.assertAlmostEqual(result.ne, self.ne, delta=0.35)

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


if __name__ == "__main__":
    unittest.main()
