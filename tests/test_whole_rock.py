"""Scientific invariants for the Gard et al. whole-rock composition prior."""

import csv
import unittest

import numpy as np

from alibz.elements import ATOMIC_NUMBER
from alibz.synthetic import (
    PeriodicCoverageScheduler,
    SyntheticSpectrumGenerator,
    WholeRockSceneSampler,
)
from alibz.whole_rock import (
    WholeRockCompositionMixture,
    WholeRockCompositionModel,
)
from scripts.build_whole_rock_prior import (
    _major_element_ppm,
    _rock_stratum,
    _volatile_major_element_ppm,
)


class TestWholeRockCompositionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WholeRockCompositionModel.load("db/whole_rock_prior_v1.npz")

    def test_artifact_retains_schema_and_source_counts(self):
        self.assertEqual(len(self.model.training_mask), 92)
        self.assertEqual(self.model.source_doi, "10.5281/zenodo.2592823")
        self.assertEqual(int(self.model.stratum_sample_count[0]), 1_023_490)
        self.assertEqual(self.model.manifest["paper"]["database_sample_count_reported"],
                         1_022_092)
        self.assertEqual(
            self.model.manifest["fit"]["valid_major_rows_85_to_120_wt_percent"],
            626_902,
        )

    def test_se_th_u_are_modeled_but_exclusions_are_not(self):
        for element in ("Se", "Th", "U"):
            ei = ATOMIC_NUMBER[element] - 1
            self.assertTrue(np.all(self.model.modeled_mask[:, ei]))
        for element in ("Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Pa"):
            ei = ATOMIC_NUMBER[element] - 1
            self.assertFalse(self.model.training_mask[ei])
            self.assertFalse(np.any(self.model.modeled_mask[:, ei]))

    def test_sample_replays_and_preserves_fraction_invariants(self):
        first = self.model.sample(8271, stratum="igneous_volcanic")
        second = self.model.sample(8271, stratum="igneous_volcanic")
        np.testing.assert_array_equal(first.mass_fraction, second.mass_fraction)
        np.testing.assert_array_equal(first.nuclei_fraction, second.nuclei_fraction)
        np.testing.assert_array_equal(
            first.below_detection_mask, second.below_detection_mask
        )
        self.assertAlmostEqual(float(first.mass_fraction.sum()), 1.0)
        self.assertAlmostEqual(float(first.nuclei_fraction.sum()), 1.0)
        self.assertGreater(first.nuclei_fraction[ATOMIC_NUMBER["O"] - 1], 0.2)
        for element in ("Tc", "Fr", "Ra", "Ac"):
            self.assertEqual(first.nuclei_fraction[ATOMIC_NUMBER[element] - 1], 0.0)

    def test_stage_fractions_are_direct_and_sum_to_nuclei(self):
        composition, abundance = self.model.sample_stage_abundance(
            9283, stratum="metamorphic_sedimentary"
        )
        self.assertEqual(abundance.shape, (92, 3))
        np.testing.assert_allclose(
            abundance.sum(axis=1), composition.nuclei_fraction, atol=1e-14
        )
        self.assertAlmostEqual(float(abundance.sum()), 1.0)

    def test_scheduler_uses_joint_prior_without_changing_focus_abundance(self):
        generator = SyntheticSpectrumGenerator("db")
        scheduler = PeriodicCoverageScheduler(
            generator,
            whole_rock_model=self.model,
            whole_rock_policy="balanced",
        )
        cell = next(
            value for value in scheduler.cells()
            if value.element == "Th" and value.ion_stage == 2
            and value.abundance == 1e-4
        )
        scene = scheduler.scene(cell, seed=771)
        th = ATOMIC_NUMBER["Th"] - 1
        self.assertEqual(scene.target.stage_abundance[th, 1], 1e-4)
        self.assertEqual(float(scene.target.stage_abundance[th].sum()), 1e-4)
        self.assertAlmostEqual(float(scene.target.stage_abundance.sum()), 1.0)
        self.assertIn(scene.metadata["whole_rock_stratum"], self.model.selectable_strata)
        self.assertEqual(
            scene.metadata["whole_rock_source_doi"], "10.5281/zenodo.2592823"
        )
        for element in ("Tc", "Fr", "Ra", "Ac"):
            ei = ATOMIC_NUMBER[element] - 1
            self.assertEqual(float(scene.target.stage_abundance[ei].sum()), 0.0)

    def test_natural_scene_retains_bulk_prior_and_separate_ambient_gas(self):
        sampler = WholeRockSceneSampler(
            self.model, gas_probability=1.0, stratum_policy="balanced"
        )
        first = sampler.scene(8472, stratum="sedimentary_clastic")
        second = sampler.scene(8472, stratum="sedimentary_clastic")
        np.testing.assert_array_equal(
            first.target.stage_abundance, second.target.stage_abundance
        )
        self.assertIsNotNone(first.ambient_gas)
        self.assertEqual(first.metadata["whole_rock_stratum"],
                         "sedimentary_clastic")
        self.assertEqual(first.metadata["bulk_to_plasma_transfer"],
                         "identity-shape-proxy-not-calibrated")
        self.assertEqual(len(first.metadata["whole_rock_mass_fraction"]), 92)
        self.assertAlmostEqual(sum(first.metadata["whole_rock_mass_fraction"]), 1.0)
        self.assertAlmostEqual(float(first.target.stage_abundance.sum()), 1.0)
        self.assertAlmostEqual(float(first.ambient_gas.stage_abundance.sum()), 1.0)

    def test_range_table_distinguishes_reporting_and_censoring(self):
        with open("db/whole_rock_element_ranges.csv", newline="") as handle:
            rows = {
                (row["stratum"], row["element"]): row
                for row in csv.DictReader(handle)
            }
        selenium = rows[("global", "Se")]
        self.assertGreater(int(selenium["n_detected"]), 20_000)
        self.assertGreater(int(selenium["n_censored"]), 20_000)
        self.assertGreater(float(selenium["detected_p500_ppm"]), 0.0)
        self.assertNotEqual(selenium["reporting_rate"], "1")


class TestWholeRockBuildRules(unittest.TestCase):
    def test_rock_strata_keep_major_lithologic_distinctions(self):
        self.assertEqual(_rock_stratum("igneous", "volcanic"),
                         "igneous_volcanic")
        self.assertEqual(_rock_stratum("metamorphic", "metasedimentary"),
                         "metamorphic_sedimentary")
        self.assertEqual(_rock_stratum("sedimentary", "chemical"),
                         "sedimentary_chemical_biogenic")
        self.assertEqual(_rock_stratum("", ""), "unclassified")

    def test_major_oxide_conversion_produces_elemental_mass_closure(self):
        header = (
            "major_id", "sio2", "tio2", "al2o3", "cr2o3", "fe2o3",
            "fe2o3_tot", "feo", "feo_tot", "mgo", "cao", "mno", "nio",
            "k2o", "na2o", "sro", "p2o5", "so3", "bao",
        )
        raw = np.full((2, len(header)), np.nan, dtype=float)
        raw[:, 0] = (1, 2)
        raw[0, header.index("sio2")] = 100.0
        raw[1, header.index("sio2")] = -100.0
        concentration, valid = _major_element_ppm(header, raw)
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertAlmostEqual(float(np.nansum(concentration[0])), 1_000_000.0,
                               places=1)
        self.assertGreater(concentration[0, ATOMIC_NUMBER["Si"] - 1], 0)
        self.assertGreater(concentration[0, ATOMIC_NUMBER["O"] - 1], 0)

    def test_volatile_conversion_requires_speciation_and_closes_mass(self):
        header = (
            "major_id", "sio2", "tio2", "al2o3", "cr2o3", "fe2o3",
            "fe2o3_tot", "feo", "feo_tot", "mgo", "cao", "mno", "nio",
            "k2o", "na2o", "sro", "p2o5", "so3", "bao", "co2",
            "h2o_plus", "h2o_minus", "h2o_tot", "caco3", "mgco3", "loi",
        )
        raw = np.full((3, len(header)), np.nan, dtype=float)
        raw[:, 0] = (1, 2, 3)
        raw[:, header.index("sio2")] = (55.0, 85.0, 90.0)
        raw[:, header.index("al2o3")] = (15.0, 5.0, 5.0)
        raw[:, header.index("cao")] = (10.0, 5.0, 5.0)
        raw[0, header.index("co2")] = 20.0
        raw[1, header.index("h2o_tot")] = 5.0
        raw[2, header.index("loi")] = 10.0
        concentration, valid, group, stats = _volatile_major_element_ppm(
            header, raw
        )
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_array_equal(group, [0, 1, -1])
        np.testing.assert_allclose(
            np.nansum(concentration[:2], axis=1), 1_000_000.0, rtol=1e-6
        )
        self.assertGreater(concentration[0, ATOMIC_NUMBER["C"] - 1], 0)
        self.assertGreater(concentration[1, ATOMIC_NUMBER["H"] - 1], 0)
        self.assertEqual(stats["loi_only_at_or_above_threshold"], 1)


class TestCarbonateVolatilePrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.volatile = WholeRockCompositionModel.load(
            "db/whole_rock_carbonate_volatile_prior_v1.npz"
        )
        cls.anhydrous = WholeRockCompositionModel.load(
            "db/whole_rock_prior_v1.npz"
        )
        cls.mixture = WholeRockCompositionMixture(
            (cls.anhydrous, cls.volatile),
            labels=("anhydrous", "carbonate_volatile"),
        )

    def test_separate_artifact_has_two_speciated_strata(self):
        self.assertEqual(
            self.volatile.strata,
            ("global", "carbonate_rich", "volatile_rich"),
        )
        self.assertEqual(self.volatile.stratum_sample_count.tolist(),
                         [29_492, 12_865, 16_627])
        for element in ("H", "C", "O", "Se", "Th", "U"):
            self.assertTrue(
                self.volatile.modeled_mask[0, ATOMIC_NUMBER[element] - 1]
            )

    def test_equal_weight_mixture_has_eleven_training_strata(self):
        self.assertEqual(len(self.mixture.training_strata), 11)
        self.assertEqual(self.mixture.manifest["default_policy"], "balanced")
        self.assertIn(
            "carbonate_volatile:carbonate_rich", self.mixture.training_strata
        )
        first = self.mixture.sample(719, stratum="carbonate_rich")
        second = self.mixture.sample(719, stratum="carbonate_rich")
        self.assertEqual(first.stratum, "carbonate_volatile:carbonate_rich")
        np.testing.assert_array_equal(first.nuclei_fraction,
                                      second.nuclei_fraction)
        self.assertGreater(first.nuclei_fraction[ATOMIC_NUMBER["C"] - 1], 0)
        schedule = self.mixture.balanced_stratum_schedule(3, seed=991)
        self.assertEqual(
            schedule, self.mixture.balanced_stratum_schedule(3, seed=991)
        )
        self.assertEqual(len(schedule), 33)
        self.assertTrue(all(schedule.count(name) == 3
                            for name in self.mixture.training_strata))

    def test_scene_sampler_keeps_balanced_policy_by_default(self):
        sampler = WholeRockSceneSampler(self.mixture, gas_probability=0.0)
        self.assertEqual(sampler.stratum_policy, "balanced")
        scene = sampler.scene(1802, stratum="volatile_rich")
        self.assertEqual(
            scene.metadata["whole_rock_stratum"],
            "carbonate_volatile:volatile_rich",
        )
        self.assertIsNone(scene.ambient_gas)


if __name__ == "__main__":
    unittest.main()
