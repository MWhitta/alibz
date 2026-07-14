"""Scientific invariants for the explicit-stage synthetic renderer."""

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np

from alibz.synthetic import (
    AtomicStrengthUncertainty,
    ChannelGrid,
    HierarchicalPlasmaSampler,
    InstrumentResponse,
    PeriodicCoverageScheduler,
    PlasmaComponent,
    SyntheticScene,
    SyntheticSpectrumGenerator,
    dry_air_component,
)
from alibz.utils.sahaboltzmann import SahaBoltzmann
from alibz.synthetic_calibration import (
    IndividualShotCalibration,
    calibrate_individual_shots,
)


def _uniform_grid(lo, hi, step):
    n = int(round((hi - lo) / step))
    left = lo + np.arange(n) * step
    right = left + step
    return ChannelGrid(0.5 * (left + right), left, right)


def _element_array(element, value):
    from alibz.elements import ATOMIC_NUMBER
    out = np.zeros(92)
    out[ATOMIC_NUMBER[element] - 1] = value
    return out


class TestChannelGrid(unittest.TestCase):
    def test_native_grid_matches_active_instrument_export(self):
        grid = ChannelGrid.native()
        self.assertEqual(grid.centers_nm.shape, (21600,))
        self.assertAlmostEqual(grid.centers_nm[0], 190.0)
        self.assertAlmostEqual(grid.centers_nm[-1], 909.9666666667)
        np.testing.assert_allclose(grid.widths_nm, 1.0 / 30.0)

    def test_disjoint_regions_do_not_create_one_large_channel(self):
        centers = np.r_[np.arange(200.0, 201.0, 0.1),
                        np.arange(500.0, 501.0, 0.1)]
        grid = ChannelGrid(centers)
        self.assertLess(float(grid.widths_nm.max()), 0.11)
        self.assertLess(grid.right_edges_nm[9], grid.left_edges_nm[10])

    def test_explicit_edges_are_retained(self):
        grid = ChannelGrid(
            np.array([200.1, 200.5]),
            np.array([200.0, 200.4]),
            np.array([200.2, 200.6]),
        )
        np.testing.assert_allclose(grid.widths_nm, [0.2, 0.2])


class TestExplicitStageGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticSpectrumGenerator("db")

    def test_no_saha_call_and_direct_stages(self):
        component = PlasmaComponent.from_mapping({
            ("Ca", 1): 0.4,
            ("Ca", 2): 0.6,
        })
        grid = ChannelGrid(np.r_[
            np.arange(392.5, 397.5, 0.02),
            np.arange(421.5, 423.5, 0.02),
        ])
        with patch.object(
            SahaBoltzmann,
            "ionization_distribution",
            side_effect=AssertionError("Saha must not be called"),
        ):
            result = self.generator.render(
                SyntheticScene(component, seed=5),
                grid,
                add_noise=False,
                return_line_table=True,
            )
        self.assertFalse(result.manifest["saha_ionization"])
        self.assertGreater(float(result.physical_channel_counts.max()), 0.0)
        stages = result.line_table["target_ion_stage"]
        self.assertIn(1, stages)
        self.assertIn(2, stages)

    def test_unsupported_mass_is_rejected_not_renormalized(self):
        component = PlasmaComponent.from_mapping({("Pm", 1): 1.0})
        with self.assertRaisesRegex(ValueError, "unsupported.*Pm"):
            self.generator.render(
                SyntheticScene(component), _uniform_grid(300, 301, 0.05)
            )

    def test_stage_schema_retains_five_unsupported_elements(self):
        unsupported = [
            self.generator.db.elements[i]
            for i in np.flatnonzero(~self.generator.support_mask)
        ]
        self.assertEqual(unsupported, ["Pm", "Po", "At", "Rn", "Pa"])
        self.assertEqual(self.generator.stage_support.shape, (92, 3))
        se = self.generator.db.elements.index("Se")
        th = self.generator.db.elements.index("Th")
        u = self.generator.db.elements.index("U")
        self.assertTrue(self.generator.stage_support[se, 0])
        self.assertTrue(self.generator.stage_support[th, :2].all())
        self.assertTrue(self.generator.stage_support[u, :2].all())

    def test_channel_integrals_are_resolution_consistent(self):
        component = PlasmaComponent.from_mapping({("Se", 1): 1.0})
        scene = SyntheticScene(component)
        fine = self.generator.render(
            scene, _uniform_grid(203.5, 204.5, 0.005), add_noise=False
        )
        coarse = self.generator.render(
            scene, _uniform_grid(203.5, 204.5, 0.05), add_noise=False
        )
        self.assertAlmostEqual(
            float(fine.physical_channel_counts.sum()),
            float(coarse.physical_channel_counts.sum()),
            delta=1e-6 * float(fine.physical_channel_counts.sum()),
        )

    def test_effective_column_reduces_resonance_flux(self):
        thin = PlasmaComponent.from_mapping({("Li", 1): 1.0})
        column = _element_array("Li", 3.0e-13)
        thick = PlasmaComponent.from_mapping(
            {("Li", 1): 1.0}, effective_column=column
        )
        grid = _uniform_grid(670.0, 671.5, 0.01)
        y_thin = self.generator.render(
            SyntheticScene(thin), grid, add_noise=False
        ).physical_channel_counts
        y_thick = self.generator.render(
            SyntheticScene(thick), grid, add_noise=False
        ).physical_channel_counts
        self.assertGreater(float(y_thin.sum()), float(y_thick.sum()))
        self.assertTrue(np.all(y_thick >= 0))

    def test_per_element_ne_broadens_halpha_without_changing_stage_mass(self):
        low_ne = PlasmaComponent.from_mapping(
            {("H", 1): 1.0}, log_ne_cm3=_element_array("H", 16.0) +
            (1.0 - _element_array("H", 1.0)) * 17.0,
        )
        high_ne = PlasmaComponent.from_mapping(
            {("H", 1): 1.0}, log_ne_cm3=_element_array("H", 18.0) +
            (1.0 - _element_array("H", 1.0)) * 17.0,
        )
        grid = _uniform_grid(654.0, 658.5, 0.01)
        lo = self.generator.render(
            SyntheticScene(low_ne), grid, add_noise=False
        ).physical_channel_counts
        hi = self.generator.render(
            SyntheticScene(high_ne), grid, add_noise=False
        ).physical_channel_counts
        x = grid.centers_nm

        def variance(y):
            norm = y.sum()
            mean = np.sum(x * y) / norm
            return np.sum((x - mean) ** 2 * y) / norm

        self.assertGreater(float(variance(hi)), float(variance(lo)))

    def test_seeded_individual_shot_noise_replays_and_can_be_negative(self):
        instrument = InstrumentResponse(
            read_noise_std_counts=(4.0, 4.0, 4.0),
            dark_offset_std_counts=(2.0, 2.0, 2.0),
            shot_noise=True,
        )
        generator = SyntheticSpectrumGenerator("db", instrument)
        component = PlasmaComponent.from_mapping(
            {("Se", 1): 1.0}, emission_scale=0.0
        )
        grid = _uniform_grid(500.0, 502.0, 0.02)
        scene = SyntheticScene(component, seed=123)
        first = generator.render(scene, grid, add_noise=True)
        second = generator.render(scene, grid, add_noise=True)
        other = generator.render(
            SyntheticScene(component, seed=124), grid, add_noise=True
        )
        np.testing.assert_array_equal(first.intensity_counts, second.intensity_counts)
        self.assertFalse(np.array_equal(first.intensity_counts, other.intensity_counts))
        self.assertTrue(np.any(first.intensity_counts < 0))
        self.assertTrue(np.all(first.physical_channel_counts >= 0))

    def test_ambient_gas_is_separate_from_target_normalization(self):
        target = PlasmaComponent.from_mapping({("Ca", 2): 1.0})
        gas = dry_air_component(
            0.8,
            {"N": (0.7, 0.3, 0.0),
             "O": (0.6, 0.4, 0.0),
             "Ar": (0.5, 0.5, 0.0)},
            emission_scale=1e-3,
        )
        result = self.generator.render(
            SyntheticScene(target, gas, seed=7),
            _uniform_grid(749.0, 752.0, 0.02),
            add_noise=False,
        )
        self.assertAlmostEqual(result.manifest["target_abundance_sum"], 1.0)
        self.assertTrue(result.manifest["ambient_excluded_from_target"])
        self.assertGreater(float(result.ambient_channel_counts.sum()), 0.0)
        np.testing.assert_allclose(
            result.physical_channel_counts,
            result.target_channel_counts + result.ambient_channel_counts,
        )

    def test_hierarchical_sampler_is_replayable_and_element_specific(self):
        abundance = np.zeros((92, 3))
        abundance[self.generator.db.elements.index("Ca"), 1] = 0.6
        abundance[self.generator.db.elements.index("Fe"), 0] = 0.4
        sampler = HierarchicalPlasmaSampler()
        a = sampler.component(abundance, seed=42)
        b = sampler.component(abundance, seed=42)
        np.testing.assert_array_equal(a.temperature_k, b.temperature_k)
        np.testing.assert_array_equal(a.log_ne_cm3, b.log_ne_cm3)
        np.testing.assert_array_equal(a.effective_column, b.effective_column)
        ca = self.generator.db.elements.index("Ca")
        fe = self.generator.db.elements.index("Fe")
        self.assertNotEqual(a.log_ne_cm3[ca], a.log_ne_cm3[fe])

    def test_heavy_line_uncertainty_is_source_aware_and_replayable(self):
        generator = SyntheticSpectrumGenerator(
            "db", atomic_uncertainty=AtomicStrengthUncertainty(enabled=True)
        )
        component = PlasmaComponent.from_mapping({("Th", 1): 1.0})
        grid = _uniform_grid(260.0, 262.0, 0.02)
        first = generator.render(
            SyntheticScene(component, seed=55), grid,
            add_noise=False, return_line_table=True,
        )
        replay = generator.render(
            SyntheticScene(component, seed=55), grid,
            add_noise=False, return_line_table=True,
        )
        other = generator.render(
            SyntheticScene(component, seed=56), grid,
            add_noise=False, return_line_table=True,
        )
        np.testing.assert_array_equal(
            first.line_table["target_strength_factor"],
            replay.line_table["target_strength_factor"],
        )
        self.assertTrue(np.any(
            first.line_table["target_strength_factor"] != 1.0
        ))
        self.assertFalse(np.array_equal(
            first.line_table["target_strength_factor"],
            other.line_table["target_strength_factor"],
        ))


class TestIndividualShotCalibration(unittest.TestCase):
    def test_calibration_records_signed_noise_without_claiming_dark_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            wavelength = 190.0 + np.arange(21600) / 30.0
            segment = np.searchsorted((365.0, 620.0), wavelength)
            for seed in (3, 4, 5):
                rng = np.random.default_rng(seed)
                baseline = np.asarray((40.0, 120.0, 70.0))[segment]
                intensity = baseline + rng.normal(0.0, 15.0, wavelength.size)
                path = Path(tmp) / f"shot-{seed}.csv"
                np.savetxt(
                    path,
                    np.column_stack((wavelength, intensity)),
                    delimiter=",",
                    header="wavelength,intensity",
                    comments="",
                )
                paths.append(path)
            got = calibrate_individual_shots(paths)
            self.assertEqual(got.n_spectra, 3)
            self.assertTrue(got.response.shot_noise)
            self.assertFalse(got.response.calibrated)
            self.assertAlmostEqual(
                got.segments["VIS"]["baseline_count_median"], 120.0,
                delta=2.0,
            )
            self.assertTrue(any("dark/blank" in text for text in got.limitations))
            artifact = Path(tmp) / "calibration.json"
            got.save(artifact)
            loaded = IndividualShotCalibration.load(artifact)
            self.assertEqual(loaded.n_spectra, 3)
            self.assertEqual(
                loaded.response.background_counts,
                got.response.background_counts,
            )


class TestPeriodicCoverageScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticSpectrumGenerator("db")
        cls.scheduler = PeriodicCoverageScheduler(cls.generator)

    def test_manifest_retains_all_positions_and_marks_supported_cells(self):
        manifest = self.scheduler.manifest()
        self.assertEqual(manifest["cells_total"], 92 * 3 * 10)
        self.assertEqual(manifest["cells_supported"], 87 * 3 * 10)
        self.assertEqual(manifest["cells_training_enabled"], 83 * 3 * 10)
        self.assertEqual(
            manifest["training_excluded_elements"], ["Tc", "Fr", "Ra", "Ac"]
        )
        self.assertEqual(len(manifest["support_mask"]), 92)
        self.assertEqual(len(manifest["training_mask"]), 92)

    def test_focus_abundance_is_exact_and_scene_replays(self):
        cells = [
            cell for cell in self.scheduler.cells()
            if cell.element == "Th" and cell.ion_stage == 2
            and cell.abundance == 1e-6
        ]
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        first = self.scheduler.scene(cell, replicate=4, seed=99)
        second = self.scheduler.scene(cell, replicate=4, seed=99)
        th = self.generator.db.elements.index("Th")
        self.assertEqual(first.target.stage_abundance[th, 1], 1e-6)
        self.assertAlmostEqual(float(first.target.stage_abundance.sum()), 1.0)
        np.testing.assert_array_equal(
            first.target.stage_abundance, second.target.stage_abundance
        )
        np.testing.assert_array_equal(
            first.target.temperature_k, second.target.temperature_k
        )
        self.assertEqual(first.seed, second.seed)

    def test_structurally_unobservable_stage_still_gets_training_cells(self):
        se = [
            cell for cell in self.scheduler.cells()
            if cell.element == "Se" and cell.ion_stage == 3
            and cell.abundance == 1e-4
        ][0]
        self.assertTrue(se.supported)
        self.assertFalse(se.quantitatively_observable)
        scene = self.scheduler.scene(se, seed=2)
        idx = self.generator.db.elements.index("Se")
        self.assertEqual(scene.target.stage_abundance[idx, 2], 1e-4)

    def test_unsupported_cell_cannot_be_synthesized(self):
        pm = [
            cell for cell in self.scheduler.cells()
            if cell.element == "Pm" and cell.ion_stage == 1
            and cell.abundance == 1e-3
        ][0]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.scheduler.scene(pm)

    def test_rare_natural_elements_are_schema_retained_but_training_excluded(self):
        for element in ("Tc", "Fr", "Ra", "Ac"):
            cell = [
                value for value in self.scheduler.cells()
                if value.element == element and value.ion_stage == 1
                and value.abundance == 1e-3
            ][0]
            self.assertTrue(cell.supported)
            self.assertFalse(cell.training_enabled)
            with self.assertRaisesRegex(ValueError, "excluded from training"):
                self.scheduler.scene(cell)


if __name__ == "__main__":
    unittest.main()
