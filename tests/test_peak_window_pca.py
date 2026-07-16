import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alibz.peak_window_pca import (
    PHYSICAL_COORDINATES,
    assign_independent_window_clusters,
    load_basis,
    pairwise_spatial_coherence,
    physical_templates,
    project_window,
    measure_broad_line,
    resolve_basis_path,
)
from scripts.compact_corpus_peak_pca import _compact_legacy_pickle
from scripts.run_mw2_112_peak_window_pca import _window_selection


class TestPeakShapeBasis(unittest.TestCase):
    def _basis(self, root: Path):
        x = np.linspace(-1, 1, 101)
        mean = np.exp(-0.5 * (x / 0.28) ** 2)
        mean = (mean - mean.min()) / (mean.max() - mean.min())
        shift = np.gradient(mean)
        shift /= np.linalg.norm(shift)
        broad = physical_templates(mean)["gaussian_broadening"]
        broad -= np.dot(broad, shift) * shift
        broad /= np.linalg.norm(broad)
        components = np.vstack((shift, broad))
        npz = root / "basis.npz"
        np.savez_compressed(
            npz, components=components, mean_peak=mean,
            explained_variance_ratio=np.array([0.7, 0.2]),
            score_std=np.array([0.2, 0.1]))
        (root / "basis.json").write_text(json.dumps({
            "training": {"half_window_nm": 0.2},
        }))
        return load_basis(npz)

    def test_templates_are_finite_unit_directions(self):
        mean = np.exp(-0.5 * (np.linspace(-1, 1, 101) / 0.3) ** 2)
        templates = physical_templates(mean)
        self.assertEqual(set(templates), set(PHYSICAL_COORDINATES))
        for template in templates.values():
            self.assertTrue(np.all(np.isfinite(template)))
            self.assertAlmostEqual(np.linalg.norm(template), 1.0, places=8)

    def test_projection_recovers_clean_shape_and_positive_area(self):
        with tempfile.TemporaryDirectory() as temp:
            basis = self._basis(Path(temp))
            wavelength = np.linspace(499.7, 500.3, 61)
            x = (wavelength - 500.0) / basis.half_window_nm
            peak = 30 * np.exp(-0.5 * (x / 0.28) ** 2)
            spectrum = 2 + 0.5 * (wavelength - 500) + peak
            result = project_window(wavelength, spectrum, 500.0, basis)
            self.assertIsNotNone(result)
            self.assertGreater(result.reconstruction_r2, 0.95)
            self.assertGreater(result.area, 0)
            self.assertAlmostEqual(result.centroid_offset_nm, 0.0, delta=0.01)

    def test_resolver_accepts_explicit_basis(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "basis.npz"
            path.write_bytes(b"x")
            self.assertEqual(resolve_basis_path(path), path.resolve())

    def test_broad_line_area_and_width_are_recovered(self):
        wavelength = np.linspace(497.5, 502.5, 1501)
        sigma = 0.20
        peak = 40.0 * np.exp(-0.5 * ((wavelength - 500.0) / sigma) ** 2)
        spectrum = 3.0 + 0.2 * (wavelength - 500.0) + peak
        result = measure_broad_line(wavelength, spectrum, 500.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.fwhm_nm, 2.35482 * sigma, delta=0.01)
        self.assertAlmostEqual(
            result.area, 40.0 * sigma * np.sqrt(2 * np.pi), delta=0.03)
        self.assertAlmostEqual(result.peak_offset_nm, 0.0, delta=0.005)


class TestSpatialCoherence(unittest.TestCase):
    def test_coherent_lines_rank_together(self):
        x = np.linspace(0, 4 * np.pi, 200)
        profiles = {"a": np.sin(x), "b": 2 * np.sin(x) + 0.01 * np.cos(x)}
        valid = {key: np.ones(x.size, dtype=bool) for key in profiles}
        medians, pairs = pairwise_spatial_coherence(profiles, valid)
        self.assertGreater(medians["a"], 0.99)
        self.assertGreater(pairs[("a", "b")], 0.99)

    def test_window_clusters_are_independent_and_do_not_chain_merge(self):
        records = [
            {"line_id": "a", "element": "Fe", "ion_stage": 1,
             "center_nm": 200.00},
            {"line_id": "b", "element": "Fe", "ion_stage": 1,
             "center_nm": 200.10},
            {"line_id": "c", "element": "Fe", "ion_stage": 1,
             "center_nm": 200.20},
            {"line_id": "d", "element": "Fe", "ion_stage": 2,
             "center_nm": 200.05},
        ]
        clusters = assign_independent_window_clusters(records, 0.16)
        self.assertEqual(clusters["a"], (1, 2))
        self.assertEqual(clusters["b"], (1, 2))
        self.assertEqual(clusters["c"], (2, 1))
        self.assertEqual(clusters["d"], (1, 1))

    def test_selection_counts_one_clean_representative_per_cluster(self):
        n = 200
        trend = np.linspace(1.0, 2.0, n)
        candidates = []
        for line_id, center, contested, canonical in (
                ("a", 500.00, [], 1),
                ("b", 500.10, [], 0),
                ("c", 501.00, [], 1),
                ("d", 502.00, ["Fe"], 1)):
            candidates.append({
                "line_id": line_id, "element": "Li", "ion_stage": 1,
                "center_nm": center, "wavelength_nm": center,
                "components_nm": [center], "component_weights": [1.0],
                "contested_by": contested, "competitor_lines": [],
                "all_competitor_lines": [], "stage_ambiguous_by": [],
                "reference_snr": 20.0, "rank_in_stage": 1,
                "is_canonical": canonical,
            })
        measurements = {}
        for index, candidate in enumerate(candidates):
            values = {
                "peak_snr": np.full(n, 10.0),
                "reconstruction_r2": np.full(n, 0.9),
                "reconstruction_rmse": np.full(n, 0.05),
                "area_fraction": trend * (index + 1),
            }
            for name in PHYSICAL_COORDINATES:
                values[f"{name}_z"] = np.zeros(n)
            measurements[candidate["line_id"]] = values
        args = SimpleNamespace(
            detection_snr=3.0, minimum_r2=0.0,
            deconfliction_radius_nm=0.16, self_absorption_z=2.0,
            shape_anomaly_z=2.0, minimum_reference_snr=1.0,
            minimum_detection_rate=0.25, minimum_median_r2=0.35,
            minimum_coherence_observations=100,
            minimum_line_coherence=0.25,
        )
        summary, _profiles, _valid, _pairs, stages = _window_selection(
            candidates, measurements, args)
        by_id = {row["line_id"]: row for row in summary}
        self.assertEqual(by_id["a"]["use_for_element"], 1)
        self.assertEqual(by_id["b"]["use_for_element"], 0)
        self.assertEqual(by_id["c"]["use_for_element"], 1)
        self.assertEqual(by_id["d"]["window_accepted"], 0)
        self.assertIn(("Li", 1), stages)


class TestLegacyCompaction(unittest.TestCase):
    def test_huge_tail_is_replaced_without_losing_prior_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.pkl"
            value = {
                "components": np.eye(2),
                "score_stats": {"std": np.ones(2)},
                "peak_metadata": [{"value": index} for index in range(100)],
                "peak_classifications": ["x"] * 100,
            }
            with path.open("wb") as fh:
                pickle.dump(value, fh, protocol=4)
            compact = _compact_legacy_pickle(path, "peak_metadata")
            np.testing.assert_array_equal(compact["components"], np.eye(2))
            self.assertEqual(compact["peak_metadata"], [])
            self.assertNotIn("peak_classifications", compact)


if __name__ == "__main__":
    unittest.main()
