import unittest
from types import SimpleNamespace
from unittest.mock import patch

import alibz
import numpy as np
import scipy.sparse

from alibz import PeakyIndexer, PeakyIndexerV3
from alibz.peaky_indexer_v3 import Species


class _FakeLineTable:

    def __init__(self, species, wavelengths, species_idx):
        self.species = list(species)
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.gA = np.ones_like(self.wavelengths)
        self.Ei = np.zeros_like(self.wavelengths)
        self.Ek = np.zeros_like(self.wavelengths)
        self.species_idx = np.asarray(species_idx, dtype=np.int32)
        self.n_lines = self.wavelengths.size
        self.n_species = len(self.species)

    def filter_species(self, keep_mask):
        keep_mask = np.asarray(keep_mask, dtype=bool)
        species = []
        wavelengths = []
        species_idx = []

        for old_idx, keep in enumerate(keep_mask):
            if not keep:
                continue
            sp = self.species[old_idx]
            span = slice(sp.line_start, sp.line_end)
            start = len(wavelengths)
            wavelengths.extend(self.wavelengths[span].tolist())
            end = len(wavelengths)
            species.append(
                Species(
                    element=sp.element,
                    ion=sp.ion,
                    Z=sp.Z,
                    abundance=sp.abundance,
                    line_start=start,
                    line_end=end,
                )
            )
            species_idx.extend([len(species) - 1] * (end - start))

        self.species = species
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.gA = np.ones_like(self.wavelengths)
        self.Ei = np.zeros_like(self.wavelengths)
        self.Ek = np.zeros_like(self.wavelengths)
        self.species_idx = np.asarray(species_idx, dtype=np.int32)
        self.n_lines = self.wavelengths.size
        self.n_species = len(self.species)


class TestPeakyIndexerPublicApi(unittest.TestCase):

    def test_peaky_fitter_removed_from_public_api(self):
        self.assertFalse(hasattr(alibz, "PeakyFitter"))

    def test_public_alias_points_to_v3(self):
        self.assertIs(PeakyIndexer, PeakyIndexerV3)

        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.03]]))
        self.assertIsInstance(idx, PeakyIndexerV3)
        self.assertEqual(idx.n_peaks, 1)
        self.assertAlmostEqual(idx.T_init, 10_000.0)

    def test_rebuild_overlap_respects_shift_tolerance(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx.line_table = SimpleNamespace(wavelengths=np.array([500.15]), n_lines=1)
        idx.peak_line_map = scipy.sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))

        idx._shift_tolerance = 0.1
        idx._rebuild_overlap(0.05, 0.05)
        self.assertEqual(idx.peak_line_map.nnz, 0)

        idx._shift_tolerance = 0.2
        idx._rebuild_overlap(0.05, 0.05)
        self.assertGreater(idx.peak_line_map.nnz, 0)

    def test_empty_peak_array_returns_empty_result(self):
        idx = PeakyIndexer(np.empty((0, 4)))
        result = idx.run(verbose=False, n_calls=10)

        self.assertEqual(result.temperature, idx.T_init)
        self.assertEqual(result.ne, idx.ne_init)
        self.assertEqual(result.observed.size, 0)
        self.assertEqual(result.predicted.size, 0)
        self.assertEqual(result.species, [])
        self.assertEqual(result.convergence_info["status"], "empty_peak_table")

    def test_fit_supports_small_n_calls(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx.line_table = SimpleNamespace(
            n_species=1,
            n_lines=1,
            species=[SimpleNamespace(element="Fe", ion=1)],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))

        def fake_solve(_temperature, _log_ne):
            idx._last_A_norm = np.array([[1.0]])
            return np.array([1.0]), 0.0

        idx._solve_concentrations = fake_solve
        idx._rebuild_overlap = lambda _sigma, _gamma: None

        with patch(
            "skopt.gp_minimize",
            return_value=SimpleNamespace(
                x=[idx.T_init, idx.ne_init, 0.05, 0.05],
                func_vals=np.array([0.0]),
            ),
        ) as mock_gp:
            result = idx.fit(n_calls=3, verbose=False)

        self.assertEqual(mock_gp.call_args.kwargs["n_initial_points"], 3)
        np.testing.assert_allclose(result.predicted, [1.0])

    def test_initial_strength_prefilter_drops_negligible_species(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx._shift_tolerance = 0.1
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 1),
                Species("Xe", 1, 54, 1.0, 1, 2),
            ],
            wavelengths=[500.0, 500.02],
            species_idx=[0, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 1])),
            shape=(1, 2),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.array([1.0, 1e-6])

        idx._prefilter_species_by_initial_strength(1e-3)

        self.assertEqual(idx.line_table.n_species, 1)
        self.assertEqual(idx.line_table.species[0].element, "Ca")
        self.assertEqual(idx.peak_line_map.shape, (1, 1))

    def test_pseudo_observations_penalize_missing_strong_lines(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 1),
                Species("Xe", 1, 54, 1.0, 1, 3),
            ],
            wavelengths=[500.0, 500.0, 510.0],
            species_idx=[0, 1, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([0.8, 1.0], ([0, 0], [0, 1])),
            shape=(1, 3),
        )
        idx.pseudo_line_map = scipy.sparse.csr_matrix(
            ([1.0], ([0], [2])),
            shape=(1, 3),
        )
        idx._pseudo_obs_weight = 1.0
        idx._line_weights = lambda _temperature, _log_ne: np.ones(3, dtype=float)

        concentrations, _cost = idx._solve_concentrations(10_000.0, 17.0)

        self.assertGreater(concentrations[0], concentrations[1])

    def test_pseudo_species_weights_scale_with_missing_line_mass(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx._shift_tolerance = 0.1
        idx._pseudo_obs_weight = 1.0
        idx._pseudo_line_rel_threshold = 0.25
        idx._pseudo_max_lines_per_species = 2
        idx.line_table = _FakeLineTable(
            [
                Species("A", 1, 1, 1.0, 0, 3),
                Species("B", 1, 2, 1.0, 3, 8),
            ],
            wavelengths=[500.0, 510.0, 520.0, 500.0, 510.0, 520.0, 530.0, 540.0],
            species_idx=[0, 0, 0, 1, 1, 1, 1, 1],
        )
        idx._line_weights = lambda _temperature, _log_ne: np.ones(8, dtype=float)

        idx._select_pseudo_wavelengths()

        self.assertEqual(idx.pseudo_line_map.shape[0], 4)
        self.assertAlmostEqual(idx._pseudo_species_weights[0], 1.0)
        self.assertGreater(idx._pseudo_species_weights[1], idx._pseudo_species_weights[0])

    def test_line_evidence_prefilter_drops_species_with_many_missing_lines(self):
        idx = PeakyIndexer(
            np.array(
                [
                    [1.0, 500.0, 0.05, 0.05],
                    [0.9, 501.0, 0.05, 0.05],
                ]
            )
        )
        idx._shift_tolerance = 0.1
        idx._evidence_top_k = 4
        idx._evidence_strong_line_rel_threshold = 0.5
        idx._evidence_presence_threshold = 0.25
        idx._evidence_min_coverage = 0.5
        idx._evidence_min_supported_lines = 2
        idx._evidence_max_missing_lines = 1
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 2),
                Species("Nd", 2, 60, 1.0, 2, 6),
            ],
            wavelengths=[500.0, 501.0, 500.0, 510.0, 520.0, 530.0],
            species_idx=[0, 0, 1, 1, 1, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0, 1.0], ([0, 1, 0], [0, 1, 2])),
            shape=(2, 6),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.ones(
            idx.line_table.n_lines,
            dtype=float,
        )

        idx._prefilter_species_by_line_evidence(10_000.0, 17.0)

        self.assertEqual(idx.line_table.n_species, 1)
        self.assertEqual(idx.line_table.species[0].element, "Ca")
        self.assertEqual(idx.peak_line_map.shape, (2, 2))

    def test_species_missing_line_penalties_reduce_sparse_surrogate_weight(self):
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx._evidence_top_k = 4
        idx._evidence_strong_line_rel_threshold = 0.5
        idx._evidence_presence_threshold = 0.25
        idx._evidence_missing_mass_weight = 0.25
        idx._evidence_missing_count_weight = 0.1
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 2),
                Species("Nd", 2, 60, 1.0, 2, 6),
            ],
            wavelengths=[500.0, 510.0, 500.0, 520.0, 530.0, 540.0],
            species_idx=[0, 0, 1, 1, 1, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 2])),
            shape=(1, 6),
        )
        idx.pseudo_line_map = scipy.sparse.csr_matrix((0, 6))
        idx._pseudo_obs_weight = 0.0
        idx._line_weights = lambda _temperature, _log_ne: np.ones(
            idx.line_table.n_lines,
            dtype=float,
        )

        concentrations, _cost = idx._solve_concentrations(10_000.0, 17.0)

        self.assertGreater(concentrations[0], concentrations[1])

    def test_postfit_prune_refit_removes_bad_active_species(self):
        idx = PeakyIndexer(
            np.array(
                [
                    [1.0, 500.0, 0.05, 0.05],
                    [0.9, 501.0, 0.05, 0.05],
                ]
            )
        )
        idx._shift_tolerance = 0.1
        idx._evidence_top_k = 4
        idx._evidence_strong_line_rel_threshold = 0.5
        idx._evidence_presence_threshold = 0.25
        idx._evidence_min_coverage = 0.5
        idx._evidence_min_supported_lines = 2
        idx._evidence_max_missing_lines = 1
        idx._evidence_min_net = 0.0
        idx._evidence_max_refits = 2
        idx._evidence_missing_mass_weight = 0.25
        idx._evidence_missing_count_weight = 0.1
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 2),
                Species("Nd", 2, 60, 1.0, 2, 6),
            ],
            wavelengths=[500.0, 501.0, 500.0, 510.0, 520.0, 530.0],
            species_idx=[0, 0, 1, 1, 1, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0, 1.0], ([0, 1, 0], [0, 1, 2])),
            shape=(2, 6),
        )
        idx._pseudo_obs_weight = 0.0
        idx._line_weights = lambda _temperature, _log_ne: np.ones(
            idx.line_table.n_lines,
            dtype=float,
        )

        concentrations, _cost = idx._prune_and_refit(
            10_000.0,
            17.0,
            0.05,
            0.05,
            np.array([1.0, 1.0]),
            1.0,
        )

        self.assertEqual(idx.line_table.n_species, 1)
        self.assertEqual(idx.line_table.species[0].element, "Ca")
        self.assertEqual(concentrations.shape, (1,))
        self.assertGreater(concentrations[0], 0.0)


if __name__ == "__main__":
    unittest.main()
