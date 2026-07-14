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
            idx._last_A = np.array([[1.0]])
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

    def test_solve_at_is_fixed_state_and_skips_optimiser(self):
        # solve_at must produce a full FitResult at a GIVEN (T, ne, sig, gam)
        # WITHOUT calling the Bayesian optimiser (the basin-safe re-solve
        # used by the iterative deepening loop).
        idx = PeakyIndexer(np.array([[2.0, 500.0, 0.05, 0.05]]))
        idx.line_table = SimpleNamespace(
            n_species=1, n_lines=1,
            species=[SimpleNamespace(element="Fe", ion=1)],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0], ([0], [0])), shape=(1, 1))
        idx._evidence_top_k = 0            # _prune_and_refit returns early
        idx._last_gamma_inst = 0.0
        idx._sa_tau_scale = 0.0
        idx._sa_fit = False
        idx._sa_doublet_info = {}
        idx._sa_log_tau_bounds = (-2.0, 3.0)
        idx._last_sa_converged = None
        idx._rebuild_overlap = lambda _sigma, _gamma: None
        idx._width_cost = lambda _log_ne: 0.0

        def fake_solve(_temperature, _log_ne):
            idx._last_A = np.array([[1.0]])
            return np.array([2.0]), 0.0

        idx._solve_concentrations = fake_solve

        with patch("skopt.gp_minimize") as mock_gp:
            result = idx.solve_at(7000.0, 16.5, 0.05, 0.05, verbose=False)

        mock_gp.assert_not_called()
        self.assertEqual(result.temperature, 7000.0)
        self.assertEqual(result.ne, 16.5)
        self.assertTrue(result.convergence_info["fixed_plasma_state"])
        np.testing.assert_allclose(result.predicted, [2.0])
        np.testing.assert_allclose(result.residuals, [0.0])

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

    def test_concentrations_are_physical_not_column_normalised(self):
        """With disjoint single-line species of very different design scale,
        the solution must satisfy A @ c = y in UN-normalised (physical)
        units, not return the column-normalised NNLS coefficients."""
        idx = PeakyIndexer(
            np.array(
                [
                    [2.0, 500.0, 0.05, 0.05],
                    [10.0, 600.0, 0.05, 0.05],
                ]
            )
        )
        idx.line_table = _FakeLineTable(
            [
                Species("Ca", 1, 20, 1.0, 0, 1),
                Species("Fe", 1, 26, 1.0, 1, 2),
            ],
            wavelengths=[500.0, 600.0],
            species_idx=[0, 1],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 1], [0, 1])),
            shape=(2, 2),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.array([4.0, 0.5])

        concentrations, _cost = idx._solve_concentrations(10_000.0, 17.0)

        # A = [[4, 0], [0, 0.5]], y = [2, 10]  =>  c = [0.5, 20.0]
        np.testing.assert_allclose(concentrations, [0.5, 20.0], rtol=1e-10)

    def _aggregation_fixture(self):
        idx = PeakyIndexer(
            np.array(
                [
                    [6.0, 500.0, 0.05, 0.05],
                    [10.0, 510.0, 0.05, 0.05],
                ]
            )
        )
        idx.line_table = _FakeLineTable(
            [
                Species("Fe", 1, 26, 1.0, 0, 1),
                Species("Fe", 2, 26, 1.0, 1, 2),
                Species("Ca", 1, 20, 1.0, 2, 3),
            ],
            wavelengths=[500.0, 500.0, 510.0],
            species_idx=[0, 1, 2],
        )
        return idx

    def test_element_aggregation_reduces_to_precision_weighted_average_when_disjoint(self):
        idx = self._aggregation_fixture()
        # Fe I and Fe II on disjoint peaks; per-stage estimates from the
        # data are 2 and 1 (the passed concentrations are the exact NNLS
        # solution of this system).
        A = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        idx._obs_amp = np.array([2.0, 3.0])
        concentrations = np.array([2.0, 1.0, 0.0])

        element_c, _element_f, disagreement = idx._aggregate_elements(concentrations, A)

        # Precision weights ||A[:,s]||^2 = 1 and 9: (1*2 + 9*1)/10 = 1.1
        # (a sum of stages would give 3).
        self.assertAlmostEqual(element_c["Fe"], 1.1)
        self.assertAlmostEqual(disagreement["Fe"], (2.0 - 1.0) / 3.0)

    def test_noise_relative_gate_drops_phantom_keeps_trace_under_dominant_line(self):
        """A near-invisible species (huge concentration, tiny emission -->
        ill-conditioned) is dropped by the noise-relative detection gate,
        while a genuine low-abundance trace survives even when a single
        dominant line would sink a fraction-of-max floor."""
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx.line_table = _FakeLineTable(
            [Species("Si", 1, 14, 1.0, 0, 1),   # dominant major
             Species("Cs", 1, 55, 1.0, 1, 2),   # real low-conc trace
             Species("Bi", 1, 83, 1.0, 2, 3)],  # phantom: near-zero column
            wavelengths=[500.0, 501.0, 502.0], species_idx=[0, 1, 2])
        # design: Si a huge line (2000), Cs a small real line (40), Bi a
        # near-zero column so its density is unmeasurable
        A = np.array([[2000.0, 0.0, 0.0],
                      [0.0, 40.0, 0.0],
                      [0.0, 0.0, 1e-4]])
        idx._obs_amp = np.array([2000.0, 40.0, 5.0])
        # NNLS-style vertex: Bi gets a huge concentration for ~zero emission
        conc = np.array([1.0, 1.0, 5.0e4])
        amp_sigma = np.array([12.0, 12.0, 12.0])   # area noise ~12 counts

        ec, ef, _ = idx._aggregate_elements(conc, A, amp_sigma=amp_sigma)
        # phantom Bi (best line 5 counts, 0.4 sigma) is gone; the real trace
        # Cs (best line 40 counts, 3.3 sigma) survives despite the 2000-count
        # dominant Si line that a fraction-of-max floor would key on
        self.assertNotIn("Bi", ef)
        self.assertIn("Cs", ef)
        self.assertIn("Si", ef)
        self.assertLess(idx._last_detection_z["Bi"], 3.0)
        self.assertGreaterEqual(idx._last_detection_z["Cs"], 3.0)

    def test_no_emission_gate_without_noise(self):
        """A bare indexer (no per-peak noise) applies no emission gate, so
        a species with a real if faint column is kept — the noise-relative
        detectability gate is opt-in via amp_sigma, and must not over-
        exclude a genuine minor element under a dominant line."""
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx.line_table = _FakeLineTable(
            [Species("Si", 1, 14, 1.0, 0, 1),
             Species("Al", 1, 13, 1.0, 1, 2)],
            wavelengths=[500.0, 501.0], species_idx=[0, 1])
        A = np.array([[2000.0, 0.0], [0.0, 30.0]])
        idx._obs_amp = np.array([2000.0, 30.0])
        _ec, ef, _ = idx._aggregate_elements(np.array([1.0, 1.0]), A)
        self.assertIn("Al", ef)      # faint but real under a dominant Si
        self.assertIn("Si", ef)

    def test_element_aggregation_is_stage_order_invariant_under_blends(self):
        """When both Fe stages feed the same peak, NNLS returns an arbitrary
        vertex (one stage exactly zero).  The tied re-solve must recover the
        same element density from either vertex — averaging raw coefficients
        would give 0.4 or 1.2 depending on species order."""
        idx = self._aggregation_fixture()
        A = np.array(
            [
                [1.0, 3.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        # LTE-consistent truth: n_Fe = 1 (peak 0 amplitude 1*1 + 3*1 = 4)...
        idx._obs_amp = np.array([4.0, 10.0])
        for vertex in (np.array([4.0, 0.0, 5.0]), np.array([0.0, 4.0 / 3.0, 5.0])):
            element_c, element_f, _ = idx._aggregate_elements(vertex, A)
            self.assertAlmostEqual(element_c["Fe"], 1.0)
            self.assertAlmostEqual(element_c["Ca"], 5.0)
            self.assertAlmostEqual(element_f["Fe"], 1.0 / 6.0)

    def test_stage_disagreement_ignores_solver_excluded_stages(self):
        """A stage whose design column is below the solver's activity
        threshold was never estimated; its placeholder zero must not fire
        the non-LTE diagnostic."""
        idx = self._aggregation_fixture()
        A = np.array(
            [
                [5.0, 1e-32, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        idx._obs_amp = np.array([5.0, 10.0])
        concentrations = np.array([1.0, 0.0, 5.0])

        element_c, _element_f, disagreement = idx._aggregate_elements(concentrations, A)

        self.assertAlmostEqual(element_c["Fe"], 1.0)
        self.assertNotIn("Fe", disagreement)
        self.assertNotIn("Ca", disagreement)  # single stage: no diagnostic

    def test_solve_concentrations_keeps_rows_aligned_when_evidence_and_pseudo_cofire(self):
        """Regression (unit level): evidence-penalty rows and pseudo rows must
        both extend y_aug; rebuilding y_aug from _obs_amp at the pseudo step
        hands nnls mismatched dimensions."""
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx._evidence_top_k = 4
        idx._evidence_strong_line_rel_threshold = 0.5
        idx._evidence_presence_threshold = 0.25
        idx._evidence_missing_mass_weight = 0.25
        idx._evidence_missing_count_weight = 0.1
        idx._pseudo_obs_weight = 1.0
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
        idx.pseudo_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 1], [1, 3])),
            shape=(2, 6),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.ones(6, dtype=float)

        concentrations, cost = idx._solve_concentrations(10_000.0, 17.0)

        self.assertEqual(concentrations.shape, (2,))
        self.assertTrue(np.all(np.isfinite(concentrations)))
        self.assertTrue(np.all(concentrations >= 0.0))
        self.assertTrue(np.isfinite(cost))

    def test_degenerate_solve_cost_is_sum_of_squares(self):
        """When every species column is numerically inactive the returned
        cost must be the SSE of the all-zero model (squared units), not the
        L2 norm — otherwise degenerate (T, ne, sigma, gamma) trials look
        spuriously good to the outer optimiser."""
        idx = PeakyIndexer(
            np.array(
                [
                    [2.0, 500.0, 0.05, 0.05],
                    [3.0, 510.0, 0.05, 0.05],
                ]
            )
        )
        idx.line_table = _FakeLineTable(
            [Species("Ca", 1, 20, 1.0, 0, 1)],
            wavelengths=[500.0],
            species_idx=[0],
        )
        idx.peak_line_map = scipy.sparse.csr_matrix(
            ([1.0], ([0], [0])),
            shape=(2, 1),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.zeros(1, dtype=float)

        concentrations, cost = idx._solve_concentrations(10_000.0, 17.0)

        np.testing.assert_array_equal(concentrations, [0.0])
        self.assertAlmostEqual(cost, 2.0 ** 2 + 3.0 ** 2)

    def test_returned_cost_matches_full_inner_objective(self):
        """The cost handed to the outer optimiser must equal the functional
        the NNLS minimised: data SSE + evidence ridge + pseudo cost."""
        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.05]]))
        idx._evidence_top_k = 4
        idx._evidence_strong_line_rel_threshold = 0.5
        idx._evidence_presence_threshold = 0.25
        idx._evidence_missing_mass_weight = 0.25
        idx._evidence_missing_count_weight = 0.1
        idx._pseudo_obs_weight = 1.0
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
        idx.pseudo_line_map = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 1], [1, 3])),
            shape=(2, 6),
        )
        idx._line_weights = lambda _temperature, _log_ne: np.ones(6, dtype=float)

        concentrations, cost = idx._solve_concentrations(10_000.0, 17.0)

        data_sse = float(np.sum((idx._obs_amp - idx._last_A @ concentrations) ** 2))
        evidence = idx._last_species_evidence
        c_norm = concentrations * idx._last_col_max
        ridge = float(
            np.sum(
                idx._evidence_missing_mass_weight
                * np.maximum(evidence["missing_mass"], 0.0)
                * c_norm ** 2
            )
        ) + float(
            np.sum(
                idx._evidence_missing_count_weight
                * np.maximum(evidence["strong_missing_count"], 0.0)
                * c_norm ** 2
            )
        )
        self.assertGreater(ridge, 0.0)
        self.assertGreater(idx._last_pseudo_cost, 0.0)
        self.assertAlmostEqual(cost, data_sse + ridge + idx._last_pseudo_cost, places=10)

    def test_fit_handles_empty_species_table(self):
        idx = PeakyIndexer(np.array([[2.0, 500.0, 0.05, 0.05]]))
        idx.line_table = _FakeLineTable([], wavelengths=[], species_idx=[])
        idx.peak_line_map = scipy.sparse.csr_matrix((1, 0))
        idx._line_weights = lambda _temperature, _log_ne: np.empty(0, dtype=float)
        idx._rebuild_overlap = lambda _sigma, _gamma: None

        with patch(
            "skopt.gp_minimize",
            return_value=SimpleNamespace(
                x=[10_000.0, 17.0, 0.05, 0.05],
                func_vals=np.array([4.0]),
            ),
        ):
            result = idx.fit(n_calls=3, verbose=False)

        self.assertEqual(result.concentrations.size, 0)
        self.assertIsNone(result.peak_assignments[0]["element"])
        self.assertEqual(result.element_fractions, {})

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
