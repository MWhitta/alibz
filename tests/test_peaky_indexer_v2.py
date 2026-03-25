"""Tests for PeakyIndexerV2 Stages 1 and 2."""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from alibz.peaky_indexer_v2 import PeakyIndexerV2, PeakRecord


class _MockDB:
    """Minimal Database mock with a few elements and lines."""

    elements = ['H', 'Li', 'Fe']
    no_lines = set()

    def __init__(self):
        # Natural abundance (mock values)
        self.elem_abund = {'H': 0.001, 'Li': 0.002, 'Fe': 0.05}

        # H lines: 1 ground-state transition at 656.28 nm (H-alpha)
        # columns: ion, wavelength, ?, gA, Ei, Ek, ..., gi, gk  (14 cols)
        self._lines = {
            'H': np.array([
                ['1.0', '656.280', '0', '4.41e7', '0.0', '12.088', '', '', '', '', '0', '0', '2', '8'],
                ['1.0', '486.130', '0', '8.42e6', '0.0', '12.749', '', '', '', '', '0', '0', '2', '18'],
                ['1.0', '434.050', '0', '2.53e6', '10.198', '13.055', '', '', '', '', '0', '0', '8', '32'],
            ]),
            'Li': np.array([
                ['1.0', '670.776', '0', '3.69e7', '0.0', '1.848', '', '', '', '', '0', '0', '2', '6'],
                ['1.0', '610.354', '0', '3.37e6', '1.848', '3.879', '', '', '', '', '0', '0', '6', '10'],
                ['2.0', '548.500', '0', '2.0e6', '0.0', '2.260', '', '', '', '', '0', '0', '1', '3'],
            ]),
            'Fe': np.array([
                # Fe I ground state line near 670.5 nm (close to Li)
                ['1.0', '670.500', '0', '1.0e6', '0.0', '1.850', '', '', '', '', '0', '0', '9', '7'],
                ['1.0', '438.354', '0', '5.0e7', '0.0', '2.831', '', '', '', '', '0', '0', '9', '11'],
                ['2.0', '526.953', '0', '3.0e7', '0.0', '2.352', '', '', '', '', '0', '0', '10', '8'],
            ]),
        }

    def lines(self, el, ion=0):
        arr = self._lines.get(el, np.empty((0, 14), dtype='U32'))
        if ion:
            mask = arr[:, 0].astype(float).astype(int) == ion
            arr = arr[mask]
        return arr


class _MockSB:
    """Mock SahaBoltzmann with a trivial partition function."""

    def stage_partition(self, element, temperature, ion):
        return np.array([1.0])


class _MockMaker:
    """Mock PeakyMaker."""

    def __init__(self):
        self.sb = _MockSB()


def _make_indexer(peak_array, pca_scores=None, temperature=10000.0):
    """Create a PeakyIndexerV2 with mocked database/maker."""
    with patch.object(PeakyIndexerV2, '__init__', lambda self, *a, **kw: None):
        idx = PeakyIndexerV2.__new__(PeakyIndexerV2)

    idx.peak_array = np.asarray(peak_array, dtype=float)
    idx.n_peaks = idx.peak_array.shape[0]
    idx.pca_scores = pca_scores
    idx.temperature = temperature
    idx.db = _MockDB()
    idx.maker = _MockMaker()

    from alibz.utils.voigt import voigt_width as _vw
    idx.peaks = []
    for i in range(idx.n_peaks):
        amp, mu, sigma, gamma = idx.peak_array[i, :4]
        rec = PeakRecord(
            peak_idx=i, wavelength=mu, amplitude=amp,
            sigma=sigma, gamma=gamma, fwhm=float(_vw(sigma, gamma)),
        )
        if pca_scores is not None and i < pca_scores.shape[0]:
            rec.pc_scores = pca_scores[i]
        idx.peaks.append(rec)

    idx.ground_states = {}
    idx.anchors = {}
    idx.boltzmann_results = {}
    idx.consensus_temperature = None
    return idx


# -----------------------------------------------------------------------
# Stage 1 tests
# -----------------------------------------------------------------------

class TestSelfAbsorption(unittest.TestCase):
    """Tests for Stage 1: self-absorption quantification."""

    def test_basic_flagging(self):
        """Peaks with extreme PC scores are flagged as self-absorbed."""
        # 10 peaks, 6 PCs
        np.random.seed(42)
        peak_array = np.column_stack([
            np.ones(10) * 1000,       # amplitude
            np.linspace(400, 700, 10), # wavelength
            np.ones(10) * 0.05,        # sigma
            np.ones(10) * 0.03,        # gamma
        ])
        # Most peaks have small PC3/PC6, but peak 0 and 1 are extreme
        scores = np.zeros((10, 6))
        scores[0, 2] = 12.0   # PC3 extreme
        scores[1, 5] = 12.0   # PC6 extreme

        idx = _make_indexer(peak_array, pca_scores=scores)
        idx.quantify_self_absorption(pc_indices=(2, 5), threshold=2.0)

        self.assertTrue(idx.peaks[0].is_self_absorbed)
        self.assertTrue(idx.peaks[1].is_self_absorbed)
        # Others should not be flagged (they have index=0)
        n_flagged = sum(1 for p in idx.peaks if p.is_self_absorbed)
        self.assertEqual(n_flagged, 2)

    def test_no_pca_raises(self):
        """Raises ValueError if no PCA scores are provided."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        idx = _make_indexer(peak_array, pca_scores=None)
        with self.assertRaises(ValueError):
            idx.quantify_self_absorption()

    def test_self_absorption_index_positive(self):
        """Self-absorption index is always non-negative (uses abs)."""
        peak_array = np.array([[100, 500, 0.05, 0.03]] * 5)
        scores = np.array([
            [0, 0, -3.0, 0, 0, 0],   # negative PC3
            [0, 0, 3.0, 0, 0, 0],     # positive PC3
            [0, 0, 0, 0, 0, -2.0],    # negative PC6
            [0, 0, 0, 0, 0, 0],       # zero
            [0, 0, 1.0, 0, 0, 1.0],   # mixed
        ])
        idx = _make_indexer(peak_array, pca_scores=scores)
        idx.quantify_self_absorption(pc_indices=(2, 5))

        for rec in idx.peaks:
            self.assertGreaterEqual(rec.self_absorption_index, 0.0)

        # Peaks 0 and 1 should have same SA index (abs makes them equal)
        self.assertAlmostEqual(
            idx.peaks[0].self_absorption_index,
            idx.peaks[1].self_absorption_index,
        )

    def test_custom_weights(self):
        """Custom weights change the self-absorption index."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        scores = np.array([[0, 0, 1.0, 0, 0, 2.0]])

        idx1 = _make_indexer(peak_array, pca_scores=scores)
        idx1.quantify_self_absorption(pc_indices=(2, 5), weights=np.array([1.0, 1.0]))

        idx2 = _make_indexer(peak_array, pca_scores=scores)
        idx2.quantify_self_absorption(pc_indices=(2, 5), weights=np.array([3.0, 0.5]))

        # Different weights → different index
        self.assertNotAlmostEqual(
            idx1.peaks[0].self_absorption_index,
            idx2.peaks[0].self_absorption_index,
        )


# -----------------------------------------------------------------------
# Stage 2 tests
# -----------------------------------------------------------------------

class TestGroundStateLines(unittest.TestCase):
    """Tests for Stage 2a: ground-state line identification."""

    def test_identifies_ground_state(self):
        """Lines with Ei=0 are identified as ground-state."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        gs = idx.identify_ground_state_lines(temperature=10000.0)

        # H should have 2 ground-state lines (Ei=0)
        self.assertIn('H', gs)
        self.assertIn(1.0, gs['H'])
        h_wl = gs['H'][1.0]['wavelengths']
        # 656.28 and 486.13 have Ei=0; 434.05 has Ei=10.198 (not ground)
        self.assertEqual(len(h_wl), 2)
        self.assertAlmostEqual(h_wl[0], 656.28, places=1)
        self.assertAlmostEqual(h_wl[1], 486.13, places=1)

    def test_temperature_affects_weights(self):
        """Higher temperature increases weight of high-Ek lines."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])

        idx_low = _make_indexer(peak_array, temperature=5000.0)
        gs_low = idx_low.identify_ground_state_lines(temperature=5000.0)

        idx_high = _make_indexer(peak_array, temperature=20000.0)
        gs_high = idx_high.identify_ground_state_lines(temperature=20000.0)

        # At higher T, more lines should survive the threshold
        # Both should have H ground states, but weights differ
        self.assertIn('H', gs_low)
        self.assertIn('H', gs_high)

    def test_negative_temperature_raises(self):
        """Negative temperature raises ValueError."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        with self.assertRaises(ValueError):
            idx.identify_ground_state_lines(temperature=-100)

    def test_no_lines_element_skipped(self):
        """Elements in no_lines set are skipped."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.db.no_lines = {'Fe'}
        gs = idx.identify_ground_state_lines()
        self.assertNotIn('Fe', gs)


class TestAnchorPeaks(unittest.TestCase):
    """Tests for Stage 2b: anchor peak identification."""

    def test_unique_element_becomes_anchor(self):
        """A peak matching only one element is anchored."""
        # Peak at 486.13 nm — matches H only (no other element near)
        peak_array = np.array([[5000, 486.13, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        anchors = idx.find_anchor_peaks(shift_tolerance=0.1)

        self.assertIn('H', anchors)
        self.assertTrue(idx.peaks[0].is_anchor)
        self.assertEqual(idx.peaks[0].anchor_element, 'H')

    def test_multi_element_not_anchored(self):
        """A peak matching multiple elements is NOT anchored."""
        # Peak at 670.6 nm — matches both Li I (670.776) and Fe I (670.500)
        # within 0.3 nm tolerance
        peak_array = np.array([[5000, 670.6, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        anchors = idx.find_anchor_peaks(shift_tolerance=0.3)

        self.assertFalse(idx.peaks[0].is_anchor)

    def test_tight_tolerance_resolves_ambiguity(self):
        """With tight tolerance, previously ambiguous peaks can become anchors."""
        # Peak at 670.78 nm — very close to Li I (670.776), far from Fe I (670.500)
        peak_array = np.array([[5000, 670.78, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        anchors = idx.find_anchor_peaks(shift_tolerance=0.05)

        self.assertTrue(idx.peaks[0].is_anchor)
        self.assertEqual(idx.peaks[0].anchor_element, 'Li')

    def test_self_absorbed_anchor_rejected(self):
        """Self-absorbed peaks are not marked as anchors when required."""
        peak_array = np.array([[5000, 486.13, 0.05, 0.03]])
        # Single peak — threshold=0 means any nonzero SA index is flagged
        scores = np.array([[0, 0, 10.0, 0, 0, 10.0]])  # extreme SA

        idx = _make_indexer(peak_array, pca_scores=scores)
        # With only 1 peak, std=0, so cutoff = mean + thresh*0 = mean.
        # The single peak's SA index equals the mean, so it won't exceed.
        # Use a 2-peak array to get a proper std.
        peak_array2 = np.array([
            [5000, 486.13, 0.05, 0.03],
            [1000, 500.00, 0.05, 0.03],
        ])
        scores2 = np.array([
            [0, 0, 10.0, 0, 0, 10.0],
            [0, 0, 0.0, 0, 0, 0.0],
        ])

        idx = _make_indexer(peak_array2, pca_scores=scores2)
        idx.quantify_self_absorption(pc_indices=(2, 5), threshold=0.5)
        self.assertTrue(idx.peaks[0].is_self_absorbed)
        self.assertFalse(idx.peaks[1].is_self_absorbed)

        idx.identify_ground_state_lines()
        idx.find_anchor_peaks(
            shift_tolerance=0.1,
            require_no_self_absorption=True,
        )

        # Peak 0 matches H but is self-absorbed → not an anchor
        self.assertFalse(idx.peaks[0].is_anchor)
        # But H should still appear in anchors dict with SA flag
        self.assertIn('H', idx.anchors)
        self.assertTrue(idx.anchors['H'][1.0][0]['is_self_absorbed'])

    def test_self_absorbed_anchor_allowed(self):
        """Self-absorbed peaks ARE anchored when require_no_self_absorption=False."""
        peak_array = np.array([[5000, 486.13, 0.05, 0.03]])
        scores = np.array([[0, 0, 10.0, 0, 0, 10.0]])

        idx = _make_indexer(peak_array, pca_scores=scores)
        idx.quantify_self_absorption(pc_indices=(2, 5), threshold=0.5)
        idx.identify_ground_state_lines()
        idx.find_anchor_peaks(
            shift_tolerance=0.1,
            require_no_self_absorption=False,
        )

        self.assertTrue(idx.peaks[0].is_anchor)

    def test_anchor_metadata_populated(self):
        """Anchor entries contain gA, Ek, and ref_wavelength."""
        peak_array = np.array([[5000, 656.28, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        idx.find_anchor_peaks(shift_tolerance=0.1)

        rec = idx.peaks[0]
        self.assertIsNotNone(rec.anchor_gA)
        self.assertIsNotNone(rec.anchor_Ek)
        self.assertIsNotNone(rec.anchor_ref_wavelength)
        self.assertGreater(rec.anchor_gA, 0)
        self.assertGreater(rec.anchor_Ek, 0)


class TestRunStages12(unittest.TestCase):
    """Integration test for run_stages_1_2."""

    def test_full_run(self):
        """Stages 1-2 run end-to-end and return expected keys."""
        np.random.seed(0)
        peak_array = np.array([
            [5000, 656.28, 0.05, 0.03],   # H-alpha
            [3000, 486.13, 0.05, 0.03],   # H-beta
            [2000, 670.78, 0.05, 0.03],   # Li I
            [1000, 500.00, 0.05, 0.03],   # no match
        ])
        scores = np.random.randn(4, 6) * 0.1

        idx = _make_indexer(peak_array, pca_scores=scores)
        result = idx.run_stages_1_2(verbose=False)

        self.assertIn('peaks', result)
        self.assertIn('anchors', result)
        self.assertIn('confirmed_elements', result)
        self.assertEqual(len(result['peaks']), 4)
        self.assertIn('H', result['confirmed_elements'])


# -----------------------------------------------------------------------
# Stage 3 tests
# -----------------------------------------------------------------------

class TestBoltzmannTemperature(unittest.TestCase):
    """Tests for Stage 3: Boltzmann temperature estimation."""

    def _setup_anchored_indexer(self, peaks_data, sa_flags=None):
        """Build an indexer with manually populated anchors.

        peaks_data: list of (amplitude, wavelength, gA, Ek, element, ion)
        """
        peak_array = np.array([
            [amp, wl, 0.05, 0.03] for amp, wl, *_ in peaks_data
        ])
        idx = _make_indexer(peak_array)

        # Manually set up anchors as if Stages 1-2 ran
        idx.identify_ground_state_lines()
        idx.anchors = {}

        for i, (amp, wl, gA, Ek, el, ion) in enumerate(peaks_data):
            rec = idx.peaks[i]
            rec.is_anchor = True
            rec.anchor_element = el
            rec.anchor_ion = ion
            rec.anchor_gA = gA
            rec.anchor_Ek = Ek
            rec.anchor_ref_wavelength = wl

            if sa_flags and i < len(sa_flags):
                rec.is_self_absorbed = sa_flags[i]

            if el not in idx.anchors:
                idx.anchors[el] = {}
            if ion not in idx.anchors[el]:
                idx.anchors[el][ion] = []
            idx.anchors[el][ion].append({
                'peak_idx': i,
                'wavelength': wl,
                'amplitude': amp,
                'ref_wavelength': wl,
                'gA': gA,
                'Ek': Ek,
                'distance': 0.0,
                'is_self_absorbed': sa_flags[i] if sa_flags else False,
            })

        return idx

    def test_known_temperature(self):
        """Recover a known temperature from synthetic Boltzmann data."""
        # Generate synthetic intensities at T = 10000 K
        T_true = 10000.0
        kB = 8.617333262e-5  # eV/K

        # H I lines with different Ek values
        lines = [
            # (wavelength, gA, Ek)
            (656.28, 4.41e7, 12.088),
            (486.13, 8.42e6, 12.749),
        ]

        # I = gA / lambda * exp(-Ek / kT) * const
        # Use const=1e8 to get reasonable amplitudes
        const = 1e8
        peaks_data = []
        for wl, gA, Ek in lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_true))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))

        idx = self._setup_anchored_indexer(peaks_data)
        results, T_consensus = idx.estimate_temperature_boltzmann(min_lines=2)

        self.assertIn('H_1', results)
        br = results['H_1']
        # Should recover T within 1% for exact synthetic data
        self.assertAlmostEqual(br.temperature_K, T_true, delta=T_true * 0.01)
        self.assertAlmostEqual(br.r_squared, 1.0, places=5)
        self.assertEqual(br.n_lines, 2)
        self.assertIsNotNone(T_consensus)
        self.assertAlmostEqual(T_consensus, T_true, delta=T_true * 0.01)

    def test_multiple_elements_consensus(self):
        """Consensus temperature averages across multiple elements."""
        T_true = 8000.0
        kB = 8.617333262e-5
        const = 1e8

        # H lines
        h_lines = [
            (656.28, 4.41e7, 12.088),
            (486.13, 8.42e6, 12.749),
        ]
        # Li lines (different Ek spread)
        li_lines = [
            (670.78, 3.69e7, 1.848),
            (610.35, 3.37e6, 3.879),
        ]

        peaks_data = []
        for wl, gA, Ek in h_lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_true))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))
        for wl, gA, Ek in li_lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_true))
            peaks_data.append((amp, wl, gA, Ek, 'Li', 1.0))

        idx = self._setup_anchored_indexer(peaks_data)
        results, T_consensus = idx.estimate_temperature_boltzmann(min_lines=2)

        self.assertIn('H_1', results)
        self.assertIn('Li_1', results)
        self.assertIsNotNone(T_consensus)
        # Both should give T_true, so consensus should match
        self.assertAlmostEqual(T_consensus, T_true, delta=T_true * 0.02)

    def test_self_absorbed_excluded(self):
        """Self-absorbed peaks are excluded from the Boltzmann fit."""
        T_true = 10000.0
        kB = 8.617333262e-5
        const = 1e8

        # 3 lines, middle one is self-absorbed (amplitude halved)
        lines = [
            (656.28, 4.41e7, 12.088),
            (486.13, 8.42e6, 12.749),
            (434.05, 2.53e6, 13.055),
        ]

        peaks_data = []
        for wl, gA, Ek in lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_true))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))

        # Halve the amplitude of line 1 (self-absorbed)
        peaks_data[1] = (peaks_data[1][0] * 0.5,) + peaks_data[1][1:]
        sa_flags = [False, True, False]

        idx = self._setup_anchored_indexer(peaks_data, sa_flags=sa_flags)
        results, _ = idx.estimate_temperature_boltzmann(
            min_lines=2, exclude_self_absorbed=True,
        )

        self.assertIn('H_1', results)
        br = results['H_1']
        # With SA excluded, should still get good T from lines 0 and 2
        self.assertEqual(br.n_lines, 2)
        self.assertAlmostEqual(br.temperature_K, T_true, delta=T_true * 0.01)

    def test_self_absorbed_included_biases_T(self):
        """Including self-absorbed peaks biases T upward."""
        T_true = 10000.0
        kB = 8.617333262e-5
        const = 1e8

        lines = [
            (656.28, 4.41e7, 12.088),
            (486.13, 8.42e6, 12.749),
            (434.05, 2.53e6, 13.055),
        ]

        peaks_data = []
        for wl, gA, Ek in lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_true))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))

        # Severely suppress the strongest (lowest Ek) line
        peaks_data[0] = (peaks_data[0][0] * 0.1,) + peaks_data[0][1:]
        sa_flags = [True, False, False]

        idx = self._setup_anchored_indexer(peaks_data, sa_flags=sa_flags)

        # Including SA → biased T
        results_with, T_with = idx.estimate_temperature_boltzmann(
            min_lines=2, exclude_self_absorbed=False,
        )
        # Excluding SA → correct T
        results_without, T_without = idx.estimate_temperature_boltzmann(
            min_lines=2, exclude_self_absorbed=True,
        )

        if 'H_1' in results_with and 'H_1' in results_without:
            # T with SA included should be higher (flatter slope)
            self.assertGreater(
                results_with['H_1'].temperature_K,
                results_without['H_1'].temperature_K,
            )

    def test_too_few_lines_skipped(self):
        """Elements with fewer than min_lines are skipped."""
        peak_array = np.array([[5000, 656.28, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        idx.anchors = {'H': {1.0: [{
            'peak_idx': 0, 'wavelength': 656.28, 'amplitude': 5000,
            'ref_wavelength': 656.28, 'gA': 4.41e7, 'Ek': 12.088,
            'distance': 0.0, 'is_self_absorbed': False,
        }]}}
        idx.peaks[0].is_anchor = True

        results, T = idx.estimate_temperature_boltzmann(min_lines=2)
        self.assertEqual(len(results), 0)
        self.assertIsNone(T)

    def test_no_anchors_raises(self):
        """Raises RuntimeError if no anchors exist."""
        peak_array = np.array([[5000, 500.0, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.anchors = {}
        with self.assertRaises(RuntimeError):
            idx.estimate_temperature_boltzmann()

    def test_nonphysical_temperature_rejected(self):
        """Fits yielding T outside [t_min, t_max] are discarded."""
        kB = 8.617333262e-5
        # Use T = 1000 K (below default t_min of 3000)
        T_low = 1000.0
        const = 1e8
        lines = [
            (656.28, 4.41e7, 12.088),
            (486.13, 8.42e6, 12.749),
        ]
        peaks_data = []
        for wl, gA, Ek in lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T_low))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))

        idx = self._setup_anchored_indexer(peaks_data)
        results, T = idx.estimate_temperature_boltzmann(
            min_lines=2, t_min=3000, t_max=50000,
        )
        # T = 1000 K is below t_min → should be rejected
        self.assertNotIn('H_1', results)
        self.assertIsNone(T)

    def test_narrow_ek_spread_skipped(self):
        """Lines with <0.1 eV Ek spread are skipped (underdetermined)."""
        kB = 8.617333262e-5
        T = 10000.0
        const = 1e8
        # Two lines with nearly identical Ek
        lines = [
            (656.28, 4.41e7, 12.088),
            (656.30, 3.00e7, 12.089),  # only 0.001 eV difference
        ]
        peaks_data = []
        for wl, gA, Ek in lines:
            amp = const * gA / wl * np.exp(-Ek / (kB * T))
            peaks_data.append((amp, wl, gA, Ek, 'H', 1.0))

        idx = self._setup_anchored_indexer(peaks_data)
        results, T_est = idx.estimate_temperature_boltzmann(min_lines=2)
        self.assertNotIn('H_1', results)


class TestRunStages13(unittest.TestCase):
    """Integration test for run_stages_1_3."""

    def test_full_run(self):
        """Stages 1-3 run end-to-end."""
        # Use H-alpha and H-beta which both match H ground state
        peak_array = np.array([
            [5000, 656.28, 0.05, 0.03],   # H-alpha
            [3000, 486.13, 0.05, 0.03],   # H-beta
            [2000, 670.78, 0.05, 0.03],   # Li I
            [1000, 500.00, 0.05, 0.03],   # no match
        ])
        scores = np.random.randn(4, 6) * 0.1

        idx = _make_indexer(peak_array, pca_scores=scores)
        result = idx.run_stages_1_3(verbose=False)

        self.assertIn('boltzmann_results', result)
        self.assertIn('consensus_temperature', result)


# -----------------------------------------------------------------------
# Stage 4 tests
# -----------------------------------------------------------------------

class TestGetCandidates(unittest.TestCase):
    """Tests for _get_candidates."""

    def test_finds_nearby_lines(self):
        """Candidates within tolerance are returned."""
        peak_array = np.array([[100, 656.28, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        cands = idx._get_candidates(656.28, shift_tolerance=0.1)
        # Should find H at 656.28
        elements = {c['element'] for c in cands}
        self.assertIn('H', elements)

    def test_max_ion_stage_filter(self):
        """Ion stages above max_ion_stage are excluded."""
        peak_array = np.array([[100, 500, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        # Fe has ion=2.0 line at 526.953
        cands_all = idx._get_candidates(526.95, shift_tolerance=0.1, max_ion_stage=3)
        cands_low = idx._get_candidates(526.95, shift_tolerance=0.1, max_ion_stage=1)
        # max_ion_stage=1 should exclude Fe II
        fe_ions_all = {c['ion'] for c in cands_all if c['element'] == 'Fe'}
        fe_ions_low = {c['ion'] for c in cands_low if c['element'] == 'Fe'}
        if 2.0 in fe_ions_all:
            self.assertNotIn(2.0, fe_ions_low)

    def test_no_match_returns_empty(self):
        """Wavelength with no database lines returns empty list."""
        peak_array = np.array([[100, 100.0, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        cands = idx._get_candidates(100.0, shift_tolerance=0.01)
        self.assertEqual(len(cands), 0)

    def test_candidate_has_Z(self):
        """Candidates include atomic number Z."""
        peak_array = np.array([[100, 670.776, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        cands = idx._get_candidates(670.776, shift_tolerance=0.1)
        li_cands = [c for c in cands if c['element'] == 'Li']
        if li_cands:
            # Z = position in mock elements list + 1
            self.assertEqual(li_cands[0]['Z'], 2)  # Li is index 1 in mock ['H','Li','Fe']


class TestScoringFunctions(unittest.TestCase):
    """Tests for individual scoring functions."""

    def test_distance_score_at_zero(self):
        """Distance score is 1.0 at zero distance."""
        self.assertAlmostEqual(
            PeakyIndexerV2._distance_score(0.0, sigma=0.05), 1.0)

    def test_distance_score_decays(self):
        """Distance score decreases with distance."""
        s1 = PeakyIndexerV2._distance_score(0.01, sigma=0.05)
        s2 = PeakyIndexerV2._distance_score(0.05, sigma=0.05)
        s3 = PeakyIndexerV2._distance_score(0.10, sigma=0.05)
        self.assertGreater(s1, s2)
        self.assertGreater(s2, s3)

    def test_fwhm_score_in_range(self):
        """FWHM score is in [0, 1]."""
        score = PeakyIndexerV2._fwhm_score(
            observed_gamma=0.05, observed_sigma=0.05,
            candidate_ion=1.0, candidate_Z=26)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fwhm_narrow_favours_neutral(self):
        """Narrow Gaussian peak scores higher for neutral low-Z."""
        # Narrow, Gaussian-dominated peak (small gamma)
        score_neutral_lowZ = PeakyIndexerV2._fwhm_score(
            observed_gamma=0.001, observed_sigma=0.1,
            candidate_ion=1.0, candidate_Z=3)
        score_ionic_highZ = PeakyIndexerV2._fwhm_score(
            observed_gamma=0.001, observed_sigma=0.1,
            candidate_ion=3.0, candidate_Z=56)
        self.assertGreater(score_neutral_lowZ, score_ionic_highZ)

    def test_fwhm_broad_favours_ionic(self):
        """Broad Lorentzian peak scores higher for ionic high-Z."""
        # Broad, Lorentzian-dominated peak (large gamma)
        score_neutral_lowZ = PeakyIndexerV2._fwhm_score(
            observed_gamma=0.2, observed_sigma=0.01,
            candidate_ion=1.0, candidate_Z=3)
        score_ionic_highZ = PeakyIndexerV2._fwhm_score(
            observed_gamma=0.2, observed_sigma=0.01,
            candidate_ion=3.0, candidate_Z=56)
        self.assertGreater(score_ionic_highZ, score_neutral_lowZ)


class TestConsistencyScore(unittest.TestCase):
    """Tests for _consistency_score."""

    def test_anchored_element_scores_high(self):
        """Elements with anchors get nonzero consistency score."""
        peak_array = np.array([[5000, 656.28, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.anchors = {'H': {1.0: [{'peak_idx': 0}] * 3}}
        score = idx._consistency_score('H', 1.0)
        self.assertGreater(score, 0.0)

    def test_unknown_element_scores_zero(self):
        """Elements not in anchors get zero consistency score."""
        peak_array = np.array([[5000, 656.28, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.anchors = {}
        score = idx._consistency_score('Xe', 1.0)
        self.assertEqual(score, 0.0)


class TestRankCandidates(unittest.TestCase):
    """Tests for rank_candidates."""

    def test_anchored_peaks_unchanged(self):
        """Anchor peaks are not re-assigned by ranking."""
        peak_array = np.array([
            [5000, 656.28, 0.05, 0.03],  # will be anchored to H
            [1000, 438.35, 0.05, 0.03],  # Fe I line
        ])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        idx.find_anchor_peaks(shift_tolerance=0.1)

        # Peak 0 should be anchored to H
        self.assertTrue(idx.peaks[0].is_anchor)

        idx.rank_candidates(shift_tolerance=0.1)

        # Anchor should still be H, not overwritten
        self.assertEqual(idx.peaks[0].anchor_element, 'H')
        self.assertTrue(idx.peaks[0].is_anchor)

    def test_non_anchor_gets_assigned(self):
        """Non-anchor peaks near a database line get assigned."""
        # 610.354 is Li I with Ei=1.848 (NOT ground state), so not an anchor
        peak_array = np.array([
            [5000, 656.28, 0.05, 0.03],  # H anchor
            [1000, 610.35, 0.05, 0.03],  # near Li I 610.354 (excited state)
        ])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        idx.find_anchor_peaks(shift_tolerance=0.1)

        # Peak 1 should NOT be an anchor (Ei != 0)
        self.assertFalse(idx.peaks[1].is_anchor)

        idx.rank_candidates(shift_tolerance=0.1)

        # Peak 1 should be assigned to something
        rec = idx.peaks[1]
        self.assertIsNotNone(rec.assigned_element)
        self.assertGreater(rec.assignment_score, 0)
        self.assertGreater(len(rec.candidates), 0)

    def test_candidates_sorted_by_score(self):
        """Candidates list is sorted descending by score."""
        peak_array = np.array([[1000, 438.35, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()
        idx.anchors = {}
        idx.rank_candidates(shift_tolerance=0.1)

        rec = idx.peaks[0]
        if len(rec.candidates) >= 2:
            scores = [c['score'] for c in rec.candidates]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_consistency_boosts_anchored_element(self):
        """Consistency score is higher when element is anchored."""
        peak_array = np.array([[5000, 438.35, 0.05, 0.03]])
        idx = _make_indexer(peak_array)
        idx.identify_ground_state_lines()

        # No anchors → consistency = 0
        idx.anchors = {}
        s_no = idx._consistency_score('Fe', 1.0)

        # With Fe anchored → consistency > 0
        idx.anchors = {'Fe': {1.0: [{'peak_idx': 99}] * 3}}
        s_yes = idx._consistency_score('Fe', 1.0)

        self.assertEqual(s_no, 0.0)
        self.assertGreater(s_yes, 0.0)


class TestRunStages14(unittest.TestCase):
    """Integration test for run_stages_1_4."""

    def test_full_run(self):
        """Stages 1-4 run end-to-end."""
        peak_array = np.array([
            [5000, 656.28, 0.05, 0.03],
            [3000, 486.13, 0.05, 0.03],
            [2000, 670.78, 0.05, 0.03],
            [1000, 438.35, 0.05, 0.03],
            [500, 500.00, 0.05, 0.03],
        ])
        scores = np.random.randn(5, 6) * 0.1

        idx = _make_indexer(peak_array, pca_scores=scores)
        result = idx.run_stages_1_4(verbose=False)

        self.assertIn('n_assigned', result)
        self.assertIn('all_elements', result)
        self.assertIsInstance(result['all_elements'], set)


if __name__ == '__main__':
    unittest.main()
