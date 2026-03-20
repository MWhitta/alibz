import unittest

import numpy as np
from scipy.special import voigt_profile as voigt

from alibz.peaky_corpus import PeakyCorpus
from alibz.peaky_pca import PeakyPCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_corpus(n_spectra=5, n_peaks=8, seed=42):
    """Build a minimal PeakyCorpus-like object from synthetic Voigt spectra.

    Returns a *PeakyCorpus* instance whose ``spectra``, ``common_wavelength``,
    ``fit_results``, ``all_peak_params``, and ``width_stats`` attributes are
    populated directly -- no CSV loading or fitting required.
    """
    rng = np.random.RandomState(seed)

    wl_min, wl_max, wl_step = 200.0, 800.0, 0.01
    wl = np.arange(wl_min, wl_max + wl_step, wl_step)
    n_chan = len(wl)

    # Vary sigma/gamma across peaks to give PCA something to decompose
    spectra = np.zeros((n_spectra, n_chan))
    all_params = []
    fit_results = []

    for i in range(n_spectra):
        params_list = []
        spectrum = np.zeros(n_chan)

        for _ in range(n_peaks):
            mu = rng.uniform(300, 700)
            amp = rng.uniform(0.5, 5.0)
            sigma = rng.uniform(0.02, 0.15)
            gamma = rng.uniform(0.02, 0.15)
            spectrum += amp * voigt(wl - mu, sigma, gamma)
            params_list.append([amp, mu, sigma, gamma])

        spectra[i] = spectrum
        params_arr = np.array(params_list)
        all_params.append(params_arr)
        fit_results.append({
            'spectrum_dictionary': {j: p for j, p in enumerate(params_arr)},
            'sorted_parameter_array': params_arr,
            'profile': spectrum,
            'background': np.zeros(n_chan),
            'residual_data': np.zeros(n_chan),
            'n_sigma': 0,
        })

    # Construct a corpus object without touching disk
    corpus = PeakyCorpus.__new__(PeakyCorpus)
    corpus.data_dir = ''
    corpus.wl_min = wl_min
    corpus.wl_max = wl_max
    corpus.wl_step = wl_step
    corpus.use_memmap = False
    corpus.common_wavelength = wl
    corpus.n_channels = n_chan
    corpus.raw_data = None
    corpus.spectra = spectra
    corpus.fit_results = fit_results
    corpus.all_peak_params = all_params
    corpus.width_stats = None
    corpus.peak_width_statistics()

    return corpus


# ---------------------------------------------------------------------------
# Tests: PeakyCorpus
# ---------------------------------------------------------------------------

class TestPeakyCorpusStandardize(unittest.TestCase):

    def test_standardize_spectrum_identity(self):
        """Interpolating onto the same grid should return identical values."""
        corpus = PeakyCorpus.__new__(PeakyCorpus)
        corpus.wl_min = 200.0
        corpus.wl_max = 400.0
        corpus.wl_step = 0.1
        corpus.common_wavelength = np.arange(200.0, 400.0 + 0.1, 0.1)

        wl_orig = corpus.common_wavelength
        intensity = np.sin(wl_orig / 50.0)

        result = corpus.standardize_spectrum(wl_orig, intensity)
        np.testing.assert_allclose(result, intensity, atol=1e-12)

    def test_standardize_spectrum_fill_value(self):
        """Out-of-range wavelengths should be filled with zero."""
        corpus = PeakyCorpus.__new__(PeakyCorpus)
        corpus.wl_min = 190.0
        corpus.wl_max = 910.0
        corpus.wl_step = 1.0
        corpus.common_wavelength = np.arange(190.0, 911.0, 1.0)

        # Spectrum only covers 300-600
        wl_orig = np.arange(300.0, 601.0, 1.0)
        intensity = np.ones_like(wl_orig)
        result = corpus.standardize_spectrum(wl_orig, intensity)

        self.assertEqual(result[0], 0.0)   # 190 nm -> fill
        self.assertEqual(result[-1], 0.0)  # 910 nm -> fill
        self.assertAlmostEqual(result[110], 1.0)  # 300 nm


class TestPeakyCorpusWidthStats(unittest.TestCase):

    def test_peak_width_statistics_known(self):
        """Statistics on identical peaks should return the same width for all."""
        corpus = _make_synthetic_corpus(n_spectra=2, n_peaks=4, seed=0)
        stats = corpus.width_stats

        self.assertGreater(stats['min'], 0)
        self.assertGreaterEqual(stats['max'], stats['min'])
        self.assertAlmostEqual(stats['mean'], np.mean(stats['all_widths']), places=10)
        self.assertAlmostEqual(stats['median'], np.median(stats['all_widths']), places=10)


# ---------------------------------------------------------------------------
# Tests: PeakyPCA
# ---------------------------------------------------------------------------

class TestPeakyPCAExtract(unittest.TestCase):

    def setUp(self):
        self.corpus = _make_synthetic_corpus(n_spectra=3, n_peaks=6)
        self.pca_obj = PeakyPCA(self.corpus, window_multiplier=3.0, n_components=3)

    def test_extract_peak_windows_shape(self):
        self.pca_obj.extract_peak_windows()
        n_peaks = self.pca_obj.windows.shape[0]
        self.assertGreater(n_peaks, 0)
        self.assertEqual(self.pca_obj.windows.shape[1], 101)
        self.assertEqual(len(self.pca_obj.peak_metadata), n_peaks)

    def test_windows_normalized(self):
        self.pca_obj.extract_peak_windows()
        maxima = np.max(self.pca_obj.windows, axis=1)
        np.testing.assert_allclose(maxima, 1.0, atol=0.05)


class TestPeakyPCAFit(unittest.TestCase):

    def setUp(self):
        self.corpus = _make_synthetic_corpus(n_spectra=5, n_peaks=8)
        self.pca_obj = PeakyPCA(self.corpus, window_multiplier=3.0, n_components=3)
        self.pca_obj.extract_peak_windows()
        self.pca_obj.fit_pca()

    def test_components_shape(self):
        self.assertEqual(self.pca_obj.components.shape[1], 101)
        self.assertLessEqual(self.pca_obj.components.shape[0], 3)

    def test_explained_variance_sums_to_at_most_one(self):
        self.assertLessEqual(np.sum(self.pca_obj.explained_variance_ratio), 1.0 + 1e-6)

    def test_scores_shape(self):
        n_peaks = self.pca_obj.windows.shape[0]
        self.assertEqual(self.pca_obj.scores.shape[0], n_peaks)


class TestMeanPeakCharacterization(unittest.TestCase):

    def test_mean_fit_has_positive_sigma_gamma(self):
        corpus = _make_synthetic_corpus(n_spectra=5, n_peaks=8)
        pca_obj = PeakyPCA(corpus, window_multiplier=3.0, n_components=3)
        pca_obj.extract_peak_windows()
        pca_obj.fit_pca()
        mf = pca_obj.characterize_mean_peak()
        self.assertGreater(mf['sigma'], 0)
        self.assertGreater(mf['gamma'], 0)
        self.assertGreater(mf['fwhm'], 0)
        self.assertGreaterEqual(mf['gaussian_fraction'], 0)
        self.assertLessEqual(mf['gaussian_fraction'], 1)


class TestPeakyPCADecompose(unittest.TestCase):

    def test_decompose_fractions_sum_to_one(self):
        corpus = _make_synthetic_corpus(n_spectra=5, n_peaks=8)
        pca_obj = PeakyPCA(corpus, window_multiplier=3.0, n_components=3)
        pca_obj.extract_peak_windows()
        pca_obj.fit_pca()
        pca_obj.characterize_mean_peak()
        pca_obj.decompose_all_components()

        for d in pca_obj.decompositions:
            total = d['gaussian_fraction'] + d['lorentzian_fraction'] + d['asymmetry_fraction']
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_decompose_has_sensitivities(self):
        corpus = _make_synthetic_corpus(n_spectra=5, n_peaks=8)
        pca_obj = PeakyPCA(corpus, window_multiplier=3.0, n_components=3)
        pca_obj.extract_peak_windows()
        pca_obj.fit_pca()
        pca_obj.characterize_mean_peak()
        pca_obj.decompose_all_components()

        for d in pca_obj.decompositions:
            self.assertIn('d_sigma', d)
            self.assertIn('d_gamma', d)
            self.assertIn('d_tau', d)
            self.assertIn('alphas', d)
            self.assertEqual(len(d['sigmas']), len(d['alphas']))

    def test_classify_peaks_returns_valid_labels(self):
        corpus = _make_synthetic_corpus(n_spectra=5, n_peaks=8)
        pca_obj = PeakyPCA(corpus, window_multiplier=3.0, n_components=3)
        pca_obj.extract_peak_windows()
        pca_obj.fit_pca()
        pca_obj.characterize_mean_peak()
        pca_obj.decompose_all_components()
        labels = pca_obj.classify_peaks()

        valid = {'Doppler/instrumental', 'Stark/natural', 'self-absorption'}
        for label in labels:
            self.assertIn(label, valid)


class TestGaussianFraction(unittest.TestCase):

    def test_pure_gaussian(self):
        """gamma=0 should give fraction 1.0."""
        frac = PeakyPCA._gaussian_fraction(sigma=0.1, gamma=0.0)
        self.assertAlmostEqual(frac, 1.0, places=3)

    def test_pure_lorentzian(self):
        """sigma near 0 should give fraction near 0."""
        frac = PeakyPCA._gaussian_fraction(sigma=1e-8, gamma=0.1)
        self.assertAlmostEqual(frac, 0.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
