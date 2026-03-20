import os
import glob

import numpy as np
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

from alibz.peaky_finder import PeakyFinder
from alibz.utils.voigt import voigt_width as _voigt_width


class PeakyCorpus:
    """Batch loading, standardization, and peak fitting across a corpus of LIBS spectra.

    Parameters
    ----------
    data_dir : str
        Root directory containing CSV spectrum files (searched recursively).
    wl_min : float, optional
        Minimum wavelength in nm for the common grid.
    wl_max : float, optional
        Maximum wavelength in nm for the common grid.
    wl_step : float, optional
        Wavelength step size in nm for the common grid.
    memmap : bool, optional
        Use numpy memory mapping for large datasets.
    pattern : str, optional
        Glob pattern to filter CSV files.  The default
        ``'**/*AverageSpectrum.csv'`` selects only shot-averaged spectra.
        Set to ``'**/*.csv'`` to include individual shots.
    delimiter : str, optional
        Column delimiter used in the CSV files.
    skip_header : int, optional
        Number of header rows to skip when reading each CSV.
    """

    def __init__(self, data_dir, wl_min=190.0, wl_max=910.0, wl_step=0.01,
                 memmap=True, pattern='**/*AverageSpectrum.csv',
                 delimiter=',', skip_header=1):
        self.data_dir = data_dir
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.wl_step = wl_step
        self.use_memmap = memmap
        self.pattern = pattern
        self.delimiter = delimiter
        self.skip_header = skip_header

        self.common_wavelength = np.arange(wl_min, wl_max + wl_step, wl_step)
        self.n_channels = len(self.common_wavelength)

        self.csv_files = []
        self.raw_data = None
        self.spectra = None
        self.fit_results = []
        self.all_peak_params = []
        self.width_stats = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_corpus(self):
        """Recursively discover and load CSV spectra from ``data_dir``.

        Each CSV is expected to have columns ``wavelength, intensity``
        (matching the Z-300 / SciAps export format).  Files are found
        using ``self.pattern`` and loaded into a memory-mapped or in-RAM
        array of shape ``(num_files, 2, num_features)``.

        Returns
        -------
        ndarray
            The raw data array.
        """
        # Discover files
        self.csv_files = sorted(
            glob.glob(os.path.join(self.data_dir, self.pattern), recursive=True)
        )
        if len(self.csv_files) == 0:
            raise FileNotFoundError(
                f"No files matching '{self.pattern}' found under {self.data_dir}"
            )

        # Peek at the first file to determine the number of spectral channels
        sample = np.loadtxt(
            self.csv_files[0], delimiter=self.delimiter,
            skiprows=self.skip_header, dtype=float,
        )
        n_rows, n_cols = sample.shape
        n_files = len(self.csv_files)

        shape = (n_files, n_cols, n_rows)  # (files, channels, features)

        if self.use_memmap:
            self.raw_data = np.memmap(
                'corpus_raw.dat', dtype='float64', mode='w+', shape=shape,
            )
        else:
            self.raw_data = np.zeros(shape, dtype='float64')

        for i, fpath in enumerate(self.csv_files):
            data = np.loadtxt(
                fpath, delimiter=self.delimiter,
                skiprows=self.skip_header, dtype=float,
            )
            self.raw_data[i, :, :] = data.T  # transpose to (channels, features)

        print(f"Loaded {n_files} spectra from {self.data_dir}")
        return self.raw_data

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def standardize_spectrum(self, wavelength, intensity):
        """Interpolate a single spectrum onto the common wavelength grid.

        Parameters
        ----------
        wavelength : array_like
            Original wavelength axis.
        intensity : array_like
            Original intensity values.

        Returns
        -------
        ndarray
            Intensity evaluated on ``self.common_wavelength``.
        """
        interpolator = interp1d(
            wavelength, intensity,
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        return interpolator(self.common_wavelength)

    def standardize_all(self):
        """Standardize every spectrum in the corpus to the common grid.

        Stores the result in ``self.spectra`` with shape
        ``(num_spectra, len(common_wavelength))``.  When *memmap* is enabled
        the array is backed by a temporary memory-mapped file.

        Returns
        -------
        ndarray
            The standardized spectra array.
        """
        if self.raw_data is None:
            raise RuntimeError("Call load_corpus() before standardize_all().")

        n_spectra = self.raw_data.shape[0]

        if self.use_memmap:
            self.spectra = np.memmap(
                'corpus_standardized.dat',
                dtype='float64',
                mode='w+',
                shape=(n_spectra, self.n_channels),
            )
        else:
            self.spectra = np.zeros((n_spectra, self.n_channels), dtype='float64')

        for i in range(n_spectra):
            wavelength = self.raw_data[i, 0, :]
            intensity = self.raw_data[i, 1, :]
            self.spectra[i, :] = self.standardize_spectrum(wavelength, intensity)
            if (i + 1) % 100 == 0 or i == n_spectra - 1:
                print(f"  standardized {i + 1}/{n_spectra}")

        return self.spectra

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_all_spectra(self, n_sigma=0, subtract_background=True):
        """Fit peaks for every standardized spectrum.

        Uses :meth:`PeakyFinder.fit_spectrum` with ``fast=True`` internally.
        Results are accumulated in ``self.fit_results`` (list of *fit_dict*)
        and ``self.all_peak_params`` (list of parameter arrays).

        Parameters
        ----------
        n_sigma : float, optional
            Detection threshold passed to :meth:`PeakyFinder.fit_spectrum`.
        subtract_background : bool, optional
            Whether to estimate and remove the background.

        Returns
        -------
        list
            The list of fit result dictionaries.
        """
        if self.spectra is None:
            raise RuntimeError("Call standardize_all() before fit_all_spectra().")

        # Create a lightweight PeakyFinder without triggering Data loading
        finder = PeakyFinder.__new__(PeakyFinder)

        n_spectra = self.spectra.shape[0]
        self.fit_results = []
        self.all_peak_params = []

        for i in range(n_spectra):
            x = self.common_wavelength
            y = self.spectra[i]

            try:
                fit_dict = finder.fit_spectrum(
                    x, y,
                    n_sigma=n_sigma,
                    subtract_background=subtract_background,
                    plot=False,
                )
                self.fit_results.append(fit_dict)

                params = np.array(list(fit_dict['spectrum_dictionary'].values()))
                self.all_peak_params.append(params)
            except Exception as e:
                print(f"Spectrum {i} fitting failed: {e}")
                self.fit_results.append(None)
                self.all_peak_params.append(np.empty((0, 4)))

            if (i + 1) % 10 == 0 or i == n_spectra - 1:
                print(f"  fitted {i + 1}/{n_spectra}")

        return self.fit_results

    # ------------------------------------------------------------------
    # Width statistics
    # ------------------------------------------------------------------

    def peak_width_statistics(self):
        """Compute FWHM statistics across all fitted peaks in the corpus.

        Uses the Voigt FWHM approximation from
        :meth:`PeakyFinder.voigt_width`.

        Returns
        -------
        dict
            Dictionary with keys ``'all_widths'``, ``'min'``, ``'max'``,
            ``'mean'``, ``'median'``, ``'std'``, and ``'per_spectrum_median'``.
        """
        if not self.all_peak_params:
            raise RuntimeError("Call fit_all_spectra() before peak_width_statistics().")

        all_widths = []
        per_spectrum_median = []

        for params in self.all_peak_params:
            if params.size == 0:
                per_spectrum_median.append(np.nan)
                continue
            sigmas = params[:, 2]
            gammas = params[:, 3]
            widths = _voigt_width(sigmas, gammas)
            all_widths.extend(widths.tolist())
            per_spectrum_median.append(float(np.median(widths)))

        all_widths = np.array(all_widths)

        self.width_stats = {
            'all_widths': all_widths,
            'min': float(np.min(all_widths)) if len(all_widths) > 0 else np.nan,
            'max': float(np.max(all_widths)) if len(all_widths) > 0 else np.nan,
            'mean': float(np.mean(all_widths)) if len(all_widths) > 0 else np.nan,
            'median': float(np.median(all_widths)) if len(all_widths) > 0 else np.nan,
            'std': float(np.std(all_widths)) if len(all_widths) > 0 else np.nan,
            'per_spectrum_median': np.array(per_spectrum_median),
        }
        return self.width_stats

    def detect_width_modes(self, max_components=5):
        """Detect modes in the FWHM distribution using Gaussian Mixture Models.

        Fits GMMs with 1 to ``max_components`` components and selects the
        best by BIC.  Results are added to ``self.width_stats`` under the
        keys ``'n_modes'``, ``'mode_means'``, ``'mode_stds'``,
        ``'mode_weights'``, and ``'smallest_mode_mean'``.

        Parameters
        ----------
        max_components : int, optional
            Maximum number of Gaussian components to try.

        Returns
        -------
        dict
            The updated ``self.width_stats``.
        """
        if self.width_stats is None:
            self.peak_width_statistics()

        widths = self.width_stats['all_widths']
        if len(widths) < 10:
            self.width_stats.update({
                'n_modes': 1,
                'mode_means': np.array([self.width_stats['median']]),
                'mode_stds': np.array([self.width_stats['std']]),
                'mode_weights': np.array([1.0]),
                'smallest_mode_mean': self.width_stats['median'],
            })
            return self.width_stats

        X = widths.reshape(-1, 1)
        best_bic = np.inf
        best_gmm = None

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=0, max_iter=200)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        means = best_gmm.means_.ravel()
        stds = np.sqrt(best_gmm.covariances_.ravel())
        weights = best_gmm.weights_.ravel()
        order = np.argsort(means)

        self.width_stats.update({
            'n_modes': len(means),
            'mode_means': means[order],
            'mode_stds': stds[order],
            'mode_weights': weights[order],
            'smallest_mode_mean': float(means[order[0]]),
            'bic_values': {k: GaussianMixture(n_components=k, random_state=0, max_iter=200).fit(X).bic(X)
                           for k in range(1, max_components + 1)},
        })

        n = self.width_stats['n_modes']
        print(f"  Detected {n} mode{'s' if n > 1 else ''} in FWHM distribution:")
        for i in range(n):
            print(f"    mode {i+1}: mean={self.width_stats['mode_means'][i]:.4f} nm, "
                  f"std={self.width_stats['mode_stds'][i]:.4f} nm, "
                  f"weight={self.width_stats['mode_weights'][i]:.1%}")

        return self.width_stats

    def get_median_peak_width(self):
        """Return the corpus-wide median peak FWHM in nm.

        Returns
        -------
        float
        """
        if self.width_stats is None:
            self.peak_width_statistics()
        return self.width_stats['median']

    def get_smallest_mode_width(self):
        """Return the mean FWHM of the narrowest mode in the width distribution.

        Calls :meth:`detect_width_modes` if not already run.

        Returns
        -------
        float
        """
        if self.width_stats is None or 'n_modes' not in self.width_stats:
            self.detect_width_modes()
        return self.width_stats['smallest_mode_mean']
