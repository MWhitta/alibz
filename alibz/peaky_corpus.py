import os
import glob
import multiprocessing as mp

import numpy as np
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

from alibz.peaky_finder import PeakyFinder
from alibz.utils.voigt import voigt_width as _voigt_width
from alibz.gpu import gpu_available

if gpu_available:
    from alibz.gpu import batch_interpolate_gpu


# ------------------------------------------------------------------
# Module-level worker for multiprocessing (must be picklable)
# ------------------------------------------------------------------

class _FitTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _FitTimeout("spectrum fit timed out")


def _worker_init():
    """Pool initializer — disable GPU in forked worker processes.

    CUDA contexts cannot survive ``fork()``.  We patch all modules that
    cache the ``gpu_available`` flag so no worker attempts GPU calls.
    """
    import alibz.gpu as _gpu_mod
    _gpu_mod.gpu_available = False
    _gpu_mod.cp = None
    # Patch cached copies in downstream modules
    import alibz.utils.voigt as _voigt_mod
    _voigt_mod.gpu_available = False


def _fit_one_spectrum(args):
    """Fit peaks for a single spectrum.  Called by Pool.imap_unordered."""
    import signal
    idx, x, y, n_sigma, subtract_background, timeout = args
    finder = PeakyFinder.__new__(PeakyFinder)

    old_handler = None
    if timeout and timeout > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        fit_dict = finder.fit_spectrum(
            x, y,
            n_sigma=n_sigma,
            subtract_background=subtract_background,
            plot=False,
            skip_profile=True,
        )
        params = finder._parameter_array(fit_dict['spectrum_dictionary'])
        return idx, fit_dict, params, None
    except _FitTimeout:
        return idx, None, np.empty((0, 4)), 'timeout'
    except Exception as e:
        return idx, None, np.empty((0, 4)), str(e)
    finally:
        if timeout and timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


class PeakyCorpus:
    """Batch loading, standardization, and peak fitting across a corpus of LIBS spectra.

    Parameters
    ----------
    data_dir : str or list of str
        Root directory (or directories) containing CSV spectrum files
        (searched recursively).  When a list is given, files from every
        directory are combined into a single corpus.
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
    use_gpu : bool or None, optional
        Force GPU (True), force CPU (False), or auto-detect (None).
    """

    def __init__(self, data_dir, wl_min=190.0, wl_max=910.0, wl_step=0.01,
                 memmap=True, pattern='**/*AverageSpectrum.csv',
                 delimiter=',', skip_header=1, use_gpu=None):
        if isinstance(data_dir, (list, tuple)):
            self.data_dirs = list(data_dir)
            self.data_dir = data_dir[0]
        else:
            self.data_dirs = [data_dir]
            self.data_dir = data_dir
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.wl_step = wl_step
        self.use_memmap = memmap
        self.pattern = pattern
        self.delimiter = delimiter
        self.skip_header = skip_header

        if use_gpu is None:
            self.use_gpu = gpu_available
        else:
            self.use_gpu = use_gpu and gpu_available

        self.common_wavelength = np.arange(wl_min, wl_max + wl_step, wl_step)
        self.n_channels = len(self.common_wavelength)

        self.csv_files = []
        self.raw_data = None
        self.spectra = None
        self.fit_results = []
        self.all_peak_params = []
        self.width_stats = None

    def _cache_path(self, filename):
        """Return a stable cache-file path rooted at the corpus input directory."""
        cache_root = os.path.join(os.path.abspath(self.data_dirs[0]), ".alibz_cache")
        os.makedirs(cache_root, exist_ok=True)
        return os.path.join(cache_root, filename)

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
        # Discover files across all data directories
        self.csv_files = []
        for d in self.data_dirs:
            self.csv_files.extend(
                glob.glob(os.path.join(d, self.pattern), recursive=True)
            )
        self.csv_files = sorted(self.csv_files)
        if len(self.csv_files) == 0:
            dirs = ', '.join(self.data_dirs)
            raise FileNotFoundError(
                f"No files matching '{self.pattern}' found under {dirs}"
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
                self._cache_path('corpus_raw.dat'),
                dtype='float64',
                mode='w+',
                shape=shape,
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

        When ``use_gpu=True`` all spectra are interpolated in a single
        GPU batch via :func:`alibz.gpu.batch_interpolate_gpu`.

        Returns
        -------
        ndarray
            The standardized spectra array.
        """
        if self.raw_data is None:
            raise RuntimeError("Call load_corpus() before standardize_all().")

        n_spectra = self.raw_data.shape[0]

        # --- GPU batch interpolation ---
        if self.use_gpu:
            raw_wl = self.raw_data[:, 0, :]   # (n_spectra, n_raw)
            raw_int = self.raw_data[:, 1, :]
            result = batch_interpolate_gpu(raw_wl, raw_int,
                                           self.common_wavelength)
            if self.use_memmap:
                self.spectra = np.memmap(
                    self._cache_path('corpus_standardized.dat'),
                    dtype='float64',
                    mode='w+',
                    shape=(n_spectra, self.n_channels),
                )
                self.spectra[:] = result
            else:
                self.spectra = result
            print(f"  standardized {n_spectra}/{n_spectra} (GPU)")
            return self.spectra

        # --- CPU path (unchanged) ---
        if self.use_memmap:
            self.spectra = np.memmap(
                self._cache_path('corpus_standardized.dat'),
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

    def fit_all_spectra(self, n_sigma=0, subtract_background=True, workers=1,
                        timeout=None):
        """Fit peaks for every standardized spectrum.

        Parameters
        ----------
        n_sigma : float, optional
            Detection threshold passed to :meth:`PeakyFinder.fit_spectrum`.
        subtract_background : bool, optional
            Whether to estimate and remove the background.
        workers : int, optional
            Number of parallel worker processes.  ``1`` (default) runs
            sequentially in the main process.  Values >1 use a
            :class:`multiprocessing.Pool` — each spectrum is fitted in
            its own process, giving near-linear speedup on multi-core
            machines.
        timeout : int or None, optional
            Per-spectrum timeout in seconds.  Spectra that exceed this
            are recorded as failed with reason ``'timeout'``.

        Returns
        -------
        list
            The list of fit result dictionaries.
        """
        if self.spectra is None:
            raise RuntimeError("Call standardize_all() before fit_all_spectra().")

        self.failed_fits = []  # list of (idx, reason) for failed spectra

        n_spectra = self.spectra.shape[0]

        if workers > 1:
            return self._fit_all_parallel(n_sigma, subtract_background,
                                          workers, timeout)

        # --- Sequential path ---
        finder = PeakyFinder.__new__(PeakyFinder)
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

                params = finder._parameter_array(fit_dict['spectrum_dictionary'])
                self.all_peak_params.append(params)
            except Exception as e:
                self.fit_results.append(None)
                self.all_peak_params.append(np.empty((0, 4)))
                self.failed_fits.append((i, str(e)))

            if (i + 1) % 10 == 0 or i == n_spectra - 1:
                print(f"  fitted {i + 1}/{n_spectra}")

        return self.fit_results

    def _fit_all_parallel(self, n_sigma, subtract_background, workers,
                          timeout=None):
        """Parallel peak fitting using multiprocessing.Pool."""
        import time as _time

        n_spectra = self.spectra.shape[0]
        x = self.common_wavelength
        to = timeout or 0

        # Pre-build argument tuples — each worker gets a copy of the
        # wavelength array and one spectrum's intensity array.
        tasks = [
            (i, x, np.array(self.spectra[i]), n_sigma, subtract_background, to)
            for i in range(n_spectra)
        ]

        # Pre-allocate output lists (indexed by spectrum order)
        self.fit_results = [None] * n_spectra
        self.all_peak_params = [np.empty((0, 4))] * n_spectra

        to_label = f", timeout={to}s" if to else ""
        print(f"  launching {workers} workers for {n_spectra} spectra{to_label}")
        t0 = _time.time()
        done = 0
        n_timeout = 0
        n_error = 0

        with mp.Pool(processes=workers, initializer=_worker_init) as pool:
            for idx, fit_dict, params, fail_reason in pool.imap_unordered(
                _fit_one_spectrum, tasks, chunksize=4
            ):
                self.fit_results[idx] = fit_dict
                self.all_peak_params[idx] = params
                if fail_reason is not None:
                    self.failed_fits.append((idx, fail_reason))
                    if fail_reason == 'timeout':
                        n_timeout += 1
                    else:
                        n_error += 1
                done += 1
                if done % 50 == 0 or done == n_spectra:
                    elapsed = _time.time() - t0
                    rate = done / elapsed
                    eta = (n_spectra - done) / rate if rate > 0 else 0
                    print(f"  fitted {done}/{n_spectra}  "
                          f"({rate:.1f} spectra/s, ETA {eta/60:.0f}m)")

        elapsed = _time.time() - t0
        failed = sum(1 for r in self.fit_results if r is None)
        print(f"  done in {elapsed/60:.1f}m  "
              f"({n_spectra / elapsed:.1f} spectra/s, "
              f"{failed} failed: {n_timeout} timeout, {n_error} error)")
        if self.failed_fits:
            print(f"  failed spectra: {self.failed_fits}")
        return self.fit_results

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_fit_results(self, path):
        """Save fit results and peak params to a pickle checkpoint.

        The archive stores:
        - ``all_peak_params``: list of per-spectrum (n_peaks, 4) arrays
        - ``fit_results``: list of fit dictionaries (or None)
        - ``csv_files``: file list for consistency checking
        - ``spectra_shape``: shape of the standardized spectra array

        Parameters
        ----------
        path : str
            Output file path.
        """
        import pickle as _pickle

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        blob = _pickle.dumps({
            'fit_results': self.fit_results,
            'all_peak_params': self.all_peak_params,
            'csv_files': self.csv_files,
            'spectra_shape': self.spectra.shape if self.spectra is not None else None,
            'failed_fits': getattr(self, 'failed_fits', []),
        })
        with open(path, 'wb') as f:
            f.write(blob)
        print(f"  saved fit checkpoint to {path} "
              f"({len(self.fit_results)} spectra, "
              f"{os.path.getsize(path) / 1e6:.1f} MB)")

    def load_fit_results(self, path):
        """Load fit results from a checkpoint file.

        Validates that the checkpoint matches the current corpus
        (same number of spectra and same file list).

        Parameters
        ----------
        path : str
            Path to the checkpoint file written by :meth:`save_fit_results`.

        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        import pickle as _pickle

        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            data = _pickle.loads(f.read())

        saved_files = data.get('csv_files', [])
        if len(saved_files) != len(self.csv_files):
            print(f"  checkpoint mismatch: {len(saved_files)} vs "
                  f"{len(self.csv_files)} spectra — refitting")
            return False

        if saved_files != self.csv_files:
            print("  checkpoint file list differs — refitting")
            return False

        self.fit_results = data['fit_results']
        self.all_peak_params = data['all_peak_params']
        self.failed_fits = data.get('failed_fits', [])
        n_fitted = sum(1 for r in self.fit_results if r is not None)
        n_failed = len(self.fit_results) - n_fitted
        print(f"  loaded fit checkpoint from {path} "
              f"({n_fitted}/{len(self.fit_results)} succeeded, "
              f"{n_failed} failed)")
        if self.failed_fits:
            n_to = sum(1 for _, r in self.failed_fits if r == 'timeout')
            n_err = len(self.failed_fits) - n_to
            print(f"  failures: {n_to} timeout, {n_err} error")
        return True

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
