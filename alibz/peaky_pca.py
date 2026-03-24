import multiprocessing as mp

import numpy as np
from scipy.special import voigt_profile as voigt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from sklearn.decomposition import PCA, IncrementalPCA

from alibz.utils.voigt import voigt_width as _voigt_width
from alibz.gpu import gpu_available, to_numpy

if gpu_available:
    from alibz.gpu import pca_gpu, extract_windows_gpu


# ------------------------------------------------------------------
# Module-level worker for parallel window extraction
# ------------------------------------------------------------------

def _extract_windows_for_spectrum(args):
    """Extract peak windows from a single spectrum.  Picklable for Pool."""
    spec_idx, spectrum, wl, params, half_window, n_window_points = args
    wl_lo, wl_hi = wl[0], wl[-1]
    x_fixed = np.linspace(0, 1, n_window_points)

    windows = []
    metadata = []

    for peak_idx in range(params.shape[0]):
        amp, mu, sigma, gamma = params[peak_idx]
        lo = mu - half_window
        hi = mu + half_window

        if lo < wl_lo or hi > wl_hi:
            continue

        mask = (wl >= lo) & (wl <= hi)
        wl_win = wl[mask]
        int_win = spectrum[mask].copy()

        if len(wl_win) < 5:
            continue

        baseline = np.linspace(int_win[0], int_win[-1], len(int_win))
        int_win -= baseline

        # Normalise to unit peak height.  For narrow windows the peak
        # may sit *below* the linear baseline (endpoints are on the
        # peak flanks), so use absolute range.
        peak_range = np.max(int_win) - np.min(int_win)
        if peak_range <= 0:
            continue
        int_win -= np.min(int_win)
        int_win /= peak_range

        x_norm = np.linspace(0, 1, len(wl_win))
        resampler = interp1d(x_norm, int_win, kind='linear')
        window = resampler(x_fixed)

        fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
        fwhm_l = 2 * gamma
        fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)

        windows.append(window)
        metadata.append({
            'spectrum_idx': spec_idx,
            'peak_idx': peak_idx,
            'mu': mu, 'sigma': sigma, 'gamma': gamma, 'amp': amp,
            'fwhm': fwhm,
        })

    return windows, metadata


class PeakyPCA:
    """PCA decomposition of peak shapes from a fitted LIBS corpus.

    Extracts normalized spectral windows around every fitted peak, runs PCA,
    and decomposes principal components into Gaussian (Doppler/instrumental),
    Lorentzian (Stark/natural), and asymmetric (self-absorption) contributions.

    Parameters
    ----------
    corpus : :class:`~peaky_corpus.PeakyCorpus`
        A corpus object with ``fit_all_spectra()`` and
        ``peak_width_statistics()`` already called.
    window_multiplier : float, optional
        Half-window width expressed as multiples of the corpus median FWHM.
        Total window = ``2 * window_multiplier * median_fwhm``.
    n_components : int, optional
        Number of principal components to retain.
    n_window_points : int, optional
        Number of evenly-spaced sample points in each extracted window.
    """

    def __init__(self, corpus, window_multiplier=3.0, n_components=5,
                 n_window_points=101, half_window_nm=None, use_gpu=None,
                 workers=1):
        self.corpus = corpus
        self.workers = workers
        self.window_multiplier = window_multiplier
        self.n_components = n_components
        self.n_window_points = n_window_points

        # GPU flag: None = auto-detect, True = force, False = disable
        if use_gpu is None:
            self.use_gpu = gpu_available
        else:
            self.use_gpu = use_gpu and gpu_available

        self.median_fwhm = corpus.get_median_peak_width()
        if half_window_nm is not None:
            self.half_window = half_window_nm
        else:
            # Default to the smallest detected mode as the window basis
            smallest = corpus.get_smallest_mode_width()
            self.half_window = window_multiplier * smallest

        self.windows = None
        self.peak_metadata = []

        self.pca = None
        self.scores = None
        self.components = None
        self.explained_variance_ratio = None
        self.mean_peak = None

        self.decompositions = []
        self.peak_classifications = []

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def extract_peak_windows(self):
        """Extract and normalize spectral windows around every fitted peak.

        For each peak the method:

        1. Selects a wavelength window of width
           ``2 * window_multiplier * median_fwhm`` centred on the peak.
        2. Subtracts a linear baseline interpolated between the window edges.
        3. Normalises to unit peak height.
        4. Resamples to a fixed number of points (``n_window_points``).

        Peaks whose windows extend beyond the spectral bounds are skipped.

        When ``use_gpu=True`` the extraction runs on GPU via
        :func:`alibz.gpu.extract_windows_gpu`.

        Stores
        ------
        self.windows : ndarray, shape (n_peaks, n_window_points)
        self.peak_metadata : list of dict
        """
        # --- Build task list (shared by serial and parallel paths) ---
        wl = self.corpus.common_wavelength
        tasks = []
        for spec_idx, (params, fit_dict) in enumerate(
            zip(self.corpus.all_peak_params, self.corpus.fit_results)
        ):
            if fit_dict is None or params.size == 0:
                continue
            tasks.append((
                spec_idx,
                np.array(self.corpus.spectra[spec_idx]),
                wl,
                params,
                self.half_window,
                self.n_window_points,
            ))

        # --- Parallel path ---
        if self.workers > 1 and len(tasks) > 0:
            import time as _time
            print(f"  extracting windows with {self.workers} workers "
                  f"({len(tasks)} spectra)")
            t0 = _time.time()

            windows = []
            metadata = []
            with mp.Pool(processes=self.workers) as pool:
                for win_batch, meta_batch in pool.imap(
                    _extract_windows_for_spectrum, tasks, chunksize=8
                ):
                    windows.extend(win_batch)
                    metadata.extend(meta_batch)

            elapsed = _time.time() - t0
            print(f"  extracted {len(windows)} windows in {elapsed:.1f}s")
            self.windows = np.array(windows) if windows else np.empty(
                (0, self.n_window_points))
            self.peak_metadata = metadata
            return self.windows

        # --- Sequential path ---
        windows = []
        metadata = []

        for task in tasks:
            win_batch, meta_batch = _extract_windows_for_spectrum(task)
            windows.extend(win_batch)
            metadata.extend(meta_batch)

        self.windows = np.array(windows) if windows else np.empty(
            (0, self.n_window_points))
        self.peak_metadata = metadata
        return self.windows

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def fit_pca(self):
        """Fit PCA to the extracted peak windows.

        When ``use_gpu=True`` the SVD is computed on the GPU via CuPy,
        bypassing scikit-learn entirely.  The resulting attributes are
        always NumPy arrays regardless of backend.

        Stores
        ------
        self.pca : sklearn PCA object (CPU) or ``None`` (GPU)
        self.scores : ndarray (n_peaks, n_components)
        self.components : ndarray (n_components, n_window_points)
        self.explained_variance_ratio : ndarray (n_components,)
        self.mean_peak : ndarray (n_window_points,)
        """
        if self.windows is None or len(self.windows) == 0:
            raise RuntimeError("Call extract_peak_windows() first.")

        n_comp = min(self.n_components, self.windows.shape[0],
                     self.windows.shape[1])

        # GPU PCA requires uploading the full windows matrix to VRAM.
        gpu_pca_feasible = (self.use_gpu and
                            self.windows.nbytes < 4 * 1024**3)  # < 4 GB

        n_samples = self.windows.shape[0]
        # Use IncrementalPCA for large datasets (>1M samples) to avoid
        # allocating a full SVD working set; batch_size chosen so each
        # batch fits comfortably in L3 cache.
        use_incremental = (n_samples > 1_000_000)

        import time as _time
        t0 = _time.time()

        if gpu_pca_feasible:
            result = pca_gpu(self.windows, n_comp)
            self.pca = None
            self.mean_peak = result['mean']
            self.components = result['components']
            self.scores = result['scores']
            self.explained_variance_ratio = result['explained_variance_ratio']
        elif use_incremental:
            batch_size = max(n_comp * 10, min(10_000, n_samples))
            print(f"  IncrementalPCA: {n_samples} samples, "
                  f"batch_size={batch_size}")
            self.pca = IncrementalPCA(n_components=n_comp,
                                      batch_size=batch_size)
            self.scores = self.pca.fit_transform(self.windows)
            self.components = self.pca.components_
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            self.mean_peak = self.pca.mean_
        else:
            self.pca = PCA(n_components=n_comp, svd_solver='randomized',
                           random_state=42)
            self.scores = self.pca.fit_transform(self.windows)
            self.components = self.pca.components_
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            self.mean_peak = self.pca.mean_

        elapsed = _time.time() - t0
        print(f"  PCA completed in {elapsed:.1f}s")

        return self.pca

    # ------------------------------------------------------------------
    # Mean peak characterisation
    # ------------------------------------------------------------------

    @staticmethod
    def _asymmetric_voigt_model(x, amp, center, sigma, gamma, tau):
        """Voigt profile with an optional exponential asymmetric tail."""
        profile = amp * voigt(x - center, max(sigma, 1e-12), max(gamma, 1e-12))
        if abs(tau) > 1e-10:
            tail = np.zeros_like(x)
            mask = x > center
            tail[mask] = np.exp(-(x[mask] - center) / abs(tau))
            profile = profile * (1 + tau / abs(tau) * tail)
        return profile

    def _fit_voigt(self, x, y, x0=None):
        """Fit an asymmetric Voigt to *y(x)*.  Returns (amp, center, sigma, gamma, tau)."""
        if x0 is None:
            i_max = np.argmax(y)
            x0 = np.array([y[i_max], x[i_max], 0.2, 0.2, 0.0])

        def residual(p):
            return y - self._asymmetric_voigt_model(x, *p)

        lower = np.array([0, x[0], 1e-6, 1e-6, -2.0])
        upper = np.array([np.inf, x[-1], 2.0, 2.0, 2.0])
        try:
            result = least_squares(residual, x0=x0, bounds=(lower, upper),
                                   loss='soft_l1')
            return result.x, result.cost
        except Exception:
            return x0, np.inf

    @staticmethod
    def _find_peak_bounds(y):
        """Find local minima flanking the global maximum of *y*.

        Returns ``(left, right)`` indices bounding the central peak.
        Falls back to the full array if no local minima are found.
        """
        i_max = int(np.argmax(y))

        # search left for first local minimum
        left = 0
        for k in range(i_max - 1, 0, -1):
            if y[k] <= y[k - 1] and y[k] <= y[k + 1]:
                left = k
                break

        # search right for first local minimum
        right = len(y) - 1
        for k in range(i_max + 1, len(y) - 1):
            if y[k] <= y[k - 1] and y[k] <= y[k + 1]:
                right = k
                break

        return left, right

    def characterize_mean_peak(self):
        """Fit the PCA mean peak with an asymmetric Voigt profile.

        The mean is first offset so that its minimum value is zero.
        The Voigt fit is performed only between the local minima flanking
        the maximum, avoiding wing artefacts from neighbouring peaks or
        baseline residuals.

        Stores
        ------
        self.mean_offset : float
            Constant subtracted from the PCA mean (its original minimum).
        self.mean_peak_zeroed : ndarray
            Offset-corrected mean peak.
        self.mean_fit : dict
        """
        if self.mean_peak is None:
            raise RuntimeError("Call fit_pca() first.")

        x = np.linspace(-1, 1, self.n_window_points)

        # offset so minimum = 0
        self.mean_offset = float(np.min(self.mean_peak))
        self.mean_peak_zeroed = self.mean_peak - self.mean_offset

        # find local minima flanking the peak
        left, right = self._find_peak_bounds(self.mean_peak_zeroed)
        x_fit = x[left:right + 1]
        y_fit = self.mean_peak_zeroed[left:right + 1]

        params, cost = self._fit_voigt(x_fit, y_fit)
        amp, center, sigma, gamma, tau = params

        # evaluate fitted profile over the full x range
        fitted_full = self._asymmetric_voigt_model(x, *params)
        fitted_region = self._asymmetric_voigt_model(x_fit, *params)
        g_frac = self._gaussian_fraction(sigma, gamma)

        self.mean_fit = {
            'amp': float(amp),
            'center': float(center),
            'sigma': float(sigma),
            'gamma': float(gamma),
            'tau': float(tau),
            'gaussian_fraction': float(g_frac),
            'fwhm': float(self._voigt_width(sigma, gamma)),
            'fitted_profile': fitted_full,
            'fit_left': int(left),
            'fit_right': int(right),
            'residual_norm': float(np.linalg.norm(y_fit - fitted_region)),
        }
        return self.mean_fit

    # ------------------------------------------------------------------
    # Component decomposition (perturbation approach)
    # ------------------------------------------------------------------

    def decompose_component(self, component_index, n_alphas=11):
        """Analyse how a PC perturbs the mean peak's Voigt parameters.

        For a range of score values ``alpha`` the perturbed profile
        ``mean + alpha * PC_i`` is re-fitted with an asymmetric Voigt.
        The resulting sigma(alpha), gamma(alpha), tau(alpha) trajectories
        reveal which broadening mechanism the PC modulates.

        Parameters
        ----------
        component_index : int
        n_alphas : int, optional
            Number of perturbation strengths to sample (symmetric around 0).

        Returns
        -------
        dict
        """
        if self.components is None or self.mean_fit is None:
            raise RuntimeError("Call fit_pca() and characterize_mean_peak() first.")

        pc = self.components[component_index]
        x = np.linspace(-1, 1, self.n_window_points)

        # Fit region from mean peak characterisation
        left = self.mean_fit['fit_left']
        right = self.mean_fit['fit_right']
        x_fit = x[left:right + 1]

        # Sample alpha from the actual score distribution for this PC
        sc = self.scores[:, component_index]
        lo, hi = np.percentile(sc, [5, 95])
        alphas = np.linspace(lo, hi, n_alphas)

        # Seed from the mean fit
        seed = np.array([self.mean_fit['amp'], self.mean_fit['center'],
                         self.mean_fit['sigma'], self.mean_fit['gamma'],
                         self.mean_fit['tau']])

        sigmas, gammas, taus, amps = [], [], [], []
        fitted_profiles = {}

        for alpha in alphas:
            # Apply same offset as mean, then fit in the same region
            perturbed = (self.mean_peak + alpha * pc) - self.mean_offset
            perturbed = np.clip(perturbed, 0, None)
            y_fit = perturbed[left:right + 1]
            params, _ = self._fit_voigt(x_fit, y_fit, x0=seed.copy())
            amps.append(params[0])
            sigmas.append(params[2])
            gammas.append(params[3])
            taus.append(params[4])
            if alpha == alphas[0] or alpha == alphas[-1] or abs(alpha) < 1e-10:
                fitted_profiles[float(alpha)] = self._asymmetric_voigt_model(x, *params)

        sigmas = np.array(sigmas)
        gammas = np.array(gammas)
        taus = np.array(taus)

        # Linear sensitivities: d(param)/d(alpha) via least-squares line fit
        def slope(alphas, values):
            A = np.vstack([alphas, np.ones(len(alphas))]).T
            m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
            return m

        d_sigma = slope(alphas, sigmas)
        d_gamma = slope(alphas, gammas)
        d_tau = slope(alphas, taus)

        # Fractional sensitivities (how much each mechanism changes per unit score)
        abs_sum = abs(d_sigma) + abs(d_gamma) + abs(d_tau)
        if abs_sum < 1e-15:
            abs_sum = 1.0

        g_frac = abs(d_sigma) / abs_sum
        l_frac = abs(d_gamma) / abs_sum
        a_frac = abs(d_tau) / abs_sum

        if g_frac >= l_frac and g_frac >= a_frac:
            interp = 'Doppler/instrumental (Gaussian width variation)'
        elif l_frac >= g_frac and l_frac >= a_frac:
            interp = 'Stark/natural (Lorentzian width variation)'
        else:
            interp = 'Self-absorption (asymmetry variation)'

        return {
            'alphas': alphas,
            'sigmas': sigmas,
            'gammas': gammas,
            'taus': taus,
            'd_sigma': float(d_sigma),
            'd_gamma': float(d_gamma),
            'd_tau': float(d_tau),
            'gaussian_fraction': float(g_frac),
            'lorentzian_fraction': float(l_frac),
            'asymmetry_fraction': float(a_frac),
            'fitted_profiles': fitted_profiles,
            'physical_interpretation': interp,
        }

    def decompose_all_components(self, n_alphas=11):
        """Decompose every principal component via perturbation analysis.

        When ``use_gpu=True`` the perturbed profiles are generated on the
        GPU in a single batched operation before being fitted on CPU.
        The Voigt re-fitting itself remains on CPU (iterative
        least-squares), but the profile generation — which dominates at
        high peak counts — is fully GPU-accelerated.

        Returns
        -------
        list of dict
        """
        if not hasattr(self, 'mean_fit') or self.mean_fit is None:
            self.characterize_mean_peak()

        n = self.components.shape[0] if self.components is not None else 0

        if self.use_gpu and n > 0:
            self.decompositions = self._decompose_all_gpu(n_alphas)
        else:
            self.decompositions = []
            for i in range(n):
                self.decompositions.append(
                    self.decompose_component(i, n_alphas=n_alphas))

        return self.decompositions

    # ------------------------------------------------------------------
    # GPU-accelerated batch decomposition
    # ------------------------------------------------------------------

    def _decompose_all_gpu(self, n_alphas=11):
        """Batch-generate perturbed profiles on GPU, then fit on CPU.

        For *n_components* PCs and *n_alphas* perturbation levels this
        builds all ``n_components × n_alphas`` perturbed profiles in a
        single GPU kernel launch, transfers back to CPU, then runs the
        Voigt re-fits using :mod:`scipy.optimize`.
        """
        import cupy as cp

        n_comp = self.components.shape[0]
        x = np.linspace(-1, 1, self.n_window_points)
        left = self.mean_fit['fit_left']
        right = self.mean_fit['fit_right']
        x_fit = x[left:right + 1]

        seed = np.array([self.mean_fit['amp'], self.mean_fit['center'],
                         self.mean_fit['sigma'], self.mean_fit['gamma'],
                         self.mean_fit['tau']])

        # --- build all perturbed profiles on GPU ---
        mean_d = cp.asarray(self.mean_peak, dtype=cp.float64)
        comps_d = cp.asarray(self.components, dtype=cp.float64)
        offset = self.mean_offset

        all_alphas = []
        for ci in range(n_comp):
            sc = self.scores[:, ci]
            lo, hi = np.percentile(sc, [5, 95])
            all_alphas.append(np.linspace(lo, hi, n_alphas))

        # shape: (n_comp, n_alphas, n_window_points)
        alphas_d = cp.asarray(np.array(all_alphas), dtype=cp.float64)
        perturbed = (mean_d[None, None, :]
                     + alphas_d[:, :, None] * comps_d[:, None, :])
        perturbed = perturbed - offset
        perturbed = cp.clip(perturbed, 0, None)
        perturbed_np = cp.asnumpy(perturbed)

        # --- fit on CPU ---
        decompositions = []
        for ci in range(n_comp):
            alphas = all_alphas[ci]
            sigmas, gammas, taus, amps = [], [], [], []
            fitted_profiles = {}

            for ai, alpha in enumerate(alphas):
                y_fit = perturbed_np[ci, ai, left:right + 1]
                params, _ = self._fit_voigt(x_fit, y_fit, x0=seed.copy())
                amps.append(params[0])
                sigmas.append(params[2])
                gammas.append(params[3])
                taus.append(params[4])
                if ai == 0 or ai == n_alphas - 1 or abs(alpha) < 1e-10:
                    fitted_profiles[float(alpha)] = \
                        self._asymmetric_voigt_model(x, *params)

            sigmas = np.array(sigmas)
            gammas = np.array(gammas)
            taus = np.array(taus)

            def slope(alphas, values):
                A = np.vstack([alphas, np.ones(len(alphas))]).T
                m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
                return m

            d_sigma = slope(alphas, sigmas)
            d_gamma = slope(alphas, gammas)
            d_tau = slope(alphas, taus)

            abs_sum = abs(d_sigma) + abs(d_gamma) + abs(d_tau)
            if abs_sum < 1e-15:
                abs_sum = 1.0

            g_frac = abs(d_sigma) / abs_sum
            l_frac = abs(d_gamma) / abs_sum
            a_frac = abs(d_tau) / abs_sum

            if g_frac >= l_frac and g_frac >= a_frac:
                interp = 'Doppler/instrumental (Gaussian width variation)'
            elif l_frac >= g_frac and l_frac >= a_frac:
                interp = 'Stark/natural (Lorentzian width variation)'
            else:
                interp = 'Self-absorption (asymmetry variation)'

            decompositions.append({
                'alphas': alphas,
                'sigmas': sigmas,
                'gammas': gammas,
                'taus': taus,
                'd_sigma': float(d_sigma),
                'd_gamma': float(d_gamma),
                'd_tau': float(d_tau),
                'gaussian_fraction': float(g_frac),
                'lorentzian_fraction': float(l_frac),
                'asymmetry_fraction': float(a_frac),
                'fitted_profiles': fitted_profiles,
                'physical_interpretation': interp,
            })

        return decompositions

    # ------------------------------------------------------------------
    # Peak classification
    # ------------------------------------------------------------------

    def classify_peaks(self):
        """Classify each peak by its dominant broadening mechanism.

        For each peak the component with the largest absolute PCA score
        determines the classification via that component's decomposition.

        Returns
        -------
        list of str
        """
        if self.scores is None or not self.decompositions:
            raise RuntimeError("Call fit_pca() and decompose_all_components() first.")

        self.peak_classifications = []
        for row in self.scores:
            dominant = int(np.argmax(np.abs(row)))
            decomp = self.decompositions[dominant]

            fracs = [
                decomp['gaussian_fraction'],
                decomp['lorentzian_fraction'],
                decomp['asymmetry_fraction'],
            ]
            labels = ['Doppler/instrumental', 'Stark/natural', 'self-absorption']
            self.peak_classifications.append(labels[int(np.argmax(fracs))])

        return self.peak_classifications

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _voigt_width(sigma, gamma):
        """Approximate FWHM of a Voigt profile (Thompson formula)."""
        return _voigt_width(sigma, gamma)

    @staticmethod
    def _gaussian_fraction(sigma, gamma):
        """Compute the Gaussian fraction of a Voigt profile.

        Returns a value between 0 (pure Lorentzian) and 1 (pure Gaussian)
        using the Thompson pseudo-Voigt approximation.
        """
        if sigma <= 0 and gamma <= 0:
            return 0.5
        if gamma <= 0:
            return 1.0
        if sigma <= 1e-12:
            return 0.0
        fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
        fwhm_l = 2 * gamma
        fwhm_v = (fwhm_g ** 5 + 2.69269 * fwhm_g ** 4 * fwhm_l
                   + 2.42843 * fwhm_g ** 3 * fwhm_l ** 2
                   + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3
                   + 0.07842 * fwhm_g * fwhm_l ** 4
                   + fwhm_l ** 5) ** 0.2
        if fwhm_v == 0:
            return 0.5
        ratio = fwhm_l / fwhm_v
        eta_l = 1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3
        return float(np.clip(1 - eta_l, 0, 1))

