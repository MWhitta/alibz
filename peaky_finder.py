
import numpy as np

from matplotlib import pyplot as plt

from scipy.special     import voigt_profile as voigt
from scipy.optimize    import least_squares
from scipy.integrate   import cumulative_trapezoid
from scipy.interpolate import interp1d

from sklearn.preprocessing import PowerTransformer

from utils.dataloader import Data


class PeakyFinder():
    """Utilities for indexing and classifying laser plasma spectra.

    Parameters
    ----------
    data_dir : str
        Path to the data directory used when loading data.

    Attributes
    ----------
    data : :class:`~utils.dataloader.Data`
        Loaded data object.
    """

    def __init__(self, data_dir) -> None:
        """Initialize the finder and load data.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        """
        self.data_dir = data_dir
        self.data = Data(data_dir)


    def multi_voigt(self, x, params):
        """Vectorized multi-Voigt summation.

        Parameters
        ----------
        x : array_like
            1-D array of x-values.
        params : array_like
            1-D array of parameters of length ``4*n`` with
            ``[amp, mu, sigma, gamma]`` repeated ``n`` times.

        Returns
        -------
        ndarray
            Summed Voigt profile evaluated at ``x``.
        """
        param_array = params.reshape(-1, 4)
        amps   = param_array[:, 0]
        mus    = param_array[:, 1]
        sigmas = param_array[:, 2]
        gammas = param_array[:, 3]
        X_shifted = x[:, None] - mus[None, :]
        profiles  = voigt(X_shifted, sigmas[None, :], gammas[None, :])
        total     = np.sum(profiles * amps[None, :], axis=1)
        return total
    

    def voigt_width(self, sigma, gamma):
        """Return the approximate FWHM of a Voigt profile.

        Parameters
        ----------
        sigma : float
            Gaussian component width.
        gamma : float
            Lorentzian component width.

        Returns
        -------
        float
            Approximate full width at half maximum.
        """
        f = 0.5346 * 2 * gamma + np.sqrt(0.2166 * 4 * gamma + 8 * sigma**2 * np.log(2))
        return f


    def residual(self, params, x, y, func):
        """Compute the residual between data and a model function.

        Parameters
        ----------
        params : array_like
            Model parameters.
        x : array_like
            Independent variable.
        y : array_like
            Observed values.
        func : callable
            Model function returning ``y`` for ``x`` and ``params``.

        Returns
        -------
        ndarray
            Difference ``y - func(x, params)``.
        """
        return y - func(x, params)


# Find peaks and peak properties -----------------------------------------------------------------------------------------------------------------
    
    def find_peaks(self, y, window: int = 5):
        """Return indices of local maxima and minima in ``y``.

        Parameters
        ----------
        y : array_like
            1-D data array.
        window : int, optional
            Size of the comparison window. ``window`` is clipped to the length of
            ``y``.

        Returns
        -------
        tuple of ndarray
            ``(peaks, minima)`` indices of peaks and minima respectively.
        """

        y = np.asarray(y)
        n = len(y)
        window = int(max(1, min(window, n)))
        half = window // 2

        offsets = np.arange(-half, half + 1)
        padded = np.pad(y, half)
        rolled = np.vstack([np.roll(padded, i) for i in offsets]).T[half:-half]
        cmp = (y[:, None] > rolled)

        lsum = cmp.sum(axis=1)
        peaks = np.flatnonzero(lsum == window - 1)
        minima = np.flatnonzero(lsum == 0)

        for i, p in enumerate(peaks):
            s = slice(max(p - half, 0), min(p + half + 1, n))
            peaks[i] = s.start + np.argmax(y[s])

        return np.unique(peaks), minima
    

    def find_background(self, x, y, range=5, n_sigma=1, plot=False):
        """Estimate the background of a spectrum.

        Parameters
        ----------
        x : array_like
            X values of the spectrum.
        y : array_like
            Intensity values.
        range : int, optional
            Window size for peak searching.
        n_sigma : float, optional
            Sigma multiplier for filtering background anchors.
        plot : bool, optional
            If ``True`` plot the intermediate and final background.

        Returns
        -------
        ndarray
            Estimated background values.
        """

        # ------------------------------------------------------------------
        # Step 1: locate peaks and minima
        # ------------------------------------------------------------------
        p_peaks, p_minima = self.find_peaks(y, window=range)

        peak_values = np.zeros_like(y)
        minima_values = np.zeros_like(y)
        minima_values[p_minima] = y[p_minima]
        peak_values[p_peaks] = y[p_peaks]

        # ------------------------------------------------------------------
        # Step 2: create a rough set of background anchors in frequency space
        # ------------------------------------------------------------------
        diff_extrema = np.fft.fft(peak_values) - np.fft.fft(minima_values)
        extrema = np.abs(np.fft.ifft(diff_extrema))
        anchors = np.nonzero(extrema > 1)[0]
        anchors = anchors[~np.in1d(anchors, p_peaks)]

        # ------------------------------------------------------------------
        # Step 3: remove spurious anchors associated with split peaks
        # ------------------------------------------------------------------
        cumint = cumulative_trapezoid(x[anchors], y[anchors], initial=0)
        intdiff = np.diff(np.insert(cumint, 0, 0))
        intmean, intstd = np.mean(intdiff), np.std(intdiff)
        valid = np.abs(intdiff) <= intmean + 2 * n_sigma * intstd
        anchors = anchors[valid]

        # ------------------------------------------------------------------
        # Step 4: interpolate to obtain a first pass background
        # ------------------------------------------------------------------
        interp = interp1d(x[anchors], y[anchors], bounds_error=False, fill_value=0)
        rough_bg = interp(x)

        # ------------------------------------------------------------------
        # Step 5: remove high outliers and re-interpolate
        # ------------------------------------------------------------------
        overshoot = (y - rough_bg) < 0
        nodes = (y - rough_bg) == 0
        overnodes = np.cumsum(overshoot.astype(int) + 2 * nodes.astype(int))
        diffnodes = np.diff(overnodes) > 1
        diffnodes = np.pad(diffnodes, (1, 0), mode='constant', constant_values=1)
        diffnodes = diffnodes.astype(bool)

        y_nodes = y[diffnodes]
        keep_nodes = np.argmax(
            np.stack((y_nodes, np.roll(y_nodes, -1)), axis=-1), axis=1
        ).astype(bool)
        filtered_bg_anchors = np.arange(len(y))[nodes][keep_nodes]

        interp = interp1d(
            x[filtered_bg_anchors],
            y[filtered_bg_anchors],
            bounds_error=False,
            fill_value=0,
        )
        filtered_bg = np.clip(interp(x), 0, np.inf)

        if plot:
            plt.figure(figsize=(35, 5))
            plt.plot(x, rough_bg, color="k")
            plt.plot(x, filtered_bg, color="r")
            plt.show()

        return filtered_bg
    

    def filter_peaks(self, y, n_sigma: float = 2):
        """Detect and rank significant peaks.

        Parameters
        ----------
        y : array_like
            Intensity values of a spectrum.
        n_sigma : float, optional
            Number of standard deviations above the mean required for a peak to
            be kept.

        Returns
        -------
        tuple
            ``(peak_indices, minima_indices, transformer)`` where ``peak_indices``
            are sorted from highest to lowest intensity.
        """

        peaks, minima = self.find_peaks(y)

        transformer = PowerTransformer()
        transformed = transformer.fit_transform(np.asarray(y).reshape(-1, 1))[:, 0]
        transformed = np.clip(transformed, transformed[0], np.inf)

        power_lambda = transformer.lambdas_

        if len(peaks) == 0:
            return peaks, minima, transformer

        peak_intensities = transformed[peaks]
        threshold = np.mean(peak_intensities) + n_sigma * np.std(peak_intensities)
        valid = peak_intensities > threshold
        filtered_peaks = peaks[valid]

        order = np.argsort(y[filtered_peaks])[::-1]
        peak_indices = np.take_along_axis(filtered_peaks, order, axis=0)

        return peak_indices, minima, transformer


    def fourier_peaks(self, y, n_sigma=0, plot=False):
        """Return peak information using Fourier and cepstral analysis.

        Parameters
        ----------
        y : array_like
            Intensity values of a spectrum.
        n_sigma : float, optional
            Threshold used when calling :meth:`filter_peaks` if peak indices are
            not already available.
        plot : bool, optional
            When ``True`` plot the autocorrelation fit used for determining
            Fourier parameters.

        Returns
        -------
        tuple
            ``(peak_indices, transformer, minima_indices, peak_limit, cep_maxs,``
            ``cep_mins, popt, pcov)``. ``popt`` contains the optimal parameters
            for the autocorrelation fit and ``pcov`` the associated covariance
            matrix.
        """

        peak_indices, p_minima, transformer = self.filter_peaks(y, n_sigma=n_sigma)
        
        # ------------------------------------------------------------------
        # Cepstral analysis to determine relevant Fourier domain region
        # ------------------------------------------------------------------
        fft = np.fft.fft(y)
        log_amp = np.log(np.abs(fft))
        cepstrum = np.abs(np.fft.ifft(log_amp))
        log_cepstrum = np.nan_to_num(np.log(cepstrum))

        peak_limit = np.argmax(log_cepstrum < np.mean(log_cepstrum))
        cep_maxs, cep_mins = self.find_peaks(log_cepstrum[:peak_limit])

        peak_limit = peak_limit

        # ------------------------------------------------------------------
        # Autocorrelation fitting in Fourier space
        # ------------------------------------------------------------------
        autokernel = np.log(np.real(np.fft.ifft(np.abs(fft) ** 2)))
        autokernel_norm = autokernel[:peak_limit] - np.min(autokernel[peak_limit])
        x_autokernel = np.arange(len(autokernel_norm))

        x0 = (1, 0, cep_mins[0], cep_mins[0])
        bounds = ([0] * 4, [np.inf] * 4)

        result = least_squares(
            self.residual,
            x0=x0,
            bounds=bounds,
            args=(x_autokernel, autokernel_norm, self.multi_voigt),
        )
        params = result.x
        pcov = np.linalg.inv(result.jac.T.dot(result.jac))

        if plot:
            plt.plot(x_autokernel, self.multi_voigt(x_autokernel, params))
            plt.plot(x_autokernel, autokernel_norm)

        return peak_indices, transformer, p_minima, peak_limit, cep_maxs, cep_mins, params, pcov
    

    def peak_parameter_guess(self, data, idx):
        """Estimate basic peak parameters.

        Parameters
        ----------
        data : ``array_like``
            1-D array containing the spectrum intensities.
        idx : ``int`` or ``array_like``
            Index or indices of the peak positions within ``data``.

        Returns
        -------
        tuple
            ``(amplitude, fwhm, hwhm, right_edge, left_edge)``.  When ``idx`` is
            an array each entry is returned as an array with matching shape.
        """

        idx = np.atleast_1d(idx)
        full_max = data[idx]
        half_max = full_max / 2

        fwhm = np.zeros_like(full_max, dtype=float)
        hwhm = np.zeros_like(full_max, dtype=float)
        nr = np.zeros_like(full_max, dtype=int)
        nl = np.zeros_like(full_max, dtype=int)

        for i, p in enumerate(idx):
            d1 = data[p:]
            d2 = data[:p]

            if np.any(d1 < half_max[i]):
                half_max_right = np.argmax(d1 < half_max[i])
                nr[i] = p + np.argmax(d1 <= 0)
            else:
                half_max_right = 0
                nr[i] = 0

            if np.any(d2 < half_max[i]):
                half_max_left = np.argmax(d2 < half_max[i])
                nl[i] = p - np.argmax(d2[::-1] <= 0)
            else:
                half_max_left = 0
                nl[i] = 0

            fwhm[i] = half_max_right + half_max_left
            hwhm[i] = fwhm[i] / 2

        if len(idx) == 1:
            return full_max[0], fwhm[0], hwhm[0], nr[0], nl[0]
        return full_max, fwhm, hwhm, nr, nl

    
    def fit_peaks(self, x, y, peak_indices, plot=False):
        """Fit Voigt profiles to detected peaks.

        Parameters
        ----------
        x : array_like
            Wavelength axis.
        y : array_like
            Intensity values.
        peak_indices : array_like
            Indices of peaks to fit.
        peak_dictionary : dict, optional
            Initial peak parameters keyed by index.
        plot : bool, optional
            When ``True`` plot intermediate fits.

        Returns
        -------
        dict
            Updated dictionary of fitted peak parameters.
        """
        # step size
        inc = np.median(np.diff(x))

        _, fwhm_array, *_ = self.peak_parameter_guess(y, peak_indices)
        median_fwhm = np.median(fwhm_array) * inc
        fwhm_limit =  5 * median_fwhm

        peak_dictionary = {}

        # loop over peaks to fit individually
        for i, p in enumerate(peak_indices):
            full_max, fwhm, hwhm, node_right, node_left = self.peak_parameter_guess(y, p)
            
            x_window = x[node_left:node_right]
            y_window = y[node_left:node_right]
            
            if len(x_window) > 5: # search for close peaks
                cumint = cumulative_trapezoid(x_window, y_window, initial=0) # x and y reversed. works better
                cumint2 = cumulative_trapezoid(x_window[::-1], y_window[::-1], initial=0) # x and y reversed. works better
                intdiff1 = np.diff(np.insert(cumint, 0, 0))
                intdiff2 = np.diff(np.insert(cumint2, 0, 0))[::-1]
                intdiff = intdiff1+intdiff2
                intdiff_peaks, _ = self.find_peaks(intdiff * (intdiff>0))
                
                window_peaks = intdiff_peaks
                window_peak_num = len(window_peaks)

                # distinguish known peaks (sometimes fit ranges overlap)
                known_peaks = np.array(list(peak_dictionary.keys())).astype(int)
                known_peaks_inrange = known_peaks[(known_peaks >= node_left) & (known_peaks <= node_right)]
                known_peak_num = len(known_peaks_inrange)

                # identify which peaks are new
                new_peaks = ~np.in1d(window_peaks, known_peaks_inrange - node_left)
                new_peak_num = np.sum(new_peaks)
                
                if new_peak_num > 0:
                    # define fit bounds
                    lower_bounds = np.array([0, x[node_left], 0, 0] * window_peak_num )
                    upper_bounds = np.array([1, x[node_right], fwhm_limit, fwhm_limit] * window_peak_num )

                    # define initial parameter guesses
                    x0 = np.zeros((window_peak_num, 4))
                    x0[0] = [full_max / 2, x[p], inc * median_fwhm, inc * median_fwhm/2]

                    if window_peak_num > 1:
                        for i, pp in enumerate(window_peaks[new_peaks]):
                            full_max = y_window[pp]
                            x0[i + known_peak_num] = np.array([full_max / 2, x[node_left + pp], median_fwhm, median_fwhm])
                        for i, ppp in enumerate(known_peaks_inrange):
                            x0[i] = peak_dictionary[ppp]
                    
                    x0 = np.ravel(x0) # flatten to pass to least squares fit
                    upper_bounds[::4] = 2 * x0[::4]
                    lower_bounds[1::4], upper_bounds[1::4] = x0[1::4] - inc, x0[1::4] + inc
                    bounds = (lower_bounds, upper_bounds)
                    
                    try:
                        popt = least_squares(self.residual, x0=x0, bounds=bounds, args=(x_window, y_window, self.multi_voigt), x_scale='jac', loss='linear')            

                        if window_peak_num > 1:
                            popt = np.reshape(popt.x, (window_peak_num, 4))
                            for k, v in zip(window_peaks, popt):
                                key = k + node_left
                                peak_dictionary.update({key : v})
                        else:
                            popt = popt.x
                            peak_dictionary.update({p : popt})

                        if window_peak_num > 1:
                            popt = np.ravel(popt)
                            
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(f"An error occurred during initial fitting: {e}")
        
        return peak_dictionary
    

    def fit_shoulders(self, x, y, peak_indices, residuals, peak_dictionary, rng=5):
        """Fit peak shoulders by modelling the residuals.

        Parameters
        ----------
        x : array_like
            x-values of the spectrum.
        y : array_like
            y-values of the spectrum.
        peak_indices : array_like
            Indices of previously fitted peaks.
        residuals : array_like
            Residual spectrum from initial fits.
        peak_dictionary : dict
            Dictionary of peak parameters.
        rng : int, optional
            Search window used when locating new peaks.

        Returns
        -------
        tuple
            ``(peak_dictionary, new_peak_indices)`` with updated parameters and
            indices of newly fitted peaks.
        """
        
        inc = np.median(np.diff(x))

        # identify position of residuals that resemble, but don't coincide with, peaks
        residual_mean, residual_std = np.mean(residuals), np.std(residuals)
        large_residual = residuals > residual_mean + 2 * residual_std
        residual_masked = residuals * large_residual
        residual_peaks, _ = self.find_peaks(residual_masked)
        offsets = np.array(np.arange(-rng//2, rng//2, 1))
        near_peak_indices = (peak_indices[:, None] + offsets).ravel()
        new_peak_indices = residual_peaks[~np.in1d(residual_peaks, near_peak_indices)]

        if len(new_peak_indices) == 0:
            return peak_dictionary, new_peak_indices

        fm, fwhm_arr, hwhm_arr, nr_arr, nl_arr = self.peak_parameter_guess(residuals, new_peak_indices)

        for p, full_max, fwhm, hwhm, node_right, node_left in zip(new_peak_indices, fm, fwhm_arr, hwhm_arr, nr_arr, nl_arr):
            x_window = x[node_left:node_right]
            y_window = y[node_left:node_right]
            
            window_shoulders_mask = (new_peak_indices >= node_left) & (new_peak_indices <= node_right)
            window_shoulders = new_peak_indices[window_shoulders_mask] - node_left
            window_shoulders_num = len(window_shoulders)
            known_peaks = np.array(list(peak_dictionary.keys())).astype(int)
            known_peaks_inrange = known_peaks[(known_peaks >= node_left) & (known_peaks <= node_right)]
            known_parameters = np.ravel(np.array([peak_dictionary[key] for key in known_peaks_inrange]))
            
            # define fit bounds
            dx = inc * (node_right - node_left)
            lower_bounds = [0, x[node_left], 0, 0] * window_shoulders_num
            upper_bounds = [0, x[node_right], dx, dx] * window_shoulders_num
            
            # define initial parameter guesses
            x0 = np.zeros((window_shoulders_num, 4))
            if window_shoulders_num:
                fm_ws, fwhm_ws, hwhm_ws, _, _ = self.peak_parameter_guess(y_window, window_shoulders)

                # ``peak_parameter_guess`` returns scalars when the input consists
                # of a single index.  Convert these values to 1-D arrays so that
                # they can be iterated over uniformly in the loop below.
                if window_shoulders_num == 1:
                    fm_ws = np.array([fm_ws])
                    fwhm_ws = np.array([fwhm_ws])
                    hwhm_ws = np.array([hwhm_ws])

                for i, (full_max, fwhm, hwhm, pp) in enumerate(zip(fm_ws, fwhm_ws, hwhm_ws, window_shoulders)):
                    x0[i] = np.array([full_max, x[node_left + pp], inc * fwhm, inc * hwhm])
                    
            x0 = np.ravel(x0) # flatten to pass to least squares fit
            upper_bounds[::4] = 5 * x0[::4]
            lower_bounds[1::4], upper_bounds[1::4] = x0[1::4] - inc, x0[1::4] + inc
            bounds = (lower_bounds, upper_bounds)

            fit_function = lambda x, p1: self.multi_voigt(x, known_parameters) + self.multi_voigt(x, p1)
                
            try:
                popt = least_squares(self.residual, x0=x0, bounds=bounds, args=(x_window, y_window, fit_function), x_scale='jac', loss='linear')            

                if window_shoulders_num > 1:
                    popt = np.reshape(popt.x, (window_shoulders_num, 4))
                    for k, v in zip(window_shoulders, popt):
                        key = k + node_left
                        peak_dictionary.update({key : v})

                else:
                    popt = popt.x
                    peak_dictionary.update({p : popt})
                    
            except (RuntimeError, TypeError, ValueError) as e:
                print(f"An error occurred during initial fitting: {e}")

        return peak_dictionary, new_peak_indices
    

    def fit_all(self, x, y, peak_dictionary, plot=True):
        """Refine all peak parameters simultaneously.

        Parameters
        ----------
        x : array_like
            Wavelength values.
        y : array_like
            Intensity values.
        peak_dictionary : dict
            Initial peak parameters keyed by index.
        plot : bool, optional
            When ``True`` display diagnostic plots.

        Returns
        -------
        dict
            Updated peak parameters after the global fit.
        """
        inc = np.median(np.diff(x))
        idxs = np.sort(np.array(list(peak_dictionary.keys())))
        if len(idxs) == 0:
            return peak_dictionary

        _, _, _, node_right_arr, node_left_arr = self.peak_parameter_guess(y, idxs)

        for idx, node_right, node_left in zip(idxs, node_right_arr, node_left_arr):
            if idx > 0:
                idx_left_range, idx_right_range = idx - node_left, node_right - idx
                if idx_left_range > 50:
                    node_left = idx - 50
                if idx_right_range > 50:
                    node_right = idx + 50
                x_window = x[node_left:node_right]
                y_window = y[node_left:node_right]
                known_peaks_inrange = idxs[(idxs >= node_left) & (idxs <= node_right)]
                refit_peak_num = len(known_peaks_inrange)
                
                x0 = np.ravel(np.array([peak_dictionary[key] for key in known_peaks_inrange]))
                lower_bounds, upper_bounds = np.zeros_like(x0), np.inf * np.ones_like(x0)

                lower_bounds[1::4], upper_bounds[1::4] = x0[1::4] - inc, x0[1::4] + inc
                bounds = (lower_bounds, upper_bounds)

                try:
                    popt = least_squares(self.residual, x0=x0, bounds=bounds, args=(x_window, y_window, self.multi_voigt), x_scale='jac', loss='linear')            

                    if refit_peak_num > 1:
                        popt = np.reshape(popt.x, (refit_peak_num, 4))
                        for k, v in zip(known_peaks_inrange, popt):
                            peak_dictionary.update({k : v})
                    else:
                        peak_dictionary.update({known_peaks_inrange[0] : popt.x})
                            
                except (RuntimeError, TypeError, ValueError) as e:
                    print(f"An error occurred during full fitting: {e}")

                idxs *= idxs > node_right

        return peak_dictionary


    def fit_spectrum(self, x, y, n_sigma=0, subtract_background=True, plot=False, *kwargs):
        """ Fit a LIBS spectrum using the :meth:`fourier_peaks`, :meth:`fit_peaks`, :meth:`fit_shoulders`, and :meth:`fit_all` methods.
        Parameters
        ----------
        x : array_like
            Wavelength or x-values of the spectrum.
        y : array_like
            Intensity or y-values of the spectrum.
        n_sigma : float, optional
            Number of standard deviations above the mean required for a peak to
            be kept.
        subtract_background : bool, optional
            If ``True``, subtract the background from the spectrum before fitting.
        plot : bool, optional
            If ``True``, plot the intermediate and final results.
        *kwargs : dict
            Additional keyword arguments to pass to the background subtraction or
            fitting methods.
        """
        if subtract_background:
            bg = self.find_background(x, y, *kwargs)
            y_bgsub = y - bg
        else:
            bg = np.zeros_like(y)
            y_bgsub = y

        # find peaks and profile parameters
        peak_indices, transformer, *rest = self.fourier_peaks(y, n_sigma=n_sigma)
        print('fourier peaks done')
        peak_dictionary = self.fit_peaks(x, y_bgsub, peak_indices, plot=plot)
        print('fit peaks done')
        
        # filter and sort fit parameters
        parameters = np.array(list(peak_dictionary.values()))

        profile = self.multi_voigt(x, np.ravel(parameters))
        profile_outliers = profile > 1.5 * np.max(y)
        profile[profile_outliers] = 0
        residual_data = y_bgsub - profile

        peak_dictionary, new_peak_indices = self.fit_shoulders(x, y_bgsub, peak_indices, residual_data, peak_dictionary)
        print('fit shoulders done')
        peak_dictionary = self.fit_all(x, y_bgsub, peak_dictionary)
        print('fit all done')
        
        spectrum_dictionary = dict({})
        inc = np.median(np.diff(x))
        sigmas, gammas = np.array(list(peak_dictionary.values()))[:, 2:4].T
        widths = self.voigt_width(sigmas, gammas)
        median_fwhm = np.median(widths) * inc
        for (key, value) in zip(peak_dictionary.keys(), peak_dictionary.values()):
            if value[2] < inc / 2:
                value[2] = 0
            if value[2] > 100 * median_fwhm:
                value[0] = 0
            if value[3] < inc / 2:
                value[3] = 0
            if value[3] > 100 * median_fwhm:
                value[0] = 0
            if value[2]==value[3]==0:
                value[0] = 0
            if value[0] < 1:
                value[0] = 0
            if value[0] >= 1:
                spectrum_dictionary.update({key: value})

        parameters = np.array(list(spectrum_dictionary.values()))
        profile = self.multi_voigt(x, np.ravel(parameters))
        p_sort = np.argsort(parameters[:, 0])
        sorted_parameter_array = parameters[p_sort][::-1] # descending order

        fit_dict = {'peak_dictionary': peak_dictionary,
                    'spectrum_dictionary': spectrum_dictionary,
                    'profile': profile,
                    'residual_data': residual_data,
                    'background': bg,
                    'sorted_parameter_array': sorted_parameter_array}

        yj_data = transformer.fit_transform(y_bgsub.reshape(-1,1))[:,0]
        power_lambda = transformer.lambdas_

        if plot:
            self.plot(
                "spectrum",
                x=x,
                y=y,
                y_bgsub=y_bgsub,
                peak_indices=peak_indices,
                profile=profile,
                yj_data=yj_data,
                residual_data=residual_data,
                new_peak_indices=new_peak_indices,
                bg=bg,
                power_lambda=power_lambda,
            )

        return fit_dict
    

    def fit_spectrum_data(self, s, n_sigma=0, subtract_background=True, plot=True, *kwargs):
        """Fit a LIBS spectrum using the :meth:`fit_spectrum` method with data loaded from a :class:`~utils.dataloader.Data` object.

        Parameters
        ----------
        s : slice or int
            Slice or index of the spectrum to fit from the loaded data.
        n_sigma : float, optional
            Number of standard deviations above the mean required for a peak to
            be kept.
        subtract_background : bool, optional
            If ``True``, subtract the background from the spectrum before fitting.
        plot : bool, optional
            If ``True``, plot the intermediate and final results.
        *kwargs : dict
            Additional keyword arguments to pass to the background subtraction or
            fitting methods.
        """
        if self.data.data is None:
            data = self.data.load_data()
        else:
            data = self.data.data

        data = data[s]

        if isinstance(s, int):
            x = data[0]
            y = data[1]
            fit_dict = self.fit_spectrum(x, y, n_sigma=n_sigma, subtract_background=subtract_background, plot=plot, *kwargs)
            
        elif isinstance(s, slice):
            data_length = len(data)
            if data_length == 0:
                raise ValueError("No data available for the specified slice.")
            else:
                
                for d in range(data_length):
                    x = data[0]
                    y = data[1]
                    dict = self.fit_spectrum(x, y, n_sigma=n_sigma, subtract_background=subtract_background, plot=plot, *kwargs)
                    fit_dict['spectrum_' + str(d)] = dict

            self.fit_spectrum(x, y, n_sigma=n_sigma, subtract_background=subtract_background, plot=plot, *kwargs)

        return fit_dict
    

    def plot(self, kind, **kwargs):
        """Central plotting utility for :class:`PeakyFinder`.

        Parameters
        ----------
        kind : {"background", "fourier", "spectrum"}
            Type of figure to produce.
        **kwargs : dict
            Data required for the selected plot. When ``kind`` is ``"spectrum"``
            ``power_lambda`` should be provided to label the transformed
            intensity histogram.
        """

        if kind == "background":
            x = kwargs.get("x")
            full_bg = kwargs.get("full_bg")
            full_filtered_bg = kwargs.get("full_filtered_bg")

            plt.figure(figsize=(35, 5))
            plt.plot(x, full_bg, color="k")
            plt.plot(x, full_filtered_bg, color="r")
            plt.show()

        elif kind == "fourier":
            x = kwargs.get("x")
            fit = kwargs.get("fit")
            y = kwargs.get("y")

            plt.figure(figsize=(8, 5))
            plt.plot(x, fit)
            plt.plot(x, y)
            plt.show()

        elif kind == "spectrum":
            x = kwargs.get("x")
            y = kwargs.get("y")
            y_bgsub = kwargs.get("y_bgsub")
            peak_indices = kwargs.get("peak_indices")
            profile = kwargs.get("profile")
            yj_data = kwargs.get("yj_data")
            residual_data = kwargs.get("residual_data")
            new_peak_indices = kwargs.get("new_peak_indices")
            bg = kwargs.get("bg")
            power_lambda = kwargs.get("power_lambda")

            plt.rcParams.update({"font.size": 14})
            fig, axs = plt.subplots(
                3,
                2,
                figsize=(30, 10),
                gridspec_kw={"width_ratios": [4, 1], "height_ratios": [4, 1, 1]},
            )
            axs1, axs2, axs3 = axs
            ax11, ax12 = axs1
            ax21, ax22 = axs2
            ax31, ax32 = axs3

            # ROW 1 - Background subtracted data
            ax11.plot(x, y_bgsub, color="k", alpha=0.9)
            ax11.scatter(x[peak_indices], y_bgsub[peak_indices], color="r", alpha=0.7)
            ax11.plot(x, profile, color="r", lw=0.5)
            ax11.fill_between(x, profile, np.zeros_like(x), color="r", alpha=0.5)
            ax11.tick_params(axis="x", which="minor", bottom=True)
            ax11.set_xlabel("wavelength [nm]")
            ax11.set_ylabel("intensity [counts]")

            ax12.hist(yj_data, bins=int(np.sqrt(len(y))), density=True, color="r", alpha=0.5)
            if power_lambda is not None:
                ax12.set_xlabel(
                    rf"(intensity + 1)$^{{{float(power_lambda):.2g}}}$ / {float(power_lambda):.2g}"
                )
            else:
                ax12.set_xlabel("transformed intensity")
            ax12.set_ylabel("prob. density")

            # ROW 2 - Residual
            ax21.plot(x, residual_data, alpha=0.5)
            ax21.scatter(x[new_peak_indices], residual_data[new_peak_indices], color="k", alpha=0.5)
            ax21.set_xlabel("wavelength [nm]")
            ax21.set_ylabel("residual [counts]")

            ax22.hist(residual_data, bins=int(np.sqrt(len(residual_data))), density=True, alpha=0.5)
            ax22.set_xlabel("residual [counts]")
            ax22.set_ylabel("prob. density")

            # ROW 3 - Background
            ax31.plot(x, bg, color="k", alpha=0.5)
            ax31.set_xlabel("wavelength [nm]")
            ax31.set_ylabel("background [counts]")

            ax32.hist(bg, bins=int(np.sqrt(len(bg))), density=True, color="k", alpha=0.5)
            ax32.set_xlabel("background [counts]")
            ax32.set_ylabel("prob. density")

            plt.tight_layout()
            plt.show()

        else:
            raise ValueError(f"Unknown plot kind: {kind}")
