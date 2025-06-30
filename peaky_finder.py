import time
from itertools import groupby
from collections import defaultdict

import numpy as np

from matplotlib import pyplot as plt

from scipy.special     import voigt_profile as voigt
from scipy.optimize    import least_squares
from scipy.integrate   import cumulative_trapezoid
from scipy.interpolate import interp1d

from sklearn.preprocessing import PowerTransformer

from utils.dataloader import Data


class PeakyFinder():
    """ PeakyFinder is a python class that contains tools for indexing and classifying laser plasma spectroscopy data.
    Args:
        data_dir (str literal): Path to the data directory upon which the `PeakyFinder` class loads data.
    Attributes:
        data (object):
    Methods:
    """

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.data = Data(data_dir)


    def multi_voigt(self, x, params):
        """
        Vectorized multi-Voigt summation.
        :param x:      1D array of x-values.
        :param params: 1D array of parameters of length 4*n, with [amp, mu, sigma, gamma] repeated n times.
        :return:       1D array of the summed Voigt profiles at each x.
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
        """ Width of a Voigt profile to within ~1%
        """
        f = 0.5346 * 2 * gamma + np.sqrt(0.2166 * 4 * gamma + 8 * sigma**2 * np.log(2))
        return f


    def residual(self, params, x, y, func):
        return y - func(x, params)


# Find peaks and peak properties -----------------------------------------------------------------------------------------------------------------
    
    def find_peaks(self, y, range=5):
        """" calculate natural window size and peak indices for experimental spectrum """
        # assumes `data` is a 1D array (no wavelength column)
        dlen = len(y)
        if dlen < range:
            range = dlen # ensure range doesn't extend beyond data
        
        # search for local maxima
        pad = range // 2
        range_array = np.arange(-pad, pad + 1, 1)
        y_pad = np.pad(y, (pad, pad))
        data_roll = np.hstack([np.roll(y_pad[:, None], i, axis=0) for i in range_array])  # shift data 'pad' indices forward and backward
        data_diff = y_pad[:, None] - data_roll[:, :]  # find differences with shifted data
        data_lmax = data_diff[pad:-pad] > 0
        data_lsum = np.sum(data_lmax, axis=1)

        # identify local maxima and minima by summing bools
        lmax_ind = data_lsum == 4
        lmin_ind = data_lsum == 0
        
        # find local max and minima
        p_peaks  = np.flatnonzero(lmax_ind)
        p_minima = np.flatnonzero(lmin_ind)

        # remove false maxima
        for i, p in enumerate(p_peaks):
            while True:
                start = max(p - pad, 0)
                end = min(p + pad + 1, dlen)
                d = y[start:end]

                max_idx = np.argmax(d)
                st = start + max_idx

                if st == p or y[st] <= y[p]:
                    break 

                p = st
                p_peaks[i] = st

        return p_peaks, p_minima
    

    def find_background(self, x, y, range=5, n_sigma=1, plot=False):
        """
        """
        # peak and minima indices
        p_peaks, p_minima = self.find_peaks(y, range=range)

        # isolate peak and minima indices
        mindata, peakdata = np.zeros_like(y), np.zeros_like(y)
        mindata[p_minima] = y[p_minima]
        peakdata[p_peaks] = y[p_peaks]

        # Fourier transform
        mindata_fft = np.fft.fft(mindata)
        peakdata_fft = np.fft.fft(peakdata)

        # Inverse Fourier transform the difference between peaks and minima
        extrema = np.abs(np.fft.ifft(peakdata_fft - mindata_fft))
        extrema_bool = extrema > 1
        bgema   = np.nonzero(extrema_bool)[0]
        bg_anchors = bgema[~np.in1d(bgema, p_peaks)]

        # filter split peaks
        cumint = cumulative_trapezoid(x[bg_anchors], y[bg_anchors], initial=0) # x and y reversed. works better
        intdiff = np.diff(np.insert(cumint, 0, 0))
        intmean, intstd = np.mean(intdiff), np.std(intdiff)
        intlier = np.abs(intdiff) > intmean + 2*n_sigma * intstd
        bg_anchors = bg_anchors[~intlier]

        # 1d linear interpolation on background anchor points
        interp = interp1d(x[bg_anchors], y[bg_anchors], bounds_error=False, fill_value=0)
        full_bg = interp(x)

        # filter highliers
        overshoot = (y - full_bg) < 0
        nodes = (y - full_bg) == 0
        overnodes = np.cumsum(overshoot.astype(int) + 2 * nodes.astype(int))
        diffnodes = np.diff(overnodes) > 1
        diffnodes = np.pad(diffnodes, (1,0), mode='constant', constant_values=1).astype(bool)
        y_nodes = y[diffnodes]
        keep_nodes = np.argmax(np.stack((y_nodes, np.roll(y_nodes, -1)), axis=-1), axis=1).astype(bool)
        filtered_bg_anchors = np.arange(len(y))[nodes][keep_nodes]
        
        interp = interp1d(x[filtered_bg_anchors], y[filtered_bg_anchors], bounds_error=False, fill_value=0)
        full_filtered_bg = np.clip(interp(x), 0, np.inf)

        if plot:
            self.plot(
                "background",
                x=x,
                full_bg=full_bg,
                full_filtered_bg=full_filtered_bg,
            )

        return full_filtered_bg
    

    def filter_peaks(self, y, n_sigma=2):
        """  """
        p_peaks, p_minima = self.find_peaks(y)

        # power transform data
        transformer = PowerTransformer()
        transformed_data = transformer.fit_transform(y.reshape(-1,1))[:,0]
        transformed_data = np.clip(transformed_data, transformed_data[0], np.inf) # remove values very close to 0
        
        self.power_lambda = transformer.lambdas_
        self.transformed_data = transformed_data

        # keep only the most prominent maxima
        i_peaks = transformed_data[p_peaks]
        i_peaks_mean = np.mean(i_peaks) # assumes exponential distribution of peak intensities
        i_peaks_std = np.std(i_peaks)
        v_peaks = i_peaks > (i_peaks_mean + n_sigma * i_peaks_std)
        p_peaks = p_peaks[v_peaks]

        p_sort = np.argsort(y[p_peaks])[::-1]    # sort by decreasing peak intensity
        peak_indices = np.take_along_axis(p_peaks, p_sort, axis=0)

        return peak_indices, p_minima, transformer


    def fourier_peaks(self, y, n_sigma=0, plot=False, *kwargs):
        """ """
        if not hasattr(self, 'peak_indices'):
            peak_indices, p_minima, transformer = self.filter_peaks(y, n_sigma=n_sigma, *kwargs)

        # cepstral analysis for peak filtering parameters
        y_fft = np.fft.fft(y)
        y_fft_logamp = np.log(np.abs(y_fft))
        y_cepstrum = np.abs(np.fft.ifft(y_fft_logamp))
        y_logcepstrum = np.nan_to_num(np.log(y_cepstrum))

        peak_limit = np.argmax(y_logcepstrum < np.mean(y_logcepstrum), axis=0)
        cep_maxs, cep_mins = self.find_peaks(y_logcepstrum[:peak_limit])

        self.peak_limit = peak_limit
        self.cep_maxs = cep_maxs
        self.cep_mins = cep_mins

        # power spectrum autocorrelation fitting
        y_autokernel = np.log(np.real(np.fft.ifft(np.abs(y_fft)**2)))
        y_autokernel_norm = y_autokernel[:peak_limit] - np.min(y_autokernel[peak_limit])
        x_autokernel = np.arange(len(y_autokernel_norm))
        x0 = (1, 0, cep_mins[0], cep_mins[0])
        upper_bounds = [np.inf]*4
        bounds = ([0]*4, upper_bounds)

        popt = least_squares(self.residual, x0=x0, bounds=bounds, args=(x_autokernel, y_autokernel_norm, self.multi_voigt))
        pcov = np.linalg.inv(popt.jac.T.dot(popt.jac))

        if plot:
            self.plot(
                "fourier",
                x=x_autokernel,
                fit=self.multi_voigt(x_autokernel, *popt),
                y=y_autokernel_norm,
            )

        return peak_indices, transformer, p_minima, peak_limit, cep_maxs, cep_mins, popt, pcov
    

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

    
    def fit_peaks(self, x, y, peak_indices, peak_dictionary=dict({}), plot=False):
        """  """
        # step size
        inc = np.median(np.diff(x))

        _, fwhm_array, *_ = self.peak_parameter_guess(y, peak_indices)
        median_fwhm = np.median(fwhm_array) * inc
        fwhm_limit =  5 * median_fwhm

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
                            # print(f'guess: fit: {np.array([x0, popt.x]).T}')
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
        """ Fit peak shoulders by fitting peaks to the residuals
        :param x: x-values of the spectrum
        :param y: y-values of the spectrum
        :param peak_indices: indices of the peaks
        :param residuals: residuals of the spectrum
        :param peak_dictionary: dictionary of peak parameters
        :param rng: range of the window to search for new peaks
        
        :return: updated peak dictionary and new peak indices
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
        """
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
                print(f'x window length: {len(x_window)}')
                known_peaks_inrange = idxs[(idxs >= node_left) & (idxs <= node_right)]
                refit_peak_num = len(known_peaks_inrange)
                
                x0 = np.ravel(np.array([peak_dictionary[key] for key in known_peaks_inrange]))
                # x0[2::4] = np.min([x0[2::4], (fwhm_limit/5)*np.ones_like(x0[2::4])], axis=0)
                # x0[3::4] = np.min([x0[3::4], (fwhm_limit/5)*np.ones_like(x0[3::4])], axis=0)
                lower_bounds, upper_bounds = np.zeros_like(x0), np.inf * np.ones_like(x0)

                lower_bounds[1::4], upper_bounds[1::4] = x0[1::4] - inc, x0[1::4] + inc
                # upper_bounds[2::4] = fwhm_limit * np.ones_like(upper_bounds[2::4])
                # upper_bounds[3::4] = fwhm_limit * np.ones_like(upper_bounds[3::4])
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
        """
        """
        if subtract_background:
            bg = self.find_background(x, y, *kwargs)
            y_bgsub = y - bg
        else:
            y = y_bgsub

        # find peaks and profile parameters
        peak_indices, transformer, *rest = self.fourier_peaks(y, n_sigma=n_sigma)
        print(f'fourier peaks done')
        peak_dictionary = self.fit_peaks(x, y_bgsub, peak_indices, plot=plot)
        print(f'fit peaks done')
        
        # filter and sort fit parameters
        parameters = np.array(list(peak_dictionary.values()))

        profile = self.multi_voigt(x, np.ravel(parameters))
        profile_outliers = profile > 1.5 * np.max(y)
        profile[profile_outliers] = 0
        residual_data = y_bgsub - profile

        peak_dictionary, new_peak_indices = self.fit_shoulders(x, y_bgsub, peak_indices, residual_data, peak_dictionary)
        print(f'fit shoulders done')
        peak_dictionary = self.fit_all(x, y_bgsub, peak_dictionary)
        print(f'fit all done')
        
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

        self.peak_dictionary = peak_dictionary
        self.spectrum_dictionary = spectrum_dictionary
        self.profile = profile
        self.residual_data = residual_data
        self.backgound = bg
        self.sorted_parameter_array = sorted_parameter_array

        yj_data = transformer.fit_transform(y_bgsub.reshape(-1,1))[:,0]

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
            )

        # return fig

    def plot(self, kind, **kwargs):
        """Central plotting utility for :class:`PeakyFinder`.

        Parameters
        ----------
        kind : {"background", "fourier", "spectrum"}
            Type of figure to produce.
        **kwargs : dict
            Data required for the selected plot.
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
            ax12.set_xlabel(
                rf"(intensity + 1)$^{{{float(self.power_lambda):.2g}}}$ / {float(self.power_lambda):.2g}"
            )
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
