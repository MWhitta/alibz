from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import huber
from scipy.special import voigt_profile as voigt

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor

from peaky_finder import PeakyFinder

class PeakyFitter():
    """ Fit object for storing and updating PeakyFitter fit parameters
    """

    def __init__(self, data_dir) -> None:
        self.elem = ['H', 'He', #row1
                    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', #row2
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', #row3
                    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', #row4
                    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', #row5
                    'Cs', 'Ba', #row6 alkali/alkaline earth
                    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', #row6 rare earths
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', #row6 transition metals
                    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'] #row 7 stable actinide elements
        
        self.finder = PeakyFinder(data_dir)
        self.finder.data.load_data()
        self.peak_dictionary = defaultdict(dict)
        self.peak_dictionary_refined = defaultdict(dict)
        self.fit_dictionary = defaultdict(dict)
        self.fit_dictionary_refined = defaultdict(dict)
        self.temperature_fit_dictionary = defaultdict(dict)
        self.blf = BaselineFinder()


    def line(self, x, m, b):
        """
        """
        y = m * x + b
        return y
    

    def huber_filter(self, data, delta=2):
        if len(data) < 3:
            return data

        # Split data into X and y
        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1]
        
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones_like(X), X])
    
        # Define the Huber loss function
        def huber_loss(params):
            intercept, slope = params
            pred_y = intercept + slope * X[:, 1]
            residuals = y - pred_y
            return np.sum(huber(delta, residuals))
    
        # Initial guess for intercept and slope
        initial_params = np.array([0.0, 1.0])
        
        # Minimize the Huber loss function
        result = minimize(huber_loss, initial_params)
        intercept, slope = result.x
        
        # Predict y values using the fitted model
        pred_y = intercept + slope * X[:, 1]
        
        # Calculate residuals
        residuals = np.abs(y - pred_y)
        
        # Filter data to keep only inliers (residuals within 2 standard deviations)
        std_residuals = np.std(residuals)
        inlier_mask = residuals < 2 * std_residuals
        
        filtered_data = data[inlier_mask]
        
        return filtered_data


    def multi_voigt(self, x, *params):
        total = np.zeros_like(x)
        for i in range(0, len(params) - 3, 4):
            amplitude = params[i]
            mean = params[i + 1]
            stddev = params[i + 2]
            gamma = params[i + 3]
            component = amplitude * voigt(x - mean, stddev, gamma)
            total += component
        return total
    

    def voigt_width(self, sigma, gamma):
        """
        """
        fwhm_gauss = 2*np.sqrt(2*np.log(2)) * sigma
        fwhm_lorentz = 2*gamma
        fwhm = 0.5346 * fwhm_lorentz + np.sqrt(0.2166 * fwhm_lorentz**2 + fwhm_gauss**2)
        return fwhm
    

    def get_peak_fit(self, fit_dict, element, ion):
        """
        """
        intensities = []
        wavelengths = []
        sigmas = []
        gammas = []
        covs = []
        try:
            fit_dict_list = fit_dict[element][ion]
            for fit in fit_dict_list:
                fit_popt, fit_pcov = fit
                fit_len = len(fit_popt)
                fit_inc = fit_len // 4
                intensities.append(fit_popt[:fit_inc+1])
                wavelengths.append(fit_popt[fit_inc:2*fit_inc+1])
                sigmas.append(fit_popt[2*fit_inc:3*fit_inc+1])
                gammas.append(fit_popt[3*fit_inc:4*fit_inc+1])
                covs.append(fit_pcov)

            intensities = np.array([x for xs in intensities for x in xs])
            wavelengths = np.array([x for xs in wavelengths for x in xs])
            sigmas = np.array([x for xs in sigmas for x in xs])
            gammas = np.array([x for xs in gammas for x in xs])

            return intensities, wavelengths, sigmas, gammas, covs
    
        except KeyError:
            print(f"Element '{element}' or ion '{ion}' not found in fit dictionary.")
            return np.array([]), np.array([]), np.array([]), np.array([]), []


    def get_temperature_fit(self, temperature_dict, element, ion):
        """
        """
        eopts, ecovs = [], []
        try:
            temperature_fit = temperature_dict[element][ion]
            for efit, ecov in temperature_fit:
                if len(efit) > 4:
                    efit = efit[:5]
                eopts.append(efit)
                ecovs.append(ecov)

            eopts = np.squeeze(eopts)
            ecovs = np.squeeze(ecovs)

            return eopts, ecovs
        
        except KeyError:
            print(f"Element '{element}' or ion '{ion}' not found in fit dictionary.")
            return np.array([]), np.array([])
        

    def peak_background_fit(self, 
                            data, 
                            peak_profile='voigt',
                            s=1,):
        """
        """

        x, y_raw = data
        x_min, x_max = np.argmin(x), np.argmax(x)
        y_min, y_max = np.min(y_raw), np.max(y_raw)
        inc = np.median(np.diff(x))
        
        peak_width, peak_indices = self.finder.calculate_peaks(y_raw)
        peak_num = len(peak_indices)
        peak_cutoff = np.sqrt(s) * np.mean(y_raw)
        y_mask = y_raw > peak_cutoff
        
        intensity, wavelength, sigma, gamma = y_raw[peak_indices], x[peak_indices], 2*peak_width*np.ones(peak_num), peak_width*np.ones(peak_num)

        initial_guesses = np.concatenate([intensity, wavelength, sigma, gamma])

        lower_bounds = np.concatenate([peak_num*[0], peak_num*[x[x_min]], peak_num*[inc], peak_num*[0]])  # intensity, wavelength, sigma, gamma
        upper_bounds = np.concatenate([peak_num*[y_max], peak_num*[x[x_max]], peak_num*[4*peak_width], peak_num*[4*peak_width]]) # intensity, wavelength, sigma, gamma

        try:
            popt, pcov = curve_fit(lambda x, *params: self.multi_voigt(x, *params) * y_mask, x, y_raw * y_mask, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
        except (RuntimeError, TypeError) as e:
            print(f"An error occurred during fitting: {e}")

        baseline = y_raw - self.multi_voigt(x, *popt)

        fig, axs = plt.subplots(2, 1, figsize=(22,8), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axs
        
        ax1.plot(x, y_raw, 'k', linewidth=0.5)
        ax1.plot(x[peak_indices], y_raw[peak_indices], 'ro', alpha=0.5)
        ax1.set_ylabel('intensity [au]')

        ax2.plot(x, baseline, 'b', linewidth=0.5)
        ax2.set_xlabel('wavelength [nm]')
        ax2.set_ylabel('intensity [au]')
        plt.show()


    def peak_fit(self, 
                x,
                y,
                r=15,
                peak_profile='voigt',
                peak_width=1,
                peak_inds=[],
                plot_all=False,
                ):
        """
        """
        inc = np.median(np.diff(x))
        width_inc = peak_width * inc
        all_mask = np.zeros_like(x)
        all_lower_bounds = []
        all_upper_bounds = []

        for i, p in enumerate(peak_inds):
            mask1, mask2 = x > (x[p] - 1*width_inc), x < (x[p] + 1*width_inc)
            peak_mask = mask1 * mask2

            fit_peaks, _ = find_peaks(y * peak_mask, width=2)

            y_max = np.max(y)
            width_scale = 2.5 * np.exp(-((y_max - y[p])/y_max)**2)

            if np.any(fit_peaks):
                lo_peak, hi_peak = np.min(fit_peaks), np.max(fit_peaks)
                peak_num = len(fit_peaks)
            else:
                fit_peaks = [np.argmax(y * peak_mask)]
                lo_peak, hi_peak = p, p
                peak_num = 1

            x_min = int(np.min([lo_peak - peak_width * width_scale, np.argmax(x * peak_mask>0)]))
            x_max = int(np.max([hi_peak + peak_width * width_scale, hi_peak + np.abs(hi_peak - x_min)]))
            mask1, mask2 = x > x[x_min], x < x[x_max]
            peak_mask = mask1 * mask2
            all_mask += peak_mask

            y0 = np.clip(y[fit_peaks], 10**-6, np.inf)
            intensity, wavelength, sigma, gamma = y0, x[fit_peaks], 2*width_inc*np.ones(peak_num), 2*width_inc*np.ones(peak_num)

            initial_guesses = np.concatenate([intensity, wavelength, sigma, gamma])

            lower_bounds = np.concatenate([peak_num*[0], peak_num*[x[x_min]], peak_num*[inc], peak_num*[inc]])  # intensity, wavelength, sigma, gamma
            upper_bounds = np.concatenate([peak_num*[2*y_max], peak_num*[x[x_max]], peak_num*[4*peak_width], peak_num*[4*width_inc]]) # intensity, wavelength, sigma, gamma
            all_lower_bounds.append(lower_bounds)
            all_upper_bounds.append(upper_bounds)

            if peak_profile == 'gaussian': #set gamma to zero
                array_mask = np.ones_like(initial_guesses).astype(bool)
                array_mask[3::4] = False
                initial_guesses[array_mask] = 0
                lower_bounds[array_mask], upper_bounds[array_mask] = 0
            if peak_profile == 'lorentzian': # set sigma to zero
                array_mask = np.ones(len(initial_guesses)).astype(bool)
                array_mask[2::4] = False
                initial_guesses[array_mask] = 0
                lower_bounds[array_mask], upper_bounds[array_mask] = 0

            try:
                print(f'initial_guesses {initial_guesses}')
                popt, pcov = curve_fit(lambda x, *params: self.multi_voigt(x, *params), x[peak_mask], y[peak_mask], p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
                self.peak_dictionary[i] = [popt, pcov]
            except (RuntimeError, TypeError) as e:
                print(f"An error occurred during initial fitting: {e}")

        for i, p in enumerate(peak_inds[:-1]):
            step = len(self.peak_dictionary[i][0]) // 4
            peak_id = np.argmin(np.abs(x[p] - self.peak_dictionary[i][0][step:2*step]))

            a, mu, sigma, gamma, *rest = self.peak_dictionary[i][0][peak_id::step]
            search_range = r * sigma
            search_min, search_max = mu - search_range, mu + search_range

            fit_keys = np.array(list(self.peak_dictionary.keys()))                                  # np arry of all keys in peak_dictionary
            mus = [self.peak_dictionary[f][0][peak_id+1] for f in fit_keys]                         # flattened list of wavelength values for each peak
            mus_range = (mus > search_min) * (mus < search_max)                                     # wavelengths within search range
            mus_window = (x > search_min) * (x < search_max)                                        # wavelengths in spectrum within search range
            fit_range_inds = fit_keys[mus_range]                                                    # fit keys within search range
            
            print(f'len(peak_inds) {len(peak_inds)}')
            print(f'mus {mus}')
            print(f'peak_id {peak_id}')
            print(f'fit_keys {fit_keys}')
            print(f'fit_range_inds {fit_range_inds}')
            print(f'all_lower_bounds {all_lower_bounds}')
            initial_guesses = [g for gs in [self.peak_dictionary[f][0][int(peak_id*4):int(peak_id*4)+4] for f in fit_range_inds] for g in gs]
            all_lower_bounds_i = [l for ls in [all_lower_bounds[f][int(peak_id*4):int(peak_id*4)+4] for f in fit_range_inds] for l in ls]
            all_upper_bounds_i = [u for us in [all_upper_bounds[f][int(peak_id*4):int(peak_id*4)+4] for f in fit_range_inds] for u in us]

            if x[p] > 0:
                try:
                    plt.plot(x[mus_window], y[mus_window], color='k')
                    plt.show()
                    popt, pcov = curve_fit(lambda x, *params: self.multi_voigt(x, *params), x[mus_window], y[mus_window], p0=initial_guesses, bounds=(all_lower_bounds_i, all_upper_bounds_i))
                    shape = len(popt) // 4
                    popt = np.reshape(popt, (shape,4))
                    sort_opt = np.argsort(popt[:,1])
                    popt = popt[sort_opt]
                    fit_range_inds = fit_range_inds[sort_opt]
                    for ind, pop in zip(fit_range_inds, popt):
                        self.peak_dictionary_refined[ind] = pop
                        if plot_all:
                            plt.figure(figsize=(5,5))
                            plt.plot(x[mus_window], y[mus_window], color='k')
                            plt.plot(x[mus_window], self.multi_voigt(x[mus_window], *pop), color='r', alpha=0.5)
                            plt.plot(x[mus_window], self.multi_voigt(x[mus_window], *self.peak_dictionary[ind][0]), color='0.5')
                            plt.xlabel('wavelength [nm]')
                            plt.ylabel('intensity [au]')
                            plt.show()
                    print(f'final fit {i} done')
                except (RuntimeError, TypeError) as e:
                    print(f"An error occurred during refined fitting: {e}")
                    for ind in fit_range_inds:
                        self.peak_dictionary_refined[ind] = np.array([a, mu, sigma, gamma])
                else:
                    for ind in fit_range_inds:
                        self.peak_dictionary_refined[ind] = np.array([a, mu, sigma, gamma])


    def ion_fit(self, 
                x,
                y,
                t_lo=5000,
                t_hi=15000,
                t_inc=500,
                wid=0.5, 
                rang=0.5,
                peak_frac=0,
                peak_profile='voigt',
                s=1,
                r=15,
                ex=[0,0],
                kernel_width=25,
                fit_threshold=0.01,
                plot_all=False,
                ):
        """
        """
        # initial peak calculation
        y_fit = np.copy(y)
        peak_width, peak_inds = self.finder.calculate_peaks(y_fit, s=s)
        peak_ind_argsort = np.argsort(y_fit[peak_inds])[::-1]
        peak_ind_argsort_frac = int(len(peak_ind_argsort) * peak_frac)
        peak_inds = peak_inds[peak_ind_argsort[:peak_ind_argsort_frac]]

        abc=0
        while abc< 1: #np.max(y_fit) > fit_threshold * np.max(y):
            # index guesses for peaks
            self.finder.rank(x,
                            y_fit,
                            t_lo=t_lo,
                            t_hi=t_hi,
                            t_inc=t_inc,
                            s=s,
                            ex=ex,
                            wid=wid,
                            rang=rang,
                            peak_frac=peak_frac, 
                            )
            
            probability_dictionary = self.finder.indexer.peak_probability_dictionary[0]
            element, ion, line_wavelength, distance, peak_match, peak_probability, absolute_intensity, *rest = probability_dictionary[:1][0]
            
            print(f'element: {element}[{ion}]')

            if element in self.fit_dictionary and ion in self.fit_dictionary[element]:
                continue
            else:
                try:
                    _, peak_inds = self.finder.calculate_peaks(y_fit, s=s)
                    peak_ind_argsort = np.argsort(y_fit[peak_inds])[::-1]
                    peak_ind_argsort_frac = int(len(peak_ind_argsort) * peak_frac)
                    peak_inds = peak_inds[peak_ind_argsort[:peak_ind_argsort_frac]]
                    
                    self.peak_fit(x,
                                    y_fit,
                                    r=r,
                                    peak_profile=peak_profile,
                                    peak_width=peak_width,
                                    peak_inds=peak_inds,   ##### NEED TO UPDATE PEAK INDICIES AFTER FITTING #######
                                    plot_all=plot_all,
                                    )
                    
                    peak_index_inds = np.array(list(self.peak_dictionary_refined.keys()))

                    peak_intensity = np.array([self.peak_dictionary_refined[f][0] for f in peak_index_inds])
                    peak_wavelength = np.array([self.peak_dictionary_refined[f][1] for f in peak_index_inds])
                    peak_sigma = np.array([self.peak_dictionary_refined[f][2] for f in peak_index_inds])
                    peak_gamma = np.array([self.peak_dictionary_refined[f][3] for f in peak_index_inds])

                    locs, amps = self.finder.indexer.intensity_probability_dictionary[element][ion]
                    
                    ritz_wavelength = []
                    valid_intensity, valid_wavelength, valid_sigma, valid_gamma = [], [], [], []
                    for a, w, s, g in zip(peak_intensity, peak_wavelength, peak_sigma, peak_gamma):
                        peak_diff = np.abs(w - locs)
                        if np.min(peak_diff) < rang:
                            close_wavelength = peak_diff.argmin()
                            ritz_wavelength.append(locs[close_wavelength])
                            valid_wavelength.append(w)
                            valid_intensity.append(a)
                            valid_sigma.append(s)
                            valid_gamma.append(g)

                    ritz_wavelength = np.array(ritz_wavelength)
                    valid_wavelength = np.array(valid_wavelength)

                    try:
                        plt.stem(ritz_wavelength, ritz_wavelength-valid_wavelength, linefmt='grey', markerfmt='k')
                        plt.xlabel('$\lambda_{Ritz}$ [nm]')
                        plt.ylabel('$\delta \lambda$ [nm]')
                        plt.annotate(f'{element}:{ion}', xy=(0.03, 0.95), xycoords='axes fraction')
                        plt.show()
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(f"An error occurred during fitting {element}[{ion}]: {e}")

                    valid_fit_params = np.array([valid_intensity, valid_wavelength, valid_sigma, valid_gamma])
                    self.fit_dictionary[element] = defaultdict(dict)
                    self.fit_dictionary[element][ion] = defaultdict(dict)
                    self.fit_dictionary[element][ion]['Ritz wavelength'] = ritz_wavelength
                    for key, value in zip(['intensity', 'wavelength', 'sigma', 'gamma'], valid_fit_params):
                        self.fit_dictionary[element][ion][key] = value

                    element_fit = self.multi_voigt(x, *valid_fit_params.T.flatten())
                    
                    plt.figure(figsize=(22,8))
                    plt.plot(x, y_fit, color='k', linewidth=0.5)
                    plt.fill_between(x, element_fit, np.zeros_like(x), color='r', alpha=0.5)
                    # plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
                    # plt.xlim([650, 690])
                    plt.show()
                    
                    y_fit -= element_fit
                    s *= 10

                    abc += 0.34
                    
                except (RuntimeError, TypeError) as e:
                    print(f"An error occurred during fitting: {e}")
                

        print(f'---------   fitting complete    ---------')

                    
    def emission_fit(self, t_lo=5000, t_hi=15000, t_inc=500, rang=0.7):
        """
        """
        t_array = np.arange(t_lo, t_hi, t_inc)[:, None] # make column vector for proper broadcasting, with temps as rows
        kT = self.finder.boltzmann_constant * t_array

        keys = self.finder.indexer.peak_probability_dictionary.keys()
        for key in list(keys):
            for i, (element, ion, line_wavelength, distance, peak_match, peak_probability, absolute_intensity, *rest) in enumerate(self.finder.indexer.peak_probability_dictionary[key][:1]):
                if element in self.temperature_fit_dictionary and ion in self.temperature_fit_dictionary[element]:
                    print(f'element {element}[{ion}] already fit')
                    continue
                
                else:
                    # import fit parameters
                    intensities, wavelengths, sigmas, gammas, covs = self.get_peak_fit(self.fit_dictionary, element, ion)

                    # import calculated data from database
                    ionization, peak_loc, gA, Ek = self.finder.db.lines(element)[:, [0, 1, 3, 5]].T.astype(float)
                    Zind = ionization == float(ion)

                    wavelength_inds = []
                    location_comparison = []
                    emission_array = []
                    intensity_array = []
                    for w, a, s in zip(wavelengths, intensities, sigmas):
                        peak_diff = np.abs(w - peak_loc[Zind])
                        if np.min(peak_diff) < rang:
                            close_wavelength = peak_diff.argmin()
                            
                            wavelength_inds.append(close_wavelength)
                            location_comparison.append((peak_loc[Zind][close_wavelength], w))

                            i_ind = int(float(ion)-1)
                            emission_array.append(peak_loc[Zind][close_wavelength]**-1 * gA[Zind][close_wavelength] * np.exp(-Ek[Zind][close_wavelength] / kT) / self.finder.maker.sb.partition(element, t_array, i_ind))
                            intensity_array.append(a*s*np.sqrt(np.pi))

                    emission_array = np.squeeze(emission_array)
                    intensity_array = np.squeeze(intensity_array)
                    
                    fitted1 = intensity_array > 10**1
                    emission_array = emission_array[fitted1]
                    intensity_array = intensity_array[fitted1]
                    location_comparison = np.array(location_comparison)[fitted1]
                    sorted = np.argsort(location_comparison[:, 0], axis=0)
                    location_comparison = location_comparison[sorted]

                    if np.shape(location_comparison)[0] > 1:
                        location_comparison_linear = self.huber_filter(location_comparison)
                        if np.shape(location_comparison_linear)[0] > 1:
                            lopt, lcov = curve_fit(self.line, location_comparison_linear[:, 0], location_comparison_linear[:, 1])
                        else:
                            lopt, lcov = curve_fit(self.line, location_comparison[:, 0], location_comparison[:, 1])

                        location_fit_residuals = location_comparison[:, 1] - self.line(location_comparison[:, 0], *lopt)
                        location_exclude = np.abs(location_fit_residuals) > 0.25

                        plt.stem(location_comparison[~location_exclude, 0], location_fit_residuals[~location_exclude], linefmt='grey', markerfmt='k')
                        plt.stem(location_comparison[location_exclude, 0], location_fit_residuals[location_exclude], linefmt='r', markerfmt='r')
                        plt.xlabel('Ritz wavelength [nm]')
                        plt.ylabel('fit residual [nm]')
                        plt.annotate(f'{element}:{ion}', xy=(0.03, 0.95), xycoords='axes fraction')
                        plt.show()

                        emission_array = emission_array[location_exclude]
                        intensity_array = intensity_array[location_exclude]

                    fitted2 = emission_array / np.max(emission_array, axis=0, keepdims=True) > 10**-3
                    t_mask = []
                    for i, (emission, fitted) in enumerate(zip(emission_array.T, fitted2.T)):
                        emission_comparison = np.array((emission[fitted], intensity_array[fitted])).T

                        if np.shape(emission_comparison)[0] > 2:
                            emission_comparison_linear = self.huber_filter(emission_comparison)
                            if np.shape(emission_comparison_linear)[0] > 2:
                                try:
                                    eopt, ecov = curve_fit(self.line, np.log10(emission_comparison_linear[:, 0]), np.log10(emission_comparison_linear[:, 1]))
                                    if element in self.temperature_fit_dictionary and ion in self.temperature_fit_dictionary[element]:
                                        self.temperature_fit_dictionary[element][ion].append([eopt, ecov])
                                    elif ion not in self.temperature_fit_dictionary[element]:
                                        self.temperature_fit_dictionary[element][ion] = [[eopt, ecov]]
                                    t_mask.append(True)

                                except (RuntimeError, TypeError) as e:
                                    print(f"An error occurred during fitting {element}[{ion}] with Huber regression: {e}")
                                    print(f'Trying linear regression')
                                    try:
                                        eopt, ecov = curve_fit(self.line, np.log10(emission_comparison[:, 0]), np.log10(emission_comparison[:, 1]))
                                        print(f'Linear regression successful')
                                        t_mask.append(True)
                                    except (RuntimeError, TypeError) as e:
                                        print(f"An error occurred during fitting {element}[{ion}]: {e}")
                                        t_mask.append(False)

                            elif np.shape(emission_comparison_linear)[0] <= 2:
                                t_mask.append(False)

                            # plt.scatter(emission_comparison[:, 0], emission_comparison[:, 1], color='r')
                            # plt.scatter(emission_comparison_linear[:, 0], emission_comparison_linear[:, 1], color='k')
                            # plt.plot(emission_comparison[:, 0], 10**(self.line(np.log10(emission_comparison[:, 0]), *eopt)), 'k-')

                            # plt.xscale('log')
                            # plt.yscale('log')
                            # plt.xlabel('calculated')
                            # plt.ylabel('measured')
                            # plt.annotate(f'{element}:{ion}', xy=(0.03, 0.95), xycoords='axes fraction')
                            # plt.show()
                        else:
                            t_mask.append(False)
                            # print(f'element {element}[{ion}] not fit - only {np.shape(emission_comparison)[0]} valid points')
                try:
                    temperature_fit, temperature_covs = self.get_temperature_fit(self.temperature_fit_dictionary, element, ion)
                    minKT_ind = np.argmin(temperature_covs[:,0,0])
                    min_kT, min_cov = kT[t_mask][minKT_ind], temperature_covs[minKT_ind,0,0]

                    fig, ax = plt.subplots(constrained_layout=True)
                    ax.plot(kT[t_mask], temperature_covs[:,0,0], color='k')
                    ax.scatter(min_kT, min_cov, color='k')
                    ax.set_xlabel('kT [eV]')
                    ax.set_ylabel('|$\sigma$|')
                    secax = ax.secondary_xaxis('top', functions=(self.finder.maker.K2eV, self.finder.maker.eV2K))
                    secax.set_xlabel('T [K]')
                    
                    ax2 = ax.twinx()
                    ax2.plot(kT[t_mask], temperature_fit[:,0], color='b')
                    ax2.set_ylabel('|$\epsilon_{line} / \epsilon_{calculated}$|')
                    ax2.yaxis.label.set_color('b')
                    ax2.spines['right'].set_color('b')
                    ax2.tick_params(axis='y', colors='b')

                    ax.set_box_aspect(aspect=1)
                    plt.show()
                    print(f'temperature, temperature_fit {min_kT, temperature_fit[minKT_ind]}')

                except (ValueError) as e:
                    print(f"Invalid dictionary values for {element}[{ion}]: {e}")


class BaselineFinder():
    """
    """

    def __init__(self) -> None:
        """
        """
        pass


    def calculate_baseline(self, x, y, peaks, kernel_width):
        """ calculates an intensity baseline function to subtract from data
        """

        def uniform_kernel(kernel_width):
            kernel = np.ones(kernel_width) / kernel_width
            return kernel
        
        def find_minima_between_peaks(data, peaks):
            sorted_peaks = np.sort(peaks)
            minima_indices = []
            for i in range(len(sorted_peaks) - 1):
                start = sorted_peaks[i]
                end = sorted_peaks[i+1]
                if end > start:
                    test_range = data[start:end+1]
                    min_test_range = min(test_range)
                    if (start > min_test_range) & (end > min_test_range):
                        min_index = np.argmin(data[start:end+1]) + start
                        minima_indices.append(min_index)
            minima_indices.sort()
            return minima_indices
        
        def find_local_minima_indices(arr):
            padded_arr = np.pad(arr, (1, 1), mode='constant', constant_values=np.inf)
            local_minima_mask = (padded_arr[1:-1] < padded_arr[:-2]) & (padded_arr[1:-1] < padded_arr[2:])
            return np.where(local_minima_mask)[0]

        mask = np.zeros_like(x, dtype=bool)
        mask[peaks] = True
        mask_weight = y[mask]
        mask[peaks] = mask[peaks].astype(float) * mask_weight

        kernel = uniform_kernel(kernel_width)
        density = np.correlate(mask, kernel, mode='same')
        density_mask = density > 0
        true_indices = np.where(~density_mask)[0]

        subsets = y[~density_mask]

        # Find local minima indices in the subset
        local_minima_subset_indices = find_local_minima_indices(subsets)

        # Map local minima indices back to the original indices
        local_minima_indices = true_indices[local_minima_subset_indices]

        baseline = np.interp(x, x[local_minima_indices], y[local_minima_indices])
        overestimate = baseline > y
        baseline[overestimate] = y[overestimate]

        fig, axs = plt.subplots(3, 1, figsize=(22,8), gridspec_kw={'height_ratios': [4, 1, 4]})
        ax1, ax2, ax3 = axs
        
        ax1.plot(x, y, 'k', linewidth=0.5)
        ax1.plot(x[peaks], y[peaks], 'ro', alpha=0.5)
        ax1.set_ylabel('intensity [au]')

        ax2.plot(x, baseline, 'b', linewidth=0.5)
        ax2.set_xlabel('wavelength [nm]')
        ax2.set_ylabel('intensity [au]')

        ax3.plot(x, y-baseline, 'k', linewidth=0.5)
        # ax1.plot(x[peaks], y[peaks], 'ro', alpha=0.5)
        ax3.set_ylabel('intensity - baseline [au]')
        plt.show()

        return baseline
        


class FitElement():
    """ Class to store element indexing information for use in classification and quantificiation
    """

    def __init__(self, element, ion=None) -> None:
        self.element = element
        self.ions = [ion]
        self.match_wavelengths = {}
        self.match_peak_indices = {}
        self.unmatch_peaks = {}


    def matching_peaks(self, ion, peak_match):
        """ retains `peaks` in spectrum matching those of `ion`
        """
        if ion not in self.match_peak_indices:
            self.match_peak_indices[ion] = peak_match
            self.ions.append(ion)
        self.ions = [x for i, x in enumerate(self.ions) if x not in self.ions[:i]]

        if ion not in self.unmatch_peaks:
            self.unmatch_peaks[ion] = ~peak_match


class FitPeak():
    """ Class to store element indexing information for use in classification and quantificiation
    """

    def __init__(self, peak) -> None:
        self.peak = peak
        self.candidates = {}

    def matching_elements(self, element, ion):
        if element not in self.candidates:
            self.candidates[element] = [ion]
        else:
            self.candidates[element].append(ion)