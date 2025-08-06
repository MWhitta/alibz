import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.special import voigt_profile as voigt

from utils.database import Database
from peaky_maker import PeakyMaker


class PeakyIndexer():
    """ PeakyIndexer is a python class that contains tools for indexing and classifying laser plasma spectroscopy data.
    Args:
        data_dir (str literal): Path to the data directory upon which the `PeakyFinder` class loads data.
    Attributes:
        data (object):
    Methods:
    """

    def __init__(self, PeakyFinder, dbpath="db") -> None:
        self.finder = PeakyFinder
        self.maker = PeakyMaker(dbpath)
        self.db = Database(dbpath)

        # Physical Constants
        self.k = 8.617333262 * 10 ** -5 # Boltzmann constant (eV/K)
        self.h = 4.135667696 * 10 ** -15 # Plank constant eV s
        self.c = 2.99792458 * 10**8 # Speed of light in a vacuum (m/s)
        self.me = 0.51099895000 * 10**6 # electron mass eV / c^2


    def ground_state(self, threshold=0.001):
        """ identify ground state transitions for each element
        """
        self.ground_states = {}
        for el in self.db.elements:
            ionization, peak_loc, gA, Ei, Ek, gi, gk = self.db.lines(el)[:, [0, 1, 3, 4, 5, 12, 13]].T.astype(float)
            ions = np.unique(ionization)
            
            for ion in ions:
                Eind = ionization == ion #indices of single ionization state
                groundEi = Ei[Eind] == 0 #indices of ground state for ionization state
                groundpeak = peak_loc[Eind][groundEi] #ground state peak wavelengths
                groundEk = gk[Eind][groundEi]*np.exp(-Ek[Eind][groundEi]/1000) #ground state upper level occupation probabilities
                groundgA = 1/gk[Eind][groundEi] * gA[Eind][groundEi]
                
                if len(groundEk) > 0:
                    if np.max(groundEk) > 0:
                        groundT = groundgA * groundEk
                        groundEkprob = np.divide(groundT, np.max(groundgA * groundEk), out=np.zeros_like(groundEk), where=groundEk > 0) > threshold
                        self.ground_states[el][ion] = groundpeak[groundEkprob], groundT[groundEkprob]
    

    def distance_decay(self, w1, w2, s):
        """ Gaussian distance metric for proximate peaks
        """
        d = np.exp(-(w1 - w2)**2 / (2 * s**2))
        return d


    def peak_proximity(self, peaks, reference, shift_tolerance):
        """ determine minimum proximity between data and reference peak locations
        """
        peak_prox = peaks[:, np.newaxis] - reference
        peak_match = np.min(np.abs(peak_prox), axis=-1) <= shift_tolerance
        return peak_match
    
    
    def peak_interference(self, 
                             x,
                         wid=1,
                        rang=1,
               log_amp_limit=6,
             ground_state=True,
             element_list=None
                               ):
        """ Determine which ions are most likely to interfere at a given peak location
        """
        
        if element_list is None:
            element_list = self.db.elements

        close_peaks = []
        for el in element_list: # elements
            ionization, peak_loc, gA, Ei = self.db.lines(el)[:,[0, 1, 3, 4]].T.astype(float)
            ions = np.sort(np.unique(ionization))
            for ion in ions:
                i_ind = ionization == ion
                if ground_state:
                    E_ind = Ei == 0
                else:
                    E_ind = Ei > 0
                i_peak_loc = peak_loc[i_ind * E_ind]
                i_gA = gA[i_ind * E_ind]

                if len(i_peak_loc) > 0:
                    close_candidates = i_peak_loc - x <= wid
                    i_peak_loc_close = i_peak_loc[close_candidates]
                    i_gA_close = np.log10(i_gA[close_candidates])
                else:
                    i_peak_loc_close = []
                    i_gA_close = []
                
                if len(i_peak_loc_close) > 0:
                    i_peak_loc_round = np.round(i_peak_loc_close, 3)
                    distance_metric = self.distance_decay(x, np.array(i_peak_loc_round), wid)
                    amplitude_metric = np.max(i_gA_close) - log_amp_limit

                    for location, distance, A in zip(i_peak_loc_close, distance_metric, i_gA_close):
                        if np.abs(x-location) < rang:
                            if A > amplitude_metric:
                                close_peaks.append([el, ion, location, x, distance, A]) # log10(A)

        if len(close_peaks) > 0:
            close_peaks.sort(key=lambda dp: dp[-1], reverse=True)
        else:
            close_peaks = [[]]
        
        return close_peaks


    def peak_match(self, peak_array, ground_state=True, element_list=None):
        """
        """
        peak_idx = np.arange(len(peak_array))
        
        peak_match_dictionary = dict({}) # store matched element peak information

        for idx in peak_idx:
            close_peaks = self.peak_interference(peak_array[idx, 1], ground_state=ground_state, element_list=element_list)
            
            if any(close_peaks[0]): # if any close peaks are found, match them
                
                for peak in close_peaks:
                    el, *rest = peak
                    match_data = Element(el)
                    match_data.update(peak, idx)

                    if el not in peak_match_dictionary:
                        peak_match_dictionary[el] = match_data 
                    else:
                        peak_match_dictionary[el].update(peak, idx)
        
        return peak_match_dictionary



    def spectrum_match(self, peak_array, ground_state=True, plot=False):
        """ 
        peak_array: (n, 4) array of peak data (like PeakyFinder.sorted_parameter_array)
        """
        
        data_peak_centers = peak_array[:, 1]
        data_peak_widths = self.PeakyFinder.voigt_width(peak_array[:, 2], peak_array[:, 3])
        mean_width = 2 * np.mean(data_peak_widths)

        min_int = np.max(peak_array[:, 0]) # maximum peak intensity in fitted data
        peak_idx = 0
        while min_int > 200: 
            print(f'min_int: {min_int}')
            close_peaks = self.peak_interference(peak_array[peak_idx, 1], ground_state=ground_state)
            
            if any(close_peaks[0]): # if any close peaks are found, match them
                close_elements, close_ions = np.array(close_peaks)[:, 0:2].T
                close_elion = []
                for (e, i) in zip(close_elements, close_ions):
                    close_elion.append(e+i)

                scores = []
                for peak in close_peaks: # loop over close peaks to look for spectrum matches
                    element, ion, loc, _ = peak
                    # retreive element data from database
                    ionization, element_peaks, gA, Ei, Ek, gi, gk = self.db.lines(element)[:, [0, 1, 3, 4, 5, 12, 13]].T.astype(float)
                    
                    # search for experimental peaks that are close to predicted peaks and compare intensities for similarity
                    if float(ion) < 6: # hard coded ionization cutoff. Incorporate physical model with ionization energy
                        i_ind = np.array(ionization).astype(float) == float(ion)
                        w_ind = (element_peaks > 240) & (element_peaks < 910)
                        i_ind *= w_ind
                        i_peaks = element_peaks[i_ind] # predicted peak locations
                        i_amps = element_peaks[i_ind]**-1 * gA[i_ind] / gk[i_ind] * np.exp(-Ek[i_ind] / 1) # predicted peak amplitudes at 10000K
                        n_amps = np.divide(i_amps, np.max(i_amps), where=np.max(i_amps)>0, out=np.zeros_like(i_amps))
                        signal_bool = n_amps > 10**-3
                        n_amps = n_amps[signal_bool]
                        i_peaks = element_peaks[i_ind][signal_bool] # predicted peak locations > 10^-3

                        peak_diff = np.abs(data_peak_centers[:, np.newaxis] - i_peaks[np.newaxis, :]) # pairwise comparison of experimental and predicted peaks
                        diffs1, diffs2 = np.min(peak_diff, axis=1), np.min(peak_diff, axis=0) # closest peaks in data, prediction
                        close1, close2 = diffs1 < data_peak_widths, diffs2 < mean_width # indeces of the close peaks in data, prediction
                        
                        peak_locs = data_peak_centers[close1]
                        peak_amps = peak_array[close1, 0] # peaks found in data close to ion peaks
                    
                        peak_data = np.stack((peak_locs, peak_amps)).T # (x, y) pairs of (wavelength, intensity)
                        peak_sort = np.argsort(peak_locs) # sort by increasing wavelength
                        peak_data = peak_data[peak_sort] # sorted data
                        
                        amps = n_amps[close2] # predicted peaks close to those found in data
                        peak_locations = i_peaks[close2] # predicted peak wavelengths close to those found in data

                        # combine overlapping predicted amplitudes within experimental resolution
                        split_indices = np.where(np.diff(peak_locations) > 1/15)[0] + 1
                        groups = np.split(peak_locations, split_indices)
                        amp_groups = np.split(amps, split_indices)
                        group_means = np.array([np.average(group, weights=amp) for (amp, group) in zip(amp_groups, groups)])
                        amp_sums = np.array([amp.sum() for amp in amp_groups])

                        # rescale nonzero peak amps such that the matched peak is set to 1 
                        if len(peak_amps) > 0:
                            predicted_peak_location = np.argmin(np.abs(group_means - peak_array[0, 1])) # index of predicted peak
                            predicted_amplitude_sum = amp_sums[predicted_peak_location] # amplitude of predicted peak
                            matched_wavelength      = group_means[predicted_peak_location] # wavelength of predicted peak
                            matched_data_location   = np.argmin(np.abs(peak_locs - matched_wavelength)) # index of matched data
                            predicted_scale = np.divide(peak_amps[matched_data_location], predicted_amplitude_sum, where=amp_sums[predicted_peak_location]>0, out=np.zeros_like(peak_amps[matched_data_location]))
                            
                            amp_sums *= predicted_scale
                            amp_data = np.stack((group_means, amp_sums)).T # predicted (wavelength, intensity) scaled to match experimental data
                            amp_sort = np.argsort(group_means)
                            amp_data = amp_data[amp_sort]

                            match_score = np.max(amp_sums)
                            scores.append(match_score)

                            print(f'match: {element}[{ion}]  match_score {match_score}')
                            print(f'amp_sums: {amp_sums}')

                        if plot:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,5))
                            ax1.scatter(peak_locs[peak_sort], peak_amps[peak_sort])
                            ax1.set_xlabel('wavelength [nm]')
                            ax1.set_ylabel('measured amp')
                            ax1.text(0.95, 0.95, f'{element}[{ion}]', transform=ax1.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
                            ax1.minorticks_on()
                            
                            ax2.scatter(group_means[amp_sort], amp_sums[amp_sort])
                            ax2.set_xlabel('wavelength [nm]')
                            ax2.set_ylabel('predicted amp')
                            ax2.minorticks_on()
                            
                            plt.plot()
                            plt.show()

                    else: # ionization state too high to be reasonable
                        scores.append(-np.inf) # goes to zero when softmax is applied

                score_probabilities = np.exp(scores)/sum(np.exp(scores)) # softmax
                matched_ion_idx = np.abs(score_probabilities) > 10**-3 # hard coded cutoff - find better method
                matched_ions = close_peaks[matched_ion_idx]

                for peak in matched_ions:
                    element, ion, loc, _ = peak
                    spectrum_dictionary.setdefault(str(element), {}).setdefault(str(ion), []).append(loc)

                peak_idx += 1
                min_int = search_peak_array[peak_idx, 0]
            
            else: # no matching peaks were found 
                if ground_state == True:
                    excited_dictionary['unindexed'].append(search_peak_array[0, 1])
                else:
                    print('hmmmmmm. There are really no peaks in this vicinity?')

                peak_idx += 1
                min_int = search_peak_array[peak_idx, 0]

        return spectrum_dictionary, excited_dictionary


    def spectrum_match2(self, peak_array, t_lo=10000, t_hi=11000, t_inc=500, i_tol=100, ground_state=True, plot=False):
        """
        """
        t_array = np.arange(t_lo, t_hi, t_inc)[:, np.newaxis] # make column vector for proper broadcasting, with temps as rows
        kT = self.k * t_array

        data_peak_centers = peak_array[:, 1]
        data_peak_widths = self.PeakyFinder.voigt_width(peak_array[:, 2], peak_array[:, 3])
        mean_width = 2 * np.mean(data_peak_widths)

        spectrum_dictionary  = dict({}) # store matched element peak information
        candidate_dictionary = dict({}) # store matched element peak information
        intensity_dictionary = dict({}) # store modeled intensity information
        element_dictionary   = dict({}) # store peak information that matches element
        rejected_dictionary  = dict({}) # store rejected elements
        excited_dictionary   = dict({'unindexed' : []}) # store rejected peaks
        search_peak_array    = peak_array.copy()
        peak_array_idx       = np.arange(len(peak_array))

        min_int = np.max(peak_array[:, 0]) # maximum peak intensity in fitted data
        peak_idx = 0
        while min_int > i_tol: 
            print(f'min_int: {min_int}')
            close_peaks = self.peak_interference(search_peak_array[peak_idx, 1], ground_state=ground_state)
            
            if any(close_peaks[0]): # if any close peaks are found, match them
                close_elements, close_ions = np.array(close_peaks)[:, 0:2].T
                close_elion = []
                for (e, i) in zip(close_elements, close_ions):
                    close_elion.append(e+i)

                # eleminate previously rejected cendidates from search
                if np.any(rejected_dictionary.items()):
                    reject_elements, reject_ions = np.array([
                        [outer_key, inner_key]
                        for outer_key, inner_dict in rejected_dictionary.items()
                        for inner_key in inner_dict.keys()]).T
                
                    close_rejected_species = []
                    for (e, i) in zip(reject_elements, reject_ions):
                            close_rejected_species.append(e+i)
                else:
                    close_rejected_species = [[]]                

                close_peaks_rejected = np.in1d(close_elion, close_rejected_species)
                close_peaks = np.array(close_peaks)[~close_peaks_rejected]
                print(f'close_peaks: {close_peaks}')
                scores = []
                for peak in close_peaks: # loop over close peaks to look for spectrum matches
                    element, ion, loc, _ = peak
                    
                    if element in intensity_dictionary and ion in intensity_dictionary[element]:
                        element_dictionary.setdefault(element, {}).setdefault(ion, []).append(loc) # save peak location to element dictionary
                        search_peak_array = search_peak_array[1:] # advance down peak list
                    
                    # retreive element data from database
                    ionization, element_peaks, gA, Ei, Ek, gi, gk = self.db.lines(element)[:, [0, 1, 3, 4, 5, 12, 13]].T.astype(float)
                    
                    # search for experimental peaks that are close to predicted peaks and compare intensities for similarity
                    
                    if float(ion) < 6: # hard coded ionization cutoff. Incorporate physical model with ionization energy?
                        i_ind = np.array(ionization).astype(float) == float(ion)
                        w_ind = (element_peaks > 240) & (element_peaks < 910)
                        i_ind *= w_ind
                        i_peaks = element_peaks[i_ind]
                        peak_diff = np.abs(data_peak_centers[:, np.newaxis] - i_peaks[np.newaxis, :]) # pairwise comparison of experimental and predicted peaks
                        diffs1, diffs2 = np.min(peak_diff, axis=1), np.min(peak_diff, axis=0) # closest peaks in data, prediction
                        close1, close2 = diffs1 < data_peak_widths, diffs2 < mean_width # indeces of the close peaks
                        
                        peak_locs = data_peak_centers[close1]
                        peak_amps = peak_array[close1, 0] # peaks found in data close to ion peaks
                    
                        peak_data = np.stack((peak_locs, peak_amps)).T # (x, y) pairs of (wavelength, intensity)
                        peak_sort = np.argsort(peak_locs) # sort by increasing wavelength
                        peak_data = peak_data[peak_sort] # sorted data

                        all_amps = element_peaks[i_ind]**-1 * gA[i_ind] / gk[i_ind] * np.exp(-Ek[i_ind] / kT) # predicted amplitudes
                        amps = all_amps[0, close2] # ion peaks close to those found in data
                        peak_locations = i_peaks[close2] # ion peak wavelengths close to those found in data

                        # combine overlapping predicted amplitudes within experimental resolution
                        split_indices = np.where(np.diff(peak_locations) > 1/15)[0] + 1
                        groups = np.split(peak_locations, split_indices)
                        amp_groups = np.split(amps, split_indices)
                        group_means = np.array([np.average(group, weights=amp) for (amp, group) in zip(amp_groups, groups)])
                        amp_sums = np.array([amp.sum() for amp in amp_groups])

                        # rescale nonzero peak amps such that the matched peak is set to 1 
                        if len(peak_amps) > 0:
                            predicted_peak_location = np.argmin(np.abs(group_means - search_peak_array[0, 1])) # index of predicted peak
                            predicted_amplitude_sum = amp_sums[predicted_peak_location] # amplitude of predicted peak
                            matched_wavelength      = group_means[predicted_peak_location] # wavelength of predicted peak
                            matched_data_location = np.argmax(np.abs(peak_locs - matched_wavelength) < mean_width) # index of matched data
                            predicted_scale = np.divide(peak_amps[matched_data_location], predicted_amplitude_sum, where=amp_sums[predicted_peak_location]>0, out=np.zeros_like(peak_amps[matched_data_location]))
                            
                            amp_sums *= predicted_scale
                            amp_data = np.stack((group_means, amp_sums)).T # predicted (wavelength, intensity) scaled to match experimental data
                            amp_sort = np.argsort(group_means)
                            amp_data = amp_data[amp_sort]

                        match_amps = peak_amps[peak_amps < min_int]
                        match_amp_sums = amp_sums[amp_sums < min_int]
                        match_score = np.log(np.divide(np.sum(match_amps),  np.sum(match_amp_sums), where=np.sum(match_amp_sums)>0, out=np.zeros_like(np.sum(match_amp_sums)))) # ratio of experimental to predicted intensity
                        scores.append(match_score)

                        print(f'match: {element}[{ion}]  match_score {match_score}')

                        if plot:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,5))
                            ax1.scatter(peak_locs[peak_sort], peak_amps[peak_sort])
                            ax1.set_xlabel('wavelength [nm]')
                            ax1.set_ylabel('measured amp')
                            ax1.text(0.95, 0.95, f'{element}[{ion}]', transform=ax1.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
                            ax1.minorticks_on()
                            
                            ax2.scatter(group_means[amp_sort], amp_sums[amp_sort])
                            ax2.set_xlabel('wavelength [nm]')
                            ax2.set_ylabel('predicted amp')
                            ax2.minorticks_on()
                            
                            plt.plot()
                            plt.show()

                    else: # ionization state too high to be reasonable
                        scores.append(-np.inf) # goes to zero when softmax is applied

                score_probabilities = np.exp(scores)/sum(np.exp(scores)) # softmax
                matched_ion_idx = np.abs(score_probabilities) > 10**-3 # hard coded cutoff - find better method
                matched_ions = close_peaks[matched_ion_idx]

                for peak in matched_ions:
                    element, ion, loc, _ = peak
                    spectrum_dictionary.setdefault(str(element), {}).setdefault(str(ion), []).append(loc)

                peak_idx += 1
                min_int = search_peak_array[peak_idx, 0]
            
            else: # no matching peaks were found 
                if ground_state == True:
                    excited_dictionary['unindexed'].append(search_peak_array[0, 1])
                else:
                    print('hmmmmmm. There are really no peaks in this vicinity?')

                peak_idx += 1
                min_int = search_peak_array[peak_idx, 0]

        return spectrum_dictionary, excited_dictionary

    
    def peak_compare(self, x, y, p, ploc, locations, amps, peaks, limit, rang):
        """ compare the expected peaks of an ion to the observed peaks and return the fraction found
        """
        if np.sum(ploc) == 0:
            peaks_found = np.zeros_like(amps[:, 0])
            return peaks_found
        
        else:
            min, max = np.min(y), np.max(y)
            intensities_max = max - min
            relint = (y[p]-min) / intensities_max

            amp = np.sum(amps[:, ploc], axis=1) # sum accounts for multiple peaks in same location
            a_max = np.max(amps, axis=1)
            a_norm = np.divide(amp, a_max, where=a_max>0)

            if np.all(amp < (limit * a_max)):
                peaks_found = np.zeros_like(amps[:, 0])
                return peaks_found
                
            else:
                dqe_mask = (locations > 240) * (locations < 920)
                a_mask = a_norm > limit

                valid_peaks = amps > ((limit / relint) * (a_max * a_mask)[:, np.newaxis])
                valid_peaks *= dqe_mask
                valid_locations = locations[np.newaxis, :] * valid_peaks

                differences = np.abs(valid_locations - x[peaks][:, np.newaxis, np.newaxis])
                index_match = np.min(differences, axis=0) < rang

                sum_match, sum_all = np.sum(index_match, axis=1), np.sum(valid_peaks, axis=1)
                peaks_found = np.divide(sum_match, sum_all, out=np.zeros_like(sum_all).astype(float), where=sum_all>0)

                return peaks_found
      
    
    def index_probabilities(self,  
              x,
              y,
              peak_indices,
              peak_array,
              wid=1, 
              rang=0.5,
              t_lo=10000,
              t_hi=11000,
              t_inc=500,
              ground_state=True,
              verbose=True,
              ):
        """ `index_probabilities` is a method to make and store peak index data
        """
        t_array = np.arange(t_lo, t_hi, t_inc)[:, np.newaxis] # make column vector for proper broadcasting, with temps as rows
        kT = self.k * t_array

        total_peaks = int(np.count_nonzero(peak_indices) * 0.1)
        self.peak_lim = 0.05
        for i, (p, v) in enumerate(zip(peak_indices, peak_array)):
            
            if verbose:
                print(f'peak {i+1} / {total_peaks+1} successfully indexed')
            px = self.peak_interference(x[p], wid=wid, rang=rang, ground_state=ground_state)
            
            if len(px[0]) < 1:
                if verbose:
                    print(f'no ground state transitions for found for peak {i+1} at {x[p]}')
                    print(f'excited orphan created for later assignment')
                self.peak_probability_dictionary[i] = []
            
            else: 
                peak_match = []
                temperature_match = []
                absolute_intensity = []

                if ground_state:
                    if not hasattr(self, 'ground_state'):
                        self.ground_state()

                for element, ion, wavelength, distance in px:
                    if element in self.intensity_probability_dictionary and ion in self.intensity_probability_dictionary[element]:
                        ploc_all, amps = self.intensity_probability_dictionary[element][ion]

                        ionization, peak_loc = self.db.lines(element)[:, [0, 1]].T.astype(float)
                        Zind = ionization == ion
                        ploc = peak_loc[Zind] == wavelength
                        
                        if ground_state:
                            gs = self.ground_states[element]
                            if ion in gs:
                                ploc_ground = [ploc == gsw for gsw in gs[ion]]
                                amp = np.sum(amps[:, ploc_ground], axis=1) # sum accounts for multiple peaks in same location
                        else:
                            amp = np.sum(amps[:, ploc], axis=1) # sum accounts for multiple peaks in same location

                    else:
                        ionization, peak_loc, gA, Ei, Ek, gi, gk = self.db.lines(element)[:, [0, 1, 3, 4, 5, 12, 13]].T.astype(float)
                        Zind = ionization == ion
                        
                        if ground_state:
                            Eind = Ei == 0 #indices of ground states
                            Zind = Zind * Eind    
                        
                        amps = peak_loc[Zind]**-1 * gA[Zind] * np.exp(-Ek[Zind] / kT)
                        ploc = peak_loc[Zind] == wavelength
                        ploc_all = peak_loc[Zind]
                        amp = np.sum(amps[:, ploc], axis=1) # sum accounts for multiple peaks in same location

                        if np.sum(amps) == 0:
                            amps = np.zeros_like(t_array, dtype=float)
                            ploc = np.zeros_like(t_array, dtype=int)
                            
                        ion_cutoff = int(float(ion))
                        if ion_cutoff >= 5:
                            amps = np.zeros_like(amps)

                        self.intensity_probability_dictionary[element][str(ion)] = (ploc_all, amps)

                    peak_matches = self.peak_compare(x, y, p, ploc, ploc_all, amps, peak_indices, self.peak_lim, rang=rang)
                    peak_match.append(peak_matches)                
                    absolute_intensity.append(amp)

                peak_match = np.swapaxes(peak_match, 0, 1)
                
                absolute_intensity = np.array(absolute_intensity)
                max_intensity = np.max(absolute_intensity, axis=0)
                max_intensity_expanded = max_intensity[np.newaxis, :]  # Shape: (1, n)
                max_intensity_expanded = np.broadcast_to(max_intensity_expanded, absolute_intensity.shape)

                # Create a where condition with the same shape
                where_condition = max_intensity_expanded > 0

                # Prepare the output array
                relative_intensity = np.zeros_like(absolute_intensity, dtype=float)

                # Perform the division
                np.divide(
                    absolute_intensity,
                    max_intensity_expanded,
                    where=where_condition,
                    out=relative_intensity)

                # Swap the axes
                relative_intensity = np.swapaxes(relative_intensity, 0, 1)
                relative_intensity[relative_intensity > 1] = 1
                peak_prob = np.array([distance for _, _, _, distance in px]) * peak_match # * relative_intensity
                max_ind = np.argmax(np.max(peak_prob, axis=1))

                max_peak_prob = peak_prob[max_ind]
                max_peak_match = peak_match[max_ind]
                max_relative_intensity = relative_intensity[max_ind]
                
                sort_prob = np.argsort(max_peak_prob)[::-1]

                new_peak_prob = max_peak_prob[sort_prob]
                new_peak_match = max_peak_match[sort_prob]
                new_max_relative_intensity = max_relative_intensity[sort_prob]
                new_max_t = t_array[max_ind]
                new_px = np.array(px)[sort_prob]
                
                peak_probabilitiy = np.hstack((new_px, new_peak_match[:, np.newaxis], new_peak_prob[:, np.newaxis], new_max_relative_intensity[:, np.newaxis]))
                self.peak_probability_dictionary[i] = peak_probabilitiy

class Element():
    """ Object to hold element information
    """

    def __init__(self, name):
        self.name = name
        self.ions = dict({})


    def update(self, data, idx):
        """
        """
        el, ion, location, x, *rest = data
        
        if ion not in self.ions: # add a new ion
            self.ions[ion] = dict({idx: np.array([location, x])})
        else: # update existing ion dictionary with new peak location
            if idx not in self.ions[ion].keys():
                self.ions[ion][idx] = np.array([location, x])


    def ion_match(self, x, y, peak_array, element, ion, plot=False, x_lo=None, x_hi=None):
        """
        """
        ionization, element_peaks, Ei = self.db.lines(element)[:,[0, 1, 4]].T.astype(float)
        i_ind = ionization == ion
        i_peaks = element_peaks[i_ind]

        data_peak_centers = peak_array[:, 1]
        data_peak_widths = self.PeakyFinder.voigt_width(peak_array[:, 2], peak_array[:, 3])

        diffs = np.min(np.abs(data_peak_centers[:, np.newaxis] - i_peaks[np.newaxis, :]), axis=1)
        close = diffs < 1 * data_peak_widths

        peaks = peak_array[close]
        profile = self.PeakyFinder.multi_voigt(x, np.ravel(peaks)) 