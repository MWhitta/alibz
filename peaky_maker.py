import pickle
import datetime
import numpy as np
from scipy.special import voigt_profile as voigt

from utils.database import Database
from utils.sahaboltzmann import SahaBoltzmann

class PeakyMaker():
    """ forward model for optical emission spectra """
    
    #class attributes
    def __init__(self, dbpath) -> None:
        # database
        self.db = Database(dbpath)
        self.sb = SahaBoltzmann(dbpath)
        self.max_z = len(self.db.elements)

        # physical constants
        self.plank_constant = 4.135667696 * 10 ** -15 # eV s
        self.boltzmann_constant = 8.617333262 * 10 ** -5 # Boltzmann Constant (eV/K)
        self.speed_c = 2.99792458 * 10**8 # Speed of Light in a Vacuum (m/s)
        self.me = 0.51099895000 * 10**6 # eV / c^2

    def K2eV(self, temperature):
        """ convert K to eV
        """
        TeV = temperature / self.boltzmann_constant
        return TeV

    def eV2K(self, temperature):
        """ convert ev to K
        """
        TK = temperature * self.boltzmann_constant
        return TK


    # Peak maker function
    def peak_maker(self,
                    fracs, #array with element abundance fractions
                    inc=1/30, #spectral wavelength resolution, i.e., increment between datapoints (nm)
                    w_lo=180, #lower limit of wavelength range (nm)
                    w_hi=961+(1/30), #upper limit of wavelength range (nm) - need an extra `inc` here to span full range
                    voigt_sig=0.1, #stdev of normal part of voigt convolution
                    voigt_gam=0.1, #half-width at half max parameter of cauchy part of convolution
                    temperature=10000, # K
                    ne=17, # log10 ne in cm**-3 - electron density
                    decimal_precision=10,
                    gsra=False,
                    time_gated=False, 
                    time_0=20, 
                    time_f=100, 
                    t_step=1,
                    verbose=False):
        """ peak_maker generates multi-element LIBS spectra
            args:
                fracs (str) - element symbol for which spectrum will be generated
            kwargs:
                inc (float) - spectral wavelength resolution, i.e., increment between datapoints (nm)
                w_lo (float) - lower limit of wavelength range (nm)
                w_hi (float) - upper limit of wavelength range (nm)
                voigt_sig (float) - stdev of normal part of voigt peak profile
                voigt_gam (float) - half-width at half max parameter of cauchy part of convolution
                temp (float) - temperature of plasma from which LIBS spectrum for this element is generated in (K)
                ne (float) - number density of electrons in the plasma (cm**-3)
                shift (bool) - whether to apply a shift to peak positions
                shift_type (str) - 'random' applies a random shift to each peak; 'sys' applies a systematic shift
                height (bool) - whether to jitter peak heights
                height_type (str) - 'random' jitters peak heights randomly
                height_mag (float) - maximum magnitude by which to vary peak heights
                plot (bool) - whether to plot the resulting spetrum
        """

        wave = np.arange(w_lo, w_hi, inc).astype('float32')

        if len(fracs) != self.max_z:
            raise ValueError(f"First {self.max_z} elements configured, {len(fracs)} provided.")
        if not (all(x >=0 for x in fracs) and np.sum(fracs) > 0):
            raise ValueError("Element fractions must be non-negative and sum must be non-zero")
                
        exclude = np.logical_not([item in ['Se', 'Pm', 'Po', 'At', 'Rn', 'Th', 'Pa', 'U'] for item in self.db.elements])
        fracs = fracs * exclude

        #scale fractions to sum to 1.0
        frac_sum = np.sum(fracs)
        if np.isclose(frac_sum,1):
            pass
        else:
            if verbose:
                print('Element fractions need renormalization because they dont sum to 1 - was this expected?')
            if frac_sum > 0:
                fracs = fracs / frac_sum
            else:
                if verbose:
                    print('zero intensity in spectrum, possibly due to absence of lines in database')

        spec_array = np.zeros((len(fracs), len(wave)))

        for i, (el, fr) in enumerate(zip(self.db.elements, fracs)):
            if fr > 0:
                x, y = self.sb.calculate(el, temperature, ne, decimal_precision=decimal_precision, gsra=gsra, time_gated=time_gated, time_0=time_0, time_f=time_f, t_step=t_step)
                peaks = np.array([yy[:, None] * voigt(wave - xx[:, None], voigt_sig, voigt_gam) for xx, yy in zip(x, y)])
                spec = fr * np.sum(peaks, axis=1)
                spec_array[i] = spec

        spec = np.sum(spec_array, axis=0)

        return wave, spec, spec_array


    def batch_maker(self,
                        focus_el=[], #optional list of specific elements
                        n_elem=4, #defines the mean number of elements included
                        n_delta=2, #defines the +/- range for number of elements to vary
                        abundance='equal',
                        abund_scale=0.5, #max variation factor on natural abundance (<=1)
                        temp=10000, # K
                        temp_vary=False,
                        ne=10**17, # cm**-3 - electron density
                        inc=1/30,
                        w_lo=180,
                        w_hi=961+(1/30), # need an extra `inc` here to span full range
                        voigt_sig=0.5, #stdev of normal part of voigt convolution
                        voigt_gam=0.5, #half-width at half max parameter of cauchy part of convolution
                        voigt_vary=False,
                        voigt_range=0.1,
                        batch=16 #number of samples to create
                        ): 
        """batch_spectra generates a batch of composite LIBS spectrum with optional kwargs for sampling a range of element fractions
            args:
                focus_el (string list) - possible elements to include in spectra
            kwargs (for batch_spectra):
                n_elem (int) -  defines the mean number of elements included
                n_delta (int) -  defines the +/- range for number of elements to vary
                abundance (str) - how to define element abundances across batch 
                                    'equal'
                abund_scale=0.5, max variation factor on abundance (<=1)
            kwargs (for peak_maker):
                inc (float) - spectral wavelength resolution, i.e., increment between datapoints (nm)
                w_lo (float) - lower limit of wavelength range (nm)
                w_hi (float) - upper limit of wavelength range (nm)
                voigt_sig (float) - stdev of normal part of voigt peak profile
                voigt_gam (float) - half-width at half max parameter of cauchy part of convolution
                temp (float) - temperature of plasma from which LIBS spectrum for this element is generated in (K)
                ne (float) - number density of electrons in the plasma (cm**-3)
        """
        #check element choices for consistency with database
        max_elem = self.max_z
        if len(focus_el):
            max_elem = len(focus_el)
            if not all (x in self.elements for x in focus_el):
                print(self.elements)
                raise ValueError(f"Elements must be in the valid elements list")
        else:
            focus_el = self.elements
            max_elem = len(focus_el)
        if n_elem + n_delta > max_elem:
            raise ValueError("n_elem + n_delta cannot exceed available elements") 
        if n_delta > n_elem-1:
            raise ValueError("n_delta must be less than n_elem to avoid empty samples")
        if abund_scale < 0 or abund_scale > 1:
            raise ValueError(f"abund_scale must lie on interval [0,1], {abund_scale} given")
        
        #generate the element fractions
        num_elem = (n_elem + np.round(2 * (n_delta+0.5) * np.random.rand(batch) - (n_delta+0.5))).astype(int) #number of elements drawn from possibilities
        sample_el = [np.random.choice(focus_el, num_elem[i]) for i in range(batch)] #list, not array
        samp_mask = np.array([np.in1d(self.elements, sample_el[i]) for i in range(batch)]) #fracs arrays for make_spectra - shape (batch, max_z)
        
        if abundance == 'natural': #pull natural crustal element abundance data from .csv
            sample_abund = self.elem_abund * samp_mask #rightmost dims = max_z for broadcasting
        else: #randomly assign element abundance 
            sample_abund = np.random.rand(len(self.elements)) * samp_mask # randomly assign abundance - rightmost dims = max_z for broadcasting
        
        sample_var = 2 * abund_scale * (np.random.rand(batch, len(self.elements))-0.5) # allowed variation in sample abundance
        sample_fracs = sample_abund * (1 + sample_var) # varied fractions
        fracs = sample_fracs / np.sum(sample_fracs, axis=1, keepdims=True) #normalize fractions to one
        
        wave = np.arange(w_lo,w_hi,inc) #only needed for correct length
        x_data = np.zeros((batch, len(wave)))
        y_data = np.zeros((batch, len(focus_el), len(wave)))
        elem_symb = focus_el
        
        if temp_vary==True:
            temp = temp + 1000*(np.random.randint(-5,6,size=batch))
        else:
            temp = temp*np.ones(batch)

        if voigt_vary==True:
            voigt_sig = voigt_sig + (np.random.rand(batch) - voigt_range)
            voigt_gam = voigt_gam + (np.random.rand(batch) - voigt_range)
        else:
            voigt_sig = voigt_sig*np.ones(batch)
            voigt_gam = voigt_gam*np.ones(batch)

        for i in np.arange(batch):
            wave, x_data[i], y_data[i] = self.peak_maker(fracs=fracs[i], 
                                                            inc=inc, 
                                                            temp=temp[i],
                                                            w_lo=w_lo, 
                                                            w_hi=w_hi,
                                                            ne=ne,
                                                            voigt_sig=voigt_sig[i], #stdev of normal part of voigt convolution
                                                            voigt_gam=voigt_gam[i], #half-width at half max parameter of cauchy part of convolution
                                                            )

        return elem_symb, fracs, wave, x_data, y_data
    

    def export_batch(self, 
                     batchnum, 
                     temp=10000, 
                     voigt_vary=True, 
                     temp_vary=False, 
                     els=0, 
                     n_elem=1, 
                     n_delta=0, 
                     dtype='float16', 
                     keep_y=True):
        
        """ export a batch of spectra as a pickle file """
        if els:
            pass
        else:
            els = ['H', 'He', #row1
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', #row2
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', #row3
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', #row4 #Se removed bw As/Br
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', #row5
            'Cs', 'Ba', #row6 alkali/alkaline earth
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', #row6 rare earths # Pm removed bw Nd/Sm
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',    #row6 transition metals #Po,At,Rn removed bw Bi/Rn
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']# Th,Pa,U removed
        
        elem_symb, fracs, wave, x_data, y_data = self.batch_maker(batch=batchnum,
                                                         focus_el=els, 
                                                         inc=1/30,
                                                         w_lo=180,
                                                         w_hi=962,
                                                         n_elem=n_elem, 
                                                         n_delta=n_delta,
                                                         abund_scale=1,
                                                         voigt_vary=voigt_vary,
                                                         temp=temp,
                                                         temp_vary=temp_vary,
                                                         )
        if keep_y == False:
            y_data = None

        elnum = np.sum(fracs>0)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') #timestamp string
        dname = f'{batchnum}spectra_{elnum}els_{int(temp/1000)}kK_Tvary{temp_vary}_{now_time}.pickle'
        with open(f'training/' + dname, 'wb') as f:
            pickle.dump(fracs, f)
            pickle.dump(wave, f)
            pickle.dump(x_data, f)
            pickle.dump(y_data, f)
            pickle.dump(elem_symb, f)
            pickle.dump(temp, f)