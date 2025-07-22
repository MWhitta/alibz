import pickle
import datetime
import numpy as np
from scipy.special import voigt_profile as voigt

from utils.database import Database
from utils.sahaboltzmann import SahaBoltzmann

class PeakyMaker():
    """Forward model for optical emission spectra."""

    def __init__(self, dbpath) -> None:
        """Initialize the maker and load the database.

        Parameters
        ----------
        dbpath : str
            Path to the database used for generating spectra.
        """

        # Database interfaces
        self.db = Database(dbpath)
        self.sb = SahaBoltzmann(dbpath)
        self.max_z = len(self.db.elements)

        # Physical constants
        self.plank_constant = 4.135667696 * 10 ** -15  # eV s
        self.boltzmann_constant = 8.617333262 * 10 ** -5  # eV/K
        self.speed_c = 2.99792458 * 10**8  # speed of light in vacuum (m/s)
        self.me = 0.51099895000 * 10**6  # eV / c^2

    def K2eV(self, temperature):
        """Convert Kelvin to electronvolts.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Temperature in electronvolts.
        """

        TeV = temperature / self.boltzmann_constant
        return TeV

    def eV2K(self, temperature):
        """Convert electronvolts to Kelvin.

        Parameters
        ----------
        temperature : float
            Temperature in electronvolts.

        Returns
        -------
        float
            Temperature in Kelvin.
        """

        TK = temperature * self.boltzmann_constant
        return TK

    def peak_maker(
        self,
        fracs,
        inc=1 / 30,
        w_lo=180,
        w_hi=961 + (1 / 30),
        voigt_sig=0.1,
        voigt_gam=0.1,
        temperature=10000,
        ne=17,
        decimal_precision=10,
        gsra=False,
        time_gated=False,
        time_0=20,
        time_f=100,
        t_step=1,
        verbose=False,
    ):
        """Generate a multi-element LIBS spectrum.

        Parameters
        ----------
        fracs : array_like
            Element abundance fractions.
        inc : float, optional
            Spectral wavelength resolution in nanometers.
        w_lo : float, optional
            Lower bound of the wavelength range in nanometers.
        w_hi : float, optional
            Upper bound of the wavelength range in nanometers.
        voigt_sig : float, optional
            Standard deviation of the Gaussian part of the Voigt profile.
        voigt_gam : float, optional
            Half width at half maximum of the Lorentzian part of the Voigt
            profile.
        temperature : float, optional
            Plasma temperature in Kelvin.
        ne : float, optional
            Base-10 logarithm of the electron density in ``cm**-3``.
        decimal_precision : int, optional
            Precision passed to :class:`~utils.sahaboltzmann.SahaBoltzmann`.
        gsra : bool, optional
            When ``True`` use the ground state recombination assumption.
        time_gated : bool, optional
            If ``True`` use time gated calculations.
        time_0 : float, optional
            Start time for gated calculations.
        time_f : float, optional
            End time for gated calculations.
        t_step : float, optional
            Time step for gated calculations.
        verbose : bool, optional
            When ``True`` print diagnostic information.

        Returns
        -------
        tuple
            ``(wave, spectrum, element_spectra)`` containing the wavelength
            axis, the summed spectrum and the individual element spectra.
        """

        wave = np.arange(w_lo, w_hi, inc).astype('float32')

        if len(fracs) != self.max_z:
            raise ValueError(f"First {self.max_z} elements configured, {len(fracs)} provided.")
        if not (all(x >=0 for x in fracs) and np.sum(fracs) > 0):
            raise ValueError("Element fractions must be non-negative and sum must be non-zero")
                
        exclude = np.logical_not([item in ['Se', 'Pm', 'Po', 'At', 'Rn', 'Th', 'Pa', 'U'] for item in self.db.elements])
        fracs = fracs * exclude

        # Scale fractions to sum to 1.0
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


    def batch_maker(
        self,
        focus_el=None,
        n_elem=4,
        n_delta=2,
        abundance="equal",
        abund_scale=0.5,
        temp=10000,
        temp_vary=False,
        ne=10**17,
        inc=1 / 30,
        w_lo=180,
        w_hi=961 + (1 / 30),
        voigt_sig=0.5,
        voigt_gam=0.5,
        voigt_vary=False,
        voigt_range=0.1,
        batch=16,
    ):
        """Generate a batch of composite LIBS spectra.

        Parameters
        ----------
        focus_el : list of str, optional
            Specific elements to include in the spectra.
        n_elem : int, optional
            Mean number of elements included in each spectrum.
        n_delta : int, optional
            Allowed variation in the number of elements.
        abundance : str, optional
            Method used to assign element abundances.
        abund_scale : float, optional
            Maximum variation factor applied to abundances.
        temp : float, optional
            Plasma temperature in Kelvin.
        temp_vary : bool, optional
            If ``True`` vary the temperature of each spectrum.
        ne : float, optional
            Electron density in ``cm**-3``.
        inc : float, optional
            Spectral wavelength resolution in nanometers.
        w_lo : float, optional
            Lower bound of the wavelength range in nanometers.
        w_hi : float, optional
            Upper bound of the wavelength range in nanometers.
        voigt_sig : float, optional
            Standard deviation of the Gaussian part of the Voigt profile.
        voigt_gam : float, optional
            Half width at half maximum of the Lorentzian part.
        voigt_vary : bool, optional
            If ``True`` vary the Voigt parameters for each spectrum.
        voigt_range : float, optional
            Maximum variation applied when ``voigt_vary`` is ``True``.
        batch : int, optional
            Number of spectra to create.

        Returns
        -------
        tuple
            ``(elem_symb, fracs, wave, x_data, y_data)`` where ``elem_symb`` are
            the element symbols used.
        """
        # Check element choices for consistency with the database
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
        
        # Generate the element fractions
        num_elem = (
            n_elem
            + np.round(2 * (n_delta + 0.5) * np.random.rand(batch) - (n_delta + 0.5))
        ).astype(int)  # Number of elements drawn from possibilities
        sample_el = [np.random.choice(focus_el, num_elem[i]) for i in range(batch)]  # List, not array
        samp_mask = np.array(
            [np.in1d(self.elements, sample_el[i]) for i in range(batch)]
        )  # Fraction arrays for ``peak_maker`` with shape ``(batch, max_z)``
        
        if abundance == "natural":  # Pull natural crustal element abundance data from .csv
            sample_abund = self.elem_abund * samp_mask  # Rightmost dimension is ``max_z`` for broadcasting
        else:  # Randomly assign element abundance
            sample_abund = np.random.rand(len(self.elements)) * samp_mask  # Rightmost dimension is ``max_z`` for broadcasting
        
        sample_var = 2 * abund_scale * (np.random.rand(batch, len(self.elements)) - 0.5)  # Allowed variation in sample abundance
        sample_fracs = sample_abund * (1 + sample_var)  # Varied fractions
        fracs = sample_fracs / np.sum(sample_fracs, axis=1, keepdims=True)  # Normalize fractions to one

        wave = np.arange(w_lo, w_hi, inc)  # Only needed for correct length
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
            wave, x_data[i], y_data[i] = self.peak_maker(
                fracs=fracs[i],
                inc=inc,
                temp=temp[i],
                w_lo=w_lo,
                w_hi=w_hi,
                ne=ne,
                voigt_sig=voigt_sig[i],  # Standard deviation of the Gaussian part of the Voigt profile
                voigt_gam=voigt_gam[i],  # Half width at half maximum of the Lorentzian part
            )

        return elem_symb, fracs, wave, x_data, y_data
    

    def export_batch(
        self,
        batchnum,
        temp=10000,
        voigt_vary=True,
        temp_vary=False,
        els=0,
        n_elem=1,
        n_delta=0,
        dtype="float16",
        keep_y=True,
    ):
        """Export a batch of spectra as a pickle file.

        Parameters
        ----------
        batchnum : int
            Number of spectra to generate.
        temp : float, optional
            Plasma temperature in Kelvin.
        voigt_vary : bool, optional
            If ``True`` vary the Voigt parameters for each spectrum.
        temp_vary : bool, optional
            If ``True`` vary the temperature of each spectrum.
        els : list of str or int, optional
            Elements to include. If ``0`` use all available elements.
        n_elem : int, optional
            Mean number of elements per spectrum.
        n_delta : int, optional
            Allowed variation in the number of elements.
        dtype : str, optional
            Data type used when writing the file.
        keep_y : bool, optional
            When ``False`` omit the per-element spectra from the pickle file.
        """
        if els:
            pass
        else:
            els = [
                'H', 'He',  # Row 1
                'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  # Row 2
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',  # Row 3
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',  # Row 4  # Se removed between As/Br
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',  # Row 5
                'Cs', 'Ba',  # Row 6 alkali/alkaline earth
                'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Row 6 rare earths  # Pm removed between Nd/Sm
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',  # Row 6 transition metals  # Po,At,Rn removed between Bi/Rn
                'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  # Th,Pa,U removed
            ]
        
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

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # Timestamp string
        dname = f'{batchnum}spectra_{elnum}els_{int(temp/1000)}kK_Tvary{temp_vary}_{now_time}.pickle'
        with open(f'training/' + dname, 'wb') as f:
            pickle.dump(fracs, f)
            pickle.dump(wave, f)
            pickle.dump(x_data, f)
            pickle.dump(y_data, f)
            pickle.dump(elem_symb, f)
            pickle.dump(temp, f)
