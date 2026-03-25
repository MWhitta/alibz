import pickle
import datetime
import os
import numpy as np
from scipy.special import voigt_profile as voigt

from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import SahaBoltzmann
from alibz.utils.constants import BOLTZMANN, PLANCK, SPEED_OF_LIGHT, ELECTRON_MASS

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

        # Physical constants (from shared module)
        self.plank_constant = PLANCK
        self.boltzmann_constant = BOLTZMANN
        self.speed_c = SPEED_OF_LIGHT
        self.me = ELECTRON_MASS

    def _normalize_focus_elements(self, focus_el):
        """Validate and de-duplicate the public element-selection input."""
        if focus_el is None:
            return list(self.db.elements)

        if isinstance(focus_el, str):
            focus_list = [focus_el]
        else:
            focus_list = list(focus_el)

        if len(focus_list) == 0:
            raise ValueError("focus_el must contain at least one element when provided")

        # Preserve order while removing duplicates.
        focus_list = list(dict.fromkeys(focus_list))
        invalid = [el for el in focus_list if el not in self.db.elements]
        if invalid:
            raise ValueError(f"Unknown element symbols: {invalid}")

        return focus_list


    def _abundance_array(self):
        """Return natural abundances in the database element order."""
        return np.array([self.db.elem_abund[el] for el in self.db.elements], dtype=float)


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
        TeV = np.asarray(temperature) * self.boltzmann_constant
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
        TK = np.asarray(temperature) / self.boltzmann_constant
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
        ne=17,
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
            Method used to assign element abundances. Supported values are
            ``"equal"``, ``"random"``, and ``"natural"``.
        abund_scale : float, optional
            Maximum variation factor applied to abundances.
        temp : float, optional
            Plasma temperature in Kelvin.
        temp_vary : bool, optional
            If ``True`` vary the temperature of each spectrum.
        ne : float, optional
            Base-10 logarithm of the electron density in ``cm**-3``.
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
            ``(elem_symb, fracs, wave, x_data, y_data)`` where:
            - ``elem_symb`` is the element list corresponding to the returned
              fractions and per-element spectra
            - ``fracs`` has shape ``(batch, len(elem_symb))``
            - ``x_data`` is the summed spectrum array with shape
              ``(batch, len(wave))``
            - ``y_data`` is the per-element spectrum array with shape
              ``(batch, len(elem_symb), len(wave))``
        """
        elem_symb = self._normalize_focus_elements(focus_el)
        max_elem = len(elem_symb)
        if batch < 1:
            raise ValueError("batch must be at least 1")
        if n_elem < 1:
            raise ValueError("n_elem must be at least 1")
        if n_elem + n_delta > max_elem:
            raise ValueError("n_elem + n_delta cannot exceed available elements") 
        if n_delta > n_elem-1:
            raise ValueError("n_delta must be less than n_elem to avoid empty samples")
        if abund_scale < 0 or abund_scale > 1:
            raise ValueError(f"abund_scale must lie on interval [0,1], {abund_scale} given")
        if abundance not in {"equal", "random", "natural"}:
            raise ValueError(f"Unsupported abundance mode {abundance!r}")
        if inc <= 0:
            raise ValueError("inc must be positive")
        if w_hi <= w_lo:
            raise ValueError("w_hi must be greater than w_lo")
        if temp <= 0:
            raise ValueError("temp must be positive")

        all_elements = np.asarray(self.db.elements, dtype=object)
        focus_indices = np.array([self.db.elements.index(el) for el in elem_symb], dtype=int)
        
        # Generate the element fractions
        min_elem = max(1, n_elem - n_delta)
        max_draw = min(max_elem, n_elem + n_delta)
        num_elem = np.random.randint(min_elem, max_draw + 1, size=batch)
        sample_el = [
            np.random.choice(elem_symb, num_elem[i], replace=False)
            for i in range(batch)
        ]
        samp_mask_full = np.array(
            [np.isin(all_elements, sample_el[i]) for i in range(batch)],
            dtype=float,
        )
        
        if abundance == "natural":  # Pull natural crustal element abundance data from .csv
            sample_abund = self._abundance_array()[None, :] * samp_mask_full
        elif abundance == "equal":
            sample_abund = samp_mask_full.copy()
        else:
            sample_abund = np.random.rand(batch, self.max_z) * samp_mask_full
        
        sample_var = 2 * abund_scale * (np.random.rand(batch, self.max_z) - 0.5)
        sample_fracs_full = sample_abund * (1 + sample_var)

        zero_rows = np.sum(sample_fracs_full, axis=1) <= 0
        if np.any(zero_rows):
            sample_fracs_full[zero_rows] = samp_mask_full[zero_rows]

        fracs_full = sample_fracs_full / np.sum(sample_fracs_full, axis=1, keepdims=True)
        fracs = fracs_full[:, focus_indices]

        wave = np.arange(w_lo, w_hi, inc)  # Only needed for correct length
        x_data = np.zeros((batch, len(wave)), dtype=float)
        y_data = np.zeros((batch, len(elem_symb), len(wave)), dtype=float)
        
        if temp_vary:
            temperatures = float(temp) + 1000 * np.random.randint(-5, 6, size=batch)
        else:
            temperatures = float(temp) * np.ones(batch)

        if voigt_vary:
            voigt_sig_arr = voigt_sig + np.random.uniform(-voigt_range, voigt_range, size=batch)
            voigt_gam_arr = voigt_gam + np.random.uniform(-voigt_range, voigt_range, size=batch)
        else:
            voigt_sig_arr = voigt_sig * np.ones(batch)
            voigt_gam_arr = voigt_gam * np.ones(batch)

        voigt_sig_arr = np.clip(voigt_sig_arr, np.finfo(float).eps, None)
        voigt_gam_arr = np.clip(voigt_gam_arr, np.finfo(float).eps, None)

        for i in np.arange(batch):
            wave, x_data[i], element_spectra = self.peak_maker(
                fracs=fracs_full[i],
                inc=inc,
                temperature=temperatures[i],
                w_lo=w_lo,
                w_hi=w_hi,
                ne=ne,
                voigt_sig=voigt_sig_arr[i],
                voigt_gam=voigt_gam_arr[i],
            )
            y_data[i] = element_spectra[focus_indices]

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
        out_dir="training",
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
        out_dir : str, optional
            Directory used for the exported pickle file. Created if needed.

        Returns
        -------
        str
            Path to the written pickle file.
        """
        if els:
            focus_el = els
        else:
            focus_el = [
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
        
        elem_symb, fracs, wave, x_data, y_data = self.batch_maker(
            batch=batchnum,
            focus_el=focus_el,
            inc=1 / 30,
            w_lo=180,
            w_hi=962,
            n_elem=n_elem,
            n_delta=n_delta,
            abund_scale=1,
            voigt_vary=voigt_vary,
            temp=temp,
            temp_vary=temp_vary,
        )

        dtype_np = np.dtype(dtype)
        fracs = fracs.astype(dtype_np)
        wave = wave.astype(dtype_np)
        x_data = x_data.astype(dtype_np)

        if keep_y is False:
            y_data = None
        else:
            y_data = y_data.astype(dtype_np)

        active_elements = int(np.count_nonzero(np.any(fracs > 0, axis=0)))

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # Timestamp string
        dname = f'{batchnum}spectra_{active_elements}els_{int(float(temp)/1000)}kK_Tvary{temp_vary}_{now_time}.pickle'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, dname)
        with open(out_path, 'wb') as f:
            pickle.dump(fracs, f)
            pickle.dump(wave, f)
            pickle.dump(x_data, f)
            pickle.dump(y_data, f)
            pickle.dump(elem_symb, f)
            pickle.dump(temp, f)

        return out_path
