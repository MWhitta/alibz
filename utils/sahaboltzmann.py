import numpy as np
from utils.database import Database

class SahaBoltzmann():
    """ Calculate ionization distribution and partition functions using the Saha-Boltzmann equation
    """

    def __init__(self, dbpath='db', temperature_lo=10000, temperature_hi=10500, temperature_inc=500, ne_lo=0., ne_hi=25., ne_step=1) -> None:
        # database
        self.db = Database(dbpath)

        # physical constants
        self.plank_constant = 4.135667696 * 10 ** -15 # eV s
        self.boltzmann_constant = 8.617333262 * 10 ** -5 # Boltzmann Constant (eV/K)
        self.speed_c = 2.99792458 * 10**8 # Speed of Light in a Vacuum (m/s)
        self.me = 0.51099895000 * 10**6 # eV / c^2

        self.temperature_array = np.arange(temperature_lo, temperature_hi, temperature_inc)
        self.ne_array = 10**np.arange(ne_lo, ne_hi, ne_step)


    def partition(self, element, temperature, ion=None):
        """ `partition` calculates the parition functions for an element
        """
        temperature = np.array(temperature)
        if np.shape(temperature)[0] == 1:
            temperature = temperature[:, None]
        kT = self.boltzmann_constant * temperature[:, None]

        # line data from database
        ionization, Ek, gi, gk = self.db.lines(element)[:, [0, 5, 12, 13]].T.astype(float)
        
        # filter elements without lines data
        ions = np.sort(np.unique(ionization))
        if len(ions) == 0:
            return 0

        # preallocate partition function array
        Zi = np.ones((len(temperature), len(ions)))  # Partition function for ionization state i
        
        # calculate partition functions
        if len(ions) == 1:  # Hydrogen
            Eion = self.db.ionization_energy(element)
            Zi = (938.272**(3/2) / (2 * self.me**(3/2))) * np.exp(-Eion[0][-1] / kT)

        else: # Non-hydrogen elements
            if ion == None:
                for ii, ion in enumerate(ions):
                    Zind = (ionization == ion)
                    Zie = (gk[Zind] / gi[Zind]) * np.exp(-Ek[Zind] / kT[:, None])
                    Zi[:, ii] = np.squeeze(np.sum(Zie, axis=-1))
                return Zi
            else:
                Zind = (ionization == ions[ion])
                Zie = (gk[Zind] / gi[Zind]) * np.exp(-Ek[Zind] / kT[:, None])
                Zi = np.sum(Zie, axis=-1)
                return Zi


    def ionization_distribution(self, element, temperature, ne, decimal_precision=10):
        """ `ionization_ distribution` method determines the distribution of ionization states
        """
        temperature = np.asarray(temperature)

        # Broadcasting kT calculation
        kT = self.boltzmann_constant * temperature[:, None]
        lamb = self.plank_constant / np.sqrt(2 * np.pi * self.me * kT / self.speed_c**2)
        loglamb = np.log(lamb)

        # ionization data from database
        ionization = self.db.lines(element)[:, 0].astype(float)
        Eion = self.db.ionization_energy(element)
        
        ions = np.sort(np.unique(ionization))
        if len(ions) == 0:
            return np.array([]), np.array([])

        # preallocate ionization energy and ionization fraction arrays
        Ei = np.ones((len(temperature), len(ions)))  # Ionization energy for ionization state i
        ci = np.ones((len(temperature), len(ions)))  # Weighting constant for ionization state i
        
        if len(ions) == 1:  # Hydrogen
            Zi = self.partition(element, temperature)
            ci = 1 / (1 + (Zi / 10**ne))

        else: # Non-hydrogen elements
            for ii, ion in enumerate(ions):
                Eind = Eion[:, 1] == ion - 1 # ionization energies are 0 indexed, ionization lines are 1 indexed
                if np.any(Eind):
                    Ei[:, ii] = Eion[Eind, -1][0]

            Zi = self.partition(element, temperature)
            for iii in range(len(ions) - 1):
                if np.any(Zi[:, iii] == 0):
                    a = np.zeros_like(Zi[:, iii]).astype(float)
                    Zz = np.divide(Zi[:, iii + 1], Zi[:, iii], where=Zi[:, iii]>0, out=a)
                    a = 2 * Zz * lamb**-3 * np.exp(-Ei[:, iii] / kT)
                    print(f'a Zi==0 {a}')
    
                elif np.any(Zi[:, iii] < 10**-decimal_precision):
                    loga = (np.log(2) + np.log(Zi[:, iii + 1]) - np.log(Zi[:, iii]) + loglamb[:, 0] - Ei[:,iii] / kT[:, 0])
                    a = np.exp(loga)
                else:
                    a = 2 * (Zi[:, iii + 1] / Zi[:, iii]) * lamb[:, 0]**-3 * np.exp(-Ei[:, iii] / kT[:, 0])
                ci[:, iii + 1] = a * ci[:, iii] / 10**ne # NIST makes some arbitrary choice for N? solid angle? that changes ne by 10**6
            
            ci /= np.sum(ci, axis=1, keepdims=True)
            ci = np.round(ci, decimal_precision)

        return ci, Zi
    

    def time_gated(self, element, time_0=20, time_f=100, t_step=1):
        """ LTE spectrum from time gated measurement
        """

        ionization, peak_loc, gA, Ei, Ek, conf_i, conf_k = self.db.lines(element)[:, [0, 1, 3, 4, 5, 7, 9]].T

        ions = np.sort(np.unique(ionization))
        if len(ions) == 0:
            return np.array([]), np.array([])
        
        lifetime_array = np.ones_like(gA, dtype=float)
        for ion in ions:
            Zind = (ionization == ion)
            confs = np.unique(conf_k[Zind])
            for conf in confs:
                Ck_ind = (conf_k == conf)
                tau_inv = np.sum(np.array(gA[Ck_ind]).astype(float)) # s**-1
                lifetime_array[Ck_ind] = tau_inv

        time_array = np.arange(0, time_f, t_step)[:, None] * 10**-9
        decay_array = np.exp(-(lifetime_array * time_array))
        detect_start = int(time_0 / t_step)
        cumulative_array = np.sum(decay_array[detect_start:], axis=0)
        normalization_factor = np.max(cumulative_array)
        time_gated_array = 1 - (cumulative_array / normalization_factor)

        return time_gated_array
    

    def ground_state_resonant_absorption(self, element):
        """ Account for the resonant absorbtion of emission due to relaxation to the ground state
        """

        ionization, Ei = self.db.lines(element)[:, [0,4]].T.astype(float)
        Zind = ionization == 1
        Gind = Ei == 0
        gsra_array = np.ones_like(Ei)
        gsra_array[Gind * Zind] = 0
        return gsra_array


    def calculate(self, element, temperature, ne, decimal_precision=10, gsra=False, time_gated=False, time_0=20, time_f=100, t_step=1):
        """Calculate Saha-Boltzmann distribution for a given element.
        """
        temperature = np.asarray(temperature)

        # Broadcasting kT calculation
        kT = self.boltzmann_constant * temperature[:, None]
        intensity_constant = (4 * np.pi)**-1 * self.plank_constant * self.speed_c * 10**9

        ionization, peak_loc, gA, Ek = self.db.lines(element)[:, [0, 1, 3, 5]].T.astype(float)

        if time_gated:
            gA = self.time_gated(element, time_0=time_0, time_f=time_f, t_step=t_step)
        
        if gsra:
            gA = gA * self.ground_state_resonant_absorption(element)
        
        ions = np.sort(np.unique(ionization))
        if len(ions) == 0:
            return np.array([]), np.array([])

        ci, Zi = self.ionization_distribution(element, temperature, ne, decimal_precision=decimal_precision)
        location = []
        spectra = []

        if len(ions) == 1:  # Hydrogen
            Zind = (ionization == ions[0])
            peak_intensity = intensity_constant * peak_loc[Zind][None,:]**-1 * gA[Zind][None,:] * np.exp(-Ek[Zind][None,:] / kT)
            overall_intensity = ci[:, None] * peak_intensity
            spectra.append(overall_intensity)
            location.append(np.tile(peak_loc[Zind], (len(temperature), 1)))

        else: # Non-hydrogen elements
            for iv, ion in enumerate(ions):
                Zind = (ionization == ion)
                peak_intensity = intensity_constant * peak_loc[Zind][None,:]**-1 * gA[Zind][None,:] * np.exp(-Ek[Zind][None,:] / kT)
                overall_intensity = ci[:, iv, None] * peak_intensity
                spectra.append(overall_intensity)
                location.append(np.tile(peak_loc[Zind], (len(temperature), 1)))

        x = np.concatenate(location, axis=-1)
        y = np.concatenate(spectra, axis=-1)

        return x, y