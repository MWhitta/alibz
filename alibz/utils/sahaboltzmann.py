from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from alibz.utils.database import Database
from alibz.utils.constants import BOLTZMANN, PLANCK, SPEED_OF_LIGHT, ELECTRON_MASS

class SahaBoltzmann():
    """ Calculate ionization distribution and partition functions using the Saha-Boltzmann equation
    """

    def __init__(self, dbpath='db', temperature_lo=10000, temperature_hi=10500, temperature_inc=500, ne_lo=0., ne_hi=25., ne_step=1) -> None:
        # database
        self.db = Database(dbpath)

        # Physical constants (from shared module)
        self.plank_constant = PLANCK
        self.boltzmann_constant = BOLTZMANN
        self.speed_c = SPEED_OF_LIGHT
        self.me = ELECTRON_MASS

        self.temperature_array = np.arange(temperature_lo, temperature_hi, temperature_inc)
        self.ne_array = 10**np.arange(ne_lo, ne_hi, ne_step)


    @staticmethod
    def _temperature_array(temperature: ArrayLike) -> NDArray[np.float64]:
        """Return *temperature* as a 1-D positive float array."""
        temperature = np.atleast_1d(np.asarray(temperature, dtype=float)).reshape(-1)
        if temperature.size == 0:
            raise ValueError("temperature must contain at least one value")
        if np.any(temperature <= 0):
            raise ValueError("temperature must be strictly positive")
        return temperature


    @staticmethod
    def _resolve_ion_index(ion, ions: NDArray[np.float64]) -> int:
        """Resolve either a zero-based ion index or a database ion label."""
        ion_value = float(ion)
        exact_match = np.nonzero(np.isclose(ions, ion_value))[0]
        if exact_match.size:
            return int(exact_match[0])

        ion_index = int(ion_value)
        if np.isclose(ion_value, ion_index) and 0 <= ion_index < len(ions):
            return ion_index

        raise ValueError(f"Unknown ion specifier {ion!r}; expected one of {ions.tolist()} or a zero-based index")


    @staticmethod
    def _stage_levels(
        Ei: NDArray[np.float64],
        Ek: NDArray[np.float64],
        gi: NDArray[np.float64],
        gk: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Build a deduplicated level table ``[energy_eV, degeneracy]`` for one ion stage."""
        energies = np.concatenate((Ei[mask], Ek[mask]))
        degeneracies = np.concatenate((gi[mask], gk[mask]))

        if energies.size == 0:
            return np.empty((0, 2), dtype=float)

        levels = np.column_stack((np.round(energies, 8), np.round(np.clip(degeneracies, 1e-30, None), 8)))
        return np.unique(levels, axis=0)


    def partition(self, element, temperature, ion=None):
        """Return stage partition functions derived from the line database.

        The partition function is built from the union of lower and upper
        levels present in the line list for each ion stage:

        ``U_i(T) = sum_k g_k exp(-E_k / kT)``

        Parameters
        ----------
        element : str
            Element symbol.
        temperature : ArrayLike
            One or more temperatures in kelvin.
        ion : float or int, optional
            Either a database ion label (e.g. ``1`` for the first stage) or a
            zero-based ion index. When omitted, partitions for all ion stages
            are returned.

        Returns
        -------
        tuple
            ``(Zi, level_weights)`` where ``Zi`` is either ``(n_temp, n_ions)``
            or ``(n_temp,)`` and ``level_weights`` contains the corresponding
            per-level Boltzmann factors.
        """
        temperature = self._temperature_array(temperature)
        kT = self.boltzmann_constant * temperature[:, None]

        lines = np.asarray(self.db.lines(element))
        if lines.size == 0:
            empty = np.zeros((len(temperature), 0), dtype=float)
            if ion is None:
                return empty, []
            return np.zeros(len(temperature), dtype=float), np.empty((len(temperature), 0), dtype=float)

        ionization, Ei, Ek, gi, gk = lines[:, [0, 4, 5, 12, 13]].T.astype(float)
        ions = np.sort(np.unique(ionization))

        Zi = np.zeros((len(temperature), len(ions)), dtype=float)
        level_weights: List[NDArray[np.float64]] = []

        for ii, ion_stage in enumerate(ions):
            Zind = ionization == ion_stage
            levels = self._stage_levels(Ei, Ek, gi, gk, Zind)
            if levels.size == 0:
                level_weights.append(np.empty((len(temperature), 0), dtype=float))
                continue

            level_energy = levels[:, 0]
            level_degeneracy = levels[:, 1]
            Zie = level_degeneracy[None, :] * np.exp(-level_energy[None, :] / kT)
            Zi[:, ii] = np.sum(Zie, axis=-1)
            level_weights.append(Zie)

        if ion is None:
            return Zi, level_weights

        ion_index = self._resolve_ion_index(ion, ions)
        return Zi[:, ion_index], level_weights[ion_index]


    def stage_partition(self, element, temperature, ion) -> NDArray[np.float64]:
        """Return the partition function of a single ion stage as a 1-D array."""
        Zi, _ = self.partition(element, temperature, ion=ion)
        return np.asarray(Zi, dtype=float).reshape(-1)


    def ionization_distribution(self, element, temperature, ne, decimal_precision=10):
        """Determine the ionization-state distribution for one element.

        Parameters
        ----------
        element : str
            Element symbol.
        temperature : ArrayLike
            One or more temperatures in kelvin.
        ne : float
            Base-10 logarithm of the electron density in ``cm**-3``.
        decimal_precision : int, optional
            Rounding precision for the ion fractions.
        """
        temperature = self._temperature_array(temperature)
        ne_log10 = float(ne)
        ne_cm3 = 10**ne_log10

        kT = self.boltzmann_constant * temperature[:, None]
        lamb = self.plank_constant / np.sqrt(2 * np.pi * self.me * kT / self.speed_c**2)

        lines = np.asarray(self.db.lines(element))
        if lines.size == 0:
            empty = np.zeros((len(temperature), 0), dtype=float)
            return empty, empty

        ionization = lines[:, 0].astype(float)
        Eion = np.asarray(self.db.ionization_energy(element), dtype=float)

        ions = np.sort(np.unique(ionization))
        Zi, _ = self.partition(element, temperature)
        ci = np.ones((len(temperature), len(ions)), dtype=float)

        if len(ions) == 1:
            return ci, Zi

        # Ionization energies connect stage i -> i+1. Missing values simply
        # suppress that transition in the ratio chain.
        Ei = np.full((len(temperature), len(ions) - 1), np.inf, dtype=float)
        for ii, ion_stage in enumerate(ions[:-1]):
            Eind = Eion[:, 1] == ion_stage - 1
            if np.any(Eind):
                Ei[:, ii] = Eion[Eind, -1][0]

        saha_prefactor = 2.0 * lamb[:, 0]**-3 / ne_cm3
        min_partition = 10.0 ** (-decimal_precision)

        for iii in range(len(ions) - 1):
            Zi_i = np.clip(Zi[:, iii], min_partition, None)
            Zi_ip1 = np.clip(Zi[:, iii + 1], min_partition, None)
            a = saha_prefactor * (Zi_ip1 / Zi_i) * np.exp(-Ei[:, iii] / kT[:, 0])
            ci[:, iii + 1] = a * ci[:, iii]

        ci_sum = np.sum(ci, axis=1, keepdims=True)
        ci = np.divide(ci, np.clip(ci_sum, min_partition, None))
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


    def calculate(
        self,
        element: str,
        temperature: ArrayLike,
        ne: float,
        decimal_precision: int = 10,
        gsra: bool = False,
        time_gated: bool = False,
        time_0: float = 20,
        time_f: float = 100,
        t_step: float = 1,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Saha-Boltzmann distribution for a given element.

        Parameters
        ----------
        element : str
            Element symbol.
        temperature : ArrayLike
            Electron temperature values in kelvin.
        ne : float
            Base-10 logarithm of the electron density in ``cm**-3``.
        decimal_precision : int, optional
            Rounding precision for the ionization fraction, by default 10.
        gsra : bool, optional
            Include ground state resonant absorption, by default ``False``.
        time_gated : bool, optional
            Apply time gating to the intensity, by default ``False``.
        time_0 : float, optional
            Start time for time gating in nanoseconds, by default ``20``.
        time_f : float, optional
            Final time for time gating in nanoseconds, by default ``100``.
        t_step : float, optional
            Step size for time gating in nanoseconds, by default ``1``.

        Returns
        -------
        x : NDArray[np.float64]
            Emission line positions.
        y : NDArray[np.float64]
            Corresponding intensities.
        """
        temperature = self._temperature_array(temperature)

        # Broadcasting kT calculation
        kT = self.boltzmann_constant * temperature[:, None]
        intensity_constant = (4 * np.pi) ** -1 * self.plank_constant * self.speed_c * 10**9

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

        for iv, ion in enumerate(ions):
            Zind = (ionization == ion)
            stage_partition = np.clip(Zi[:, [iv]], 10.0 ** (-decimal_precision), None)
            boltzmann_weight = np.exp(-Ek[Zind][None, :] / kT)
            peak_intensity = (
                intensity_constant
                * peak_loc[Zind][None, :]**-1
                * gA[Zind][None, :]
                * boltzmann_weight
                / stage_partition
            )
            overall_intensity = ci[:, [iv]] * peak_intensity
            spectra.append(overall_intensity)
            location.append(np.tile(peak_loc[Zind], (len(temperature), 1)))

        x = np.concatenate(location, axis=-1)
        y = np.concatenate(spectra, axis=-1)

        return x, y
