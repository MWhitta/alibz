"""Approximate per-line Stark broadening for electron-density inference.

Electron-impact (quadratic) Stark widths of isolated, non-hydrogenic lines
scale approximately with the fourth power of the upper level's effective
principal quantum number and inversely with the square of the net charge
seen by the optical electron (Griem's semi-empirical scaling), and are
linear in the electron density in the impact regime:

    HWHM_j(n_e) ~= c4 * (n_eff_j**4 / z_j**2) * (n_e / n_e_ref),

with ``n_eff**2 = z**2 * Ry / (E_ion - E_k)``.  A single global scale
``c4`` (HWHM in nm at the reference density for a unit shape factor) is
deliberately the only free constant: the *relative* widths across lines of
different upper levels carry the n_e information, and ``c4`` is calibrated
against reference lines or corpus width statistics.

Hydrogen is deliberately excluded from this model: Balmer lines are
linear-Stark dominated (FWHM ~ n_e^(2/3), an order of magnitude wider),
and require a dedicated treatment.
"""

import numpy as np

#: Rydberg energy in eV.
RYDBERG_EV = 13.605693122994


def effective_quantum_number_sq(ion_energy_eV, Ek_eV, z):
    """``n_eff**2 = z**2 * Ry / (E_ion - E_k)`` of the upper level.

    Returns 0 where the level gap is non-positive (autoionising levels or
    missing ionization data) so such lines drop out of the width model
    instead of diverging.
    """
    gap = np.asarray(ion_energy_eV, dtype=float) - np.asarray(Ek_eV, dtype=float)
    z = np.asarray(z, dtype=float)
    safe_gap = np.where(gap > 0.0, gap, 1.0)
    return np.where(gap > 0.0, z ** 2 * RYDBERG_EV / safe_gap, 0.0)


def stark_shape_factor(ion_energy_eV, Ek_eV, z):
    """Dimensionless per-line factor ``n_eff**4 / z**2``.

    Zero where the upper-level binding energy is non-positive.
    """
    n_sq = effective_quantum_number_sq(ion_energy_eV, Ek_eV, z)
    z = np.asarray(z, dtype=float)
    return n_sq ** 2 / np.clip(z, 1.0, None) ** 2


def stark_hwhm(shape_factor, log_ne, c4, log_ne_ref=17.0):
    """Lorentzian Stark HWHM in nm at ``log10(n_e / cm^-3) = log_ne``.

    Linear in n_e (electron-impact regime):
    ``c4 * shape_factor * 10**(log_ne - log_ne_ref)``.
    """
    return (
        float(c4)
        * np.asarray(shape_factor, dtype=float)
        * 10.0 ** (float(log_ne) - float(log_ne_ref))
    )
