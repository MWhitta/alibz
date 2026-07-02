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


# Gigosos, Gonzalez & Cardenoso (2003) H-alpha fit: FWHM[nm] =
# 0.549 * (n_e / 1e23 m^-3)^0.67965, weakly T- and mu-dependent.
_HALPHA_FWHM_REF_NM = 0.549
_HALPHA_EXPONENT = 0.67965


def halpha_log_ne(fwhm_nm):
    """log10(n_e / cm^-3) from the H-alpha Stark FWHM (nm).

    Hydrogen Balmer lines are linear-Stark dominated, so they carry an
    ABSOLUTE electron-density scale independent of the quadratic-Stark
    ``c4`` calibration — this is the anchor that breaks the exact
    (c4, n_e) degeneracy of the width model.  Uses the Gigosos-Gonzalez-
    Cardenoso (2003) fit; the supplied width should be the Stark
    component (instrumental broadening removed).
    """
    fwhm_nm = np.asarray(fwhm_nm, dtype=float)
    return 17.0 + np.log10(fwhm_nm / _HALPHA_FWHM_REF_NM) / _HALPHA_EXPONENT


#: Air wavelength of H-alpha in nm.
HALPHA_NM = 656.28


def halpha_ne_bounds(
    peak_array,
    tolerance_nm=0.2,
    min_gamma_nm=0.05,
    half_width_dex=0.3,
):
    """Data-driven ``ne_bounds`` from an H-alpha peak in a fitted peak table.

    Scans ``peak_array`` (rows ``[amplitude, center, sigma, gamma]``) for a
    peak within ``tolerance_nm`` of H-alpha whose Lorentzian HWHM exceeds
    ``min_gamma_nm`` (below that, the width is instrumental, not Stark, and
    the anchor would be meaningless).  Returns ``(lo, hi)`` bounds in
    log10(n_e / cm^-3) centred on :func:`halpha_log_ne` of the peak's
    Lorentzian FWHM, or ``None`` when no usable H-alpha is present.

    This is the practical per-spectrum n_e measurement: at typical LIBS
    densities the quadratic-Stark widths of non-hydrogenic lines
    (~0.002 nm at 1e17 cm^-3) sit an order of magnitude below realistic
    per-peak width noise (~0.03-0.05 nm on handheld-class spectra), so the
    linear-Stark H-alpha width — an order of magnitude larger — carries
    essentially all of the accessible width information.  The
    ``half_width_dex`` default reflects the GGC-fit accuracy plus a
    realistic width-fit error.
    """
    peak_array = np.atleast_2d(np.asarray(peak_array, dtype=float))
    if peak_array.size == 0 or peak_array.shape[1] < 4:
        return None
    near = np.abs(peak_array[:, 1] - HALPHA_NM) <= tolerance_nm
    if not np.any(near):
        return None
    candidate = peak_array[near][np.argmax(peak_array[near, 0])]
    gamma = float(candidate[3])
    if not np.isfinite(gamma) or gamma < min_gamma_nm:
        return None
    center = float(halpha_log_ne(2.0 * gamma))
    return (center - half_width_dex, center + half_width_dex)


def calibrate_c4(obs_gamma, shape_factor, log_ne, weights=None, log_ne_ref=17.0):
    """Calibrate ``(c4, gamma_inst)`` from observed Lorentzian HWHMs at a
    KNOWN electron density (e.g. from :func:`halpha_log_ne`).

    Weighted least squares of ``gamma_obs = gamma_inst + c4 * x`` with
    ``x = shape_factor * 10**(log_ne - log_ne_ref)``.  Returns
    ``(c4, gamma_inst)``; both clipped at 0.  Lines with non-positive
    shape factors are excluded.
    """
    obs_gamma = np.asarray(obs_gamma, dtype=float)
    shape_factor = np.asarray(shape_factor, dtype=float)
    weights = (
        np.ones_like(obs_gamma)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    valid = (shape_factor > 0) & np.isfinite(obs_gamma) & (weights > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "calibrate_c4 needs at least two lines with positive shape factors"
        )

    x = shape_factor[valid] * 10.0 ** (float(log_ne) - float(log_ne_ref))
    y = obs_gamma[valid]
    w = weights[valid]

    w_sum = np.sum(w)
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    var_x = np.sum(w * (x - x_mean) ** 2)
    if var_x <= 0:
        raise ValueError(
            "calibrate_c4 needs a spread of shape factors (all x identical)"
        )
    c4 = float(np.sum(w * (x - x_mean) * (y - y_mean)) / var_x)
    gamma_inst = float(y_mean - c4 * x_mean)
    return max(c4, 0.0), max(gamma_inst, 0.0)
