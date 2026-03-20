"""Shared Voigt profile utilities used across the alibz package."""

import numpy as np
from scipy.special import voigt_profile as voigt


def voigt_width(sigma, gamma):
    """Approximate FWHM of a Voigt profile (Thompson formula).

    Parameters
    ----------
    sigma : float or array_like
        Gaussian component width.
    gamma : float or array_like
        Lorentzian component width.

    Returns
    -------
    float or ndarray
        Approximate full width at half maximum.
    """
    fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_l = 2 * gamma
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def multi_voigt(x, params):
    """Vectorized multi-Voigt summation.

    Parameters
    ----------
    x : array_like
        1-D array of x-values.
    params : array_like
        1-D array of parameters of length ``4*n`` with
        ``[amp, mu, sigma, gamma]`` repeated ``n`` times.

    Returns
    -------
    ndarray
        Summed Voigt profile evaluated at ``x``.
    """
    params = np.asarray(params)
    param_array = params.reshape(-1, 4)
    amps = param_array[:, 0]
    mus = param_array[:, 1]
    sigmas = param_array[:, 2]
    gammas = param_array[:, 3]
    X_shifted = x[:, None] - mus[None, :]
    profiles = voigt(X_shifted, sigmas[None, :], gammas[None, :])
    return np.sum(profiles * amps[None, :], axis=1)
