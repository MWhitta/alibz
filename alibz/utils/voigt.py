"""Shared Voigt profile utilities used across the alibz package.

When a CUDA-capable GPU is available and CuPy is installed, the
``multi_voigt`` function automatically dispatches to a GPU kernel
for large evaluations.  All public functions continue to accept and
return NumPy arrays so callers need no changes.
"""

import numpy as np
from scipy.special import voigt_profile as voigt

from alibz.gpu import gpu_available, to_numpy

if gpu_available:
    from alibz.gpu import multi_voigt_gpu as _multi_voigt_gpu


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


# GPU dispatch threshold.  The pseudo-Voigt approximation used on GPU has
# ~1% peak-area error but ~10% wing error, so GPU should only be used for
# coarse batch evaluations (e.g. synthetic spectra generation), NOT during
# iterative curve fitting where accuracy matters.  Set high to prevent
# automatic dispatch during least_squares iterations.
_GPU_THRESHOLD = 10_000_000


def multi_voigt(x, params, use_gpu=None):
    """Vectorized multi-Voigt summation.

    Parameters
    ----------
    x : array_like
        1-D array of x-values.
    params : array_like
        1-D array of parameters of length ``4*n`` with
        ``[amp, mu, sigma, gamma]`` repeated ``n`` times.
    use_gpu : bool or None
        Force GPU (``True``), force CPU (``False``), or auto-select
        (``None`` — uses GPU when available and the problem is large
        enough to amortise transfer overhead).

    Returns
    -------
    ndarray
        Summed Voigt profile evaluated at ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n_peaks = params.size // 4

    if use_gpu is None:
        use_gpu = gpu_available and (len(x) * n_peaks >= _GPU_THRESHOLD)

    if use_gpu and gpu_available:
        return _multi_voigt_gpu(x, params)

    # CPU path
    param_array = params.reshape(-1, 4)
    amps = param_array[:, 0]
    mus = param_array[:, 1]
    sigmas = param_array[:, 2]
    gammas = param_array[:, 3]
    X_shifted = x[:, None] - mus[None, :]
    profiles = voigt(X_shifted, sigmas[None, :], gammas[None, :])
    return np.sum(profiles * amps[None, :], axis=1)
