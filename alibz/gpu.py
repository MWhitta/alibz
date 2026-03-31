"""GPU backend abstraction for alibz.

Provides a unified array interface (``xp``) that resolves to CuPy when a
CUDA-capable GPU is available and falls back to NumPy otherwise.  All
GPU-accelerated routines live here so that the rest of the package can
remain backend-agnostic.

Usage
-----
>>> from alibz.gpu import xp, to_numpy, gpu_available
>>> a = xp.array([1.0, 2.0, 3.0])      # lives on GPU when available
>>> a_np = to_numpy(a)                   # always returns a NumPy array
"""

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

gpu_available: bool = False
"""``True`` when CuPy is importable and at least one CUDA device is found."""

import os as _os

if _os.environ.get('ALIBZ_NO_GPU', ''):
    cp = None
else:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()  # will raise if no GPU
        gpu_available = True
    except Exception:
        cp = None

# ``xp`` is the array module used throughout alibz when GPU mode is active.
xp = cp if gpu_available else np


def to_numpy(a):
    """Ensure *a* is a host-side NumPy array."""
    if gpu_available and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def to_device(a):
    """Move *a* to the active device (GPU if available, else no-op)."""
    if gpu_available:
        return cp.asarray(a)
    return np.asarray(a)


# ---------------------------------------------------------------------------
# GPU-accelerated Voigt profile (pseudo-Voigt approximation)
# ---------------------------------------------------------------------------
#
# Uses the Thompson pseudo-Voigt approximation — a weighted sum of
# Gaussian and Lorentzian profiles whose mixing ratio is computed from
# the component widths.  This avoids custom CUDA kernel compilation
# (no NVRTC dependency) and uses only CuPy's pre-compiled element-wise
# ops.  Accuracy is ~1% relative error, well within spectral fitting
# tolerance.

_LN2 = float(np.log(2))
_SQRT_2LN2 = float(np.sqrt(2.0 * np.log(2)))
_SQRT_PI = float(np.sqrt(np.pi))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_SQRT_LN2_PI = float(np.sqrt(np.log(2) / np.pi))


def voigt_profile_gpu(x, sigma, gamma):
    """Evaluate the Voigt profile on GPU via pseudo-Voigt.

    Uses the standard pseudo-Voigt approximation where both Gaussian and
    Lorentzian components share the Voigt FWHM ``fV``, and the mixing
    ratio ``eta`` controls the blend.  This matches the normalisation of
    ``scipy.special.voigt_profile`` (unit area).

    Parameters
    ----------
    x, sigma, gamma : cupy.ndarray or float
        Broadcasted inputs (same semantics as
        ``scipy.special.voigt_profile``).

    Returns
    -------
    cupy.ndarray
    """
    x = cp.asarray(x, dtype=cp.float64)
    sigma = cp.asarray(sigma, dtype=cp.float64)
    gamma = cp.asarray(gamma, dtype=cp.float64)

    sigma = cp.maximum(sigma, 1e-30)
    gamma = cp.maximum(gamma, 1e-30)

    # Component FWHMs
    fG = 2.0 * _SQRT_2LN2 * sigma
    fL = 2.0 * gamma

    # Thompson (1987) Voigt FWHM
    fV = (fG**5 + 2.69269 * fG**4 * fL
          + 2.42843 * fG**3 * fL**2
          + 4.47163 * fG**2 * fL**3
          + 0.07842 * fG * fL**4
          + fL**5) ** 0.2
    fV = cp.maximum(fV, 1e-30)

    # Pseudo-Voigt mixing parameter eta (Lorentzian fraction)
    r = fL / fV
    eta = 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3
    eta = cp.clip(eta, 0.0, 1.0)

    # Both components use the *Voigt* FWHM fV (not their own widths).
    # Gaussian with FWHM = fV, unit-area normalised:
    #   G(x) = (2√(ln2/π) / fV) exp(-4 ln2 (x/fV)²)
    G = (2.0 * _SQRT_LN2_PI / fV) * cp.exp(-4.0 * _LN2 * (x / fV)**2)

    # Lorentzian with FWHM = fV, unit-area normalised:
    #   L(x) = (2 / (π fV)) / (1 + 4(x/fV)²)
    L = (2.0 / (cp.pi * fV)) / (1.0 + 4.0 * (x / fV)**2)

    return eta * L + (1.0 - eta) * G


# ---------------------------------------------------------------------------
# GPU-accelerated PCA via truncated SVD
# ---------------------------------------------------------------------------

def pca_gpu(X, n_components):
    """Compute truncated PCA on the GPU.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Data matrix (will be centred internally).
    n_components : int
        Number of components to retain.

    Returns
    -------
    dict with keys:
        ``mean``        — (n_features,) mean vector
        ``components``  — (n_components, n_features) principal axes
        ``scores``      — (n_samples, n_components) projected data
        ``explained_variance_ratio`` — (n_components,) fraction of variance
    """
    X_d = cp.asarray(X, dtype=cp.float64)
    mean = X_d.mean(axis=0)
    Xc = X_d - mean

    # Full SVD, then truncate — for matrices up to ~50k×200 this is faster
    # than iterative methods on GPU.
    U, S, Vt = cp.linalg.svd(Xc, full_matrices=False)

    components = Vt[:n_components]
    scores = U[:, :n_components] * S[:n_components]
    total_var = (S ** 2).sum()
    explained = (S[:n_components] ** 2) / total_var

    return {
        'mean': cp.asnumpy(mean),
        'components': cp.asnumpy(components),
        'scores': cp.asnumpy(scores),
        'explained_variance_ratio': cp.asnumpy(explained),
    }


# ---------------------------------------------------------------------------
# GPU-accelerated batch multi-Voigt evaluation
# ---------------------------------------------------------------------------

def multi_voigt_gpu(x, params):
    """Evaluate a sum of Voigt profiles on GPU.

    Same interface as :func:`alibz.utils.voigt.multi_voigt` but runs on GPU.
    """
    x_d = cp.asarray(x, dtype=cp.float64)
    params = cp.asarray(params, dtype=cp.float64).reshape(-1, 4)
    amps = params[:, 0]
    mus = params[:, 1]
    sigmas = params[:, 2]
    gammas = params[:, 3]

    # Broadcast: (len_x, n_peaks)
    X_shifted = x_d[:, None] - mus[None, :]
    profiles = voigt_profile_gpu(X_shifted, sigmas[None, :], gammas[None, :])
    result = cp.sum(profiles * amps[None, :], axis=1)
    return cp.asnumpy(result)


# ---------------------------------------------------------------------------
# GPU-accelerated batch interpolation
# ---------------------------------------------------------------------------

def batch_interpolate_gpu(raw_wl, raw_int, common_wl):
    """Interpolate many spectra onto a common wavelength grid on GPU.

    Parameters
    ----------
    raw_wl : ndarray, shape (n_spectra, n_raw)
        Per-spectrum wavelength axes.
    raw_int : ndarray, shape (n_spectra, n_raw)
        Per-spectrum intensities.
    common_wl : ndarray, shape (n_common,)
        Target grid.

    Returns
    -------
    ndarray, shape (n_spectra, n_common)
    """
    n_spectra = raw_wl.shape[0]
    n_common = len(common_wl)

    wl_d = cp.asarray(raw_wl, dtype=cp.float64)
    int_d = cp.asarray(raw_int, dtype=cp.float64)
    cw_d = cp.asarray(common_wl, dtype=cp.float64)

    out = cp.zeros((n_spectra, n_common), dtype=cp.float64)

    for i in range(n_spectra):
        idx = cp.searchsorted(wl_d[i], cw_d).clip(1, wl_d.shape[1] - 1)
        x0 = wl_d[i][idx - 1]
        x1 = wl_d[i][idx]
        y0 = int_d[i][idx - 1]
        y1 = int_d[i][idx]
        t = (cw_d - x0) / (x1 - x0 + 1e-30)
        out[i] = y0 + t * (y1 - y0)
        # Zero outside bounds
        out[i, cw_d < wl_d[i, 0]] = 0.0
        out[i, cw_d > wl_d[i, -1]] = 0.0

    return cp.asnumpy(out)


# ---------------------------------------------------------------------------
# GPU-accelerated batch peak window extraction
# ---------------------------------------------------------------------------

def extract_windows_gpu(spectra, wavelength, peak_params_list, half_window,
                        n_window_points=101):
    """Extract, baseline-subtract, and normalise peak windows on GPU.

    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, n_channels)
    wavelength : ndarray, shape (n_channels,)
    peak_params_list : list of ndarray, each shape (n_peaks_i, 4)
        Per-spectrum ``[amp, mu, sigma, gamma]`` arrays.
    half_window : float
        Half-width of the extraction window in nm.
    n_window_points : int
        Resampled window size.

    Returns
    -------
    windows : ndarray, shape (total_peaks, n_window_points)
    metadata : list of dict
    """
    wl = cp.asarray(wavelength, dtype=cp.float64)
    wl_lo, wl_hi = float(wl[0]), float(wl[-1])
    x_fixed = np.linspace(0, 1, n_window_points)

    windows = []
    metadata = []

    for spec_idx, params in enumerate(peak_params_list):
        if params.size == 0:
            continue
        spec_d = cp.asarray(spectra[spec_idx], dtype=cp.float64)

        for peak_idx in range(params.shape[0]):
            amp, mu, sigma, gamma = params[peak_idx]
            lo, hi = mu - half_window, mu + half_window
            if lo < wl_lo or hi > wl_hi:
                continue

            mask = (wl >= lo) & (wl <= hi)
            wl_win = wl[mask]
            int_win = spec_d[mask].copy()

            if len(wl_win) < 5:
                continue

            # Linear baseline subtraction
            baseline = cp.linspace(float(int_win[0]), float(int_win[-1]),
                                   len(int_win))
            int_win -= baseline

            # Normalise to unit peak height (match CPU path in peaky_pca.py)
            peak_range = float(cp.max(int_win) - cp.min(int_win))
            if peak_range <= 0:
                continue
            int_win -= cp.min(int_win)
            int_win /= peak_range

            # Resample to fixed grid (transfer to CPU for interp1d)
            wl_np = cp.asnumpy(wl_win)
            int_np = cp.asnumpy(int_win)
            x_norm = np.linspace(0, 1, len(wl_np))
            from scipy.interpolate import interp1d
            resampler = interp1d(x_norm, int_np, kind='linear')
            window = resampler(x_fixed)

            windows.append(window)
            metadata.append({
                'spectrum_idx': spec_idx,
                'peak_idx': peak_idx,
                'mu': mu, 'sigma': sigma, 'gamma': gamma, 'amp': amp,
                'fwhm': float(_voigt_width_np(sigma, gamma)),
            })

    return np.array(windows) if windows else np.empty((0, n_window_points)), metadata


def _voigt_width_np(sigma, gamma):
    """Thompson FWHM (CPU-only helper)."""
    fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_l = 2 * gamma
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)
