"""Vacuum-air wavelength conversion.

The pickled NIST line lists store Ritz VACUUM wavelengths, while LIBS
spectrometers are calibrated to standard-air wavelengths (the ASD
convention: air above 200 nm, vacuum below).  Left unconverted, the
mismatch is the Edlen dispersion difference — 0.11 nm at 400 nm rising to
0.21 nm at 770 nm — which exceeds the indexer's matching tolerance
everywhere and makes observed peaks silently match wrong lines.
"""

import numpy as np


def vacuum_to_air(wavelength_vac_nm):
    """Convert Ritz vacuum wavelengths [nm] to standard air.

    Uses the Edlen (1966) dispersion of standard air as adopted by the
    NIST Atomic Spectra Database:

        n - 1 = 1e-8 * (8342.13 + 2406030/(130 - s^2) + 15997/(38.9 - s^2))

    with ``s = 1/lambda_vac`` in inverse micrometres.  Wavelengths below
    200 nm are returned unchanged: the ASD (and instrument-calibration)
    convention quotes vacuum wavelengths in the VUV, where the formula is
    not valid and air is opaque anyway.
    """
    wl = np.asarray(wavelength_vac_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_sq = (1.0e3 / wl) ** 2
        n = 1.0 + 1.0e-8 * (
            8342.13
            + 2406030.0 / (130.0 - sigma_sq)
            + 15997.0 / (38.9 - sigma_sq)
        )
        converted = wl / n
    return np.where(wl >= 200.0, converted, wl)
