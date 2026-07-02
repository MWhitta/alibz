"""Shared utilities for the alibz package."""

from alibz.utils.constants import BOLTZMANN, PLANCK, SPEED_OF_LIGHT, ELECTRON_MASS
from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import SahaBoltzmann, line_emissivity
from alibz.utils.stark import (
    calibrate_c4,
    halpha_log_ne,
    halpha_ne_bounds,
    stark_hwhm,
    stark_shape_factor,
)
from alibz.utils.voigt import voigt_width, multi_voigt
from alibz.utils.wavelength import estimate_wavelength_shift, vacuum_to_air
from alibz.utils.absorption import escape_factor, doublet_ratio
from alibz.utils.dataloader import Data

__all__ = [
    "BOLTZMANN",
    "PLANCK",
    "SPEED_OF_LIGHT",
    "ELECTRON_MASS",
    "Database",
    "SahaBoltzmann",
    "line_emissivity",
    "stark_hwhm",
    "stark_shape_factor",
    "halpha_log_ne",
    "halpha_ne_bounds",
    "calibrate_c4",
    "voigt_width",
    "multi_voigt",
    "vacuum_to_air",
    "estimate_wavelength_shift",
    "escape_factor",
    "doublet_ratio",
    "Data",
]
