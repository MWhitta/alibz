"""Deterministic, explicit-ion-stage LIBS spectrum synthesis.

This module is the forward model for synthetic-training data.  It is
deliberately separate from :class:`alibz.peaky_maker.PeakyMaker`: the legacy
maker obtains ion fractions from Saha equilibrium, whereas this renderer takes
the abundance of stages I--III directly and never calls a Saha ionisation
solver.

The renderer has two positivity domains:

* line/continuum emission and expected physical detector counts are
  nonnegative;
* exported counts may be signed after dark/offset subtraction, read noise,
  and a (possibly ringing) vendor export kernel.

Line areas use the repository's common Boltzmann emissivity convention.
Instrument profiles are integrated over explicit wavelength cells, so changing
the numerical grid does not reinterpret a point-sampled peak height as a
channel count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erfcinv, roots_legendre
from scipy.special import voigt_profile

from alibz.elements import ATOMIC_NUMBER, ELEMENTS_BY_ATOMIC_NUMBER
from alibz.utils.absorption import C2_NM_K as _C2_NM_K, stimulated_emission_factor
from alibz.utils.absorption import escape_factor
from alibz.utils.constants import BOLTZMANN
from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import line_emissivity
from alibz.utils.stark import (
    _HALPHA_EXPONENT,
    _HALPHA_FWHM_REF_NM,
    HALPHA_NM,
    stark_hwhm,
    stark_shape_factor,
)
from alibz.utils.voigt import voigt_width
from alibz.utils.wavelength import vacuum_to_air


N_ELEMENTS = 92
N_STAGES = 3
NATIVE_WL_LO_NM = 190.0
NATIVE_WL_HI_NM = 910.0
NATIVE_PITCH_NM = 1.0 / 30.0
GENERATOR_VERSION = "explicit-stage-v0.1"

# Standard atomic weights (or conventional mass numbers for elements without
# a standard atomic weight), H through U.  The values are used only for the
# thermal Doppler width; the atomic-line database remains the authority for
# transition physics.
ATOMIC_MASS_U = np.asarray([
    1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999,
    18.998403163, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908,
    47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934,
    63.546, 65.38, 69.723, 72.630, 74.921595, 78.971, 79.904, 83.798,
    85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.95, 98.0, 101.07,
    102.90550, 106.42, 107.8682, 112.414, 114.818, 118.710, 121.760,
    127.60, 126.90447, 131.293, 132.90545196, 137.327, 138.90547,
    140.116, 140.90766, 144.242, 145.0, 150.36, 151.964, 157.25,
    158.92535, 162.500, 164.93033, 167.259, 168.93422, 173.045,
    174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217,
    195.084, 196.966569, 200.592, 204.38, 207.2, 208.98040, 209.0,
    210.0, 222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588,
    238.02891,
], dtype=float)
if ATOMIC_MASS_U.shape != (N_ELEMENTS,):  # pragma: no cover - import guard
    raise RuntimeError("atomic-mass table must retain 92 H--U positions")

_K_B_SI = 1.380649e-23
_AMU_KG = 1.66053906660e-27
_C_M_S = 299_792_458.0


def _readonly_float_array(value, shape, name, default=None):
    if value is None:
        value = default
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=float)
    else:
        try:
            arr = np.broadcast_to(arr, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} must broadcast to {shape}, got {arr.shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


def _array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _stage_levels(lines, stage):
    """Deduplicated ``[energy_eV, degeneracy]`` table for one stage."""
    ion = lines[:, 0].astype(float)
    mask = ion == float(stage)
    energies = np.concatenate((
        lines[mask, 4].astype(float), lines[mask, 5].astype(float)
    ))
    degeneracies = np.concatenate((
        lines[mask, 12].astype(float), lines[mask, 13].astype(float)
    ))
    if energies.size == 0:
        return np.empty((0, 2), dtype=float)
    levels = np.column_stack((
        np.round(energies, 8),
        np.round(np.clip(degeneracies, 1e-30, None), 8),
    ))
    return np.unique(levels, axis=0)


@dataclass(frozen=True)
class ChannelGrid:
    """Wavelength centers and physical cell edges for one spectrum.

    Disjoint regions are represented by gaps between successive cells.  When
    edges are omitted, gaps wider than ``gap_factor`` times the median pitch
    start a new region instead of becoming one enormous synthetic channel.
    """

    centers_nm: np.ndarray
    left_edges_nm: Optional[np.ndarray] = None
    right_edges_nm: Optional[np.ndarray] = None
    gap_factor: float = 4.0

    def __post_init__(self):
        centers = np.asarray(self.centers_nm, dtype=float).reshape(-1)
        if centers.size < 1 or not np.all(np.isfinite(centers)):
            raise ValueError("centers_nm must contain finite wavelengths")
        if centers.size > 1 and np.any(np.diff(centers) <= 0):
            raise ValueError("centers_nm must be strictly increasing")

        if self.left_edges_nm is None and self.right_edges_nm is None:
            left, right = self._infer_edges(centers, float(self.gap_factor))
        elif self.left_edges_nm is None or self.right_edges_nm is None:
            raise ValueError("left and right channel edges must be supplied together")
        else:
            left = np.asarray(self.left_edges_nm, dtype=float).reshape(-1)
            right = np.asarray(self.right_edges_nm, dtype=float).reshape(-1)
            if left.shape != centers.shape or right.shape != centers.shape:
                raise ValueError("channel edge arrays must match centers_nm")

        if (not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or
                np.any(left >= centers) or np.any(centers >= right)):
            raise ValueError("every channel must satisfy left < center < right")
        if centers.size > 1 and np.any(right[:-1] > left[1:] + 1e-12):
            raise ValueError("channel cells must not overlap")

        centers = centers.copy()
        left = np.asarray(left, dtype=float).copy()
        right = np.asarray(right, dtype=float).copy()
        centers.setflags(write=False)
        left.setflags(write=False)
        right.setflags(write=False)
        object.__setattr__(self, "centers_nm", centers)
        object.__setattr__(self, "left_edges_nm", left)
        object.__setattr__(self, "right_edges_nm", right)

    @staticmethod
    def _infer_edges(centers: np.ndarray, gap_factor: float):
        if centers.size == 1:
            raise ValueError("a one-channel region requires explicit edges")
        if not np.isfinite(gap_factor) or gap_factor <= 1.0:
            raise ValueError("gap_factor must be greater than one")

        delta = np.diff(centers)
        pitch = float(np.median(delta))
        breaks = np.nonzero(delta > gap_factor * pitch)[0]
        starts = np.r_[0, breaks + 1]
        stops = np.r_[breaks + 1, centers.size]
        left = np.empty_like(centers)
        right = np.empty_like(centers)

        for start, stop in zip(starts, stops):
            region = centers[start:stop]
            if region.size == 1:
                # Use the global median pitch for a singleton embedded among
                # otherwise well-defined regions.
                local_pitch = pitch
                left[start] = region[0] - local_pitch / 2.0
                right[start] = region[0] + local_pitch / 2.0
                continue
            mids = 0.5 * (region[:-1] + region[1:])
            left[start + 1:stop] = mids
            right[start:stop - 1] = mids
            left[start] = region[0] - 0.5 * (region[1] - region[0])
            right[stop - 1] = region[-1] + 0.5 * (region[-1] - region[-2])
        return left, right

    @classmethod
    def native(cls):
        n = int(round((NATIVE_WL_HI_NM - NATIVE_WL_LO_NM) / NATIVE_PITCH_NM))
        centers = NATIVE_WL_LO_NM + np.arange(n, dtype=float) * NATIVE_PITCH_NM
        return cls(centers)

    @property
    def widths_nm(self):
        return self.right_edges_nm - self.left_edges_nm

    @property
    def digest(self):
        return _array_hash(self.centers_nm, self.left_edges_nm, self.right_edges_nm)


@dataclass(frozen=True)
class PlasmaComponent:
    """One emitting component with direct element-stage abundances.

    ``stage_abundance[e, s]`` is a nuclei fraction, with ``s=0,1,2``
    corresponding to spectroscopic stages I, II, and III.  ``effective_column``
    is the provisional per-element scale multiplying the repository's relative
    line opacity.  It is zero for an optically thin component.
    """

    stage_abundance: np.ndarray
    temperature_k: np.ndarray = 10_000.0
    log_ne_cm3: np.ndarray = 17.0
    effective_column: np.ndarray = 0.0
    emission_scale: float = 1.0e-3
    continuum_scale: float = 0.0
    label: str = "target"

    def __post_init__(self):
        abundance = _readonly_float_array(
            self.stage_abundance, (N_ELEMENTS, N_STAGES), "stage_abundance"
        )
        if np.any(abundance < 0):
            raise ValueError("stage_abundance must be nonnegative")
        temperature = _readonly_float_array(
            self.temperature_k, (N_ELEMENTS, N_STAGES), "temperature_k"
        )
        if np.any(temperature <= 0):
            raise ValueError("temperature_k must be strictly positive")
        log_ne = _readonly_float_array(
            self.log_ne_cm3, (N_ELEMENTS,), "log_ne_cm3"
        )
        column = _readonly_float_array(
            self.effective_column, (N_ELEMENTS,), "effective_column"
        )
        if np.any(column < 0):
            raise ValueError("effective_column must be nonnegative")
        if not np.isfinite(self.emission_scale) or self.emission_scale < 0:
            raise ValueError("emission_scale must be finite and nonnegative")
        if not np.isfinite(self.continuum_scale) or self.continuum_scale < 0:
            raise ValueError("continuum_scale must be finite and nonnegative")
        object.__setattr__(self, "stage_abundance", abundance)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "log_ne_cm3", log_ne)
        object.__setattr__(self, "effective_column", column)

    @property
    def element_abundance(self):
        return np.sum(self.stage_abundance, axis=1)

    @classmethod
    def from_mapping(
        cls,
        stage_abundance: Mapping[Tuple[str, int], float],
        temperature_k=10_000.0,
        log_ne_cm3=17.0,
        effective_column=0.0,
        emission_scale=1.0e-3,
        continuum_scale=0.0,
        label="target",
    ):
        abundance = np.zeros((N_ELEMENTS, N_STAGES), dtype=float)
        for (element, stage), value in stage_abundance.items():
            if element not in ATOMIC_NUMBER:
                raise KeyError(element)
            if int(stage) not in (1, 2, 3):
                raise ValueError("ion stage must be I, II, or III (1, 2, or 3)")
            abundance[ATOMIC_NUMBER[element] - 1, int(stage) - 1] = float(value)
        return cls(
            abundance,
            temperature_k=temperature_k,
            log_ne_cm3=log_ne_cm3,
            effective_column=effective_column,
            emission_scale=emission_scale,
            continuum_scale=continuum_scale,
            label=label,
        )


@dataclass(frozen=True)
class SyntheticScene:
    """Replayable physical scene for a single individual-shot spectrum."""

    target: PlasmaComponent
    ambient_gas: Optional[PlasmaComponent] = None
    seed: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if int(self.seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self):
        arrays = [
            self.target.stage_abundance,
            self.target.temperature_k,
            self.target.log_ne_cm3,
            self.target.effective_column,
        ]
        if self.ambient_gas is not None:
            arrays.extend([
                self.ambient_gas.stage_abundance,
                self.ambient_gas.temperature_k,
                self.ambient_gas.log_ne_cm3,
                self.ambient_gas.effective_column,
            ])
        extra = json.dumps(
            {
                "seed": self.seed,
                "target_scale": self.target.emission_scale,
                "target_continuum": self.target.continuum_scale,
                "gas_scale": None if self.ambient_gas is None
                else self.ambient_gas.emission_scale,
                "gas_continuum": None if self.ambient_gas is None
                else self.ambient_gas.continuum_scale,
                "metadata": self.metadata,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(_array_hash(*arrays).encode("ascii"))
        digest.update(extra)
        return digest.hexdigest()


@dataclass(frozen=True)
class InstrumentResponse:
    """Configurable current-instrument observation model.

    Defaults are deliberately provisional Voigt widths and an identity export
    kernel.  A corpus-calibrated PCA/LSF artifact will replace them without
    changing the scene or result schema.
    """

    segment_edges_nm: Tuple[float, float] = (365.0, 620.0)
    gaussian_sigma_nm: Tuple[float, float, float] = (0.030, 0.040, 0.050)
    lorentzian_gamma_nm: Tuple[float, float, float] = (0.010, 0.012, 0.015)
    segment_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    background_counts: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    export_offset_counts: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    read_noise_std_counts: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dark_offset_std_counts: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    electron_gain: float = 1.0
    shot_noise: bool = False
    export_kernel: Tuple[float, ...] = (1.0,)
    stark_c4_nm: float = 1.0e-4
    stark_log_ne_ref: float = 17.0
    tail_mass_tolerance: float = 1.0e-3
    quadrature_order: int = 8
    calibrated: bool = False
    profile_artifact: Optional[str] = None

    def __post_init__(self):
        edges = tuple(float(x) for x in self.segment_edges_nm)
        if len(edges) != 2 or edges[0] >= edges[1]:
            raise ValueError("segment_edges_nm must contain two increasing values")
        for name in (
            "gaussian_sigma_nm", "lorentzian_gamma_nm", "segment_gain",
            "background_counts", "export_offset_counts",
            "read_noise_std_counts", "dark_offset_std_counts",
        ):
            values = tuple(float(x) for x in getattr(self, name))
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)
        if np.any(np.asarray(self.gaussian_sigma_nm) <= 0):
            raise ValueError("gaussian_sigma_nm must be positive")
        if np.any(np.asarray(self.lorentzian_gamma_nm) < 0):
            raise ValueError("lorentzian_gamma_nm must be nonnegative")
        if np.any(np.asarray(self.segment_gain) <= 0):
            raise ValueError("segment_gain must be positive")
        if np.any(np.asarray(self.background_counts) < 0):
            raise ValueError("background_counts represents pre-subtraction counts")
        if np.any(np.asarray(self.read_noise_std_counts) < 0):
            raise ValueError("read_noise_std_counts must be nonnegative")
        if np.any(np.asarray(self.dark_offset_std_counts) < 0):
            raise ValueError("dark_offset_std_counts must be nonnegative")
        if not np.isfinite(self.electron_gain) or self.electron_gain <= 0:
            raise ValueError("electron_gain must be positive")
        kernel = np.asarray(self.export_kernel, dtype=float).reshape(-1)
        if (kernel.size < 1 or kernel.size % 2 != 1 or
                not np.all(np.isfinite(kernel)) or np.isclose(kernel.sum(), 0.0)):
            raise ValueError("export_kernel must be finite, odd-length, and nonzero-sum")
        kernel = kernel / kernel.sum()
        object.__setattr__(self, "export_kernel", tuple(kernel.tolist()))
        if not 0 < self.tail_mass_tolerance < 0.1:
            raise ValueError("tail_mass_tolerance must be between zero and 0.1")
        if int(self.quadrature_order) < 4:
            raise ValueError("quadrature_order must be at least four")
        object.__setattr__(self, "quadrature_order", int(self.quadrature_order))
        object.__setattr__(self, "segment_edges_nm", edges)

    def segment_index(self, wavelength_nm):
        return np.searchsorted(self.segment_edges_nm, wavelength_nm).astype(int)

    def manifest(self):
        return {
            "segment_edges_nm": list(self.segment_edges_nm),
            "gaussian_sigma_nm": list(self.gaussian_sigma_nm),
            "lorentzian_gamma_nm": list(self.lorentzian_gamma_nm),
            "segment_gain": list(self.segment_gain),
            "background_counts": list(self.background_counts),
            "export_offset_counts": list(self.export_offset_counts),
            "read_noise_std_counts": list(self.read_noise_std_counts),
            "dark_offset_std_counts": list(self.dark_offset_std_counts),
            "electron_gain": self.electron_gain,
            "shot_noise": bool(self.shot_noise),
            "export_kernel": list(self.export_kernel),
            "stark_c4_nm": self.stark_c4_nm,
            "stark_log_ne_ref": self.stark_log_ne_ref,
            "tail_mass_tolerance": self.tail_mass_tolerance,
            "quadrature_order": self.quadrature_order,
            "calibrated": bool(self.calibrated),
            "profile_artifact": self.profile_artifact,
        }

    @classmethod
    def from_manifest(cls, values: Mapping[str, object]):
        """Reconstruct a response from a saved calibration manifest."""
        fields = {
            "segment_edges_nm", "gaussian_sigma_nm",
            "lorentzian_gamma_nm", "segment_gain", "background_counts",
            "export_offset_counts", "read_noise_std_counts",
            "dark_offset_std_counts", "electron_gain", "shot_noise",
            "export_kernel", "stark_c4_nm", "stark_log_ne_ref",
            "tail_mass_tolerance", "quadrature_order", "calibrated",
            "profile_artifact",
        }
        return cls(**{key: values[key] for key in fields if key in values})

    @classmethod
    def provisional_current_instrument(cls):
        """Corpus-anchored individual-shot defaults pending the PCA refit.

        Widths use the smallest FWHM mode in each segment from
        ``peaky_data`` (0.1682/0.3426/0.3615 nm), decomposed using its median
        Gaussian fraction.  Background/noise values are medians from 100
        MW2-112 individual shots.  The export kernel and conditional PCA score
        distribution are still unavailable, so the returned model is
        intentionally marked uncalibrated.
        """
        return cls(
            gaussian_sigma_nm=(0.0404213, 0.0823326, 0.0868746),
            lorentzian_gamma_nm=(0.0558694, 0.1137982, 0.1200760),
            background_counts=(45.5345, 134.7987, 75.9372),
            # Quadrature difference between total local noise and the
            # Poisson variance of the representative background.
            read_noise_std_counts=(20.3354, 9.3553, 18.3065),
            shot_noise=True,
            profile_artifact=(
                "peaky_data-summary+MW2-112-929-individual-provisional"
            ),
            calibrated=False,
        )


@dataclass(frozen=True)
class AtomicStrengthUncertainty:
    """Replayable source-class perturbations of transition strengths.

    The normalized heavy-element catalog retains source codes for Se/Th/U.
    ``sigma_dex`` values are deliberately configurable and provisional; they
    broaden synthetic truth rather than asserting a known database error.
    A fraction of each source-class perturbation is common to all of its lines,
    preserving correlated compilation bias.
    """

    enabled: bool = False
    sigma_dex: Mapping[str, float] = field(default_factory=lambda: {
        "NIST_GA": 0.10,
        "MC": 0.35,
        "CB": 0.35,
        "MULT": 0.50,
        "GUES": 0.70,
    })
    common_fraction: float = 0.5

    def __post_init__(self):
        values = {str(key): float(value) for key, value in self.sigma_dex.items()}
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("atomic strength uncertainties must be nonnegative")
        if not 0.0 <= float(self.common_fraction) <= 1.0:
            raise ValueError("common_fraction must lie in [0, 1]")
        object.__setattr__(self, "sigma_dex", values)
        object.__setattr__(self, "common_fraction", float(self.common_fraction))

    def manifest(self):
        return {
            "enabled": bool(self.enabled),
            "sigma_dex": dict(self.sigma_dex),
            "common_fraction": self.common_fraction,
        }


@dataclass(frozen=True)
class SyntheticSpectrum:
    wavelength_nm: np.ndarray
    intensity_counts: np.ndarray
    expected_export_counts: np.ndarray
    physical_channel_counts: np.ndarray
    target_channel_counts: np.ndarray
    ambient_channel_counts: np.ndarray
    stage_observable: np.ndarray
    manifest: Mapping[str, object]
    line_table: Optional[Mapping[str, np.ndarray]] = None


def dry_air_component(
    argon_purge_fraction: float,
    stage_fractions: Mapping[str, Sequence[float]],
    temperature_k=10_000.0,
    log_ne_cm3=17.0,
    effective_column=0.0,
    emission_scale=1.0e-3,
    continuum_scale=0.0,
):
    """Build the ambient component between pure Ar and fixed dry air.

    Dry air contains N2/O2/Ar only in this development round.  Molecular
    volume fractions are converted to elemental-nuclei fractions before ion
    stages are assigned.  Stage fractions are explicit and must be supplied;
    no Saha model is used for the gas.
    """
    p = float(argon_purge_fraction)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("argon_purge_fraction must lie in [0, 1]")
    required = ("N", "O", "Ar")
    missing = [el for el in required if el not in stage_fractions]
    if missing:
        raise ValueError(f"stage_fractions missing gas elements: {missing}")

    # Mole fractions at the dry-air endpoint; CO2 and humidity are excluded
    # by the locked current-round gas contract.
    molecule = {
        "N": (1.0 - p) * 2.0 * 0.78084,
        "O": (1.0 - p) * 2.0 * 0.20946,
        "Ar": p + (1.0 - p) * 0.00934,
    }
    total = sum(molecule.values())
    abundance = np.zeros((N_ELEMENTS, N_STAGES), dtype=float)
    for element in required:
        split = np.asarray(stage_fractions[element], dtype=float).reshape(-1)
        if split.shape != (N_STAGES,) or np.any(split < 0) or not np.isclose(split.sum(), 1.0):
            raise ValueError(f"{element} gas stage fractions must be three nonnegative values summing to one")
        abundance[ATOMIC_NUMBER[element] - 1] = molecule[element] / total * split
    return PlasmaComponent(
        abundance,
        temperature_k=temperature_k,
        log_ne_cm3=log_ne_cm3,
        effective_column=effective_column,
        emission_scale=emission_scale,
        continuum_scale=continuum_scale,
        label="ambient-Ar-dry-air",
    )


class SyntheticSpectrumGenerator:
    """Render explicit-stage physical scenes onto arbitrary channel grids."""

    def __init__(
        self,
        dbpath="db",
        instrument: Optional[InstrumentResponse] = None,
        atomic_uncertainty: Optional[AtomicStrengthUncertainty] = None,
    ):
        self.db = Database(dbpath)
        self.instrument = instrument or InstrumentResponse()
        self.atomic_uncertainty = atomic_uncertainty or AtomicStrengthUncertainty()
        if tuple(self.db.elements) != tuple(ELEMENTS_BY_ATOMIC_NUMBER):
            raise RuntimeError("database element order does not match the fixed 92-position schema")
        self.support_mask = np.asarray(self.db.support_mask, dtype=bool).copy()
        self.stage_support = self._build_stage_support()
        self._partition_cache: Dict[Tuple[int, int, float], float] = {}
        self._ion_energy_cache: Dict[Tuple[int, int], Optional[float]] = {}
        self._strength_source = self._load_strength_sources()
        self._quad_nodes, self._quad_weights = roots_legendre(
            self.instrument.quadrature_order
        )

    def _load_strength_sources(self):
        sources = {}
        path = self.db.dbpath / "quantitative_lines_se_th_u.tsv"
        if not path.exists():
            return sources
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                wavelength = float(vacuum_to_air(float(row["wavelength_vacuum_nm"])))
                source = "NIST_GA" if row.get("nist_gA") == "1" else row.get(
                    "kurucz_ref", ""
                )
                key = (
                    row["element"], int(row["ion_stage"]),
                    round(wavelength, 6),
                )
                sources[key] = source
        return sources

    def _strength_factors(self, element, stage, wavelength, seed):
        factor = np.ones(len(wavelength), dtype=float)
        if not self.atomic_uncertainty.enabled:
            return factor
        source = np.asarray([
            self._strength_source.get(
                (element, int(stage), round(float(value), 6)), ""
            )
            for value in wavelength
        ], dtype=object)
        common_share = self.atomic_uncertainty.common_fraction
        independent_share = np.sqrt(max(0.0, 1.0 - common_share ** 2))
        for code in np.unique(source):
            sigma = float(self.atomic_uncertainty.sigma_dex.get(str(code), 0.0))
            if sigma <= 0:
                continue
            payload = f"{int(seed)}:{element}:{int(stage)}:{code}".encode("utf-8")
            local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
            rng = np.random.default_rng(local_seed)
            mask = source == code
            common = rng.normal(0.0, sigma * common_share)
            independent = rng.normal(
                0.0, sigma * independent_share, int(np.sum(mask))
            )
            factor[mask] = 10.0 ** (common + independent)
        return factor

    def _build_stage_support(self):
        support = np.zeros((N_ELEMENTS, N_STAGES), dtype=bool)
        for ei, element in enumerate(self.db.elements):
            if not self.support_mask[ei]:
                continue
            lines = np.asarray(self.db.lines(element))
            if lines.size == 0:
                continue
            data = lines[:, [0, 1, 3, 4, 5]].astype(float)
            for stage in range(1, N_STAGES + 1):
                mask = (
                    (data[:, 0] == stage)
                    & (data[:, 1] >= NATIVE_WL_LO_NM)
                    & (data[:, 1] <= NATIVE_WL_HI_NM)
                    & np.isfinite(data[:, 2])
                    & (data[:, 2] > 0)
                    & np.isfinite(data[:, 3])
                    & np.isfinite(data[:, 4])
                )
                support[ei, stage - 1] = bool(np.any(mask))
        support.setflags(write=False)
        return support

    def _validate_component(self, component: PlasmaComponent, name: str):
        total = float(component.stage_abundance.sum())
        if not np.isclose(total, 1.0, rtol=1e-8, atol=1e-12):
            raise ValueError(f"{name} stage abundances must sum to one, got {total:.12g}")
        unsupported_mass = component.stage_abundance[~self.support_mask].sum()
        if unsupported_mass > 0:
            bad = [
                self.db.elements[i]
                for i in np.flatnonzero(
                    component.stage_abundance.sum(axis=1) > 0
                )
                if not self.support_mask[i]
            ]
            raise ValueError(f"{name} assigns abundance to unsupported elements: {bad}")

    def _stage_partition(self, element_index: int, stage: int, temperature: float):
        key = (int(element_index), int(stage), float(temperature))
        cached = self._partition_cache.get(key)
        if cached is not None:
            return cached
        lines = np.asarray(self.db.lines(self.db.elements[element_index]))
        levels = _stage_levels(lines, stage)
        if levels.size == 0:
            partition = 0.0
        else:
            partition = float(np.sum(
                levels[:, 1] * np.exp(-levels[:, 0] / (BOLTZMANN * temperature))
            ))
        if not np.isfinite(partition) or partition <= 0:
            partition = 0.0
        self._partition_cache[key] = partition
        return partition

    def _ionization_energy(self, element_index: int, stage: int):
        key = (int(element_index), int(stage))
        if key in self._ion_energy_cache:
            return self._ion_energy_cache[key]
        rows = np.asarray(
            self.db.ionization_energy(self.db.elements[element_index], ion=stage),
            dtype=float,
        )
        value = None if rows.size == 0 else float(np.atleast_2d(rows)[0, -1])
        self._ion_energy_cache[key] = value
        return value

    def _line_widths(self, element_index, stage, wavelength, upper_energy,
                     temperature, log_ne):
        segment = self.instrument.segment_index(wavelength)
        inst_sigma = np.asarray(self.instrument.gaussian_sigma_nm)[segment]
        inst_gamma = np.asarray(self.instrument.lorentzian_gamma_nm)[segment]

        mass_kg = ATOMIC_MASS_U[element_index] * _AMU_KG
        sigma_doppler = (
            np.asarray(wavelength, dtype=float)
            * np.sqrt(_K_B_SI * float(temperature) / mass_kg)
            / _C_M_S
        )
        sigma = np.sqrt(inst_sigma ** 2 + sigma_doppler ** 2)

        z_atomic = element_index + 1
        if stage >= z_atomic:
            shape = np.zeros_like(wavelength, dtype=float)
        else:
            ion_energy = self._ionization_energy(element_index, stage)
            shape = (
                np.zeros_like(wavelength, dtype=float)
                if ion_energy is None
                else stark_shape_factor(ion_energy, upper_energy, stage)
            )
        gamma = inst_gamma + stark_hwhm(
            shape,
            log_ne,
            self.instrument.stark_c4_nm,
            self.instrument.stark_log_ne_ref,
        )

        # The repository has an absolute H-alpha linear-Stark calibration;
        # use it only for that line rather than applying the quadratic model
        # to hydrogenic stages.
        if element_index == 0 and stage == 1:
            near_halpha = np.abs(np.asarray(wavelength) - HALPHA_NM) <= 0.25
            fwhm = _HALPHA_FWHM_REF_NM * 10.0 ** (
                _HALPHA_EXPONENT * (float(log_ne) - 17.0)
            )
            gamma = np.where(near_halpha, inst_gamma + 0.5 * fwhm, gamma)
        return np.maximum(sigma, 1e-8), np.maximum(gamma, 0.0)

    def _component_lines(self, component: PlasmaComponent, atomic_seed=0):
        rows = []
        element_total = component.element_abundance
        for ei, element in enumerate(self.db.elements):
            if element_total[ei] <= 0 or not self.support_mask[ei]:
                continue
            raw = np.asarray(self.db.lines(element))
            if raw.size == 0:
                continue
            data = raw[:, [0, 1, 3, 4, 5]].astype(float)
            for stage in range(1, N_STAGES + 1):
                abundance = float(component.stage_abundance[ei, stage - 1])
                if abundance <= 0:
                    continue
                mask = (
                    (data[:, 0] == stage)
                    & np.isfinite(data[:, 1])
                    & np.isfinite(data[:, 2])
                    & (data[:, 2] > 0)
                    & np.isfinite(data[:, 3])
                    & np.isfinite(data[:, 4])
                )
                if not np.any(mask):
                    continue
                wl, gA, Ei, Ek = data[mask, 1:].T
                strength_factor = self._strength_factors(
                    element, stage, wl, atomic_seed
                )
                gA = gA * strength_factor
                temperature = float(component.temperature_k[ei, stage - 1])
                partition = self._stage_partition(ei, stage, temperature)
                if partition <= 0:
                    continue
                area = abundance * float(component.emission_scale) * line_emissivity(
                    wl, gA, Ek, temperature, partition, 1.0
                )
                stage_fraction = abundance / max(float(element_total[ei]), 1e-300)
                opacity = (
                    wl ** 2
                    * gA
                    * np.exp(-Ei / (BOLTZMANN * temperature))
                    / partition
                    * stage_fraction
                    * stimulated_emission_factor(wl, temperature)
                )
                tau = float(component.effective_column[ei]) * opacity
                area = area * escape_factor(tau)
                sigma, gamma = self._line_widths(
                    ei,
                    stage,
                    wl,
                    Ek,
                    temperature,
                    component.log_ne_cm3[ei],
                )
                valid = (
                    np.isfinite(area) & (area > 0) & np.isfinite(sigma)
                    & np.isfinite(gamma)
                )
                if not np.any(valid):
                    continue
                for j in np.flatnonzero(valid):
                    rows.append((
                        float(wl[j]), float(area[j]), float(sigma[j]),
                        float(gamma[j]), ei, stage, float(tau[j]),
                        float(strength_factor[j]),
                    ))
        if not rows:
            return {
                "wavelength_nm": np.empty(0), "area_counts": np.empty(0),
                "sigma_nm": np.empty(0), "gamma_nm": np.empty(0),
                "element_index": np.empty(0, dtype=int),
                "ion_stage": np.empty(0, dtype=int), "tau": np.empty(0),
                "strength_factor": np.empty(0),
            }
        rows = np.asarray(rows, dtype=float)
        order = np.argsort(rows[:, 0])
        rows = rows[order]
        return {
            "wavelength_nm": rows[:, 0],
            "area_counts": rows[:, 1],
            "sigma_nm": rows[:, 2],
            "gamma_nm": rows[:, 3],
            "element_index": rows[:, 4].astype(int),
            "ion_stage": rows[:, 5].astype(int),
            "tau": rows[:, 6],
            "strength_factor": rows[:, 7],
        }

    def _profile_radius(self, sigma, gamma):
        tol = float(self.instrument.tail_mass_tolerance)
        r_gauss = np.sqrt(2.0) * sigma * float(erfcinv(tol))
        r_lorentz = 0.0 if gamma <= 0 else gamma / np.tan(np.pi * tol / 2.0)
        return max(float(r_gauss), float(r_lorentz), 2.0 * NATIVE_PITCH_NM)

    def _integrated_profile(self, center, sigma, gamma, left, right):
        """Gauss-Legendre integration, subdividing cells wider than a line."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        out = np.zeros(left.size, dtype=float)
        fwhm = max(float(voigt_width(sigma, gamma)), 1e-6)
        max_sub_width = max(0.25 * fwhm, 1e-4)
        subdivisions = np.maximum(
            1, np.ceil((right - left) / max_sub_width).astype(int)
        )
        # Group equal subdivision counts.  Native current-instrument cells
        # normally all use n_sub=1, turning thousands of Python-level bin
        # integrations into one vectorized quadrature operation per line.
        for n_sub in np.unique(subdivisions):
            use = subdivisions == n_sub
            widths = (right[use] - left[use]) / int(n_sub)
            sub_index = np.arange(int(n_sub), dtype=float) + 0.5
            mids = left[use, None] + widths[:, None] * sub_index[None, :]
            halves = 0.5 * widths
            nodes = (
                mids[:, :, None]
                + halves[:, None, None] * self._quad_nodes[None, None, :]
            )
            vals = voigt_profile(nodes - center, sigma, gamma)
            out[use] = np.sum(
                halves[:, None, None]
                * self._quad_weights[None, None, :]
                * vals,
                axis=(1, 2),
            )
        return out

    def _render_lines(self, table, grid: ChannelGrid):
        out = np.zeros(grid.centers_nm.size, dtype=float)
        left = grid.left_edges_nm
        right = grid.right_edges_nm
        for center, area, sigma, gamma in zip(
            table["wavelength_nm"], table["area_counts"],
            table["sigma_nm"], table["gamma_nm"],
        ):
            radius = self._profile_radius(sigma, gamma)
            lo = int(np.searchsorted(right, center - radius, side="right"))
            hi = int(np.searchsorted(left, center + radius, side="left"))
            if hi <= lo:
                continue
            out[lo:hi] += area * self._integrated_profile(
                center, sigma, gamma, left[lo:hi], right[lo:hi]
            )
        return out

    @staticmethod
    def _free_free_continuum(component: PlasmaComponent, grid: ChannelGrid):
        """Positive Kramers-shaped continuum in calibrated nuisance units."""
        if component.continuum_scale <= 0:
            return np.zeros(grid.centers_nm.size, dtype=float)
        wl = grid.centers_nm
        density = np.zeros_like(wl, dtype=float)
        for ei in np.flatnonzero(component.element_abundance > 0):
            for si in range(N_STAGES):
                abundance = float(component.stage_abundance[ei, si])
                charge = float(si)  # I=neutral, II=+1, III=+2
                if abundance <= 0 or charge <= 0:
                    continue
                temperature = float(component.temperature_k[ei, si])
                weight = (
                    abundance * charge ** 2
                    * 10.0 ** (float(component.log_ne_cm3[ei]) - 17.0)
                    * np.sqrt(10_000.0 / temperature)
                )
                density += weight * (500.0 / wl) ** 2 * np.exp(
                    -_C2_NM_K / (wl * temperature)
                )
        return float(component.continuum_scale) * density * grid.widths_nm

    def _render_component(self, component: PlasmaComponent, grid: ChannelGrid,
                          atomic_seed=0):
        table = self._component_lines(component, atomic_seed=atomic_seed)
        line_counts = self._render_lines(table, grid)
        continuum = self._free_free_continuum(component, grid)
        physical = np.maximum(line_counts + continuum, 0.0)
        return physical, table

    def _export_blocks(self, grid: ChannelGrid):
        segment = self.instrument.segment_index(grid.centers_nm)
        gap = np.zeros(grid.centers_nm.size, dtype=bool)
        gap[0] = True
        if grid.centers_nm.size > 1:
            nominal = np.maximum(grid.widths_nm[:-1], grid.widths_nm[1:])
            gap[1:] = (
                (grid.left_edges_nm[1:] - grid.right_edges_nm[:-1])
                > 1.5 * nominal
            ) | (segment[1:] != segment[:-1])
        starts = np.flatnonzero(gap)
        stops = np.r_[starts[1:], grid.centers_nm.size]
        return segment, list(zip(starts, stops))

    def _apply_export_kernel(self, values, blocks):
        kernel = np.asarray(self.instrument.export_kernel, dtype=float)
        if kernel.size == 1:
            return np.asarray(values, dtype=float).copy()
        out = np.asarray(values, dtype=float).copy()
        half = kernel.size // 2
        for start, stop in blocks:
            block = np.asarray(values[start:stop], dtype=float)
            padded = np.pad(block, (half, half), mode="edge")
            out[start:stop] = np.convolve(padded, kernel, mode="valid")
        return out

    def _observe(self, physical_counts, grid, seed, add_noise):
        segment, blocks = self._export_blocks(grid)
        gain = np.asarray(self.instrument.segment_gain)[segment]
        cell_scale = grid.widths_nm / NATIVE_PITCH_NM
        background = (
            np.asarray(self.instrument.background_counts)[segment] * cell_scale
        )
        offset = np.asarray(self.instrument.export_offset_counts)[segment]
        pre_export = np.maximum(physical_counts * gain + background, 0.0)

        seed_words = [int(seed), int(grid.digest[:8], 16), int(grid.digest[8:16], 16)]
        rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        observed = pre_export.copy()
        if add_noise and self.instrument.shot_noise:
            lam = np.clip(pre_export * self.instrument.electron_gain, 0.0, 1e12)
            observed = rng.poisson(lam).astype(float) / self.instrument.electron_gain
        if add_noise:
            read = (
                np.asarray(self.instrument.read_noise_std_counts)[segment]
                * np.sqrt(cell_scale)
            )
            observed += rng.normal(0.0, read, observed.size)
            dark_std = np.asarray(self.instrument.dark_offset_std_counts)
            dark_draw = rng.normal(0.0, dark_std, 3)
            observed -= dark_draw[segment]
        observed += offset
        expected = pre_export + offset
        return (
            self._apply_export_kernel(expected, blocks),
            self._apply_export_kernel(observed, blocks),
        )

    def render(
        self,
        scene: SyntheticScene,
        grid: Optional[ChannelGrid] = None,
        add_noise: bool = True,
        return_line_table: bool = False,
    ) -> SyntheticSpectrum:
        """Render one replayable individual-shot spectrum."""
        grid = ChannelGrid.native() if grid is None else grid
        self._validate_component(scene.target, "target")
        if scene.ambient_gas is not None:
            self._validate_component(scene.ambient_gas, "ambient_gas")

        target, target_lines = self._render_component(
            scene.target, grid, atomic_seed=scene.seed
        )
        if scene.ambient_gas is None:
            ambient = np.zeros_like(target)
            ambient_lines = self._component_lines(
                PlasmaComponent.from_mapping({("Ar", 1): 1.0}, emission_scale=0.0),
                atomic_seed=scene.seed,
            )
        else:
            ambient, ambient_lines = self._render_component(
                scene.ambient_gas, grid, atomic_seed=scene.seed
            )
        physical = np.maximum(target + ambient, 0.0)
        expected, observed = self._observe(
            physical, grid, scene.seed, bool(add_noise)
        )

        table = None
        if return_line_table:
            table = {
                "target_" + key: value.copy()
                for key, value in target_lines.items()
            }
            table.update({
                "ambient_" + key: value.copy()
                for key, value in ambient_lines.items()
            })

        db_sources = self.db.dbpath / "atomic_line_sources.json"
        source_hash = None
        if db_sources.exists():
            source_hash = hashlib.sha256(db_sources.read_bytes()).hexdigest()
        manifest = {
            "generator_version": GENERATOR_VERSION,
            "scene_digest": scene.digest,
            "scene_seed": scene.seed,
            "grid_digest": grid.digest,
            "n_channels": int(grid.centers_nm.size),
            "atomic_source_manifest_sha256": source_hash,
            "saha_ionization": False,
            "stages": [1, 2, 3],
            "target_abundance_sum": float(scene.target.stage_abundance.sum()),
            "ambient_excluded_from_target": scene.ambient_gas is not None,
            "individual_shot": True,
            "instrument": self.instrument.manifest(),
            "atomic_strength_uncertainty": self.atomic_uncertainty.manifest(),
            "warnings": [] if self.instrument.calibrated else [
                "instrument profile/export kernel is provisional and not corpus-calibrated"
            ],
        }
        return SyntheticSpectrum(
            wavelength_nm=grid.centers_nm.copy(),
            intensity_counts=observed,
            expected_export_counts=expected,
            physical_channel_counts=physical,
            target_channel_counts=target,
            ambient_channel_counts=ambient,
            stage_observable=self.stage_support.copy(),
            manifest=manifest,
            line_table=table,
        )


class HierarchicalPlasmaSampler:
    """Provisional non-Saha sampler for per-element nuisance parameters.

    A shared log-space centre provides cross-element shrinkage, while
    Student-t deviations allow genuine element-specific plasma conditions.
    The centre is a statistical pooling reference, not a global physical
    electron density or column density.
    """

    def __init__(
        self,
        temperature_range_k=(5_000.0, 20_000.0),
        log_ne_center_range=(16.0, 18.0),
        log_ne_deviation_scale=0.25,
        log_column_per_fraction_center=-13.0,
        log_column_deviation_scale=0.5,
    ):
        self.temperature_range_k = tuple(float(x) for x in temperature_range_k)
        self.log_ne_center_range = tuple(float(x) for x in log_ne_center_range)
        self.log_ne_deviation_scale = float(log_ne_deviation_scale)
        self.log_column_per_fraction_center = float(log_column_per_fraction_center)
        self.log_column_deviation_scale = float(log_column_deviation_scale)

    def component(
        self,
        stage_abundance,
        seed,
        emission_scale=1.0e-3,
        continuum_scale=0.0,
        label="target",
    ):
        abundance = _readonly_float_array(
            stage_abundance, (N_ELEMENTS, N_STAGES), "stage_abundance"
        )
        rng = np.random.default_rng(int(seed))
        active_stage = abundance > 0
        temperature = np.full((N_ELEMENTS, N_STAGES), 10_000.0)
        lo_t, hi_t = self.temperature_range_k
        temperature[active_stage] = np.exp(rng.uniform(
            np.log(lo_t), np.log(hi_t), int(active_stage.sum())
        ))

        active_element = abundance.sum(axis=1) > 0
        lo_ne, hi_ne = self.log_ne_center_range
        centre_ne = rng.uniform(lo_ne, hi_ne)
        log_ne = np.full(N_ELEMENTS, centre_ne)
        log_ne[active_element] = (
            centre_ne
            + self.log_ne_deviation_scale
            * rng.standard_t(df=4, size=int(active_element.sum()))
        )

        element_fraction = abundance.sum(axis=1)
        column = np.zeros(N_ELEMENTS)
        deviations = self.log_column_deviation_scale * rng.standard_t(
            df=4, size=int(active_element.sum())
        )
        column[active_element] = element_fraction[active_element] * 10.0 ** (
            self.log_column_per_fraction_center + deviations
        )
        return PlasmaComponent(
            abundance,
            temperature_k=temperature,
            log_ne_cm3=log_ne,
            effective_column=column,
            emission_scale=emission_scale,
            continuum_scale=continuum_scale,
            label=label,
        )


class WholeRockSceneSampler:
    """Replayable natural-rock scene sampler backed by a composition prior.

    The whole-rock nuclei fractions are used as a provisional *shape prior*
    for emitting-plasma nuclei.  No calibrated bulk-to-plasma transfer model
    is implied.  Ion stages remain independent direct draws.
    """

    def __init__(
        self,
        composition_model,
        plasma_sampler: Optional[HierarchicalPlasmaSampler] = None,
        gas_probability: float = 0.8,
        stratum_policy: str = "balanced",
        stage_alpha=(1.0, 1.0, 1.0),
        emission_log10_range=(-2.0, -0.3),
        continuum_log10_range=(1.3, 2.7),
    ):
        if not 0.0 <= float(gas_probability) <= 1.0:
            raise ValueError("gas_probability must lie in [0, 1]")
        if stratum_policy not in ("balanced", "corpus"):
            raise ValueError("stratum_policy must be 'balanced' or 'corpus'")
        alpha = np.asarray(stage_alpha, dtype=float)
        if alpha.shape != (N_STAGES,) or np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError("stage_alpha must contain three positive values")
        self.composition_model = composition_model
        self.plasma_sampler = plasma_sampler or HierarchicalPlasmaSampler()
        self.gas_probability = float(gas_probability)
        self.stratum_policy = stratum_policy
        self.stage_alpha = tuple(alpha.tolist())
        self.emission_log10_range = tuple(float(x) for x in emission_log10_range)
        self.continuum_log10_range = tuple(float(x) for x in continuum_log10_range)

    def scene(self, seed, *, stratum=None, argon_purge_fraction=None):
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        rng = np.random.default_rng(seed)
        composition, abundance = self.composition_model.sample_stage_abundance(
            seed ^ 0xE7037ED1A0B428DB,
            stratum=stratum,
            stratum_policy=self.stratum_policy,
            stage_alpha=self.stage_alpha,
            min_nuclei_fraction=1.0e-10,
        )
        target = self.plasma_sampler.component(
            abundance,
            seed=seed ^ 0x8EBC6AF09C88C6E3,
            emission_scale=10.0 ** rng.uniform(*self.emission_log10_range),
            continuum_scale=10.0 ** rng.uniform(*self.continuum_log10_range),
        )

        if argon_purge_fraction is None:
            use_gas = rng.random() < self.gas_probability
            argon_fraction = float(rng.uniform(0.0, 1.0)) if use_gas else None
        else:
            argon_fraction = float(argon_purge_fraction)
            if not 0.0 <= argon_fraction <= 1.0:
                raise ValueError("argon_purge_fraction must lie in [0, 1]")
        gas = None
        if argon_fraction is not None:
            gas_splits = {
                element: rng.dirichlet(np.array([1.5, 1.2, 0.3]))
                for element in ("N", "O", "Ar")
            }
            base_gas = dry_air_component(
                argon_fraction,
                gas_splits,
                emission_scale=10.0 ** rng.uniform(-2.5, -0.5),
                continuum_scale=10.0 ** rng.uniform(0.5, 2.0),
            )
            gas = self.plasma_sampler.component(
                base_gas.stage_abundance,
                seed=seed ^ 0x589965CC75374CC3,
                emission_scale=base_gas.emission_scale,
                continuum_scale=base_gas.continuum_scale,
                label=base_gas.label,
            )

        return SyntheticScene(
            target=target,
            ambient_gas=gas,
            seed=seed,
            metadata={
                "whole_rock_stratum": composition.stratum,
                "whole_rock_source_doi": composition.source_doi,
                "whole_rock_policy": self.stratum_policy,
                "whole_rock_mass_fraction": composition.mass_fraction.tolist(),
                "composition_prior_nuclei_fraction": (
                    composition.nuclei_fraction.tolist()
                ),
                "bulk_to_plasma_transfer": "identity-shape-proxy-not-calibrated",
                "argon_purge_fraction": argon_fraction,
            },
        )


@dataclass(frozen=True)
class CoverageCell:
    """One required element-stage-abundance stratum in the training design."""

    element_index: int
    ion_stage: int
    abundance: float
    supported: bool
    training_enabled: bool
    quantitatively_observable: bool

    @property
    def element(self):
        return ELEMENTS_BY_ATOMIC_NUMBER[self.element_index]


class PeriodicCoverageScheduler:
    """Deterministic periodic-table and abundance-decade scene scheduler.

    This is not a random Dirichlet shortcut: every H--U position and each stage
    I--III receives an explicit cell at zero and every decade from 1e-8 through
    1. Cells with no quantitative lines are still synthesized as target states
    so the downstream evidence head learns structural unobservability. The five
    unsupported positions and the separate current-round training exclusions
    remain in the manifest but cannot produce scenes.
    """

    DEFAULT_ABUNDANCE_LEVELS = (
        0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
    )

    def __init__(
        self,
        generator: SyntheticSpectrumGenerator,
        plasma_sampler: Optional[HierarchicalPlasmaSampler] = None,
        abundance_levels: Sequence[float] = DEFAULT_ABUNDANCE_LEVELS,
        gas_probability: float = 0.8,
        excluded_elements: Sequence[str] = ("Tc", "Fr", "Ra", "Ac"),
        whole_rock_model=None,
        whole_rock_policy: str = "balanced",
    ):
        self.generator = generator
        self.plasma_sampler = plasma_sampler or HierarchicalPlasmaSampler()
        levels = tuple(float(x) for x in abundance_levels)
        if (not levels or any(not np.isfinite(x) or x < 0 or x > 1 for x in levels)
                or 0.0 not in levels or 1.0 not in levels):
            raise ValueError("abundance_levels must include zero and one within [0, 1]")
        self.abundance_levels = tuple(sorted(set(levels)))
        self.gas_probability = float(gas_probability)
        if not 0.0 <= self.gas_probability <= 1.0:
            raise ValueError("gas_probability must lie in [0, 1]")
        excluded = tuple(dict.fromkeys(str(el) for el in excluded_elements))
        unknown = [el for el in excluded if el not in ATOMIC_NUMBER]
        if unknown:
            raise ValueError(f"unknown excluded elements: {unknown}")
        self.excluded_elements = excluded
        self.training_mask = self.generator.support_mask.copy()
        for element in excluded:
            self.training_mask[ATOMIC_NUMBER[element] - 1] = False
        if whole_rock_policy not in ("balanced", "corpus"):
            raise ValueError("whole_rock_policy must be 'balanced' or 'corpus'")
        self.whole_rock_model = whole_rock_model
        self.whole_rock_policy = whole_rock_policy

    def cells(self, include_unsupported=True):
        for ei in range(N_ELEMENTS):
            supported = bool(self.generator.support_mask[ei])
            training_enabled = bool(self.training_mask[ei])
            if not include_unsupported and not supported:
                continue
            for stage in range(1, N_STAGES + 1):
                observable = bool(self.generator.stage_support[ei, stage - 1])
                for abundance in self.abundance_levels:
                    yield CoverageCell(
                        ei, stage, abundance, supported, training_enabled,
                        observable,
                    )

    @staticmethod
    def _cell_seed(cell: CoverageCell, replicate: int, seed: int):
        payload = (
            f"{cell.element_index}:{cell.ion_stage}:{cell.abundance:.17g}:"
            f"{int(replicate)}:{int(seed)}"
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def scene(self, cell: CoverageCell, replicate=0, seed=0):
        if not cell.supported:
            raise ValueError(
                f"{cell.element} is an explicit unsupported schema position"
            )
        if not cell.training_enabled:
            raise ValueError(
                f"{cell.element} is retained in the schema but excluded from training"
            )
        local_seed = self._cell_seed(cell, replicate, seed)
        rng = np.random.default_rng(local_seed)
        abundance = np.zeros((N_ELEMENTS, N_STAGES), dtype=float)
        abundance[cell.element_index, cell.ion_stage - 1] = cell.abundance

        remaining = 1.0 - cell.abundance
        whole_rock_stratum = None
        whole_rock_source_doi = None
        if remaining > 0:
            if self.whole_rock_model is None:
                candidates = np.argwhere(self.generator.stage_support).astype(int)
                candidates = candidates[self.training_mask[candidates[:, 0]]]
                # Keep the focus element's total abundance exactly equal to the
                # scheduled stage abundance; backgrounds use other elements.
                candidates = candidates[candidates[:, 0] != cell.element_index]
                if candidates.size == 0:  # pragma: no cover - database invariant
                    raise RuntimeError("no observable background stages available")
                n_background = int(rng.integers(1, min(7, len(candidates) + 1)))
                chosen = rng.choice(len(candidates), n_background, replace=False)
                weights = rng.dirichlet(np.ones(n_background)) * remaining
                for (ei, si), value in zip(candidates[chosen], weights):
                    abundance[int(ei), int(si)] += float(value)
            else:
                composition, background = self.whole_rock_model.sample_stage_abundance(
                    local_seed ^ 0xA0761D6478BD642F,
                    stratum_policy=self.whole_rock_policy,
                    min_nuclei_fraction=1.0e-10,
                )
                background = np.asarray(background, dtype=float).copy()
                if background.shape != (N_ELEMENTS, N_STAGES):
                    raise ValueError(
                        "whole-rock model must return a 92-by-3 stage abundance"
                    )
                background[~self.training_mask] = 0.0
                background[cell.element_index] = 0.0
                total = float(background.sum())
                if not np.isfinite(total) or total <= 0:
                    raise RuntimeError("whole-rock model produced no eligible background")
                abundance += remaining * background / total
                whole_rock_stratum = composition.stratum
                whole_rock_source_doi = composition.source_doi

        emission_scale = 10.0 ** rng.uniform(-2.0, -0.3)
        continuum_scale = 10.0 ** rng.uniform(1.3, 2.7)
        target = self.plasma_sampler.component(
            abundance,
            seed=local_seed ^ 0x9E3779B97F4A7C15,
            emission_scale=emission_scale,
            continuum_scale=continuum_scale,
        )

        gas = None
        argon_fraction = None
        if rng.random() < self.gas_probability:
            argon_fraction = float(rng.uniform(0.0, 1.0))
            gas_splits = {
                element: rng.dirichlet(np.array([1.5, 1.2, 0.3]))
                for element in ("N", "O", "Ar")
            }
            base_gas = dry_air_component(
                argon_fraction,
                gas_splits,
                emission_scale=10.0 ** rng.uniform(-2.5, -0.5),
                continuum_scale=10.0 ** rng.uniform(0.5, 2.0),
            )
            gas = self.plasma_sampler.component(
                base_gas.stage_abundance,
                seed=local_seed ^ 0xD1B54A32D192ED03,
                emission_scale=base_gas.emission_scale,
                continuum_scale=base_gas.continuum_scale,
                label=base_gas.label,
            )

        return SyntheticScene(
            target=target,
            ambient_gas=gas,
            seed=local_seed & 0xFFFFFFFF,
            metadata={
                "coverage_element": cell.element,
                "coverage_stage": cell.ion_stage,
                "coverage_abundance": cell.abundance,
                "coverage_quantitatively_observable": cell.quantitatively_observable,
                "coverage_replicate": int(replicate),
                "argon_purge_fraction": argon_fraction,
                "whole_rock_stratum": whole_rock_stratum,
                "whole_rock_source_doi": whole_rock_source_doi,
            },
        )

    def manifest(self):
        return {
            "schema": "alibz-periodic-coverage-v1",
            "elements": list(ELEMENTS_BY_ATOMIC_NUMBER),
            "support_mask": self.generator.support_mask.astype(int).tolist(),
            "training_mask": self.training_mask.astype(int).tolist(),
            "training_excluded_elements": list(self.excluded_elements),
            "stage_observable": self.generator.stage_support.astype(int).tolist(),
            "abundance_levels": list(self.abundance_levels),
            "gas_probability": self.gas_probability,
            "whole_rock_prior": self.whole_rock_model is not None,
            "whole_rock_policy": (
                self.whole_rock_policy if self.whole_rock_model is not None else None
            ),
            "whole_rock_source_doi": (
                self.whole_rock_model.source_doi
                if self.whole_rock_model is not None else None
            ),
            "whole_rock_training_strata": (
                list(self.whole_rock_model.training_strata)
                if self.whole_rock_model is not None else None
            ),
            "cells_total": N_ELEMENTS * N_STAGES * len(self.abundance_levels),
            "cells_supported": int(self.generator.support_mask.sum())
            * N_STAGES * len(self.abundance_levels),
            "cells_training_enabled": int(self.training_mask.sum())
            * N_STAGES * len(self.abundance_levels),
        }


__all__ = [
    "ATOMIC_MASS_U",
    "AtomicStrengthUncertainty",
    "ChannelGrid",
    "CoverageCell",
    "GENERATOR_VERSION",
    "HierarchicalPlasmaSampler",
    "InstrumentResponse",
    "N_ELEMENTS",
    "N_STAGES",
    "PeriodicCoverageScheduler",
    "PlasmaComponent",
    "SyntheticScene",
    "SyntheticSpectrum",
    "SyntheticSpectrumGenerator",
    "WholeRockSceneSampler",
    "dry_air_component",
]
