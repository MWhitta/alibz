"""Corpus calibration helpers for the explicit-stage synthetic renderer.

The calibration here is intentionally limited to quantities identifiable from
individual-shot two-column exports: grid, segment baseline/count distributions,
negative fractions, and local noise.  It records the ``peaky_data`` segment
width modes as provisional line-profile anchors, but does not claim that the
conditional PCA score distribution or vendor export kernel has been fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from alibz.peaky_finder import PeakyFinder
from alibz.synthetic import InstrumentResponse


SEGMENT_LABELS = ("UV", "VIS", "NIR")
PEAKY_DATA_FWHM_MODES_NM = (0.1682, 0.3426, 0.3615)
PEAKY_DATA_GAUSSIAN_FRACTION = 0.46


def _load_csv(path):
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    except ValueError:
        rows = []
        with Path(path).open() as handle:
            for line in handle:
                fields = line.strip().split(",")
                if len(fields) < 2:
                    continue
                try:
                    rows.append((float(fields[0]), float(fields[1])))
                except ValueError:
                    continue
        data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"{path!s} is not a two-column spectrum")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def _width_components(fwhm_modes, gaussian_fraction):
    g = float(gaussian_fraction)
    if not 0.0 < g < 1.0:
        raise ValueError("gaussian_fraction must lie between zero and one")
    l = 1.0 - g
    factor = 0.5346 * l + np.sqrt(0.2166 * l ** 2 + g ** 2)
    fwhm = np.asarray(fwhm_modes, dtype=float)
    sigma = (g * fwhm / factor) / 2.354820045
    gamma = (l * fwhm / factor) / 2.0
    return sigma, gamma


@dataclass(frozen=True)
class IndividualShotCalibration:
    response: InstrumentResponse
    n_spectra: int
    file_manifest_sha256: str
    wavelength: Mapping[str, object]
    segments: Mapping[str, Mapping[str, object]]
    limitations: Sequence[str]

    def as_dict(self):
        return {
            "schema": "alibz-individual-shot-instrument-v1",
            "n_spectra": int(self.n_spectra),
            "file_manifest_sha256": self.file_manifest_sha256,
            "wavelength": dict(self.wavelength),
            "segments": {key: dict(value) for key, value in self.segments.items()},
            "instrument_response": self.response.manifest(),
            "limitations": list(self.limitations),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        if values.get("schema") != "alibz-individual-shot-instrument-v1":
            raise ValueError("unsupported individual-shot calibration schema")
        return cls(
            response=InstrumentResponse.from_manifest(values["instrument_response"]),
            n_spectra=int(values["n_spectra"]),
            file_manifest_sha256=values["file_manifest_sha256"],
            wavelength=values["wavelength"],
            segments=values["segments"],
            limitations=tuple(values["limitations"]),
        )


def calibrate_individual_shots(
    paths: Iterable,
    segment_edges_nm=(365.0, 620.0),
    active_range_nm=(190.0, 910.0),
    profile_fwhm_modes_nm=PEAKY_DATA_FWHM_MODES_NM,
    gaussian_fraction=PEAKY_DATA_GAUSSIAN_FRACTION,
):
    """Estimate provisional observation parameters from individual shots."""
    paths = sorted(Path(path) for path in paths)
    if not paths:
        raise ValueError("at least one individual-shot spectrum is required")
    digest = hashlib.sha256()
    per_segment = [[] for _ in SEGMENT_LABELS]
    wavelength_rows = []

    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        wavelength, intensity = _load_csv(path)
        native_pitch = float(np.median(np.diff(wavelength)))
        edge_tol = 0.5 * native_pitch
        active = (
            (wavelength >= float(active_range_nm[0]) - edge_tol)
            & (wavelength < float(active_range_nm[1]) - edge_tol)
        )
        x = wavelength[active]
        y = intensity[active]
        if x.size < 2:
            raise ValueError(f"{path!s} contains no active wavelength samples")
        wavelength_rows.append((
            x.size, float(x[0]), float(x[-1]), float(np.median(np.diff(x))),
            float(np.std(np.diff(x))),
        ))
        segment_index = np.searchsorted(
            np.asarray(segment_edges_nm, dtype=float), x + 1e-9
        )
        for si in range(3):
            mask = segment_index == si
            values = y[mask]
            if values.size < 2:
                raise ValueError(f"{path!s} has no samples in {SEGMENT_LABELS[si]}")
            neg = np.flatnonzero(values < 0)
            longest = 0
            if neg.size:
                split = np.flatnonzero(np.diff(neg) > 1) + 1
                longest = max(len(run) for run in np.split(neg, split))
            per_segment[si].append((
                float(np.median(values)),
                float(PeakyFinder._noise_scale(values)),
                float(np.mean(values < 0)),
                float(np.quantile(values, 0.01)),
                float(np.quantile(values, 0.99)),
                float(np.max(values)),
                int(longest),
            ))

    wavelength_rows = np.asarray(wavelength_rows, dtype=float)
    wavelength_summary = {
        "active_rows_median": int(round(np.median(wavelength_rows[:, 0]))),
        "lo_nm_median": float(np.median(wavelength_rows[:, 1])),
        "hi_nm_median": float(np.median(wavelength_rows[:, 2])),
        "pitch_nm_median": float(np.median(wavelength_rows[:, 3])),
        "pitch_nm_std_max": float(np.max(wavelength_rows[:, 4])),
    }

    segment_summary = {}
    backgrounds = []
    total_noise = []
    for label, rows in zip(SEGMENT_LABELS, per_segment):
        rows = np.asarray(rows, dtype=float)
        background = float(np.median(rows[:, 0]))
        noise = float(np.median(rows[:, 1]))
        backgrounds.append(max(background, 0.0))
        total_noise.append(max(noise, 0.0))
        segment_summary[label] = {
            "baseline_count_median": background,
            "baseline_count_p05_p95": np.quantile(rows[:, 0], [0.05, 0.95]).tolist(),
            "local_noise_median": noise,
            "local_noise_p05_p95": np.quantile(rows[:, 1], [0.05, 0.95]).tolist(),
            "negative_fraction_median": float(np.median(rows[:, 2])),
            "negative_fraction_p95": float(np.quantile(rows[:, 2], 0.95)),
            "q01_count_median": float(np.median(rows[:, 3])),
            "q99_count_median": float(np.median(rows[:, 4])),
            "maximum_count_median": float(np.median(rows[:, 5])),
            "longest_negative_run_channels_p95": float(np.quantile(rows[:, 6], 0.95)),
        }

    # Treat the median baseline as a photon-like count only provisionally.
    # Subtract its Poisson variance from total local variance; any remainder is
    # assigned to read/export noise.  The limitations below explicitly prevent
    # this from being mistaken for a dark-frame calibration.
    read_noise = np.sqrt(np.maximum(
        np.asarray(total_noise) ** 2 - np.asarray(backgrounds), 0.0
    ))
    sigma, gamma = _width_components(profile_fwhm_modes_nm, gaussian_fraction)
    response = InstrumentResponse(
        segment_edges_nm=tuple(segment_edges_nm),
        gaussian_sigma_nm=tuple(sigma.tolist()),
        lorentzian_gamma_nm=tuple(gamma.tolist()),
        background_counts=tuple(backgrounds),
        read_noise_std_counts=tuple(read_noise.tolist()),
        shot_noise=True,
        profile_artifact=(
            "peaky_data-width-modes+individual-shot-summary:"
            + digest.hexdigest()[:16]
        ),
        calibrated=False,
    )
    return IndividualShotCalibration(
        response=response,
        n_spectra=len(paths),
        file_manifest_sha256=digest.hexdigest(),
        wavelength=wavelength_summary,
        segments=segment_summary,
        limitations=(
            "No dark/blank frames: electronic dark and plasma continuum are not separated.",
            "The vendor export/ringing kernel remains the identity.",
            "Peak widths use peaky_data summary modes, not the pending quality-gated conditional PCA refit.",
            "Count/noise statistics are valid for individual-shot spectra only.",
        ),
    )


__all__ = [
    "IndividualShotCalibration",
    "PEAKY_DATA_FWHM_MODES_NM",
    "PEAKY_DATA_GAUSSIAN_FRACTION",
    "calibrate_individual_shots",
]
