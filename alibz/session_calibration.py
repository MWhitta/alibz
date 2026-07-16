"""Data-internal detector response and wavelength-shift calibration.

Single-pulse rock spectra sometimes lack enough continuum or anchor lines to
estimate an instrument correction independently.  This module borrows only
instrument characteristics from nearby shots: it pools response steps and
atomic-line wavelength offsets over a short acquisition window, while the
full pipeline still prefers a trustworthy estimate from the shot itself.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from alibz.detector import estimate_segment_response
from alibz.peaky_finder import PeakyFinder
from alibz.utils.database import Database
from alibz.utils.wavelength import _anchor_catalog, _match_deltas


SEGMENT_EDGES_NM = (365.0, 620.0)
_CALIBRATION_CACHE = {}


def _database_anchors(dbpath: str) -> np.ndarray:
    anchors = _CALIBRATION_CACHE.get(dbpath)
    if anchors is None:
        anchors = _anchor_catalog(Database(dbpath))
        _CALIBRATION_CACHE[dbpath] = anchors
    return anchors


def _quick_peak_table(x: np.ndarray, y_bgsub: np.ndarray) -> np.ndarray:
    segment_indices = np.searchsorted(x, SEGMENT_EDGES_NM)
    noise = PeakyFinder._noise_scale_local(
        y_bgsub, segment_indices=segment_indices)
    indices, _ = find_peaks(
        y_bgsub, prominence=(4.0 * np.maximum(noise, 1e-12), None))
    if indices.size == 0:
        return np.empty((0, 4), dtype=float)
    pitch = float(np.median(np.diff(x))) if x.size > 1 else 0.0
    centers = x[indices].astype(float).copy()
    valid = (indices > 0) & (indices < y_bgsub.size - 1)
    for out_i, idx in zip(np.flatnonzero(valid), indices[valid]):
        left, middle, right = y_bgsub[idx - 1:idx + 2]
        denom = left - 2.0 * middle + right
        if abs(denom) > 1e-12:
            delta = 0.5 * (left - right) / denom
            centers[out_i] += float(np.clip(delta, -1.0, 1.0)) * pitch
    table = np.zeros((indices.size, 4), dtype=float)
    table[:, 0] = np.maximum(y_bgsub[indices], 0.0)
    table[:, 1] = centers
    for out_i, idx in enumerate(indices):
        half = 0.5 * y_bgsub[idx]
        left = idx
        while left > max(0, idx - 12) and y_bgsub[left] > half:
            left -= 1
        right = idx
        while right < min(y_bgsub.size - 1, idx + 12) \
                and y_bgsub[right] > half:
            right += 1
        fwhm = float(x[right] - x[left]) if right > left else np.nan
        if 0.04 <= fwhm <= 0.6:
            table[out_i, 2] = fwhm / 2.354820045
    return table


def measure_session_characteristics(path: str, dbpath: str) -> dict:
    """Measure correction evidence without running composition inference."""
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    x, y = values[:, 0], values[:, 1]
    if not np.any(y):
        return {
            "response_source": "invalid", "response_ratio": None,
            "response_uncertainty": None,
            "shift_deltas_nm": [[], [], []],
            "peak_fwhm_nm": [[], [], []], "n_quick_peaks": 0,
        }
    finder = PeakyFinder.__new__(PeakyFinder)
    background = finder.find_background(x, y)
    residual = y - background
    noise = 1.4826 * float(np.median(
        np.abs(residual - np.median(residual))))
    _, response_meta = estimate_segment_response(
        x, background, edges=(620.0,), noise_scale=noise,
        fallback=None, return_metadata=True)
    response = response_meta[0]

    peaks = _quick_peak_table(x, residual)
    anchors = _database_anchors(dbpath)
    if peaks.size:
        deltas, where = _match_deltas(
            peaks, anchors, tolerance_nm=0.15, n_peaks=100,
            ambiguity_ratio=3.0)
    else:
        deltas, where = np.empty(0), np.empty(0)
    segment = np.digitize(where, SEGMENT_EDGES_NM)
    shift_deltas = [deltas[segment == i].astype(float).tolist()
                    for i in range(3)]
    peak_segment = np.digitize(peaks[:, 1], SEGMENT_EDGES_NM)
    peak_fwhm = [
        (2.354820045 * peaks[(peak_segment == i) & (peaks[:, 2] > 0), 2])
        .astype(float).tolist()
        for i in range(3)
    ]
    return {
        "response_source": response["source"],
        "response_ratio": (response["ratio"]
                           if response["source"] == "measured" else None),
        "response_uncertainty": response["ratio_uncertainty"],
        "shift_deltas_nm": shift_deltas,
        "peak_fwhm_nm": peak_fwhm,
        "n_quick_peaks": int(peaks.shape[0]),
    }


def _robust_location(values: Sequence[float]) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    tolerance = max(3.0 * mad, 0.02)
    keep = np.abs(values - median) <= tolerance
    kept = values[keep]
    median = float(np.median(kept))
    mad = 1.4826 * float(np.median(np.abs(kept - median)))
    uncertainty = 1.2533 * mad / np.sqrt(max(kept.size, 1))
    return median, uncertainty, int(kept.size)


def build_shared_calibration(
    inventory: Sequence[dict],
    measurements: Sequence[dict],
    response_window: int = 25,
    shift_window: int = 10,
    minimum_response_neighbors: int = 3,
    minimum_shift_anchors: int = 15,
    acquisition_break_s: float = 120.0,
) -> list[dict]:
    """Pool neighboring measurements without crossing acquisition breaks."""
    if len(inventory) != len(measurements):
        raise ValueError("inventory and measurements must be aligned")
    groups, group = [], 0
    for i, row in enumerate(inventory):
        gap = row.get("acquisition_gap_s")
        if i and gap is not None and float(gap) > acquisition_break_s:
            group += 1
        groups.append(group)

    output = []
    for i, (entry, measurement) in enumerate(zip(inventory, measurements)):
        response_values = []
        for j in range(max(0, i - response_window),
                       min(len(inventory), i + response_window + 1)):
            if groups[j] != groups[i]:
                continue
            value = measurements[j].get("response_ratio")
            if value is not None:
                response_values.append(float(value))
        ratio, ratio_unc, ratio_n = _robust_location(response_values)
        response_valid = (ratio_n >= minimum_response_neighbors
                          and np.isfinite(ratio))

        shift_values = [[], [], []]
        for j in range(max(0, i - shift_window),
                       min(len(inventory), i + shift_window + 1)):
            if groups[j] != groups[i]:
                continue
            for segment in range(3):
                shift_values[segment].extend(
                    measurements[j]["shift_deltas_nm"][segment])
        shift_prior, shift_unc, shift_n = [], [], []
        for values in shift_values:
            location, uncertainty, count = _robust_location(values)
            valid = count >= minimum_shift_anchors and np.isfinite(location)
            shift_prior.append(location if valid else None)
            shift_unc.append(uncertainty if valid else None)
            shift_n.append(count)

        width_values = [[], [], []]
        for j in range(max(0, i - shift_window),
                       min(len(inventory), i + shift_window + 1)):
            if groups[j] != groups[i]:
                continue
            for segment in range(3):
                width_values[segment].extend(
                    measurements[j].get("peak_fwhm_nm", [[], [], []])[segment])
        profile_fwhm, profile_fwhm_n = [], []
        for values in width_values:
            location, _uncertainty, count = _robust_location(values)
            profile_fwhm.append(location if count >= 20 else None)
            profile_fwhm_n.append(count)

        output.append({
            "test_id": int(entry["test_id"]),
            "calibration_group": groups[i],
            "response_measured": measurement.get("response_ratio"),
            "response_measured_uncertainty":
                measurement.get("response_uncertainty"),
            "response_prior": ratio if response_valid else None,
            "response_prior_uncertainty": ratio_unc if response_valid else None,
            "response_prior_n": ratio_n,
            "shift_prior_nm": shift_prior,
            "shift_prior_uncertainty_nm": shift_unc,
            "shift_prior_n": shift_n,
            "profile_fwhm_nm": profile_fwhm,
            "profile_fwhm_n": profile_fwhm_n,
            "quick_peak_count": measurement.get("n_quick_peaks", 0),
        })
    return output
