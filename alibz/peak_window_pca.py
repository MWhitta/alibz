"""Peak-window PCA measurements with explicit shape-nuisance diagnostics.

The corpus PCA basis was trained on baseline-subtracted, range-normalized peak
windows.  This module reproduces that normalization, projects individual LIBS
windows into the fixed basis, and converts the reconstruction into a relative
area proxy while keeping shift, broadening, and self-absorption-like shape
coordinates separate.

The resulting amplitude is suitable for *within-line* spatial trends.  It is
not an absolute concentration and is not comparable between elements without
standards and a plasma-response model.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d


PHYSICAL_COORDINATES = (
    "shift",
    "gaussian_broadening",
    "lorentzian_wings",
    "flattening",
    "splitting",
)
DEFAULT_BASIS_NAME = "corpus_peak_shape_pca_05x_10pc.npz"


@dataclass(frozen=True)
class PeakShapeBasis:
    components: np.ndarray
    mean_peak: np.ndarray
    explained_variance_ratio: np.ndarray
    score_std: np.ndarray
    half_window_nm: float
    n_window_points: int
    templates: dict[str, np.ndarray]
    template_loadings: dict[str, np.ndarray]
    template_score_std: dict[str, float]
    manifest: dict


@dataclass(frozen=True)
class WindowProjection:
    normalized: np.ndarray
    reconstructed: np.ndarray
    scores: np.ndarray
    physical_scores: dict[str, float]
    physical_z: dict[str, float]
    peak_range: float
    area: float
    reconstruction_rmse: float
    reconstruction_r2: float
    centroid_offset_nm: float
    noise: float
    peak_snr: float


@dataclass(frozen=True)
class BroadLineMeasurement:
    """Wide-window line measurement for optically thick resonance peaks."""

    area: float
    area_sigma: float
    snr: float
    height: float
    center_height: float
    peak_offset_nm: float
    centroid_offset_nm: float
    fwhm_nm: float
    top_flatness: float
    noise: float


def resolve_basis_path(path: str | Path | None = None) -> Path:
    """Resolve the compact basis in source and installed layouts."""
    candidates = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    configured = os.environ.get("ALIBZ_PEAK_PCA_BASIS")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((
        Path.cwd() / "corrections" / DEFAULT_BASIS_NAME,
        Path(__file__).resolve().parents[1] / "corrections" / DEFAULT_BASIS_NAME,
        Path(sys.prefix) / "share" / "alibz" / "corrections"
        / DEFAULT_BASIS_NAME,
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "peak-shape PCA basis not found; pass --basis or set "
        "ALIBZ_PEAK_PCA_BASIS")


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def _range_normalize(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    output = profile - np.min(profile)
    scale = float(np.max(output))
    return output / scale if scale > 0 else np.zeros_like(output)


def _shift_profile(profile: np.ndarray, delta: float) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, profile.size)
    return np.interp(x - delta, x, profile,
                     left=float(profile[0]), right=float(profile[-1]))


def physical_templates(mean_peak: np.ndarray) -> dict[str, np.ndarray]:
    """Construct signed, unit-norm shape directions around the corpus mean."""
    mean = _range_normalize(mean_peak)
    broadened = _range_normalize(gaussian_filter1d(mean, sigma=2.0))

    x = np.linspace(-1.0, 1.0, mean.size)
    gamma = 0.08
    kernel = gamma / (np.pi * (x ** 2 + gamma ** 2))
    kernel /= np.sum(kernel)
    wings = _range_normalize(np.convolve(mean, kernel, mode="same"))

    cap = 0.78 * float(np.max(mean))
    flattened = _range_normalize(np.minimum(mean, cap))
    split = _range_normalize(
        0.5 * (_shift_profile(mean, -0.10) + _shift_profile(mean, 0.10)))
    shifted = _range_normalize(_shift_profile(mean, 0.08))
    return {
        "shift": _unit(shifted - mean),
        "gaussian_broadening": _unit(broadened - mean),
        "lorentzian_wings": _unit(wings - mean),
        "flattening": _unit(flattened - mean),
        "splitting": _unit(split - mean),
    }


def load_basis(npz_path: str | Path,
               manifest_path: str | Path | None = None) -> PeakShapeBasis:
    npz_path = Path(npz_path)
    if manifest_path is None:
        manifest_path = npz_path.with_suffix(".json")
    manifest = json.loads(Path(manifest_path).read_text())
    with np.load(npz_path) as data:
        components = np.asarray(data["components"], dtype=float)
        mean = np.asarray(data["mean_peak"], dtype=float)
        evr = np.asarray(data["explained_variance_ratio"], dtype=float)
        score_std = np.asarray(data["score_std"], dtype=float)
    if components.ndim != 2 or mean.shape != (components.shape[1],):
        raise ValueError("incompatible corpus peak-PCA basis shapes")
    if score_std.shape != (components.shape[0],):
        raise ValueError("score_std must have one value per component")
    templates = physical_templates(mean)
    loadings = {
        name: components @ template for name, template in templates.items()}
    template_std = {
        name: float(np.sqrt(np.sum((score_std * weights) ** 2)))
        for name, weights in loadings.items()
    }
    return PeakShapeBasis(
        components=components,
        mean_peak=mean,
        explained_variance_ratio=evr,
        score_std=score_std,
        half_window_nm=float(manifest["training"]["half_window_nm"]),
        n_window_points=int(components.shape[1]),
        templates=templates,
        template_loadings=loadings,
        template_score_std=template_std,
        manifest=manifest,
    )


def basis_characteristics(basis: PeakShapeBasis) -> list[dict]:
    """Return correlations of every PC with interpretable shape directions."""
    rows = []
    for index, component in enumerate(basis.components):
        correlations = {
            name: float(np.dot(component, template))
            for name, template in basis.templates.items()
        }
        dominant = max(correlations, key=lambda name: abs(correlations[name]))
        rows.append({
            "pc": index + 1,
            "explained_variance_ratio": basis.explained_variance_ratio[index],
            "score_std": basis.score_std[index],
            **{f"{name}_loading": correlations[name]
               for name in PHYSICAL_COORDINATES},
            "dominant_template": dominant,
            "dominant_template_loading": correlations[dominant],
        })
    return rows


def extract_normalized_window(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    half_window_nm: float,
    n_window_points: int,
) -> tuple[np.ndarray, float, float] | None:
    """Apply the exact baseline/range normalization used for corpus training."""
    mask = ((wavelength >= center_nm - half_window_nm)
            & (wavelength <= center_nm + half_window_nm))
    if np.sum(mask) < 5:
        return None
    values = np.asarray(intensity[mask], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    baseline = np.linspace(values[0], values[-1], values.size)
    residual = values - baseline
    peak_range = float(np.max(residual) - np.min(residual))
    if not np.isfinite(peak_range) or peak_range <= 0:
        return None
    normalized = (residual - np.min(residual)) / peak_range
    old = np.linspace(0.0, 1.0, values.size)
    new = np.linspace(0.0, 1.0, n_window_points)
    window = np.interp(new, old, normalized)
    noise = 1.4826 * float(np.median(np.abs(
        residual - np.median(residual))))
    return window, peak_range, noise


def measure_broad_line(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    half_window_nm: float = 1.0,
    baseline_edge_nm: tuple[float, float] = (0.80, 1.00),
    noise_band_nm: tuple[float, float] = (1.20, 1.80),
) -> BroadLineMeasurement | None:
    """Measure a broad resonance line without treating its wings as noise.

    A linear continuum is fitted to the outer edges of the integration window.
    Noise is estimated in a disjoint, more remote band.  The returned area is
    the positive residual equivalent area; it is a relative-intensity proxy,
    not an optical-depth-corrected concentration.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    distance = np.abs(wavelength - float(center_nm))
    window = distance <= half_window_nm
    edge = ((distance >= baseline_edge_nm[0])
            & (distance <= baseline_edge_nm[1]))
    if np.sum(window) < 9 or np.sum(edge) < 6:
        return None
    x = wavelength[window]
    y = intensity[window]
    if not np.all(np.isfinite(y)):
        return None
    coefficients = np.polyfit(wavelength[edge], intensity[edge], 1)
    residual = y - np.polyval(coefficients, x)
    positive = np.maximum(residual, 0.0)
    area = float(np.trapezoid(positive, x))
    peak_index = int(np.argmax(residual))
    height = float(residual[peak_index])
    if not np.isfinite(height) or height <= 0 or area <= 0:
        return None

    half_height = 0.5 * height
    left = peak_index
    while left > 0 and residual[left - 1] >= half_height:
        left -= 1
    right = peak_index
    while right + 1 < residual.size and residual[right + 1] >= half_height:
        right += 1
    pitch = float(np.median(np.diff(x)))
    fwhm = float(x[right] - x[left] + pitch)
    centroid = float(np.sum((x - center_nm) * positive) / np.sum(positive))
    center_height = float(np.interp(center_nm, x, residual))
    top_count = min(5, residual.size)
    top = np.sort(residual)[-top_count:]
    top_flatness = float((top[-1] - top[0]) / max(height, 1e-12))

    noise_mask = ((distance >= noise_band_nm[0])
                  & (distance <= noise_band_nm[1]))
    if np.sum(noise_mask) >= 6:
        noise_values = (intensity[noise_mask]
                        - np.polyval(coefficients, wavelength[noise_mask]))
    else:
        noise_values = residual[np.r_[0:3, -3:0]]
    noise = 1.4826 * float(np.median(np.abs(
        noise_values - np.median(noise_values))))
    area_sigma = noise * pitch * np.sqrt(residual.size)
    return BroadLineMeasurement(
        area=area,
        area_sigma=float(area_sigma),
        snr=area / max(float(area_sigma), 1e-12),
        height=height,
        center_height=center_height,
        peak_offset_nm=float(x[peak_index] - center_nm),
        centroid_offset_nm=centroid,
        fwhm_nm=fwhm,
        top_flatness=top_flatness,
        noise=noise,
    )


def project_window(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    basis: PeakShapeBasis,
    n_components: int | None = None,
) -> WindowProjection | None:
    """Project one window and reconstruct a PCA-constrained relative area."""
    extracted = extract_normalized_window(
        wavelength, intensity, center_nm, basis.half_window_nm,
        basis.n_window_points)
    if extracted is None:
        return None
    normalized, peak_range, noise = extracted
    distance = np.abs(wavelength - center_nm)
    side = ((distance >= 1.25 * basis.half_window_nm)
            & (distance <= 3.0 * basis.half_window_nm))
    if np.sum(side) >= 6:
        side_values = np.asarray(intensity[side], dtype=float)
        side_noise = 1.4826 * float(np.median(np.abs(
            side_values - np.median(side_values))))
        if np.isfinite(side_noise) and side_noise > 0:
            noise = side_noise
    count = basis.components.shape[0] if n_components is None else min(
        int(n_components), basis.components.shape[0])
    components = basis.components[:count]
    scores = (normalized - basis.mean_peak) @ components.T
    reconstructed = basis.mean_peak + scores @ components
    residual = normalized - reconstructed
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((normalized - np.mean(normalized)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / max(total, 1e-12)

    shape = _range_normalize(reconstructed)
    x_nm = np.linspace(-basis.half_window_nm, basis.half_window_nm,
                       basis.n_window_points)
    weights = np.maximum(shape, 0.0)
    centroid = (float(np.sum(x_nm * weights) / np.sum(weights))
                if np.sum(weights) > 0 else np.nan)
    area = peak_range * float(np.trapezoid(shape, x_nm))

    physical_scores = {}
    physical_z = {}
    full_scores = np.zeros(basis.components.shape[0], dtype=float)
    full_scores[:count] = scores
    for name in PHYSICAL_COORDINATES:
        value = float(full_scores @ basis.template_loadings[name])
        scale = max(basis.template_score_std[name], 1e-12)
        physical_scores[name] = value
        physical_z[name] = value / scale
    return WindowProjection(
        normalized=normalized,
        reconstructed=reconstructed,
        scores=full_scores,
        physical_scores=physical_scores,
        physical_z=physical_z,
        peak_range=peak_range,
        area=area,
        reconstruction_rmse=rmse,
        reconstruction_r2=r2,
        centroid_offset_nm=centroid,
        noise=noise,
        peak_snr=peak_range / max(noise, 1e-12),
    )


def reference_peak(
    wavelength: np.ndarray,
    reference: np.ndarray,
    expected_nm: float,
    basis: PeakShapeBasis,
    search_half_width_nm: float = 0.10,
    step_nm: float = 0.01,
    n_components: int | None = 5,
) -> dict:
    """Refine a known line center on the shift-corrected summed spectrum."""
    candidates = np.arange(
        -search_half_width_nm, search_half_width_nm + 0.5 * step_nm, step_nm)
    best = None
    for offset in candidates:
        projection = project_window(
            wavelength, reference, expected_nm + offset, basis,
            n_components=n_components)
        if projection is None:
            continue
        score = projection.peak_snr / (1.0 + projection.reconstruction_rmse)
        if best is None or score > best[0]:
            best = (score, offset, projection)
    if best is None:
        return {"center_nm": expected_nm, "offset_nm": 0.0,
                "reference_snr": 0.0, "reconstruction_r2": np.nan}
    _score, offset, projection = best
    return {
        "center_nm": expected_nm + offset,
        "offset_nm": offset,
        "reference_snr": projection.peak_snr,
        "reconstruction_r2": projection.reconstruction_r2,
        "flattening_z": projection.physical_z["flattening"],
        "splitting_z": projection.physical_z["splitting"],
    }


def pairwise_spatial_coherence(
    profiles: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    minimum_observations: int = 100,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Median same-stage Spearman coherence for each candidate line."""
    from scipy.stats import spearmanr

    line_ids = sorted(profiles)
    pairs = {}
    per_line: dict[str, list[float]] = {line_id: [] for line_id in line_ids}
    for first_index, first_id in enumerate(line_ids):
        for second_id in line_ids[first_index + 1:]:
            use = valid[first_id] & valid[second_id]
            if np.sum(use) < minimum_observations:
                continue
            first = profiles[first_id][use]
            second = profiles[second_id][use]
            if np.ptp(first) == 0 or np.ptp(second) == 0:
                continue
            rho = float(spearmanr(first, second).statistic)
            if not np.isfinite(rho):
                continue
            pairs[(first_id, second_id)] = rho
            per_line[first_id].append(rho)
            per_line[second_id].append(rho)
    medians = {
        line_id: (float(np.median(values)) if values else np.nan)
        for line_id, values in per_line.items()
    }
    return medians, pairs


def assign_independent_window_clusters(
    records: Iterable[dict],
    radius_nm: float,
    center_key: str = "center_nm",
) -> dict[str, tuple[int, int]]:
    """Group overlapping same-element/stage windows without chain merging.

    The returned mapping is ``line_id -> (cluster_index, cluster_size)``.
    Every cluster spans at most ``radius_nm`` from its first to last center;
    this avoids a chain of dense lines turning an entire spectral region into
    one nominal feature.
    """
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (str(record["element"]), int(record["ion_stage"]))
        grouped.setdefault(key, []).append(record)
    output = {}
    for features in grouped.values():
        features = sorted(features, key=lambda row: float(row[center_key]))
        clusters: list[list[dict]] = []
        for feature in features:
            center = float(feature[center_key])
            if (not clusters
                    or center - float(clusters[-1][0][center_key])
                    > radius_nm):
                clusters.append([feature])
            else:
                clusters[-1].append(feature)
        for cluster_index, cluster in enumerate(clusters, 1):
            for feature in cluster:
                output[str(feature["line_id"])] = (
                    cluster_index, len(cluster))
    return output


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(
        np.cumsum(weights), 0.5 * np.sum(weights), side="left")])
