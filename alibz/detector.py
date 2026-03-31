"""Three-segment detector model for SciAps Z-300 / similar LIBS instruments.

The CCD detector is physically split into three segments joined at
365 nm and 620 nm.  At each junction the baseline, gain, and dark
current can be discontinuous.  This module:

1. **Removes** per-segment artifacts identified by corpus PCA (components
   whose loading is concentrated at a junction).
2. **Subtracts** the baseline independently within each segment so that
   gain and offset discontinuities do not leak across segments.

The junction positions are stored in ``corrections/detector.json``
and loaded at construction time.

Typical usage
-------------
>>> from alibz.detector import DetectorModel
>>> model = DetectorModel.from_pca("data/bg_pca.pkl")
>>> corrected = model.correct(wavelength, intensity)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Locate the corrections directory relative to this file
# ---------------------------------------------------------------------------

_CORRECTIONS_DIR = Path(__file__).resolve().parent.parent / "corrections"
_DETECTOR_CONFIG = _CORRECTIONS_DIR / "detector.json"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Junction:
    """A detector-segment boundary."""
    wavelength: float          # junction wavelength (nm)
    index: int                 # index into the common wavelength grid
    zone: Tuple[float, float]  # affected wavelength range (nm)


@dataclass
class Segment:
    """One contiguous detector segment."""
    index: int                 # 0, 1, or 2
    label: str                 # "UV", "VIS", or "NIR"
    wl_lo: float               # inclusive lower bound (nm)
    wl_hi: float               # inclusive upper bound (nm)
    idx_lo: int                # index into wavelength grid
    idx_hi: int                # index into wavelength grid (exclusive)
    artifact_pcs: List[int] = field(default_factory=list)


@dataclass
class DetectorModel:
    """Three-segment detector model with PCA-derived artifact correction.

    Attributes
    ----------
    wavelength : ndarray
        Common wavelength grid from the corpus PCA.
    junctions : list of Junction
        The two junction boundaries (365 nm and 620 nm).
    segments : list of Segment
        The three detector segments (UV, VIS, NIR).
    pca_components : ndarray, shape (n_components, n_channels)
        PCA loadings from the corpus-level background PCA.
    pca_mean : ndarray, shape (n_channels,)
        Mean spectrum from the PCA.
    artifact_components : list of int
        Indices of PCA components identified as junction artifacts.
    blend_width : float
        Width of the cosine blend applied across each junction (nm).
    """
    wavelength: np.ndarray
    junctions: List[Junction]
    segments: List[Segment]
    pca_components: np.ndarray
    pca_mean: np.ndarray
    artifact_components: List[int]
    blend_width: float = 2.0

    # =================================================================
    # Construction
    # =================================================================

    @classmethod
    def from_pca(
        cls,
        pca_path: Union[str, Path],
        config_path: Union[str, Path, None] = None,
        artifact_threshold: float = 2.0,
    ) -> "DetectorModel":
        """Build a detector model from a background PCA file.

        Junction positions are read from the detector configuration
        file (``corrections/detector.json`` by default).  PCA
        components whose loading is concentrated at a junction zone
        are flagged as artifacts.

        Parameters
        ----------
        pca_path : str or Path
            Pickle file produced by ``background-pca``.
        config_path : str, Path, or None
            Detector configuration JSON.  Defaults to
            ``corrections/detector.json`` in the repository root.
        artifact_threshold : float
            A component is flagged as an artifact if the fraction of
            its squared loading in any junction zone exceeds
            ``artifact_threshold`` times the zone's wavelength share.
        """
        # --- Load config ---
        cfg_path = Path(config_path) if config_path else _DETECTOR_CONFIG
        with open(cfg_path) as f:
            cfg = json.load(f)

        junction_wavelengths = cfg["junctions_nm"]
        zone_hw = cfg.get("junction_zone_half_width_nm", 5.0)
        blend_width = cfg.get("blend_width_nm", 2.0)
        seg_cfgs = cfg.get("segments", [])

        # --- Load PCA ---
        with open(Path(pca_path), "rb") as f:
            bg = pickle.load(f)

        wl = np.asarray(bg["wavelength"], dtype=float)
        components = np.asarray(bg["components"], dtype=float)
        mean = np.asarray(bg["mean"], dtype=float)
        n = len(wl)

        # --- Build junctions from config ---
        junctions = []
        for jw in sorted(junction_wavelengths):
            idx = int(np.argmin(np.abs(wl - jw)))
            junctions.append(Junction(
                wavelength=float(jw),
                index=idx,
                zone=(jw - zone_hw, jw + zone_hw),
            ))

        # --- Build segments ---
        seg_labels = {s["index"]: s.get("label", str(s["index"])) for s in seg_cfgs}
        segments = [
            Segment(0, seg_labels.get(0, "UV"),
                    float(wl[0]), junctions[0].wavelength,
                    0, junctions[0].index),
            Segment(1, seg_labels.get(1, "VIS"),
                    junctions[0].wavelength, junctions[1].wavelength,
                    junctions[0].index, junctions[1].index),
            Segment(2, seg_labels.get(2, "NIR"),
                    junctions[1].wavelength, float(wl[-1]),
                    junctions[1].index, n),
        ]

        # --- Identify artifact components ---
        artifact_set: set = set()
        for junc in junctions:
            zone_mask = (wl >= junc.zone[0]) & (wl <= junc.zone[1])
            zone_frac = zone_mask.sum() / n

            for i, pc in enumerate(components):
                total_var = np.sum(pc ** 2)
                if total_var <= 0:
                    continue
                zone_var = np.sum(pc[zone_mask] ** 2)
                if zone_var / total_var > artifact_threshold * zone_frac:
                    artifact_set.add(i)

        artifact_components = sorted(artifact_set)

        # Tag each segment with its concentrated artifact PCs
        for seg in segments:
            seg_mask = np.zeros(n, dtype=bool)
            seg_mask[seg.idx_lo:seg.idx_hi] = True
            seg_frac = seg_mask.sum() / n
            for i in artifact_components:
                pc = components[i]
                total_var = np.sum(pc ** 2)
                if total_var <= 0:
                    continue
                seg_var = np.sum(pc[seg_mask] ** 2)
                if seg_var / total_var > 1.5 * seg_frac:
                    seg.artifact_pcs.append(i)

        return cls(
            wavelength=wl,
            junctions=junctions,
            segments=segments,
            pca_components=components,
            pca_mean=mean,
            artifact_components=artifact_components,
            blend_width=blend_width,
        )

    # =================================================================
    # Correction
    # =================================================================

    def correct(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subtract_artifacts: bool = True,
        subtract_background: bool = True,
        bg_window: int = 5,
        bg_n_sigma: float = 1.0,
    ) -> np.ndarray:
        """Apply segmented artifact removal and background subtraction.

        Parameters
        ----------
        x : ndarray
            Wavelength axis.
        y : ndarray
            Raw intensity values.
        subtract_artifacts : bool
            Project out PCA artifact components per segment.
        subtract_background : bool
            Estimate and subtract background per segment.
        bg_window : int
            Window size for background estimation.
        bg_n_sigma : float
            Sigma threshold for background anchor filtering.

        Returns
        -------
        ndarray
            Corrected intensity on the same wavelength grid as ``x``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).copy()

        on_grid = (len(x) == len(self.wavelength)
                   and np.allclose(x, self.wavelength, atol=1e-6))
        if not on_grid:
            interp_fn = interp1d(x, y, kind="linear",
                                 bounds_error=False, fill_value=0.0)
            y_grid = interp_fn(self.wavelength)
        else:
            y_grid = y

        corrected = np.zeros_like(y_grid)

        for seg in self.segments:
            sl = slice(seg.idx_lo, seg.idx_hi)
            seg_y = y_grid[sl].copy()
            seg_wl = self.wavelength[sl]

            if subtract_artifacts and seg.artifact_pcs:
                seg_centered = seg_y - self.pca_mean[sl]
                for pc_idx in seg.artifact_pcs:
                    pc_seg = self.pca_components[pc_idx, sl]
                    pc_norm = np.dot(pc_seg, pc_seg)
                    if pc_norm > 1e-30:
                        coeff = np.dot(seg_centered, pc_seg) / pc_norm
                        seg_centered -= coeff * pc_seg
                seg_y = seg_centered + self.pca_mean[sl]

            if subtract_background:
                bg = _estimate_segment_background(
                    seg_wl, seg_y,
                    window=bg_window,
                    n_sigma=bg_n_sigma,
                )
                seg_y = seg_y - bg

            corrected[sl] = seg_y

        for junc in self.junctions:
            corrected = _blend_junction(
                self.wavelength, corrected, junc,
                blend_width=self.blend_width,
            )

        if not on_grid:
            interp_back = interp1d(self.wavelength, corrected,
                                   kind="linear", bounds_error=False,
                                   fill_value=0.0)
            return interp_back(x)

        return corrected

    def correct_corpus(
        self,
        spectra: np.ndarray,
        wavelength: Optional[np.ndarray] = None,
        subtract_artifacts: bool = True,
        subtract_background: bool = True,
    ) -> np.ndarray:
        """Apply correction to every spectrum in a corpus array.

        Parameters
        ----------
        spectra : ndarray, shape (n_spectra, n_channels)
        wavelength : ndarray, optional

        Returns
        -------
        ndarray, shape (n_spectra, n_channels)
        """
        wl = wavelength if wavelength is not None else self.wavelength
        n = spectra.shape[0]
        out = np.empty_like(spectra)
        for i in range(n):
            out[i] = self.correct(
                wl, spectra[i],
                subtract_artifacts=subtract_artifacts,
                subtract_background=subtract_background,
            )
            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"  corrected {i + 1}/{n}")
        return out

    # =================================================================
    # Diagnostics
    # =================================================================

    def summary(self) -> str:
        """Return a human-readable summary of the detector model."""
        lines = ["DetectorModel"]
        lines.append(f"  wavelength range: {self.wavelength[0]:.1f} - "
                     f"{self.wavelength[-1]:.1f} nm "
                     f"({len(self.wavelength)} channels)")
        lines.append(f"  junctions ({len(self.junctions)}):")
        for j in self.junctions:
            lines.append(f"    {j.wavelength:.1f} nm  "
                         f"(zone={j.zone[0]:.1f}-{j.zone[1]:.1f} nm)")
        lines.append(f"  segments ({len(self.segments)}):")
        for s in self.segments:
            art = f", artifact PCs: {s.artifact_pcs}" if s.artifact_pcs else ""
            lines.append(f"    {s.label} (seg {s.index}): "
                         f"{s.wl_lo:.1f} - {s.wl_hi:.1f} nm "
                         f"({s.idx_hi - s.idx_lo} channels{art})")
        lines.append(f"  artifact components: {self.artifact_components} "
                     f"({len(self.artifact_components)} total)")
        return "\n".join(lines)


# =====================================================================
# Private helpers
# =====================================================================

def _estimate_segment_background(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 5,
    n_sigma: float = 1.0,
) -> np.ndarray:
    """Estimate background for a single detector segment.

    Uses a rolling-minimum approach:
    1. Compute a local minimum envelope using a median filter.
    2. Iteratively reject points above the envelope.
    3. Interpolate through the remaining anchor points.
    """
    n = len(y)
    if n < 5:
        return np.zeros_like(y)

    win = max(window, 3)
    if win % 2 == 0:
        win += 1
    large_win = max(win, n // 20)
    if large_win % 2 == 0:
        large_win += 1

    envelope = median_filter(y, size=large_win, mode="reflect")

    residual = y - envelope
    threshold = np.mean(residual) + n_sigma * np.std(residual)
    anchors = np.where(residual <= threshold)[0]

    if len(anchors) < 2:
        return envelope

    if anchors[0] != 0:
        anchors = np.concatenate(([0], anchors))
    if anchors[-1] != n - 1:
        anchors = np.concatenate((anchors, [n - 1]))

    bg = np.interp(np.arange(n), anchors, y[anchors])
    bg = np.minimum(bg, y)
    bg = np.clip(bg, 0, None)

    return bg


def _blend_junction(
    wl: np.ndarray,
    y: np.ndarray,
    junction: Junction,
    blend_width: float = 2.0,
) -> np.ndarray:
    """Smooth the intensity across a junction with a linear bridge."""
    jw = junction.wavelength
    lo = jw - blend_width
    hi = jw + blend_width

    blend_mask = (wl >= lo) & (wl <= hi)
    if blend_mask.sum() < 3:
        return y

    idx = np.where(blend_mask)[0]
    left_idx = max(idx[0] - 1, 0)
    right_idx = min(idx[-1] + 1, len(y) - 1)

    left_val = y[left_idx]
    right_val = y[right_idx]

    t = (wl[idx] - lo) / (hi - lo)
    linear = left_val + (right_val - left_val) * t

    y_out = y.copy()
    bridge_weight = 0.5
    y_out[idx] = (1 - bridge_weight) * y[idx] + bridge_weight * linear

    return y_out
