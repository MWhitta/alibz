"""Data-driven spatial-profile pipeline for LIBS line scans.

For a *profile* of many single-shot spectra of the same specimen (e.g. a
drill-core line scan at fixed spacing), the whole-pattern Saha-Boltzmann
inversion (:func:`alibz.pipeline.analyze_spectrum`) is fragile: single-shot
noise drives it into one-element / low-temperature basins, so per-position
composition is spatially INCOHERENT (measured on MW2-112: the dominant
element flips ~77 % of adjacent positions -- it is not tracking stratigraphy).

This module takes the opposite, data-driven route that exploits the profile's
own structure:

1. **Global reference** -- the mean/median over all positions has ~sqrt(N)
   better SNR, giving stable, high-confidence peaks.
2. **Regions of interest** -- detect peaks on the reference; each becomes a
   fixed wavelength window ("ROI") assigned to a database element/ion.
3. **Corpus-PCA constraint** -- deflate the corpus background/detector PCA's
   junction-artifact directions from every spectrum, so ROI intensities
   reflect chemistry rather than instrument common-mode.
4. **Per-position ROI matrix** -- integrated window intensity at every
   position, normalised to relative composition.
5. **Chemical-mode PCA** -- across-position PCA of the ROI matrix yields a few
   interpretable spatial chemical modes (loadings label them by element).

The products are *relative spatial profiles* and *detection status*, which are
spatially coherent (measured on MW2-112: ROI-profile lag-1 correlation ~0.77,
dominant-element flips ~14 %) -- NOT certified bulk weight fractions (see the
MW2-112 analysis plan re: ablation/transport/response calibration).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

_K_EV = 8.617333262e-5  # Boltzmann constant [eV/K]

#: Geochemically plausible elements for silicate/mudrock line assignment.  A
#: generous set: without it, line-rich rare earths win on raw gA alone.
DEFAULT_PLAUSIBLE = frozenset(
    "H Li B C N O F Na Mg Al Si P S Cl K Ca Ti V Cr Mn Fe Co Ni Cu Zn "
    "Rb Sr Zr Ba La Ce Cs".split()
)


@dataclass
class ProfileResult:
    """Outputs of :func:`analyze_profile`."""
    ids: np.ndarray                     # position id per row of M
    wavelength: np.ndarray              # channel grid
    roi_wl: np.ndarray                  # ROI centre wavelengths
    roi_label: List[str]               # "El ion" per ROI
    M: np.ndarray                       # (n_pos, n_roi) integrated ROI intensity
    Mn: np.ndarray                      # (n_pos, n_roi) per-position normalised
    element_profiles: dict              # element -> (n_pos,) summed relative intensity
    pca_components: np.ndarray          # (k, n_roi) chemical-mode loadings
    pca_scores: np.ndarray              # (n_pos, k) chemical-mode scores
    pca_evr: np.ndarray                 # explained-variance ratio
    coherence: float                    # median lag-1 corr of ROI profiles
    n_artifact_deflated: int = 0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1 -- reference stack
# ---------------------------------------------------------------------------
def _leading_int(path: str) -> int:
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def build_reference_stack(
    data_dir: str, pattern: str = "*.csv",
    exclude: Sequence[str] = ("summary.csv", "detections.csv"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every profile spectrum onto a common grid, ordered by position id.

    Returns ``(X, wavelength, ids)`` with ``X`` shape ``(n_pos, n_channels)``.
    """
    from alibz.pipeline import load_spectrum_csv
    files = sorted((f for f in glob.glob(os.path.join(data_dir, pattern))
                    if os.path.basename(f) not in set(exclude)),
                   key=_leading_int)
    if not files:
        raise FileNotFoundError(f"no {pattern!r} spectra in {data_dir!r}")
    wl0, _ = load_spectrum_csv(files[0])
    X = np.empty((len(files), wl0.size), dtype=np.float32)
    ids = np.empty(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        xi, yi = load_spectrum_csv(f)
        X[i] = yi if (xi.shape == wl0.shape and np.allclose(xi, wl0)) \
            else np.interp(wl0, xi, yi)
        ids[i] = _leading_int(f)
    return X, wl0, ids


# ---------------------------------------------------------------------------
# Stage 2 -- ROIs from the reference, database assignment
# ---------------------------------------------------------------------------
def detect_rois(wavelength: np.ndarray, reference: np.ndarray,
                prominence_sigma: float = 8.0, max_rois: int = 120,
                min_separation: int = 3) -> np.ndarray:
    """Peak-centre wavelengths of the strongest ROIs on the reference spectrum."""
    base = percentile_filter(reference, 25, size=151)
    r = reference - base
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
    pk, props = find_peaks(r, prominence=prominence_sigma * mad,
                           distance=min_separation)
    if pk.size == 0:
        return np.empty(0)
    keep = pk[np.argsort(props["prominences"])[::-1][:max_rois]]
    return np.sort(wavelength[keep])


def _line_table(db, plausible):
    els, ions, wls, gA, eup = [], [], [], [], []
    for e in db.elements:
        if e in db.no_lines or (plausible and e not in plausible):
            continue
        a = db.lines(e)
        if a.size == 0:
            continue
        ion = a[:, 0].astype(float)
        msk = ion <= 2
        els += [e] * int(msk.sum())
        ions += list(ion[msk].astype(int))
        wls += list(a[msk, 1].astype(float))
        gA += list(a[msk, 3].astype(float))
        eup += list(a[msk, 5].astype(float))
    o = np.argsort(wls)
    return (np.array(els, object)[o], np.array(ions)[o], np.array(wls)[o],
            np.array(gA)[o], np.array(eup)[o])


def assign_rois(roi_wl: np.ndarray, db, temperature: float = 8000.0,
                tol_nm: float = 0.08,
                plausible: Optional[frozenset] = DEFAULT_PLAUSIBLE
                ) -> List[Tuple[str, int]]:
    """Assign each ROI to its strongest Boltzmann-weighted nearby db line."""
    els, ions, wls, gA, eup = _line_table(db, plausible)
    kT = _K_EV * temperature
    out = []
    for w in roi_wl:
        lo, hi = np.searchsorted(wls, w - tol_nm), np.searchsorted(wls, w + tol_nm)
        if hi <= lo:
            out.append(("?", 0))
            continue
        score = gA[lo:hi] * np.exp(-eup[lo:hi] / kT)
        j = lo + int(np.argmax(score))
        out.append((els[j], int(ions[j])))
    return out


# ---------------------------------------------------------------------------
# Stage 3 -- corpus-PCA instrument constraint
# ---------------------------------------------------------------------------
def corpus_deflate(X: np.ndarray, wavelength: np.ndarray, bg_pca_path: str,
                   junctions_nm: Sequence[float] = (365.0, 620.0),
                   zone_hw_nm: float = 5.0, concentration: float = 2.0
                   ) -> Tuple[np.ndarray, int]:
    """Remove corpus background-PCA junction-artifact directions from ``X``.

    Components whose squared loading concentrates near a detector junction
    (more than ``concentration``x their wavelength share) are instrument
    artifacts; their subspace is projected out of every spectrum.  Returns
    ``(deflated_X, n_removed)``.  A missing/unreadable PCA file is a no-op.
    """
    import pickle
    try:
        bg = pickle.load(open(bg_pca_path, "rb"))
    except (OSError, ValueError, pickle.UnpicklingError):
        return X, 0
    cwl = np.asarray(bg["wavelength"], float)
    comps = np.asarray(bg["components"], float)
    jmask = np.zeros(cwl.size, dtype=bool)
    for j in junctions_nm:
        jmask |= np.abs(cwl - j) <= zone_hw_nm
    if not jmask.any():
        return X, 0
    share = jmask.mean()
    jfrac = (comps[:, jmask] ** 2).sum(1) / ((comps ** 2).sum(1) + 1e-30)
    art = np.where(jfrac > concentration * share)[0]
    if art.size == 0:
        return X, 0
    A = np.column_stack([np.interp(wavelength, cwl, comps[i]) for i in art])
    Q, _ = np.linalg.qr(A)
    cmean = np.interp(wavelength, cwl, np.asarray(bg["mean"], float))
    Xr = X - cmean
    return (X - (Xr @ Q) @ Q.T).astype(X.dtype), int(art.size)


# ---------------------------------------------------------------------------
# Stage 4/5 -- ROI matrix, chemical-mode PCA, coherence
# ---------------------------------------------------------------------------
def roi_matrix(X: np.ndarray, wavelength: np.ndarray, roi_wl: np.ndarray,
               half_window_nm: float = 0.10) -> np.ndarray:
    """(n_pos, n_roi) baseline-removed integrated intensity per ROI window."""
    M = np.zeros((X.shape[0], roi_wl.size))
    for k, c in enumerate(roi_wl):
        idx = np.where((wavelength >= c - half_window_nm)
                       & (wavelength <= c + half_window_nm))[0]
        if idx.size < 3:
            continue
        seg = X[:, idx]
        base = 0.5 * (seg[:, :2].mean(1, keepdims=True)
                      + seg[:, -2:].mean(1, keepdims=True))
        M[:, k] = np.clip(seg - base, 0, None).sum(1)
    return M


def profile_coherence(M: np.ndarray) -> float:
    """Median lag-1 correlation of the ROI profiles along the position axis.

    ~0 means position-independent noise; ->1 means smoothly varying (tracking
    real structure).  The single number that distinguishes a real spatial
    profile from a per-position fit artifact.
    """
    a = M - M.mean(0)
    num = (a[:-1] * a[1:]).sum(0)
    den = np.sqrt((a[:-1] ** 2).sum(0) * (a[1:] ** 2).sum(0)) + 1e-30
    return float(np.median(num / den))


def analyze_profile(
    data_dir: str, dbpath: str = "db", pattern: str = "*.csv",
    bg_pca_path: Optional[str] = None, max_rois: int = 120,
    half_window_nm: float = 0.10, n_modes: int = 6,
    temperature: float = 8000.0,
    plausible: Optional[frozenset] = DEFAULT_PLAUSIBLE,
) -> ProfileResult:
    """Run the full data-driven spatial-profile pipeline on a scan directory."""
    from sklearn.decomposition import PCA
    from alibz.utils.database import Database

    X, wl, ids = build_reference_stack(data_dir, pattern)
    reference = np.median(X, axis=0)
    n_art = 0
    if bg_pca_path:
        X, n_art = corpus_deflate(X, wl, bg_pca_path)
    roi_wl = detect_rois(wl, reference, max_rois=max_rois)
    roi_el = assign_rois(roi_wl, Database(dbpath), temperature=temperature,
                         plausible=plausible)
    roi_label = [f"{e} {'I' * i}" if i else "?" for e, i in roi_el]

    M = roi_matrix(X, wl, roi_wl, half_window_nm)
    tot = M.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    Mn = M / tot

    key = np.array([f"{e}{'I' * i}" if i else "?" for e, i in roi_el])
    eprof = {}
    for e in sorted({r[0] for r in roi_el if r[0] != "?"}):
        cols = np.array([r[0] == e for r in roi_el])
        eprof[e] = Mn[:, cols].sum(1)

    n_comp = min(n_modes, *Mn.shape)
    L = np.log10(Mn + 1e-4)
    L = L - L.mean(0)
    pca = PCA(n_components=n_comp).fit(L)
    scores = pca.transform(L)

    return ProfileResult(
        ids=ids, wavelength=wl, roi_wl=roi_wl, roi_label=roi_label,
        M=M, Mn=Mn, element_profiles=eprof,
        pca_components=pca.components_, pca_scores=scores,
        pca_evr=pca.explained_variance_ratio_,
        coherence=profile_coherence(Mn), n_artifact_deflated=n_art,
        meta=dict(n_positions=X.shape[0], n_rois=roi_wl.size,
                  roi_element_key=key),
    )
