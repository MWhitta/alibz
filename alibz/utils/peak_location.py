"""Per-segment peak-location dispatcher.

One entry point chooses the sub-pixel peak-location method **per detector
segment**, driven by the packaged Z300 shape file's ``location.location_method``
field: the empirical line-spread function (ePSF) in VIS, the point-Gaussian in
NIR (the NIR ePSF is gated for a sub-band-seam location bias -- see the file's
``gating_note``), the vendor-style cubic spline root in UV.  The choice is
data-driven, so enabling a future UV/NIR ePSF is a one-field edit in
``alibz/data/epsf-z300-20260924.json``, not a code change.

Instrument scope
----------------
The ePSF shapes are valid only for SciAps Z300 *native-grid* data (per-segment
pitch UV~0.086, VIS~0.138, NIR~0.190 nm).  :func:`is_z300_native` gates it; for
uniform 1/30 nm exports, synthetic spectra or any other grid the dispatcher
falls back to the spline root, and the caller-facing adapters
(:func:`locate_records`, :func:`locate_one_record`) delegate byte-identically
to the existing :mod:`alibz.utils.peakfit` path so non-Z300 behaviour is
unchanged.

Public API
----------
``locate(x, y, peak_indices) -> (centres_nm, uncertainties_nm, methods)``
    The specified array interface.
``locate_records`` / ``locate_one_record``
    Dict-returning adapters matching :func:`alibz.utils.peakfit.refine_peaks`
    and :func:`alibz.utils.peakfit.gaussian_subpixel_center` so existing
    callers can route through the dispatcher.

Deterministic; numpy + scipy only; never raises (returns NaN / ``ok=False``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from alibz.utils.epsf import fit_line
from alibz.utils.epsf_shapes import load_z300_shapes, segment_of, SEGMENT_EDGES
from alibz.utils.native_grid import segment_pitch_array
from alibz.utils import peakfit

METHODS = ("auto", "epsf", "spline", "parabola", "gaussian")

#: Expected per-segment native pitch bands [nm]; used to recognise Z300 native
#: data and to reject uniform 1/30 nm (~0.033 nm) exports.
_Z300_PITCH_BAND = {"UV": (0.060, 0.115), "VIS": (0.100, 0.170), "NIR": (0.140, 0.230)}


# ---------------------------------------------------------------------------
# Instrument detection
# ---------------------------------------------------------------------------
def is_z300_native(x) -> bool:
    """True when ``x`` looks like a Z300 native grid (multi-pitch, right bands).

    Requires the per-segment pitches to actually differ (native) and at least
    one present segment's median pitch to fall in its Z300 band.  A uniform
    1/30 nm export or a synthetic uniform grid returns False.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return False
    _pitch, _inc, is_native = segment_pitch_array(x)
    if not is_native:
        return False
    ok = False
    edges = [-np.inf, *SEGMENT_EDGES, np.inf]
    names = ("UV", "VIS", "NIR")
    for name, lo_e, hi_e in zip(names, edges[:-1], edges[1:]):
        m = (x > lo_e) & (x <= hi_e) & (x < 955.0)
        if np.count_nonzero(m) >= 3:
            med = float(np.median(np.diff(x[m])))
            lo, hi = _Z300_PITCH_BAND[name]
            if lo <= med <= hi:
                ok = True
    return ok


# ---------------------------------------------------------------------------
# Segment geometry on a full spectrum
# ---------------------------------------------------------------------------
def _segment_slice(x, index):
    """(lo, hi) sample bounds of the contiguous detector segment holding ``index``.

    Segments are delimited by the 365/620 nm edges and by any large pitch jump
    (seam / dead-channel gap)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    seg = segment_of(x[index])
    edges = {"UV": (-np.inf, SEGMENT_EDGES[0]), "VIS": SEGMENT_EDGES,
             "NIR": (SEGMENT_EDGES[1], np.inf)}[seg]
    lo, hi = 0, n - 1
    while lo < index and not (edges[0] < x[lo] <= edges[1]):
        lo += 1
    while hi > index and not (edges[0] < x[hi] <= edges[1]):
        hi -= 1
    # clip at pitch-jump seams inside the segment (defensive)
    d = np.diff(x)
    if hi > lo:
        med = np.median(d[lo:hi]) if hi > lo else 1.0
        j = index
        while j - 1 >= lo and d[j - 1] <= 5 * med:
            j -= 1
        lo = max(lo, j)
        j = index
        while j < hi and d[j] <= 5 * med:
            j += 1
        hi = min(hi, j)
    return lo, hi


def _spline_root(x, y, index, half=4):
    """Vendor-style cubic-spline root (continuous maximum) near ``index`` [nm]."""
    n = x.size
    s = slice(max(index - half, 0), min(index + half + 1, n))
    xs, ys = x[s], y[s]
    if xs.size < 4:
        return float(x[index])
    cs = CubicSpline(xs, ys, bc_type="not-a-knot")
    lo = x[max(index - 1, 0)]
    hi = x[min(index + 1, n - 1)]
    xx = np.linspace(lo, hi, 4001)
    return float(xx[int(np.argmax(cs(xx)))])


def _parabola(x, y, index):
    """Native 3-point parabola vertex near ``index`` [nm]."""
    n = x.size
    if index < 1 or index > n - 2:
        return float(x[index])
    xs = x[index - 1:index + 2] - x[index]
    a, b, _c = np.polyfit(xs, y[index - 1:index + 2], 2)
    return float(x[index] - b / (2 * a)) if a < 0 else float(x[index])


def _nm_at(x_seg, u_seg, c_local):
    return float(np.interp(c_local, u_seg, x_seg))


def _gauss_diag(x, y, index, seg):
    """Covariance-based centre 1-sigma [nm] + snr/width/blend from the existing
    point-Gaussian fit; attaches a principled uncertainty to the non-ePSF
    centres (spline/parabola give none) and diagnostics to every method."""
    g = peakfit.gaussian_subpixel_center(
        x, y, int(index), sigma_px=peakfit.instrument_sigma_px(seg))
    if g.get("ok"):
        return {"ok": True, "sigma_nm": float(g["center_sigma_nm"]), "snr": float(g["snr"]),
                "sigma_px": float(g["sigma_px"]), "chi2_nu": float(g["chi2_nu"]),
                "blend": bool(g["blend"]), "gauss_center_nm": float(g["center_nm"])}
    return {"ok": False, "sigma_nm": np.nan, "snr": np.nan, "sigma_px": np.nan,
            "chi2_nu": np.nan, "blend": None, "gauss_center_nm": float(x[index])}


def _locate_one(x, y, index, seg, method, shapes, sigma=None):
    """Return (center_nm, sigma_nm, method_used, extra) for a single peak.

    ``extra`` carries snr / sigma_px / chi2_nu / blend for the dict adapters.
    """
    index = int(index)
    n = x.size
    if index < 2 or index > n - 3:
        return (float(x[index]) if 0 <= index < n else np.nan), np.nan, "edge", {}
    pitch, _inc, _ = segment_pitch_array(x)
    lp = float(pitch[index])
    diag = _gauss_diag(x, y, index, seg)
    extra = {"snr": diag["snr"], "sigma_px": diag["sigma_px"],
             "chi2_nu": diag["chi2_nu"], "blend": diag["blend"]}

    if method == "parabola":
        return _parabola(x, y, index), diag["sigma_nm"], "parabola", extra
    if method == "spline":
        return _spline_root(x, y, index), diag["sigma_nm"], "spline", extra
    if method == "gaussian":
        # the validated point-Gaussian (gaussian_subpixel_center): best NIR
        # locator in the anchor isolation, and what registration uses.
        if diag.get("ok"):
            return float(diag["gauss_center_nm"]), diag["sigma_nm"], "gaussian", extra
        return _spline_root(x, y, index), diag["sigma_nm"], "spline(gauss_fallback)", extra
    # ---- epsf ----------------------------------------------------------
    if shapes is None:
        return _spline_root(x, y, index), diag["sigma_nm"], "spline(no_shapes)", extra
    lo, hi = _segment_slice(x, index)
    u = np.arange(lo, hi + 1, dtype=float)
    ep = shapes.epsf_at(float(x[index]))
    sg = None if sigma is None else np.asarray(sigma)[lo:hi + 1]
    f = fit_line(ep, u, y[lo:hi + 1], sg, c0=float(index), half=4.5,
                 fallback=True, center_bound=1.0)
    if not f.get("ok"):
        return _spline_root(x, y, index), diag["sigma_nm"], "spline(epsf_fallback)", extra
    c = _nm_at(x[lo:hi + 1], u, f["center"])
    if f.get("fallback") == "gauss":
        extra["chi2_nu"] = float(f.get("chi2nu", np.nan)); extra["blend"] = True
        return c, diag["sigma_nm"], "gaussian(epsf_fallback)", extra
    extra["chi2_nu"] = float(f.get("chi2nu", np.nan))
    cerr_nm = float(f.get("center_err", np.nan)) * lp
    return c, cerr_nm, "epsf", extra


# ---------------------------------------------------------------------------
# Public array interface
# ---------------------------------------------------------------------------
def locate(x, y, peak_indices, *, method: str = "auto",
           instrument: str = "auto", sigma=None, shapes=None, profile=None):
    """Sub-pixel peak centres, uncertainties and the method used, per segment.

    Parameters
    ----------
    x, y : full-spectrum wavelength [nm] and intensity arrays.
    peak_indices : sample indices of the peaks to locate.
    method : ``"auto"`` (per-segment from the shape flags), or one of
        ``"epsf"``, ``"spline"``, ``"parabola"``, ``"gaussian"`` to force one.
    instrument : ``"auto"`` (detect Z300 native), ``"z300"`` (assume native) or
        ``"none"`` (never use the ePSF; spline everywhere).
    sigma : optional per-sample 1-sigma for weighting (whole-spectrum array).
    shapes : a :class:`~alibz.utils.epsf_shapes.Z300Shapes`; loaded on demand.

    Returns
    -------
    centres_nm : np.ndarray
    uncertainties_nm : np.ndarray
    methods : list[str]  (the method actually used for each peak, incl. fallbacks)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = [int(i) for i in np.atleast_1d(peak_indices)]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    if instrument == "z300":
        native = True
    elif instrument == "none":
        native = False
    else:
        native = is_z300_native(x)

    if native and shapes is None:
        try:
            shapes = load_z300_shapes()
        except Exception:
            shapes = None

    centres = np.full(len(idx), np.nan)
    uncert = np.full(len(idx), np.nan)
    methods = []
    for k, i in enumerate(idx):
        if not (0 <= i < x.size):
            methods.append("edge")
            continue
        seg = segment_of(x[i])
        if method == "auto":
            # per-segment method from the shape file (UV spline, VIS epsf,
            # NIR gaussian by default); spline for non-native input.
            m = shapes.method_for(seg) if (native and shapes is not None) else "spline"
        else:
            m = method
            if m == "epsf" and (not native or shapes is None):
                m = "spline"
        c, s, used, _extra = _locate_one(x, y, i, seg, m, shapes, sigma=sigma)
        centres[k] = c
        uncert[k] = s
        methods.append(used)
    return centres, uncert, methods


# ---------------------------------------------------------------------------
# Dict adapters for existing callers
# ---------------------------------------------------------------------------
def locate_one_record(x, y, index, *, sigma_px, profile=None, baseline="linear",
                      sigma_bounds=(0.6, 2.0), instrument="auto", shapes=None,
                      noise=None):
    """Drop-in for :func:`peakfit.gaussian_subpixel_center`.

    Non-Z300 input delegates to ``gaussian_subpixel_center`` byte-identically.
    Z300 native input uses the ePSF/spline dispatcher and returns a dict with
    the same keys the callers read (``ok, center_nm, snr, sigma_px, blend`` and
    ``method_used``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    use_z300 = (instrument == "z300") or (instrument == "auto" and is_z300_native(x))
    if not use_z300:
        return peakfit.gaussian_subpixel_center(
            x, y, index, sigma_px=sigma_px, baseline=baseline,
            sigma_bounds=sigma_bounds, noise=noise)
    if shapes is None:
        try:
            shapes = load_z300_shapes()
        except Exception:
            shapes = None
    i = int(index)
    if not (0 <= i < x.size):
        return {"ok": False, "reason": "bad_index"}
    seg = segment_of(x[i])
    m = shapes.method_for(seg) if shapes is not None else "spline"
    c, s, used, extra = _locate_one(x, y, i, seg, m, shapes, sigma=None)
    if not np.isfinite(c):
        return {"ok": False, "reason": used}
    return {"ok": True, "center_nm": float(c),
            "center_sigma_nm": float(s) if np.isfinite(s) else float("nan"),
            "snr": float(extra.get("snr", np.nan)),
            "sigma_px": float(extra.get("sigma_px", np.nan)),
            "chi2_nu": float(extra.get("chi2_nu", np.nan)),
            "blend": bool(extra.get("blend")) if extra.get("blend") is not None else False,
            "method_used": used, "index": i}


def locate_records(x, y, indices, *, profile=None, baseline="linear",
                   sigma_bounds=(0.6, 2.0), instrument="auto", shapes=None):
    """Drop-in for :func:`peakfit.refine_peaks`.

    Non-Z300 input delegates to ``refine_peaks`` byte-identically.  Z300 native
    input returns one record per index from the ePSF/spline dispatcher, each
    carrying ``method_used``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    use_z300 = (instrument == "z300") or (instrument == "auto" and is_z300_native(x))
    if not use_z300:
        return peakfit.refine_peaks(x, y, indices, profile=profile,
                                    baseline=baseline, sigma_bounds=sigma_bounds)
    if shapes is None:
        try:
            shapes = load_z300_shapes()
        except Exception:
            shapes = None
    out = []
    for idx in indices:
        r = locate_one_record(x, y, int(idx), sigma_px=0.0, profile=profile,
                              baseline=baseline, sigma_bounds=sigma_bounds,
                              instrument="z300", shapes=shapes)
        r["index"] = int(idx)
        out.append(r)
    return out


__all__ = ["locate", "locate_records", "locate_one_record", "is_z300_native",
           "METHODS"]
