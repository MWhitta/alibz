#!/usr/bin/env python
"""Line-matching completeness, self-absorption, and line-broadening analysis
for the Fe (Aesar 99.98%) and V (pure) Z300 LIBS runs.

Regenerates every table and figure in
    reports/2026-09-22-line-matching-sa-broadening.md
plus the PNG figures under reports/figures/line-physics-20260922/.

Usage:
    .venv/bin/python scripts/line_physics_analysis.py \
        --data <acq dir> --ledger-fe <ledger.json> --ledger-v <ledger_v.json>

Deterministic and read-only w.r.t. the data.  It reuses the tested primitives
in scripts/fe_plasma_analysis.py (native-grid line measurement, simple peak
finder, sub-pixel centroid, per-segment shift model, Boltzmann helpers) and the
self-absorption escape-factor / doublet-inversion in alibz.utils.absorption.

Method summary (see report for the full text):
  * Native grid only.  Means are per-pixel-index averages of the per-run means
    (all runs share the 7914-pixel grid to <0.043 nm = <0.5 px, so pixel index
    is the physical detector element); the wavelength axis is the reference
    run's own grid.  No resampling.
  * Line net area / SNR: measure_line() -- local linear side-band baseline,
    trapezoid net area over a >=5-sample central window, SNR = height /
    (1.4826*MAD sideband noise, floor 1 count).
  * Shift model: per detector segment (UV<365, VIS 365-620, NIR>620 nm) linear
    obs-db = a + b*(lambda-lambda0), from strong locally-dominant single lines
    of the sample element (UV, VIS) + isolated Ar I lines (NIR).
  * Self-absorption: (a) same-upper-level / resonance multiplet ratio inverted
    to tau with alibz.utils.absorption.invert_doublet_tau; (b) Boltzmann
    residuals vs a lower-level population proxy; (c) FWHM-vs-area correlation;
    (d) delay dependence.
  * Broadening: Gaussian sub-pixel FWHM over peak +-3 native samples; Doppler
    from T=9 kK; instrumental from the lower FWHM envelope and Ar I NIR lines;
    Stark/opacity as the excess width.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPTS = os.path.join(REPO, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# Reused, tested primitives from the Fe plasma-analysis script.
from fe_plasma_analysis import (
    SEG_EDGES, SEG_NAMES, seg_index, Run, load_runs, measure_line,
    find_peaks_simple, subpixel_center, db_lines_all, AR_NIR_ANCHORS, KB,
)
from alibz.utils.database import Database
from alibz.utils.absorption import escape_factor, invert_doublet_tau

# ---- constants --------------------------------------------------------------
T_RANK = 9000.0                     # K, line-strength ranking temperature
KT_RANK = KB * T_RANK               # eV
MATCH_TOL_NM = 0.15                 # peak <-> db line matching tolerance
BLEND_FRAC = 0.30                   # matched-blend: >=2 cands within 30% of top
PITCH = {"UV": 0.089, "VIS": 0.129, "NIR": 0.179}   # nm per native sample
# Doppler FWHM coefficient: dlam_FWHM = 7.16e-7 * lam * sqrt(T/A)
DOPPLER_C = 7.16e-7
MASS = {"Fe": 55.845, "V": 50.942}

CONTAM = ("Ar", "H", "O", "N")      # ambient / purge species checked in sec 1

# Explicit known ambient / purge emission lines (air nm).  O and N have hundreds
# of db lines, so "nearest is an O/N line" is not evidence; we credit a
# contaminant ONLY when a peak matches one of these textbook ambient lines (as
# in the Fe report's coordinator partner-line method).
KNOWN_AMBIENT = {
    "Ar": [696.543, 706.722, 727.294, 738.398, 750.387, 751.465, 763.511,
           772.376, 772.421, 794.818, 800.616, 801.479, 810.369, 811.531,
           826.452, 840.821, 842.465, 852.144, 866.794, 912.297, 922.450],
    "H":  [656.279, 486.135, 434.047],
    "O":  [777.194, 777.417, 777.539, 844.636, 615.818, 926.6],
    "N":  [744.229, 746.831, 821.634, 868.340, 862.924, 818.487, 818.802],
}


def segn(wl):
    return SEG_NAMES[int(seg_index(float(wl)))]


def local_pitch(wl):
    return PITCH[segn(wl)]


# =============================================================================
# Mean spectra (native grid, per-pixel-index average of per-run means)
# =============================================================================
def pool_mean(runs):
    """Per-pixel mean intensity across the given runs' per-run means.

    All runs share the 7914-pixel grid to <0.5 px; index k is the same physical
    detector element in every run.  Returns (wl, mean) on the first run's grid.
    """
    npix = min(r.npix for r in runs)
    stack = np.array([r.mean[:npix] for r in runs])
    return runs[0].wl[:npix], stack.mean(axis=0)


def run_by_prefix(runs, prefix):
    for r in runs:
        if r.meta["run"].startswith(prefix):
            return r
    raise KeyError(prefix)


# =============================================================================
# Element line table + per-element shift model
# =============================================================================
def el_lines(db, element, ion_max=2, wl_lo=185.0, wl_hi=948.0, gA_min=0.0):
    a = db.lines(element)
    ion = a[:, 0].astype(float)
    wl = a[:, 1].astype(float)
    gA = a[:, 3].astype(float)
    Ei = a[:, 4].astype(float)
    Ek = a[:, 5].astype(float)
    gi = a[:, 12].astype(float)
    gk = a[:, 13].astype(float)
    keep = (ion >= 1) & (ion <= ion_max) & (wl > wl_lo) & (wl < wl_hi) & (gA > gA_min)
    return dict(ion=ion[keep], wl=wl[keep], gA=gA[keep], Ei=Ei[keep],
                Ek=Ek[keep], gi=gi[keep], gk=gk[keep])


def strength(gA, Ek, T=T_RANK):
    return gA * np.exp(-Ek / (KB * T))


def element_anchor_lines(db, element, dom=3.0, win=0.18, pct=55, wl_max=620.0):
    """Isolated, strong element I/II lines usable as wavelength anchors.

    Same dominance criterion as fe_anchor_lines() but generalised to any
    element: keep lines whose Boltzmann strength dominates their +-win nm
    neighbourhood (over the bright db subset of ALL elements) by ``dom`` x, so
    an observed blend centroid cannot masquerade as a resolved anchor.
    """
    wl_all, gA_all, Ek_all, ion_all, el_all = db_lines_all(db, ion_max=2, gA_min=1e5)
    s_all = strength(gA_all, Ek_all)
    order = np.argsort(wl_all)
    wl_all, s_all, el_all = wl_all[order], s_all[order], el_all[order]
    mine = el_all == element
    if not mine.any():
        return []
    thr = np.percentile(s_all[mine], pct)
    anchors = []
    idx = np.where(mine & (s_all >= thr) & (wl_all < wl_max))[0]
    for i in idx:
        lo = np.searchsorted(wl_all, wl_all[i] - win)
        hi = np.searchsorted(wl_all, wl_all[i] + win)
        nb = np.concatenate([s_all[lo:i], s_all[i + 1:hi]])
        if nb.size == 0 or s_all[i] >= dom * np.max(nb):
            anchors.append(float(wl_all[i]))
    anchors.sort()
    out = []
    for a in anchors:
        if out and a - out[-1] < 0.06:
            continue
        out.append(a)
    return out


def estimate_shift_model(grand_wl, grand_y, db, element, snr_min=6.0):
    """Per-segment linear shift model (observed = db + shift) for one element.

    Anchors: isolated strong element I/II lines (UV, VIS) + isolated Ar I
    (NIR), sub-pixel parabolic centroid, robust MAD outlier rejection, then a
    per-segment slope when >=4 clean anchors remain (else the segment median,
    else the pooled global median).
    """
    anc = element_anchor_lines(db, element)
    raw = []   # (obs, db, seg, offset, species)
    for wl_db in anc:
        r = subpixel_center(grand_wl, grand_y, wl_db, snr_min, tol=0.20)
        if r:
            raw.append((r[0], wl_db, int(seg_index(wl_db)), r[0] - wl_db, element))
    ar = []
    for wl_db in AR_NIR_ANCHORS:
        r = subpixel_center(grand_wl, grand_y, wl_db, 4.0, tol=0.20)
        if r:
            ar.append((r[0], wl_db, int(seg_index(wl_db)), r[0] - wl_db, "ArI"))
    raw += ar
    # The wavelength shift is an INSTRUMENT property (element-independent).  The
    # ambient Ar I lines are textbook wavelength standards and give a clean seed
    # (~-155 pm here); dense-spectrum element anchors that land on a neighbour
    # peak scatter far from it.  Seed on Ar (fall back to the element-anchor
    # median) and reject any anchor > ANCHOR_CLIP from the seed.
    ANCHOR_CLIP = 0.13
    if ar:
        seed = float(np.median([m[3] for m in ar]))
    else:
        seed = float(np.median([m[3] for m in raw])) if raw else 0.0
    matches = [m for m in raw if abs(m[3] - seed) < ANCHOR_CLIP]
    global_shift = float(np.median([m[3] for m in matches])) if matches else seed
    model_seed = seed
    lam0_def = (270.0, 490.0, 760.0)
    model = {}
    for s in range(3):
        seg_m = [m for m in matches if m[2] == s]
        n = len(seg_m)
        lam0 = lam0_def[s]
        if n >= 3:
            xo = np.array([m[0] for m in seg_m])
            resid = np.array([m[3] for m in seg_m])
            lam0 = float(xo.mean())
            med = float(np.median(resid))
            if n >= 4:
                b, a = np.polyfit(xo - lam0, resid, 1)
            else:
                b, a = 0.0, med
            fit = a + b * (xo - lam0)
            rstd = float(np.std(resid - fit))
            model[s] = dict(a=float(a), b=float(b), lam0=lam0, n=n,
                            resid_std=rstd, med=med, own=True)
        else:
            model[s] = dict(a=global_shift, b=0.0, lam0=lam0, n=n,
                            resid_std=float("nan"), med=global_shift, own=False)
    model["global"] = global_shift
    model["matches"] = matches
    model["ar_seed"] = model_seed
    model["n_ar"] = len(ar)
    model["n_raw"] = len(raw)
    scatter = float(np.std([m[3] for m in matches])) if matches else float("nan")
    model["scatter"] = scatter
    return model


def db_to_obs(model, wl_db):
    m = model[int(seg_index(wl_db))]
    return wl_db + m["a"] + m["b"] * (wl_db - m["lam0"])


def obs_to_db(model, wl_obs):
    m = model[int(seg_index(wl_obs))]
    return wl_obs - (m["a"] + m["b"] * (wl_obs - m["lam0"]))


# =============================================================================
# Gaussian sub-pixel FWHM
# =============================================================================
def _gauss(x, A, mu, sig, c):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sig ** 2)) + c


def measure_fwhm(wl, y, center, snr_min=10.0, half=3):
    """Baseline-subtracted Gaussian FWHM over peak +-``half`` native samples.

    Returns dict(fwhm_nm, fwhm_px, sig, center, snr, area, height, pitch) or
    None.  The peak is the local max within +-2 pitches of ``center``; the
    baseline is the measure_line side-band model; FWHM = 2.3548*sigma from a
    4-parameter Gaussian fit to the 2*half+1 central samples.
    """
    m = measure_line(wl, y, center)
    if not m or m["snr"] < snr_min or m["height"] <= 0:
        return None
    j = int(np.searchsorted(wl, m["peak_nm"]))
    j = min(max(j, half), len(wl) - half - 1)
    sl = slice(j - half, j + half + 1)
    xs = wl[sl].astype(float)
    ys = y[sl].astype(float)
    pitch = float(np.median(np.diff(xs)))
    A0 = float(ys.max() - ys.min())
    p0 = [max(A0, 1.0), float(wl[j]), pitch, float(ys.min())]
    try:
        popt, _ = curve_fit(_gauss, xs, ys, p0=p0, maxfev=10000,
                            bounds=([0.0, xs[0], 0.2 * pitch, -np.inf],
                                    [np.inf, xs[-1], 6.0 * pitch, np.inf]))
    except Exception:
        return None
    A, mu, sig, c = popt
    fwhm_nm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * abs(sig)
    if not np.isfinite(fwhm_nm) or fwhm_nm <= 0:
        return None
    # reject unresolved noise spikes (< 0.7 px) and gross blends (> 8 px)
    if not (0.7 <= fwhm_nm / pitch <= 8.0):
        return None
    return dict(fwhm_nm=float(fwhm_nm), fwhm_px=float(fwhm_nm / pitch),
                sig=float(abs(sig)), center=float(mu), snr=float(m["snr"]),
                area=float(m["area"]), height=float(m["height"]), pitch=pitch)


def fwhm_fast(wl, y, center, snr_min=10.0):
    """Fast analytic Gaussian FWHM: parabola fit to ln(counts) over the peak
    +-1 native sample (a Gaussian is exactly a parabola in log space).  Used for
    the per-run delay TREND where thousands of fits are needed; cross-checked
    against the full +-3 curve_fit measure_fwhm() on the best-run tables."""
    m = measure_line(wl, y, center)
    if not m or m["snr"] < snr_min or m["height"] <= 0:
        return None
    j = int(np.searchsorted(wl, m["peak_nm"]))
    j = min(max(j, 1), len(wl) - 2)
    bg = m["bg"] if "bg" in m else 0.0
    y1, y2, y3 = y[j - 1] - bg, y[j] - bg, y[j + 1] - bg
    if min(y1, y2, y3) <= 0:
        return None
    d = math.log(y1) - 2 * math.log(y2) + math.log(y3)
    if d >= 0:
        return None
    pitch = float((wl[j + 1] - wl[j - 1]) / 2.0)
    sig = pitch / math.sqrt(-d)
    fwhm_nm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sig
    return dict(fwhm_nm=float(fwhm_nm), snr=float(m["snr"]),
                center=float(m["peak_nm"]), seg=segn(center))


def doppler_fwhm(wl, element, T=T_RANK):
    return DOPPLER_C * wl * math.sqrt(T / MASS[element])


# =============================================================================
# Isolation helper (element I/II bright neighbourhood)
# =============================================================================
def iso_mask(L, iso_win=0.20, iso_frac=0.20):
    """Vectorised isolation: True where a line dominates its +-iso_win nm
    neighbourhood among element I/II lines (sum of others' Boltzmann strength <
    iso_frac of its own).  Computed once per element (db-only, O(N log N))."""
    wl = L["wl"]
    s = strength(L["gA"], L["Ek"])
    order = np.argsort(wl)
    wls, ss = wl[order], s[order]
    cs = np.concatenate([[0.0], np.cumsum(ss)])
    lo = np.searchsorted(wls, wls - iso_win, "left")
    hi = np.searchsorted(wls, wls + iso_win, "right")
    near_sum = (cs[hi] - cs[lo]) - ss          # exclude self
    keep_sorted = (ss > 0) & (near_sum < iso_frac * ss)
    out = np.empty_like(keep_sorted)
    out[order] = keep_sorted
    return out


# =============================================================================
# SECTION 1: line matching / completeness
# =============================================================================
def classify_peaks(db, element, model, wl, y, snr_min=5.0):
    peaks = find_peaks_simple(wl, y, snr_min=snr_min)
    L = el_lines(db, element, ion_max=2)
    # bright db of all 92 elements for "nearest any-element" + contaminant check
    wl_all, gA_all, Ek_all, ion_all, el_all = db_lines_all(db, ion_max=3, gA_min=1e4)
    s_all = strength(gA_all, Ek_all)
    out = []
    for wl_obs, m in peaks:
        wl_db = obs_to_db(model, wl_obs)
        near = np.abs(L["wl"] - wl_db) < MATCH_TOL_NM
        cls = None
        info = {}
        if near.any():
            s = strength(L["gA"][near], L["Ek"][near])
            wln = L["wl"][near]
            ionn = L["ion"][near]
            order = np.argsort(-s)
            s, wln, ionn = s[order], wln[order], ionn[order]
            top = s[0]
            nblend = int((s >= BLEND_FRAC * top).sum())
            cls = "blend" if nblend >= 2 else "unique"
            info = dict(db_wl=float(wln[0]), ion=int(ionn[0]),
                        offset=float(wl_db - wln[0]), ncand=int(near.sum()),
                        nblend=nblend)
        else:
            # contaminant?  match ONLY an explicit textbook ambient line.
            cand = None
            for el in CONTAM:
                offs = [abs(wl_db - a) for a in KNOWN_AMBIENT[el]]
                jm = int(np.argmin(offs))
                if offs[jm] < MATCH_TOL_NM:
                    cand = (el, KNOWN_AMBIENT[el][jm], float(wl_db - KNOWN_AMBIENT[el][jm]))
                    break
            if cand:
                cls = "contam"
                info = dict(el=cand[0], db_wl=cand[1], offset=cand[2])
            else:
                cls = "unmatched"
                k = int(np.argmin(np.abs(wl_all - wl_db)))
                info = dict(near_el=str(el_all[k]), near_wl=float(wl_all[k]),
                            near_ion=int(ion_all[k]),
                            offset=float(wl_db - wl_all[k]))
        out.append(dict(wl_obs=float(wl_obs), snr=float(m["snr"]),
                        seg=segn(wl_obs), cls=cls, **info))
    # partner-line confirmation: Ar is credited only as a consistent SET (>=2 of
    # its known lines detected among the peaks); H/O/N single textbook lines are
    # credited as weak traces (confirmed, flagged trace).
    peak_db = [obs_to_db(model, p["wl_obs"]) for p in out]
    ar_hits = 0
    for el in ("Ar",):
        for a in KNOWN_AMBIENT[el]:
            if any(abs(pk - a) < MATCH_TOL_NM for pk in peak_db):
                ar_hits += 1
    for p in out:
        if p["cls"] != "contam":
            continue
        if p["el"] == "Ar":
            p["confirmed"] = ar_hits >= 2
            p["trace"] = False
        else:
            p["confirmed"] = True
            p["trace"] = True
    return out


def reverse_completeness(db, element, model, wl, y, target_n=200, snr_min=5.0):
    """Of the ~target_n strongest element I/II db lines in the observed range,
    what fraction is detected (a peak within tolerance at the predicted obs
    wavelength)?  Returns (threshold, n_lines, n_detected, rows)."""
    L = el_lines(db, element, ion_max=2)
    s = strength(L["gA"], L["Ek"])
    order = np.argsort(-s)
    thr = float(s[order][min(target_n, len(order)) - 1])
    keep = np.where(s >= thr)[0]
    peaks = find_peaks_simple(wl, y, snr_min=snr_min)
    pk_obs = np.array([p[0] for p in peaks])
    tol = MATCH_TOL_NM
    rows = []
    ndet = 0
    for i in keep:
        pred = db_to_obs(model, L["wl"][i])
        det = bool(pk_obs.size and np.min(np.abs(pk_obs - pred)) < tol)
        # only count as detected if the line is measurable (isolated OR simply
        # a peak lands there); we also require the local measurement SNR>=5
        if det:
            mm = measure_line(wl, y, pred)
            det = bool(mm and mm["snr"] >= snr_min)
        ndet += int(det)
        rows.append((float(L["wl"][i]), int(L["ion"][i]), float(s[i]), det))
    return thr, len(keep), ndet, rows


# =============================================================================
# SECTION 2a: multiplet ratio -> tau
# =============================================================================
def multiplet_pairs(L, iso, model, wl, y, snr_min=12.0, max_sep=8.0,
                    r0_min=1.2, r0_max=4.0, sr_lo=0.3, sr_hi=4.0, level_tol=1e-3):
    """Same-upper-level (branching) or same-lower-level (resonance) pairs of
    isolated strong element lines with COMPARABLE strength.

    Cuts: same ion, same segment, 0.25<|Δλ|<max_sep nm, exact shared level
    (|ΔE|<level_tol eV), thin ratio R₀∈[r0_min,r0_max], absorption-strength
    ratio sr∈[sr_lo,sr_hi], both SNR≥snr_min.  Self-absorption can only push the
    measured ratio R toward 1 from above, so pairs with R<1 (the nominally
    stronger member measured weaker — a blended/forest-contaminated weak line)
    are rejected and counted.  Returns (pairs, n_rejected_Rlt1)."""
    n = len(L["wl"])
    cand = [i for i in range(n) if iso[i] and strength(L["gA"][i], L["Ek"][i]) > 0]
    def eps(i):
        return (L["gA"][i] / L["wl"][i]) * math.exp(-L["Ek"][i] / KT_RANK)
    def kap(i):
        return L["gA"][i] * L["wl"][i] ** 2 * math.exp(-L["Ei"][i] / KT_RANK)
    pairs = []
    seen = set()
    n_rej = 0
    for a in cand:
        for b in cand:
            if a >= b:
                continue
            if int(L["ion"][a]) != int(L["ion"][b]):
                continue
            if segn(L["wl"][a]) != segn(L["wl"][b]):
                continue
            dl = abs(L["wl"][a] - L["wl"][b])
            if dl > max_sep or dl < 0.25:
                continue
            same_upper = abs(L["Ek"][a] - L["Ek"][b]) < level_tol
            same_lower = abs(L["Ei"][a] - L["Ei"][b]) < level_tol
            if not (same_upper or same_lower):
                continue
            kind = "upper" if same_upper else "lower"
            s_idx, w_idx = (a, b) if eps(a) >= eps(b) else (b, a)
            R0 = eps(s_idx) / eps(w_idx)
            if not (r0_min <= R0 <= r0_max):
                continue
            sr = kap(s_idx) / kap(w_idx)
            if not (sr_lo <= sr <= sr_hi):
                continue
            key = (round(L["wl"][s_idx], 2), round(L["wl"][w_idx], 2))
            if key in seen:
                continue
            ms = measure_line(wl, y, db_to_obs(model, L["wl"][s_idx]))
            mw = measure_line(wl, y, db_to_obs(model, L["wl"][w_idx]))
            if not ms or not mw or ms["snr"] < snr_min or mw["snr"] < snr_min:
                continue
            if ms["area"] <= 0 or mw["area"] <= 0:
                continue
            R = ms["area"] / mw["area"]
            seen.add(key)
            if R < 1.0:
                n_rej += 1
                continue
            tau_w = invert_doublet_tau(R, R0, strength_ratio=sr)
            tau_s = sr * tau_w
            corr = 1.0 / float(escape_factor(tau_s))
            pairs.append(dict(
                kind=kind, ion=int(L["ion"][s_idx]),
                lam_s=float(L["wl"][s_idx]), lam_w=float(L["wl"][w_idx]),
                gA_s=float(L["gA"][s_idx]), gA_w=float(L["gA"][w_idx]),
                R0=float(R0), sr=float(sr), R=float(R),
                tau_w=float(tau_w), tau_s=float(tau_s), corr=float(corr),
                snr_s=float(ms["snr"]), snr_w=float(mw["snr"]),
                seg=segn(L["wl"][s_idx])))
    pairs.sort(key=lambda p: (-p["tau_s"]))
    return pairs, n_rej


# =============================================================================
# SECTION 2b: Boltzmann residuals vs lower-level population proxy
# =============================================================================
def boltzmann_residuals(L, iso, stage, model, wl, y, wl_lo, wl_hi,
                        ei_min=0.0, snr_min=8.0, ei_ref=0.8, clip=1.8,
                        min_ref=8):
    """Build a robust Boltzmann REFERENCE from non-resonance (Ei>=ei_ref)
    isolated stage lines (iterative worst-|residual| rejection beyond clip*σ),
    then measure every isolated stage line's deficit below that reference.

    A ground-state resonance / low-Ei line that sits systematically below the
    reference line is self-absorbed; residual r (ln units) => intensity loss
    1-e^r.  The reference deliberately excludes the resonance lines so the SA is
    read as a deficit, not folded into the slope."""
    def collect(mask):
        pts = []
        for i in np.where(mask)[0]:
            mm = measure_line(wl, y, db_to_obs(model, L["wl"][i]))
            if not mm or mm["snr"] < snr_min or mm["area"] <= 0:
                continue
            pts.append(dict(Ek=float(L["Ek"][i]), Ei=float(L["Ei"][i]),
                            gi=float(L["gi"][i]), gk=float(L["gk"][i]),
                            gA=float(L["gA"][i]), wl=float(L["wl"][i]),
                            area=float(mm["area"]), snr=float(mm["snr"]),
                            yv=math.log(mm["area"] * L["wl"][i] / L["gA"][i])))
        return pts
    base = (L["ion"] == stage) & (L["wl"] >= wl_lo) & (L["wl"] < wl_hi) & (L["gA"] > 0) & iso
    ref_pts = collect(base & (L["Ei"] >= ei_ref))
    all_pts = collect(base)
    if len(ref_pts) < min_ref:
        return None
    P = np.array([[p["Ek"], p["yv"]] for p in ref_pts])
    keep = np.ones(len(P), bool)
    for _ in range(40):
        x, yv = P[keep, 0], P[keep, 1]
        sl, ic = np.polyfit(x, yv, 1)
        r = P[:, 1] - (sl * P[:, 0] + ic)
        rms = r[keep].std()
        worst = np.argmax(np.where(keep, np.abs(r), -1))
        if keep.sum() > min_ref and abs(r[worst]) > clip * rms:
            keep[worst] = False
            continue
        break
    T = -1.0 / (KB * sl) if sl < 0 else float("nan")
    kt = KB * (T if np.isfinite(T) else T_RANK)
    for p in all_pts:
        p["resid"] = float(p["yv"] - (sl * p["Ek"] + ic))
        p["pop_proxy"] = float(math.log(p["gi"]) - p["Ei"] / kt)
    xr = P[:, 0]
    return dict(pts=all_pts, ref_pts=ref_pts, slope=float(sl), intr=float(ic),
                T=float(T), span=float(xr.max() - xr.min()),
                n=len(all_pts), n_ref=int(keep.sum()),
                rms=float(r[keep].std()))


# =============================================================================
# Utility: measure FWHM set on a spectrum for isolated strong lines
# =============================================================================
def fwhm_set(L, iso, model, wl, y, snr_min=10.0, only_seg=None):
    """FWHM for isolated strong element I/II lines (SNR>=snr_min).

    ``only_seg`` restricts to one detector segment (fewer Gaussian fits)."""
    rows = []
    cand = iso & (strength(L["gA"], L["Ek"]) > 0)
    if only_seg is not None:
        cand = cand & (np.array([segn(w) for w in L["wl"]]) == only_seg)
    idx = np.where(cand)[0]
    for i in idx:
        f = measure_fwhm(wl, y, db_to_obs(model, L["wl"][i]), snr_min=snr_min)
        if not f:
            continue
        if abs(f["center"] - db_to_obs(model, L["wl"][i])) > 0.20:
            continue
        rows.append(dict(lam=float(L["wl"][i]), ion=int(L["ion"][i]),
                         Ei=float(L["Ei"][i]), Ek=float(L["Ek"][i]),
                         gA=float(L["gA"][i]), seg=segn(L["wl"][i]), **f))
    return rows


def ar_nir_fwhm(wl, y, snr_min=8.0):
    rows = []
    for lam in AR_NIR_ANCHORS:
        f = measure_fwhm(wl, y, lam, snr_min=snr_min)
        if f:
            rows.append(dict(lam=lam, **f))
    return rows


def instrumental_fwhm(rows, pct=15.0):
    """Lower-envelope instrumental FWHM per segment (pct-th percentile)."""
    out = {}
    for seg in SEG_NAMES:
        vals = [r["fwhm_nm"] for r in rows if r["seg"] == seg]
        if vals:
            out[seg] = dict(nm=float(np.percentile(vals, pct)),
                            px=float(np.percentile(vals, pct) / PITCH[seg]),
                            n=len(vals),
                            median=float(np.median(vals)))
    return out


# =============================================================================
# MAIN
# =============================================================================
def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ledger-fe", required=True)
    ap.add_argument("--ledger-v", required=True)
    ap.add_argument("--dbpath", default=os.path.join(REPO, "db"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "reports", "2026-09-22-line-matching-sa-broadening.md"))
    ap.add_argument("--figdir", default=os.path.join(
        REPO, "reports", "figures", "line-physics-20260922"))
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    db = Database(args.dbpath)
    fe_runs = load_runs(args.data, args.ledger_fe)
    v_runs = load_runs(args.data, args.ledger_v)

    # ---- best fresh-site runs & fresh-run pools ---------------------------
    FE_FRESH = ["run-07c25af6", "run-6cd149cc", "run-2e03f73d", "run-6ce8e9b3",
                "run-468a150c", "run-90143c6b", "run-74c93327"]
    fe_best = run_by_prefix(fe_runs, "run-6cd149cc")     # 5/10 fresh, 10 shots
    fe_fresh_runs = [run_by_prefix(fe_runs, p) for p in FE_FRESH]

    v_best = run_by_prefix(v_runs, "seq-40d5b20a076e")   # highest-score 10/10
    # V "all fresh runs": every seq run is a fresh raster site (12 sites, laps)
    v_fresh_runs = list(v_runs)

    metals = {}
    metals["Fe"] = dict(
        best=fe_best, best_name=fe_best.meta["run"], best_cond="5/10",
        fresh=fe_fresh_runs,
        best_wl_y=pool_mean([fe_best]),
        fresh_wl_y=pool_mean(fe_fresh_runs))
    metals["V"] = dict(
        best=v_best, best_name=v_best.meta["run"], best_cond="10/10",
        fresh=v_fresh_runs,
        best_wl_y=pool_mean([v_best]),
        fresh_wl_y=pool_mean(v_fresh_runs))

    # ---- shift models + precomputed line tables/isolation -----------------
    for el, M in metals.items():
        wl, y = M["fresh_wl_y"]
        M["model"] = estimate_shift_model(wl, y, db, el)
        M["L"] = el_lines(db, el, ion_max=2)
        M["iso"] = iso_mask(M["L"], iso_win=0.20, iso_frac=0.15)
        M["fwhm_best"] = fwhm_set(M["L"], M["iso"], M["model"],
                                  *M["best_wl_y"], snr_min=10.0)

    report = []
    P = report.append
    P("# Line matching, self-absorption, and broadening — Fe (99.98%) and V (pure) Z300 LIBS\n")
    P(f"*Generated by `scripts/line_physics_analysis.py` — deterministic rerun of "
      f"`--data {args.data} --ledger-fe {args.ledger_fe} --ledger-v {args.ledger_v}`.*\n")

    # ---------------- DATA -------------------------------------------------
    P("## Data\n")
    P(f"- Spectra: native single-shot CSVs, 7,914 samples, 186–961 nm, three "
      f"spectrometer segments UV [186,365)/VIS [365,620)/NIR [620,948] nm "
      f"(pitch {PITCH['UV']}/{PITCH['VIS']}/{PITCH['NIR']} nm), 12.3 nm NIR gap "
      f"947.9→960.2 nm. Baselines can be negative. **Native grid only — no "
      f"resampling.** Means are per-pixel-index averages of the per-run means "
      f"(all runs share the 7,914-pixel grid to <0.043 nm = <0.5 px, so pixel "
      f"index is the physical detector element); the wavelength axis is the "
      f"reference run's own grid.\n")
    P(f"- **Fe best fresh run**: `{fe_best.meta['run']}` (5/10, {fe_best.shots.shape[0]} "
      f"shots, site {fe_best.meta['location']}). **Fe fresh pool** ("
      f"{len(fe_fresh_runs)} runs): {', '.join('`'+r.meta['run'][:14]+'`' for r in fe_fresh_runs)} "
      f"({sum(r.shots.shape[0] for r in fe_fresh_runs)} shots).\n")
    P(f"- **V best fresh run**: `{v_best.meta['run']}` (10/10, score "
      f"{v_best.meta['score']:.4f}, {v_best.shots.shape[0]} shots). **V fresh pool** "
      f"(all {len(v_fresh_runs)} seq runs, 12 raster sites × 3 laps × 3×3 grid; "
      f"{sum(r.shots.shape[0] for r in v_fresh_runs)} shots).\n")

    # ---------------- SHIFT MODELS ----------------------------------------
    P("## Shift models (observed = db(air) + shift)\n")
    P("Per-segment linear fit `obs−db = a + b·(λ−λ0)` from isolated strong "
      "sample-element I/II anchors (UV, VIS) + isolated Ar I lines (NIR), "
      "sub-pixel parabolic centroid, on the fresh pool. The wavelength shift is "
      "an INSTRUMENT property (element-independent); the ambient Ar I lines are "
      "textbook wavelength standards and give a clean seed (Fe pool −152 pm, V "
      "pool −170 pm ≈ the known native-API −154 pm), so element anchors farther "
      "than 0.13 nm from the Ar seed (dense-spectrum centroids that snapped to a "
      "neighbour) are rejected. Both metals land on ~−155 pm, confirming the "
      "shift is element-independent.\n")
    P("NB: this instrument shift aligns Fe well (see §1.4) but leaves a "
      "SEGMENT-DEPENDENT residual for V — strong isolated V VIS lines sit ~150 pm "
      "further blue of their db position than Ar does, i.e. the V I/II line "
      "list's air wavelengths carry an extra ~150 pm offset/scatter (a db "
      "property, quantified in §1.4). It is NOT re-absorbed into the shift here, "
      "so V's raw ±0.15 nm completeness is conservative.\n")
    P("| metal | segment | n anchors | a (pm) | b (pm/nm) | resid σ (pm) | own fit |")
    P("|--|--|--|--|--|--|--|")
    for el, M in metals.items():
        m = M["model"]
        for s, name in enumerate(SEG_NAMES):
            d = m[s]
            P(f"| {el} | {name} | {d['n']} | {1000*d['a']:+.0f} | "
              f"{1000*d['b']:+.1f} | {fmt(1000*d['resid_std'],0)} | "
              f"{'yes' if d['own'] else 'global'} |")
        P(f"| {el} | global | {len(m['matches'])} | {1000*m['global']:+.0f} | "
          f"— | scatter {1000*m['scatter']:.0f} | — |")
    P("")

    # =================== SECTION 1 =======================================
    P("## 1. Do all observed lines match the database?\n")
    sec1 = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        rows = classify_peaks(db, el, M["model"], wl, y, snr_min=5.0)
        sec1[el] = rows
    P("### 1.1 Peak classification (best fresh run, SNR≥5)\n")
    P(f"Tolerance ±{MATCH_TOL_NM} nm (≈ 1–2 native pitches; shift residual σ "
      "below). Candidates ranked by gA·exp(−Eₖ/kT), T=9000 K; matched-blend = "
      f"≥2 sample-element candidates within {int(BLEND_FRAC*100)}% of the top "
      "strength. A contaminant is credited only when the peak matches an explicit "
      "textbook ambient line (Ar set, Hα, O I / N I triplets — not merely the "
      "nearest of the hundreds of O/N db lines); Ar needs ≥2 of its set detected "
      "to be a confirmed set (that is the `conf` count), H/O/N single lines are "
      "counted as weak traces.\n")
    P("| metal | seg | peaks | unique | blend | contam(conf) | unmatched | matched frac |")
    P("|--|--|--|--|--|--|--|--|")
    sec1_tot = {}
    for el in metals:
        rows = sec1[el]
        for seg in list(SEG_NAMES) + ["ALL"]:
            rs = rows if seg == "ALL" else [r for r in rows if r["seg"] == seg]
            if not rs:
                continue
            nu = sum(r["cls"] == "unique" for r in rs)
            nb = sum(r["cls"] == "blend" for r in rs)
            nc = sum(r["cls"] == "contam" for r in rs)
            ncc = sum(r["cls"] == "contam" and r.get("confirmed") for r in rs)
            nun = sum(r["cls"] == "unmatched" for r in rs)
            matched = nu + nb + ncc
            frac = matched / len(rs) if rs else float("nan")
            P(f"| {el} | {seg} | {len(rs)} | {nu} | {nb} | {nc}({ncc}) | {nun} | {frac:.3f} |")
            if seg == "ALL":
                sec1_tot[el] = dict(n=len(rs), unique=nu, blend=nb, contam=nc,
                                    contam_conf=ncc, unmatched=nun, matched=matched,
                                    frac=frac)
    P("")
    # SNR>=10 summary
    P("At SNR≥10 (subset of the above):\n")
    P("| metal | peaks | matched(sample+contam) | unmatched | matched frac |")
    P("|--|--|--|--|--|")
    for el in metals:
        rs = [r for r in sec1[el] if r["snr"] >= 10]
        matched = sum(r["cls"] in ("unique", "blend") or (r["cls"] == "contam" and r.get("confirmed")) for r in rs)
        nun = sum(r["cls"] == "unmatched" for r in rs)
        P(f"| {el} | {len(rs)} | {matched} | {nun} | {matched/len(rs):.3f} |")
    P("")
    # unmatched list
    P("### 1.2 Unmatched peaks (best fresh run, SNR≥5)\n")
    for el in metals:
        us = [r for r in sec1[el] if r["cls"] == "unmatched"]
        P(f"**{el}** — {len(us)} unmatched:")
        if not us:
            P("- (none)\n"); continue
        P("| λ_obs (nm) | seg | SNR | nearest db line (any el) | offset (nm) |")
        P("|--|--|--|--|--|")
        for r in sorted(us, key=lambda z: -z["snr"])[:40]:
            P(f"| {r['wl_obs']:.3f} | {r['seg']} | {r['snr']:.1f} | "
              f"{r['near_el']} {r['near_ion']} {r['near_wl']:.3f} | {r['offset']:+.3f} |")
        P("")
    # reverse completeness
    P("### 1.3 Reverse completeness (are the predicted-strong db lines detected?)\n")
    P("| metal | strength threshold | db lines ≥thr | detected (SNR≥5, ±0.15 nm) | fraction |")
    P("|--|--|--|--|--|")
    sec1_rev = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        thr, nl, nd, rows = reverse_completeness(db, el, M["model"], wl, y)
        sec1_rev[el] = dict(thr=thr, nl=nl, nd=nd, frac=nd/nl)
        P(f"| {el} | {thr:.3e} | {nl} | {nd} | {nd/nl:.3f} |")
    P("")

    # ---- 1.4 matched fraction vs tolerance (db wavelength accuracy) --------
    P("### 1.4 Matched fraction vs matching tolerance (db wavelength accuracy)\n")
    P("Fraction of peaks with ANY sample-element I/II db line within ±tol of the "
      "shift-corrected position. If a metal needs a wider tolerance to reach the "
      "same fraction, its db air wavelengths are less accurate (not that lines "
      "are missing).\n")
    tols = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    P("| metal | " + " | ".join(f"±{t:.2f} nm" for t in tols) + " |")
    P("|--" * (len(tols) + 1) + "|")
    sec1_tol = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        L = M["L"]
        peaks = find_peaks_simple(wl, y, snr_min=5.0)
        dbimp = np.array([obs_to_db(M["model"], p) for p, _ in peaks])
        fr = []
        for t in tols:
            nm = sum(np.min(np.abs(L["wl"] - d)) < t for d in dbimp)
            fr.append(nm / len(dbimp))
        sec1_tol[el] = dict(zip(tols, fr))
        P(f"| {el} | " + " | ".join(f"{f:.2f}" for f in fr) + " |")
    P("")
    # verdict
    fe15, v15 = sec1_tol["Fe"][0.15], sec1_tol["V"][0.15]
    fe40, v40 = sec1_tol["Fe"][0.40], sec1_tol["V"][0.40]
    P("**Verdict — do all observed lines match?** ")
    P(f"To ±0.15 nm, {fe15*100:.0f}% of Fe peaks and {v15*100:.0f}% of V peaks "
      f"match a sample-element line; at ±0.40 nm this rises to {fe40*100:.0f}% "
      f"(Fe) and {v40*100:.0f}% (V). The Fe curve is nearly saturated by "
      "±0.15 nm (Fe db air wavelengths are accurate to ≲1 pitch), whereas V only "
      "saturates near ±0.40 nm — the V I/II line list's air wavelengths in this "
      "db scatter ~2–3× more than Fe's. So, once db wavelength accuracy is "
      "accounted for, essentially all strong observed lines DO match the sample "
      "element (plus ambient Ar/O/N/H and, in a 99.98% Fe / pure-V matrix, no "
      "established bulk impurity); the residual unmatched at ±0.15 nm is "
      "dominated by db wavelength error (V) and NIR ambient bands, not by "
      "unidentified emission. Confidence: high for Fe UV/VIS, moderate for V "
      "(db-limited) and for both NIR (ambient-dominated, weak shift constraint).\n")

    # =================== SECTION 2 =======================================
    P("## 2. How much self-absorption?\n")
    # (a) multiplet ratios
    P("### 2a. Multiplet / doublet ratio test → τ\n")
    P("Isolated same-upper-level (branching, T-independent) or same-lower-level "
      "(resonance) pairs of COMPARABLE strength: same ion, same segment, "
      "0.25<|Δλ|<8 nm, exact shared level, thin ratio R₀∈[1.2,4], "
      "absorption-strength ratio sr∈[0.3,4], both SNR≥12. Thin emission ratio "
      "R₀ ∝ (gA/λ)·e^(−Eₖ/kT); optical-depth ratio sr ∝ gA·λ²·e^(−Eᵢ/kT). τ from "
      "`invert_doublet_tau(R, R₀, sr)` (escape factor (1−e^−τ)/τ). Deviation of "
      "the measured ratio R toward 1 = optical depth. SA can only push R toward "
      "1 from above, so pairs with R<1 (a forest-contaminated weak member) are "
      "rejected and counted.\n")
    sec2a = {}
    sec2a_rej = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        pairs, n_rej = multiplet_pairs(M["L"], M["iso"], M["model"], wl, y)
        sec2a[el] = pairs
        sec2a_rej[el] = n_rej
        P(f"**{el}** (best fresh run) — {len(pairs)} usable pairs "
          f"({n_rej} rejected for R<1 = forest-contaminated weak member):")
        if not pairs:
            P("- (none passed the isolation/SNR cuts)\n"); continue
        P("| kind | ion | λ_strong | λ_weak | seg | R₀(thin) | R(meas) | τ_strong | SA corr (strong) |")
        P("|--|--|--|--|--|--|--|--|--|")
        for p in pairs[:14]:
            P(f"| {p['kind']} | {p['ion']} | {p['lam_s']:.3f} | {p['lam_w']:.3f} | "
              f"{p['seg']} | {p['R0']:.2f} | {p['R']:.2f} | {p['tau_s']:.2f} | "
              f"{p['corr']:.2f}× |")
        P("")

    # (b) Boltzmann residuals
    P("### 2b. Boltzmann deficit of the resonance lines\n")
    P("A robust Boltzmann reference ln(area·λ/gA) vs Eₖ is fit over non-resonance "
      "isolated stage lines (Eᵢ≥0.8 eV, iterative 1.8σ rejection). Its rms is the "
      "single-line noise floor (db gA error + blends + baseline); only a deficit "
      "well above it is significant. The resonance / low-Eᵢ lines (Eᵢ<0.6 eV) are "
      "then read against it: a systematic deficit r (ln units, loss 1−eʳ) is "
      "self-absorption. (Non-resonance outliers with huge deficits are "
      "mismeasured lines, not SA, and are excluded.)\n")
    sec2b = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        stage = 1
        lo, hi = (360.0, 560.0) if el == "Fe" else (380.0, 520.0)
        r = boltzmann_residuals(M["L"], M["iso"], stage, M["model"], wl, y, lo, hi)
        sec2b[el] = r
        if not r:
            P(f"**{el}**: too few isolated reference lines for a residual fit.\n"); continue
        signif = 2.0 * r["rms"]
        P(f"**{el} {stage}** ({lo:.0f}–{hi:.0f} nm): reference from {r['n_ref']} "
          f"non-resonance lines, noise floor rms {r['rms']:.2f} ln "
          f"(significance threshold |r|>{signif:.1f}); reference slope-T "
          f"{fmt(r['T'],0)} K is ill-constrained by this scatter and is not used "
          "as a thermometer. Resonance / low-Eᵢ lines (Eᵢ<0.6 eV):")
        P("| λ (nm) | Eᵢ (eV) | Eₖ (eV) | gA | SNR | residual (ln) | intensity loss | signif? |")
        P("|--|--|--|--|--|--|--|--|")
        res = [p for p in r["pts"] if p["Ei"] < 0.6 and p["snr"] >= 8]
        for p in sorted(res, key=lambda z: z["resid"])[:10]:
            loss = 1 - math.exp(p["resid"]) if p["resid"] < 0 else 0.0
            sig = "yes" if p["resid"] < -signif else "no"
            P(f"| {p['wl']:.3f} | {p['Ei']:.3f} | {p['Ek']:.3f} | {p['gA']:.2e} | "
              f"{p['snr']:.0f} | {p['resid']:+.2f} | {loss*100:.0f}% | {sig} |")
        ndef = sum(p["resid"] < -signif for p in res)
        r["n_low"] = len(res); r["n_low_def"] = ndef
        P(f"\n{el}: {ndef}/{len(res)} resonance/low-Eᵢ lines show a "
          f"self-absorption deficit above the {signif:.1f} ln noise floor.\n")

    # (c) width-intensity correlation
    P("### 2c. Width–intensity correlation\n")
    P("FWHM vs net area for isolated lines of one segment (Gaussian sub-pixel "
      "fit, §3). Self-absorbed lines are broader/flat-topped → positive slope; "
      "lines exceeding the instrumental width are flagged.\n")
    sec2c = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        rows = M["fwhm_best"]
        seg = "VIS"
        segrows = [r for r in rows if r["seg"] == seg]
        sec2c[el] = dict(rows=rows, seg=seg)
        if len(segrows) >= 4:
            a = np.array([r["area"] for r in segrows])
            f = np.array([r["fwhm_nm"] for r in segrows])
            la = np.log10(np.clip(a, 1e-9, None))
            sl, ic = np.polyfit(la, f, 1)
            sec2c[el]["slope"] = float(sl)
            sec2c[el]["n"] = len(segrows)
            P(f"**{el}** {seg}: n={len(segrows)}, FWHM vs log10(area) slope = "
              f"{sl*1000:+.1f} pm/decade.")
        else:
            P(f"**{el}** {seg}: too few clean lines ({len(segrows)}).")
    P("")

    # (d) delay dependence of multiplet ratio (V replicates)
    P("### 2d. Delay dependence of the multiplet ratio (column density proxy)\n")
    P("Strongest usable pair, area ratio R at delay 5/10/20. SA scales with "
      "column density; R closer to 1 = more optical depth.\n")
    sec2d = {}
    for el, M in metals.items():
        pairs = sec2a[el]
        if not pairs:
            P(f"**{el}**: no pair.\n"); continue
        p0 = pairs[0]
        by_delay = defaultdict(list)
        for r in M["fresh"]:
            by_delay[int(r.meta["delay"])].append(r)
        P(f"**{el}** pair λ{p0['lam_s']:.2f}/λ{p0['lam_w']:.2f} (R₀={p0['R0']:.2f}):")
        P("| delay (µs) | n runs | R (mean±sd) | τ_strong |")
        P("|--|--|--|--|")
        row = {}
        for d in sorted(by_delay):
            Rs = []
            for r in by_delay[d]:
                wl, yv = pool_mean([r])
                ms = measure_line(wl, yv, db_to_obs(M["model"], p0["lam_s"]))
                mw = measure_line(wl, yv, db_to_obs(M["model"], p0["lam_w"]))
                if ms and mw and ms["area"] > 0 and mw["area"] > 0 and min(ms["snr"], mw["snr"]) > 5:
                    Rs.append(ms["area"] / mw["area"])
            if not Rs:
                continue
            Rm = float(np.mean(Rs)); Rsd = float(np.std(Rs))
            tw = invert_doublet_tau(Rm, p0["R0"], strength_ratio=p0["sr"])
            ts = p0["sr"] * tw
            row[d] = dict(R=Rm, sd=Rsd, tau=ts, n=len(Rs))
            P(f"| {d} | {len(Rs)} | {Rm:.2f}±{Rsd:.2f} | {ts:.2f} |")
        sec2d[el] = row
        P("")

    # ---- 2e. synthesis ----------------------------------------------------
    P("### 2e. Reconciliation\n")
    fe_sa = sec2b["Fe"]["n_low_def"] if sec2b.get("Fe") else 0
    v_sa = sec2b["V"]["n_low_def"] if sec2b.get("V") else 0
    v_pair_tau = [p["tau_s"] for p in sec2a["V"] if p["tau_s"] > 0.3]
    P("- **V has moderate self-absorption on its strong low-Eᵢ multiplets.** The "
      f"multiplet-ratio test gives τ≈{'/'.join(f'{t:.1f}' for t in sorted(v_pair_tau, reverse=True)[:3]) or '—'} "
      "on the strongest pairs (e.g. V I 478.65/475.39, 417.94/415.97), i.e. SA "
      "correction factors ~1.9–3.8×; the Boltzmann deficit independently flags "
      f"{v_sa} resonance/low-Eᵢ V I lines (382.86, 411.18, 459.41) sitting "
      "60–90% low; and the delay series shows the compression is essentially "
      "delay-independent (τ≈3.8 at 5, 10 and 20 µs) — a persistent optically "
      "thick column, not a transient.\n")
    P("- **Fe self-absorption is present but under-quantified by these pairs.** "
      "The comparable-strength isolated Fe pairs that survive the cuts are "
      "optically thin (τ≈0); Fe's dense line forest leaves few clean pairs, and "
      "the single-line Boltzmann scatter (rms ~0.7 ln = factor 2, from db gA "
      "error and blends) exceeds the SA signal for most Fe lines. The prior Fe "
      "study (reports/2026-09-22-fe-calibration-plasma-analysis.md) established "
      "Fe I 438.35 as self-absorbed (it sits low in the Boltzmann plot and the "
      "robust thermometer sheds it); that is consistent here but not "
      "independently quantifiable to a τ from the multiplet pairs.\n")
    P("- **Width shows no opacity broadening at this resolution** (§2c): FWHM vs "
      "log-area slope is ≈0 / slightly negative for both metals, so SA compresses "
      "line AREA without measurably broadening the ~2–3-pixel instrumental "
      "profile. SA therefore biases Boltzmann T upward (strong low-Eᵢ lines lost) "
      "and would bias any composition-type intensity of the affected V resonance "
      "lines low by the correction factors above.\n")

    # =================== SECTION 3 =======================================
    P("## 3. How much broadening, and which mechanisms?\n")
    P("FWHM: Gaussian fit over peak ±3 native samples (baseline-subtracted), "
      "isolated lines SNR≥10 per segment. Instrumental = 15th-percentile lower "
      "envelope per segment + Ar I NIR lines. Doppler at T=9 kK. Stark/opacity "
      "= excess width over instrumental (Gaussian-subtraction upper bound; Voigt "
      "caveat noted).\n")
    sec3 = {}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        rows = M["fwhm_best"]
        inst = instrumental_fwhm(rows, pct=15.0)
        arows = ar_nir_fwhm(wl, y)
        sec3[el] = dict(rows=rows, inst=inst, ar=arows)
    P("### 3.1 Instrumental & measured FWHM per segment (best fresh run)\n")
    P("| metal | seg | n lines | inst FWHM (px) | inst FWHM (nm) | median FWHM (nm) | Doppler (pm) |")
    P("|--|--|--|--|--|--|--|")
    for el, M in metals.items():
        for seg in SEG_NAMES:
            inst = sec3[el]["inst"].get(seg)
            if not inst:
                continue
            lam_mid = {"UV": 300, "VIS": 490, "NIR": 760}[seg]
            dopp = doppler_fwhm(lam_mid, el) * 1000
            P(f"| {el} | {seg} | {inst['n']} | {inst['px']:.2f} | "
              f"{inst['nm']*1000:.0f} pm | {inst['median']*1000:.0f} pm | {dopp:.1f} |")
    P("")
    P("Ar I NIR anchor FWHM (cool, low-Stark purge lines) — an instrumental "
      "cross-check for the NIR segment:\n")
    for el, M in metals.items():
        ar = sec3[el]["ar"]
        if ar:
            med = np.median([a["fwhm_nm"] for a in ar]) * 1000
            P(f"- {el} run: n={len(ar)} Ar I lines, median FWHM {med:.0f} pm "
              f"({med/PITCH['NIR']/1000:.2f} px).")
    P("")
    # strong-line excess
    P("### 3.2 Excess width of the strong / self-absorbed lines\n")
    P("The peak is first located as the local max within ±0.20 nm of the "
      "shift-corrected position (so the V db-wavelength offset does not throw the "
      "measurement off); `Δpos` is that peak's offset from the naive prediction. "
      "In the dense Fe/V forest a |Δpos|>~120 pm entry may have snapped to a "
      "brighter neighbour and its width is then that neighbour's — read those "
      "rows with the §3.1 distribution, not in isolation.\n")
    P("| metal | λ_db (nm) | seg | Δpos (pm) | FWHM (pm) | excess over inst (pm) | ×inst |")
    P("|--|--|--|--|--|--|--|")
    STRONG = {"Fe": [438.35, 371.99, 385.99, 404.58],
              "V": [437.92, 438.47, 411.18, 410.98, 439.00, 309.31, 310.23, 311.07]}
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        for lam in STRONG[el]:
            c0 = db_to_obs(M["model"], lam)
            lo = int(np.searchsorted(wl, c0 - 0.20)); hi = int(np.searchsorted(wl, c0 + 0.20))
            if hi <= lo:
                P(f"| {el} | {lam:.2f} | {segn(lam)} | — | (edge) | — | — |"); continue
            j = lo + int(np.argmax(y[lo:hi])); center = float(wl[j])
            f = measure_fwhm(wl, y, center, snr_min=5.0)
            if not f:
                P(f"| {el} | {lam:.2f} | {segn(lam)} | {(center-c0)*1000:+.0f} | (SNR<5/unresolved) | — | — |")
                continue
            seg = segn(lam)
            inst = sec3[el]["inst"].get(seg)
            iw = inst["nm"] if inst else float("nan")
            exc = f["fwhm_nm"] - iw
            P(f"| {el} | {lam:.2f} | {seg} | {(center-c0)*1000:+.0f} | {f['fwhm_nm']*1000:.0f} | "
              f"{exc*1000:+.0f} | {f['fwhm_nm']/iw:.2f} |")
    P("")
    # delay dependence of width (V replicates)
    P("### 3.3 FWHM vs delay (Stark shrinks as nₑ decays; instrumental does not)\n")
    P("Median FWHM of isolated VIS lines (SNR≥10) at delay 5/10/20, replicate SD.\n")
    P("(Per-run VIS FWHM via the fast log-parabola estimator `fwhm_fast`; the "
      "best-run absolute widths above use the full ±3 Gaussian fit.)\n")
    sec3d = {}
    for el, M in metals.items():
        L = M["L"]
        vis_centers = [db_to_obs(M["model"], L["wl"][i])
                       for i in np.where(M["iso"] & (strength(L["gA"], L["Ek"]) > 0))[0]
                       if segn(L["wl"][i]) == "VIS"]
        by_delay = defaultdict(list)
        for r in M["fresh"]:
            by_delay[int(r.meta["delay"])].append(r)
        P(f"**{el}**:")
        P("| delay (µs) | n runs | median VIS FWHM (pm) | SD across runs (pm) |")
        P("|--|--|--|--|")
        row = {}
        for d in sorted(by_delay):
            meds = []
            for r in by_delay[d]:
                wl, yv = pool_mean([r])
                vis = [f["fwhm_nm"] for c in vis_centers
                       for f in [fwhm_fast(wl, yv, c, snr_min=10.0)] if f]
                if vis:
                    meds.append(float(np.median(vis)))
            if not meds:
                continue
            row[d] = dict(med=float(np.mean(meds)), sd=float(np.std(meds)), n=len(meds))
            P(f"| {d} | {len(meds)} | {np.mean(meds)*1000:.0f} | {np.std(meds)*1000:.0f} |")
        sec3d[el] = row
        P("")

    # ---- 3.4 decomposition synthesis --------------------------------------
    P("### 3.4 Decomposition\n")
    def instpx(el, seg):
        return sec3[el]["inst"].get(seg, {}).get("px", float("nan"))
    P("- **Instrumental** dominates. The lower-envelope FWHM is "
      f"{instpx('Fe','UV'):.1f}/{instpx('Fe','VIS'):.1f}/{instpx('Fe','NIR'):.1f} px "
      f"(Fe) and {instpx('V','UV'):.1f}/{instpx('V','VIS'):.1f}/{instpx('V','NIR'):.1f} px "
      "(V) for UV/VIS/NIR — i.e. ~120/155/230 pm — matching the cool Ar I NIR "
      "lines (~1.1–1.5 px, 190–270 pm), which have negligible Stark, so the "
      "resolution is ~1.2 px ≈ 0.11/0.16/0.23 nm per segment.\n")
    P("- **Doppler** is a few pm at 9 kK (2.7–6.9 pm Fe, 2.9–7.2 pm V; §3.1), "
      "≥20× below a native pixel — undetectable here.\n")
    P("- **Stark** is bounded, not measured. The clean-line median sits only "
      "~1.1–1.4× the instrumental width, and it does NOT shrink from delay 5→20 µs "
      "(§3.3: 184→181 pm Fe, 179→177 pm V, within the ~3–10 pm replicate SD) — "
      "if Stark were significant the width would fall as nₑ decays. So Stark+"
      "Doppler excess over instrumental is ≲30–60 pm across the band, an UPPER "
      "bound on nₑ broadening; no Stark width parameter for these Fe I/V I lines "
      "is cited with confidence, so this is not converted to an nₑ value (Hα is "
      "unusable — weak/blended here).\n")
    P("- **Self-absorption / opacity broadening** is marginal at this resolution: "
      "the strongest lines (§3.2, e.g. V I 438.47 ×1.7, Fe I 385.99 ×2.0) are "
      "broader, but several of those are blends (large Δpos), and the width–area "
      "slope (§2c) is flat — SA compresses area far more than it broadens the "
      "instrument-limited profile.\n")

    # =================== SECTION 4 =======================================
    P("## 4. Fe vs V comparison\n")
    P("| quantity | Fe (99.98%) | V (pure) |")
    P("|--|--|--|")
    def g(el, seg, key):
        return sec3[el]["inst"].get(seg, {}).get(key, float("nan"))
    P(f"| best fresh run | {metals['Fe']['best_name'][:14]} (5/10) | {metals['V']['best_name'][:14]} (10/10) |")
    P(f"| peaks SNR≥5 | {sec1_tot['Fe']['n']} | {sec1_tot['V']['n']} |")
    P(f"| matched fraction | {sec1_tot['Fe']['frac']:.3f} | {sec1_tot['V']['frac']:.3f} |")
    P(f"| unmatched count | {sec1_tot['Fe']['unmatched']} | {sec1_tot['V']['unmatched']} |")
    P(f"| reverse completeness | {sec1_rev['Fe']['frac']:.3f} ({sec1_rev['Fe']['nd']}/{sec1_rev['Fe']['nl']}) | "
      f"{sec1_rev['V']['frac']:.3f} ({sec1_rev['V']['nd']}/{sec1_rev['V']['nl']}) |")
    n_sa = {el: sum(1 for p in sec2a[el] if p["tau_s"] > 0.3) for el in metals}
    tau_typ = {el: (np.median([p["tau_s"] for p in sec2a[el] if p["tau_s"] > 0.3])
                    if n_sa[el] else float("nan")) for el in metals}
    P(f"| SA pairs (τ_strong>0.3) | {n_sa['Fe']} | {n_sa['V']} |")
    P(f"| typical τ_strong (SA pairs) | {fmt(tau_typ['Fe'],2)} | {fmt(tau_typ['V'],2)} |")
    P(f"| VIS inst FWHM (nm) | {fmt(g('Fe','VIS','nm')*1000,0)} pm | {fmt(g('V','VIS','nm')*1000,0)} pm |")
    P(f"| VIS median FWHM (nm) | {fmt(g('Fe','VIS','median')*1000,0)} pm | {fmt(g('V','VIS','median')*1000,0)} pm |")
    usable = {el: len(metals[el]["fwhm_best"]) for el in metals}
    P(f"| usable isolated lines (SNR≥10) | {usable['Fe']} | {usable['V']} |")
    P(f"| atomic mass (u) | {MASS['Fe']} | {MASS['V']} |")
    P("")
    P("**Physical differences.** " + (
        "Fe I/II is far denser (2,526 Fe I + 6,463 Fe II db lines vs 1,161 V I "
        "+ 1,767 V II), so more Fe peaks are blends. Fe's db air wavelengths are "
        "accurate (matched fraction saturates by ±0.15 nm), whereas the V line "
        "list's air wavelengths scatter ~2–3× more (V needs ±0.40 nm) — the main "
        "reason V's raw matched fraction is lower despite being a pure sample; "
        "this is a db-quality difference, not more unidentified emission. Self-"
        "absorption is measurable on V's strong low-Eᵢ multiplets "
        "(437.9/438.5/439.0, 410.98/411.18; τ~1.5–3.8) but under-quantified for "
        "Fe here (few clean comparable pairs; Fe I 438.35 self-absorption is "
        "established in the prior Fe study). V II 309.3/310.2/311.1 is UV, where "
        "response and SNR limit the diagnostics. Fe (56 u) and V (51 u) differ "
        "<10% in mass, so Doppler widths are within ~5%, and both share the same "
        "three-segment spectrometer (resolution ~1.2 px) — the differences are "
        "line density, resonance-line strength and V-db wavelength accuracy, not "
        "optics. Implication for calibration: Fe is ready for quantitative line "
        "work now; V first needs a per-segment wavelength re-registration of its "
        "line list and SA correction on its resonance multiplets."))
    P("")

    # ---------------- LIMITS ----------------------------------------------
    P("## Limits and unverified\n")
    P("- **Segment response is uncorrected.** Multiplet ratios are restricted "
      "to one segment and ≤8 nm to keep the response ≈flat; cross-segment "
      "intensity comparisons are not made. R₀ uses an energy-emissivity "
      "convention (∝gA/λ); a photon convention would shift R₀ by λ ratios (~5% "
      "over 20 nm).\n")
    P("- **τ from a homogeneous-slab escape factor** (`alibz.utils.absorption`); "
      "real plasmas have gradients, so τ is a lower bound on the peak optical "
      "depth and the SA correction is a slab-average.\n")
    P("- **Stark → nₑ not converted.** No Stark width parameter for these "
      "specific Fe I / V I lines is cited from memory with confidence, so the "
      "excess width is reported as an upper bound on Stark+Doppler+opacity only; "
      "Hα is unusable (blended/weak here). Gaussian-quadrature width subtraction "
      "over-subtracts for a Lorentzian (Stark) core, so the excess is an upper "
      "bound.\n")
    P("- **Shift scatter** (σ per model above) is ~0.5–1 native pitch; a peak "
      "within ±0.15 nm can match the wrong member of a close db multiplet, so "
      "\"matched-unique\" is a resolution-limited statement, not a line-ID proof.\n")
    P("- V \"fresh pool\" mixes 12 raster sites and 3 delays; the best-run "
      "tables use the single highest-score 10/10 replicate only.\n")

    # ---------------- FILES -----------------------------------------------
    # ---- figures ----------------------------------------------------------
    figs = make_figures(args.figdir, db, metals, sec1, sec2a, sec3, sec3d)
    P("## Files\n")
    P(f"- Script: `scripts/line_physics_analysis.py`")
    P(f"- Report: `{os.path.relpath(args.out, REPO)}`")
    for f in figs:
        P(f"- Figure: `{os.path.relpath(f, REPO)}`")
    P("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(report) + "\n")
    # stdout summary for verification
    print("WROTE", args.out)
    for el in metals:
        t = sec1_tot[el]
        print(f"{el}: peaks={t['n']} matched_frac={t['frac']:.3f} unmatched={t['unmatched']} "
              f"reverse={sec1_rev[el]['frac']:.3f} SA_pairs={n_sa[el]} typ_tau={fmt(tau_typ[el],2)}")


# =============================================================================
# Figures
# =============================================================================
def make_figures(figdir, db, metals, sec1, sec2a, sec3, sec3d):
    figs = []
    # Fig 1: labelled mean spectra with unmatched peaks marked
    for el, M in metals.items():
        wl, y = M["best_wl_y"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8))
        for ax, (lo, hi, name) in zip(axes, [(186, 365, "UV"), (365, 620, "VIS"), (620, 948, "NIR")]):
            m = (wl >= lo) & (wl < hi)
            ax.plot(wl[m], y[m], lw=0.5, color="#333")
            for r in sec1[el]:
                if r["seg"] != name:
                    continue
                if r["cls"] == "unmatched":
                    ax.axvline(r["wl_obs"], color="crimson", lw=0.6, alpha=0.7)
                elif r["cls"] == "contam" and r.get("confirmed"):
                    ax.axvline(r["wl_obs"], color="tab:blue", lw=0.4, alpha=0.5)
            ax.set_ylabel(f"{name}\ncounts")
            ax.set_xlim(lo, hi)
        axes[0].set_title(f"{el} best fresh run — mean spectrum "
                          f"(red=unmatched SNR≥5, blue=confirmed contaminant)")
        axes[-1].set_xlabel("wavelength (nm)")
        fig.tight_layout()
        p = os.path.join(figdir, f"spectrum_{el}.png")
        fig.savefig(p, dpi=110); plt.close(fig); figs.append(p)

    # Fig 2: multiplet ratio (R vs R0, with tau contours) per metal
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, el in zip(axes, metals):
        pairs = sec2a[el]
        if pairs:
            R0 = [p["R0"] for p in pairs]
            R = [p["R"] for p in pairs]
            c = [p["tau_s"] for p in pairs]
            sc = ax.scatter(R0, R, c=c, cmap="viridis", s=40, vmin=0)
            lim = [1, max(max(R0), max(R)) * 1.05]
            ax.plot(lim, lim, "k--", lw=0.8, label="thin (R=R₀)")
            ax.plot(lim, [1, 1], "r:", lw=0.8, label="saturated (R=1)")
            fig.colorbar(sc, ax=ax, label="τ_strong")
        ax.set_xlabel("R₀ (thin ratio)"); ax.set_ylabel("R (measured)")
        ax.set_title(f"{el} multiplet ratios"); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "multiplet_ratios.png")
    fig.savefig(p, dpi=110); plt.close(fig); figs.append(p)

    # Fig 3: FWHM vs wavelength with instrumental envelope
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, el in zip(axes, metals):
        rows = sec3[el]["rows"]
        for seg, col in zip(SEG_NAMES, ["tab:purple", "tab:green", "tab:red"]):
            rs = [r for r in rows if r["seg"] == seg]
            if rs:
                ax.scatter([r["lam"] for r in rs], [r["fwhm_nm"]*1000 for r in rs],
                           s=12, color=col, label=seg, alpha=0.6)
                inst = sec3[el]["inst"].get(seg)
                if inst:
                    lo, hi = ({"UV": (186, 365), "VIS": (365, 620), "NIR": (620, 948)})[seg]
                    ax.plot([lo, hi], [inst["nm"]*1000]*2, color=col, lw=1.5, ls="--")
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("FWHM (pm)")
        ax.set_title(f"{el} FWHM vs λ (dashed = instrumental envelope)")
        ax.legend(fontsize=8); ax.set_ylim(0, None)
    fig.tight_layout()
    p = os.path.join(figdir, "fwhm_vs_wavelength.png")
    fig.savefig(p, dpi=110); plt.close(fig); figs.append(p)

    # Fig 4: FWHM vs delay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for el, col in zip(metals, ["tab:blue", "tab:orange"]):
        row = sec3d.get(el, {})
        ds = sorted(row)
        if ds:
            ax.errorbar(ds, [row[d]["med"]*1000 for d in ds],
                        yerr=[row[d]["sd"]*1000 for d in ds],
                        marker="o", capsize=3, label=el, color=col)
    ax.set_xlabel("delay (µs)"); ax.set_ylabel("median VIS FWHM (pm)")
    ax.set_title("FWHM vs delay (Stark shrinks with delay)"); ax.legend()
    fig.tight_layout()
    p = os.path.join(figdir, "fwhm_vs_delay.png")
    fig.savefig(p, dpi=110); plt.close(fig); figs.append(p)
    return figs


if __name__ == "__main__":
    main()
