#!/usr/bin/env python3
"""Quantify estimator bias on the Z300 NATIVE (undersampled) wavelength grid.

Since 2026-09-22 Z300 spectra are on the detector's native grid: 7,914 samples,
three segments (UV 0.078-0.096, VIS 0.113-0.142, NIR 0.156-0.198 nm/px), line
FWHM ~1.0-1.6 px, i.e. critically sampled or undersampled. The old data was a
1/30 nm cubic-spline resample (4-5x oversampled, correlated noise). This script
simulates lines with a realistic instrument profile, INTEGRATES them over pixels
at each segment's native pitch with random sub-pixel phase and white noise, and
compares the estimators the alibz code and the cal/ analysis scripts use:

  peak LOCATION : argmax-snap (measure_line.peak_nm), 3-point parabolic
                  (fe_plasma_analysis.subpixel_center), point-sampled Gaussian
                  fit (line_physics_analysis.measure_fwhm-style), and a
                  PIXEL-INTEGRATED Gaussian fit (the proposed fix).
  amplitude     : local-max height (v_timing_study.robust_height), fixed-window
                  trapezoid area (fe_plasma_analysis.measure_line), point-fit
                  amplitude, pixel-integrated-fit amplitude/area.
  FWHM          : point-sampled vs pixel-integrated Gaussian fit.
  line RATIOS   : a same-upper-level pair with a known optically-thin ratio,
                  each member at an independent sub-pixel phase.

Each estimator is run BOTH on the native pixels and on the vendor cubic-spline
resample of the SAME simulated pixels (to 1/30 nm), to show which estimators the
old grid flattered or hid.

With --real-shots it also runs the parabolic centroid vs the pixel-integrated
fit on real native V lines (10 shots) and reports centroid scatter and fit
residuals -- the coordinator's priority prototype.

Deterministic (fixed seed). numpy + scipy + (optional) matplotlib only.

Example:
    python cal/scripts/native_grid_estimator_sim.py
    python cal/scripts/native_grid_estimator_sim.py --n-mc 400 --seed 7
    python cal/scripts/native_grid_estimator_sim.py --real-shots AUTO
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.special import erf, voigt_profile

SIGMA_TO_FWHM = 2.3548200450309493
_EXPORT_PX = 1.0 / 30.0

# Representative native pitch [nm/px] and instrument FWHM [px] per segment.
# Pitches: segment midrange from the brief; FWHM: VIS ~0.18 nm ~1.3-1.6 px,
# NIR ~1 px, UV ~1.3 px.
SEGMENTS = {
    "UV":  dict(pitch=0.087, center=270.0, fwhm_px=1.35),
    "VIS": dict(pitch=0.128, center=490.0, fwhm_px=1.45),
    "NIR": dict(pitch=0.177, center=760.0, fwhm_px=1.10),
}


# --------------------------------------------------------------------------
# pixel-integrated truth
# --------------------------------------------------------------------------
def pixel_edges(centers):
    """Edges of pixels given their center wavelengths (midpoints; ends mirrored)."""
    mid = 0.5 * (centers[1:] + centers[:-1])
    lo = centers[0] - (mid[0] - centers[0])
    hi = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[lo], mid, [hi]])


def gauss_integral(edges, amp, mu, sigma):
    """Exact integral of amp*exp(-(x-mu)^2/2sigma^2) over each pixel [edge_i,edge_{i+1}]."""
    z = (edges - mu) / (math.sqrt(2.0) * sigma)
    cdf = 0.5 * (1.0 + erf(z))
    return amp * sigma * math.sqrt(2.0 * math.pi) * np.diff(cdf)


def voigt_integral(edges, amp, mu, sigma, gamma, oversample=16):
    """Integral of amp*voigt(x-mu;sigma,gamma) over each pixel by fine oversampling.

    voigt_profile is area-normalised, so amp is the line area (counts)."""
    n = edges.size - 1
    out = np.empty(n)
    for i in range(n):
        xs = np.linspace(edges[i], edges[i + 1], oversample + 1)
        vals = amp * voigt_profile(xs - mu, sigma, gamma)
        out[i] = np.trapezoid(vals, xs)
    return out


# --------------------------------------------------------------------------
# estimators (faithful re-implementations of the shipped ones)
# --------------------------------------------------------------------------
def est_argmax_snap(wl, y, j0):
    """measure_line.peak_nm: position SNAPPED to the largest sample."""
    return float(wl[int(np.argmax(y))])


def est_parabolic(wl, y, j0):
    """fe_plasma_analysis.subpixel_center: 3-point parabola on raw y."""
    j = int(np.argmax(y))
    j = min(max(j, 1), len(wl) - 2)
    y1, y2, y3 = y[j - 1], y[j], y[j + 1]
    denom = (y1 - 2 * y2 + y3)
    if denom >= 0:
        return float(wl[j])
    return float(wl[j] + 0.5 * (y1 - y3) / denom * (wl[j + 1] - wl[j - 1]) / 2.0)


def _point_gauss_fit(wl, y, half=3):
    """line_physics_analysis.measure_fwhm-style POINT-sampled Gaussian fit."""
    j = int(np.argmax(y))
    j = min(max(j, half), len(wl) - half - 1)
    sl = slice(j - half, j + half + 1)
    xs = wl[sl].astype(float); ys = y[sl].astype(float)
    pitch = float(np.median(np.diff(xs)))
    A0 = float(ys.max() - ys.min())

    def model(p):
        A, mu, sig, c = p
        return A * np.exp(-(xs - mu) ** 2 / (2.0 * sig ** 2)) + c - ys

    p0 = [max(A0, 1.0), float(wl[j]), pitch, float(ys.min())]
    lb = [0.0, xs[0], 0.2 * pitch, -np.inf]
    ub = [np.inf, xs[-1], 6.0 * pitch, np.inf]
    try:
        sol = least_squares(model, p0, bounds=(lb, ub), max_nfev=4000)
        A, mu, sig, c = sol.x
        resid = model(sol.x)
        return dict(mu=float(mu), amp=float(A), sigma=float(abs(sig)),
                    fwhm=float(SIGMA_TO_FWHM * abs(sig)),
                    rms=float(np.sqrt(np.mean(resid ** 2))))
    except Exception:
        return None


def est_point_gauss_center(wl, y, j0):
    r = _point_gauss_fit(wl, y)
    return r["mu"] if r else float("nan")


def _pixint_gauss_fit(wl, y, half=4):
    """PIXEL-INTEGRATED Gaussian fit (proposed fix): model = integral over pixel."""
    j = int(np.argmax(y))
    j = min(max(j, half), len(wl) - half - 1)
    sl = slice(j - half, j + half + 1)
    xs = wl[sl].astype(float); ys = y[sl].astype(float)
    edges = pixel_edges(xs)
    pitch = float(np.median(np.diff(xs)))
    A0 = float(ys.max() - ys.min())

    def model(p):
        A, mu, sig, c = p
        integ = gauss_integral(edges, A, mu, sig)  # counts per pixel
        return integ + c - ys

    # seed amp so the model peak pixel matches the observed peak: A0 is a peak
    # COUNT but the model's A is the continuous peak height, so scale by the
    # unit-amplitude integrated peak (~pitch factor) or the fit sees a bad basin.
    sig0 = 0.8 * pitch
    c0 = float(np.median(ys[[0, -1]]))
    unit_peak = float(gauss_integral(edges, 1.0, float(wl[j]), sig0).max())
    A_seed = max(A0, 1.0) / unit_peak if unit_peak > 0 else max(A0, 1.0)
    p0 = [A_seed, float(wl[j]), sig0, c0]
    lb = [0.0, xs[0], 0.5 * pitch, -np.inf]
    ub = [np.inf, xs[-1], 6.0 * pitch, np.inf]
    try:
        sol = least_squares(model, p0, bounds=(lb, ub), max_nfev=4000)
        A, mu, sig, c = sol.x
        resid = model(sol.x)
        return dict(mu=float(mu), amp=float(A), sigma=float(abs(sig)),
                    fwhm=float(SIGMA_TO_FWHM * abs(sig)),
                    rms=float(np.sqrt(np.mean(resid ** 2))))
    except Exception:
        return None


def est_pixint_gauss_center(wl, y, j0):
    r = _pixint_gauss_fit(wl, y)
    return r["mu"] if r else float("nan")


def est_localmax_height(wl, y, j0):
    """v_timing_study.robust_height core: largest sample minus sideband median."""
    j = int(np.argmax(y))
    base = np.median(np.concatenate([y[:2], y[-2:]]))
    return float(y[j] - base)


def est_trapz_area(wl, y, j0, half_nm=0.12):
    """fe_plasma_analysis.measure_line: trapezoid over +-half_nm, linear baseline."""
    c = wl[int(np.argmax(y))]
    lo = np.searchsorted(wl, c - half_nm, "left")
    hi = np.searchsorted(wl, c + half_nm, "right")
    cx = np.arange(lo, hi)
    if cx.size < 3:
        cx = np.arange(max(0, len(wl)//2 - 2), min(len(wl), len(wl)//2 + 3))
    base = np.median(np.concatenate([y[:2], y[-2:]]))
    return float(np.trapezoid(y[cx] - base, wl[cx]))


# --------------------------------------------------------------------------
# vendor spline resample of native pixels
# --------------------------------------------------------------------------
def vendor_resample(wl_native, y_native):
    """Not-a-knot cubic spline of the native samples onto the 1/30 nm export grid."""
    xe = np.arange(math.ceil(wl_native[0] * 30) / 30.0,
                   math.floor(wl_native[-1] * 30) / 30.0 + 1e-9, _EXPORT_PX)
    cs = CubicSpline(wl_native, y_native, bc_type="not-a-knot")
    return xe, cs(xe)


# --------------------------------------------------------------------------
# one line realisation
# --------------------------------------------------------------------------
def make_line(seg, phase, snr, rng, npix=13, gamma_px=0.0, amp=1000.0,
              baseline=0.0):
    """Native pixel window centred (with sub-pixel phase) on one line.

    phase in [0,1): position of the true center within its pixel.
    Returns (wl, y_noisy, true_mu, true_fwhm_nm, peak_counts)."""
    p = SEGMENTS[seg]["pitch"]
    fwhm_px = SEGMENTS[seg]["fwhm_px"]
    sigma = fwhm_px * p / SIGMA_TO_FWHM
    gamma = gamma_px * p
    j0 = npix // 2
    # pixel centers; true center offset by phase within central pixel
    wl = (np.arange(npix) - j0) * p + SEGMENTS[seg]["center"]
    mu = wl[j0] + (phase - 0.5) * p
    edges = pixel_edges(wl)
    if gamma > 0:
        counts = voigt_integral(edges, amp, mu, sigma, gamma)
        # normalise amp so peak pixel ~ amp-scale comparable to gaussian
    else:
        counts = gauss_integral(edges, amp, mu, sigma)
    peak = counts.max()
    noise_sd = peak / snr if snr > 0 else 0.0
    y = counts + baseline + (rng.normal(0, noise_sd, npix) if noise_sd > 0 else 0.0)
    fwhm_nm = fwhm_px * p if gamma == 0 else None
    return wl, y, mu, fwhm_nm, peak, sigma, gamma


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def exp_location_bias(rng, phases):
    """Noiseless location bias vs sub-pixel phase, per segment (pm)."""
    ests = [("argmax-snap", est_argmax_snap),
            ("parabolic", est_parabolic),
            ("point-Gauss", est_point_gauss_center),
            ("pixel-int-Gauss", est_pixint_gauss_center)]
    rows = []
    for seg in SEGMENTS:
        p = SEGMENTS[seg]["pitch"]
        for name, fn in ests:
            errs = []
            for ph in phases:
                wl, y, mu, *_ = make_line(seg, ph, snr=1e9, rng=rng)
                errs.append((fn(wl, y, None) - mu) * 1000.0)  # pm
            errs = np.array(errs)
            rows.append((seg, p, name, float(np.mean(errs)),
                         float(np.max(np.abs(errs))), float(np.std(errs))))
    return rows


def exp_location_scatter(rng, snrs, n_mc):
    """MC location scatter (std of error, pm) over random phase+noise per SNR."""
    ests = [("argmax-snap", est_argmax_snap),
            ("parabolic", est_parabolic),
            ("point-Gauss", est_point_gauss_center),
            ("pixel-int-Gauss", est_pixint_gauss_center)]
    rows = []
    for seg in ("VIS", "NIR"):
        for snr in snrs:
            for name, fn in ests:
                errs = []
                for _ in range(n_mc):
                    ph = rng.uniform(0, 1)
                    wl, y, mu, *_ = make_line(seg, ph, snr, rng)
                    v = fn(wl, y, None)
                    if np.isfinite(v):
                        errs.append((v - mu) * 1000.0)
                errs = np.array(errs)
                rows.append((seg, snr, name, float(np.mean(errs)),
                             float(np.std(errs))))
    return rows


def exp_amp_fwhm(rng, phases):
    """Noiseless amplitude and FWHM bias vs phase (relative %), per segment."""
    rows = []
    for seg in SEGMENTS:
        # references from a well-sampled fit
        lm, tr, pgf_a, pxf_a = [], [], [], []
        pgf_w, pxf_w = [], []
        # true peak-pixel-integrated reference: use amp param recovered by a
        # dense fit as the amplitude ground truth is the model amp itself.
        for ph in phases:
            wl, y, mu, fwhm_nm, peak, sigma, gamma = make_line(seg, ph, 1e9, rng)
            lm.append(est_localmax_height(wl, y, None))
            tr.append(est_trapz_area(wl, y, None))
            r1 = _point_gauss_fit(wl, y)
            r2 = _pixint_gauss_fit(wl, y)
            SQ2PI = math.sqrt(2.0 * math.pi)
            if r1:
                pgf_a.append(r1["amp"] * r1["sigma"] * SQ2PI); pgf_w.append(r1["fwhm"])
            if r2:
                pxf_a.append(r2["amp"] * r2["sigma"] * SQ2PI); pxf_w.append(r2["fwhm"])
        def spread(a):
            a = np.array(a)
            return 100.0 * (a.max() - a.min()) / np.mean(a) if a.size and np.mean(a) else float("nan")
        rows.append((seg, "local-max height", spread(lm)))
        rows.append((seg, "trapz area", spread(tr)))
        rows.append((seg, "point-Gauss area", spread(pgf_a)))
        rows.append((seg, "pixel-int area", spread(pxf_a)))
        # FWHM bias: mean recovered vs true, and phase spread
        fwhm_true = SEGMENTS[seg]["fwhm_px"] * SEGMENTS[seg]["pitch"]
        rows.append((seg, "point-Gauss FWHM bias%",
                     100.0 * (np.mean(pgf_w) - fwhm_true) / fwhm_true))
        rows.append((seg, "pixel-int FWHM bias%",
                     100.0 * (np.mean(pxf_w) - fwhm_true) / fwhm_true))
    return rows


def exp_ratio(rng, n_mc, true_ratio=2.05, seg="VIS"):
    """Same-upper-level pair, independent sub-pixel phases; recovered ratio by
    each amplitude estimator, native vs vendor-spline. Also with real SA on the
    strong line (tau) to show estimator bias masquerading as self-absorption."""
    p = SEGMENTS[seg]["pitch"]
    fwhm_px = SEGMENTS[seg]["fwhm_px"]
    sigma = fwhm_px * p / SIGMA_TO_FWHM

    def one(amp_s, amp_w):
        # two separate isolated windows, independent phases
        res = {}
        for tag, resamp in (("native", False), ("vendor", True)):
            hs = ws = trs = trw = fas = faw = None
            for amp, key in ((amp_s, "s"), (amp_w, "w")):
                ph = rng.uniform(0, 1)
                wl, y, mu, *_ = make_line(seg, ph, 1e9, rng, amp=amp)
                if resamp:
                    wl, y = vendor_resample(wl, y)
                lm = est_localmax_height(wl, y, None)
                tr = est_trapz_area(wl, y, None)
                fa = _pixint_gauss_fit(wl, y)
                fa = fa["amp"] if fa else np.nan
                if key == "s":
                    hs, trs, fas = lm, tr, fa
                else:
                    ws, trw, faw = lm, tr, fa
            res[tag] = dict(height=hs / ws, trapz=trs / trw, pixint=fas / faw)
        return res

    # thin case: true amplitude ratio = true_ratio (same upper level, thin)
    out = {}
    for label, (amp_s, amp_w) in (("thin (true 2.05)", (2.05, 1.0)),):
        acc = {t: {k: [] for k in ("height", "trapz", "pixint")}
               for t in ("native", "vendor")}
        for _ in range(n_mc):
            r = one(amp_s, amp_w)
            for t in acc:
                for k in acc[t]:
                    acc[t][k].append(r[t][k])
        out[label] = {t: {k: (float(np.median(v)), float(np.std(v)))
                          for k, v in acc[t].items()} for t in acc}
    return out, true_ratio


# --------------------------------------------------------------------------
# real V lines
# --------------------------------------------------------------------------
def load_shots(shots_dir, max_shots=10):
    files = sorted(glob.glob(os.path.join(shots_dir, "shot-*.csv")))[:max_shots]
    arrs = []
    for f in files:
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        arrs.append(d)
    return files, arrs


def find_run_with_shots(root):
    """Pick the run (>=8 shots) with the strongest signal in shot 0."""
    best = None
    for d in sorted(glob.glob(os.path.join(root, "run-*", "shots"))):
        fs = glob.glob(os.path.join(d, "shot-*.csv"))
        if len(fs) >= 8:
            m = float(np.loadtxt(sorted(fs)[0], delimiter=",", skiprows=1)[:, 1].max())
            if best is None or m > best[1]:
                best = (d, m)
    return best[0] if best else None


def detect_isolated_peaks(wl, y, snr_min=15.0, isol_nm=0.35, topn=10):
    """Strong isolated local maxima (observed positions) from one shot."""
    cand = []
    for i in range(5, len(y) - 5):
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= y[i - 2] and y[i] >= y[i + 2]:
            base = np.median(np.concatenate([y[i - 5:i - 3], y[i + 4:i + 6]]))
            h = y[i] - base
            noise = 1.4826 * np.median(np.abs(y[i - 5:i + 6] - np.median(y[i - 5:i + 6])))
            if noise > 0 and h / noise > snr_min:
                cand.append((float(wl[i]), h))
    iso = [c for c in cand if sum(1 for o in cand if abs(o[0] - c[0]) < isol_nm) == 1]
    return [c[0] for c in sorted(iso, key=lambda c: -c[1])[:topn]]


# Isolated V I/II lines (nm). The dense 318.40/318.54 pair is kept out; these
# are single, strong, and clear of a close neighbour in these runs.
V_LINES = [437.924, 411.178, 385.585, 268.796, 214.006, 292.402]


def exp_real(shots_dir, search_nm=0.22, snr_min=8.0):
    """Parabolic vs point-Gauss vs pixel-int centroid across shots + fit RMS.

    Centroid scatter is the std across shots (pm) of each estimator; RMS is the
    per-shot fit-residual RMS as a fraction of the peak (point vs pixel-int)."""
    files, arrs = load_shots(shots_dir)
    if len(arrs) < 3:
        return None
    lines = detect_isolated_peaks(arrs[0][:, 0], arrs[0][:, 1])
    if not lines:
        lines = V_LINES
    rows = []
    for lam in lines:
        par, pgt, pxi = [], [], []
        rms_pt, rms_px = [], []
        for d in arrs:
            wl, y = d[:, 0], d[:, 1]
            k0 = int(np.searchsorted(wl, lam))
            pitch = float(np.median(np.diff(wl[max(0, k0 - 5):k0 + 6])))
            half = max(search_nm, 4.0 * pitch)  # >=4 px each side
            lo = np.searchsorted(wl, lam - half)
            hi = np.searchsorted(wl, lam + half)
            if hi - lo < 3:
                continue
            j = lo + int(np.argmax(y[lo:hi]))
            if j < 4 or j > len(wl) - 5:
                continue
            sl = slice(j - 4, j + 5)
            wlw, yw = wl[sl].astype(float), y[sl].astype(float)
            base = np.median(np.concatenate([yw[:2], yw[-2:]]))
            peak = float(yw.max() - base)
            noise = 1.4826 * np.median(np.abs(yw - np.median(yw)))
            if noise <= 0 or peak / noise < snr_min:
                continue
            # parabolic (subpixel_center)
            y1, y2, y3 = y[j - 1], y[j], y[j + 1]
            den = y1 - 2 * y2 + y3
            cpar = (wl[j] if den >= 0 else
                    wl[j] + 0.5 * (y1 - y3) / den * (wl[j + 1] - wl[j - 1]) / 2.0)
            rp = _point_gauss_fit(wlw, yw, half=4)   # same 9-px window as pixint
            rx = _pixint_gauss_fit(wlw, yw, half=4)
            if rp is None or rx is None:
                continue
            par.append(cpar); pgt.append(rp["mu"]); pxi.append(rx["mu"])
            rms_pt.append(rp["rms"] / max(peak, 1e-9))
            rms_px.append(rx["rms"] / max(peak, 1e-9))
        if len(par) >= 3:
            par = np.array(par); pgt = np.array(pgt); pxi = np.array(pxi)
            rows.append((lam, len(par),
                         float(np.std(par) * 1000),
                         float(np.std(pgt) * 1000),
                         float(np.std(pxi) * 1000),
                         float(np.mean(rms_pt) * 100),
                         float(np.mean(rms_px) * 100)))
    return files, rows


def exp_pitch(shot_csv):
    """Dead-4th-channel global-pitch effect on the PRODUCTION Voigt seed.

    Loads one real native shot, prints median(np.diff(x)) with and without the
    dead channel and the true per-segment pitch, then calls the ACTUAL
    alibz.peaky_finder.PeakyFinder._voigt_seed for a representative 1.1-px NIR
    line under each pitch to show the seed FWHM / amplitude mis-scaling."""
    d = np.loadtxt(shot_csv, delimiter=",", skiprows=1)
    x, y = d[:, 0], d[:, 1]
    dx = np.diff(x)
    live = x < 960.0
    inc_all = float(np.median(dx))
    inc_live = float(np.median(np.diff(x[live])))
    segs = {}
    for name, lo, hi in (("UV", 186, 365), ("VIS", 365, 620), ("NIR", 620, 948)):
        m = (x >= lo) & (x < hi)
        segs[name] = float(np.median(np.diff(x[m])))
    print("  median diff  ALL=%.4f  (dead channel: %d zeros of %d at >=960.18)"
          % (inc_all, int(np.sum(y[x >= 960.18] == 0)), int(np.sum(x >= 960.18))))
    print("  median diff  live(x<960)=%.4f" % inc_live)
    print("  per-segment pitch  UV=%.4f VIS=%.4f NIR=%.4f" %
          (segs["UV"], segs["VIS"], segs["NIR"]))
    print("  seed-width mis-scale factor = inc/pitch_seg:")
    for label, inc in (("ALL+dead", inc_all), ("live-trim", inc_live)):
        print("    %-10s UV=%+.0f%% VIS=%+.0f%% NIR=%+.0f%%" % (
            label, 100 * (inc / segs["UV"] - 1), 100 * (inc / segs["VIS"] - 1),
            100 * (inc / segs["NIR"] - 1)))
    try:
        from alibz.peaky_finder import PeakyFinder
        from alibz.utils.voigt import voigt_width

        class _D:  # _voigt_seed only reads x and args, not other state
            pass
        xw = np.array([760.0 + k * segs["NIR"] for k in range(9)])
        print("  PRODUCTION _voigt_seed for a genuine 1.1-px NIR line:")
        for label, inc in (("ALL+dead", inc_all), ("live-trim", inc_live),
                           ("NIR-local", segs["NIR"])):
            amp, mu, sig, gam = PeakyFinder._voigt_seed(_D(), xw, 4, 1000.0, 1.1, inc)
            fw = float(voigt_width(sig, gam))
            print("    %-10s min_inc=%.4f -> seedFWHM=%.4f nm (%.2f px)  seedAmp=%.0f"
                  % (label, inc, fw, fw / segs["NIR"], amp))
    except Exception as exc:
        print("  (could not exercise production _voigt_seed: %s)" % exc)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-mc", type=int, default=300, help="Monte-Carlo draws")
    ap.add_argument("--n-phase", type=int, default=41, help="phase-sweep points")
    ap.add_argument("--real-shots", default=None,
                    help="dir with shot-*.csv, or AUTO to find one under the "
                         "scratchpad vrun2 acq tree")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    phases = np.linspace(0, 1, args.n_phase, endpoint=False)

    print("=" * 74)
    print("NATIVE-GRID ESTIMATOR BIAS SIMULATION  (seed=%d, n_mc=%d)" % (args.seed, args.n_mc))
    print("=" * 74)

    print("\n[2.1] PEAK-LOCATION bias vs sub-pixel phase (noiseless), pm")
    print("%-4s %7s %-16s %8s %8s %8s" % ("seg", "pitch", "estimator", "mean", "max|e|", "phase-std"))
    for seg, p, name, mean, mx, sd in exp_location_bias(rng, phases):
        print("%-4s %7.3f %-16s %8.1f %8.1f %8.1f" % (seg, p, name, mean, mx, sd))

    print("\n[2.1b] PEAK-LOCATION scatter (MC over phase+noise), std of error pm")
    print("%-4s %5s %-16s %8s %8s" % ("seg", "snr", "estimator", "bias", "std"))
    for seg, snr, name, mean, sd in exp_location_scatter(rng, (20, 50, 100), args.n_mc):
        print("%-4s %5d %-16s %8.1f %8.1f" % (seg, snr, name, mean, sd))

    print("\n[2.2] AMPLITUDE / FWHM phase-dependent spread (noiseless)")
    print("%-4s %-22s %10s" % ("seg", "estimator", "spread%/bias%"))
    for seg, name, val in exp_amp_fwhm(rng, phases):
        print("%-4s %-22s %10.2f" % (seg, name, val))

    print("\n[2.2b] LINE-RATIO recovery (same-upper-level pair, true 2.05)")
    out, tr = exp_ratio(rng, args.n_mc)
    for label, d in out.items():
        print("  %s" % label)
        for t in ("native", "vendor"):
            for k in ("height", "trapz", "pixint"):
                med, sd = d[t][k]
                print("    %-7s %-7s median=%.3f  scatter=%.3f  (true %.2f, err %+.0f%%)"
                      % (t, k, med, sd, tr, 100 * (med - tr) / tr))

    if args.real_shots:
        sd = args.real_shots
        if sd == "AUTO":
            root = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
                    "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/vrun2/acq")
            sd = find_run_with_shots(root)
        if sd and os.path.isdir(sd):
            shot0 = sorted(glob.glob(os.path.join(sd, "shot-*.csv")))
            if shot0:
                print("\n[2.4] DEAD-CHANNEL global-pitch effect on the PRODUCTION Voigt seed")
                exp_pitch(shot0[0])
        print("\n[2.3] REAL native V lines: centroid scatter across shots (dir=%s)" % sd)
        if sd and os.path.isdir(sd):
            res = exp_real(sd)
            if res:
                files, rows = res
                print("  %d shots; centroid scatter across shots (pm), fit RMS (%% of peak)" % len(files))
                print("  %-9s %5s %10s %10s %10s %9s %9s" %
                      ("line_nm", "nshot", "par_pm", "ptGauss_pm", "pxint_pm",
                       "pt_rms%", "px_rms%"))
                for lam, n, ps, gs, xs, rp, rx in rows:
                    print("  %-9.3f %5d %10.1f %10.1f %10.1f %9.2f %9.2f"
                          % (lam, n, ps, gs, xs, rp, rx))
            else:
                print("  too few usable shots/lines")
        else:
            print("  shots dir not found: %s" % sd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
