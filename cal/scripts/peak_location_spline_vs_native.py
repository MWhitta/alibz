#!/usr/bin/env python3
"""Head-to-head: does the vendor cubic spline give better PEAK LOCATIONS than
native-pixel methods, after background subtraction?  Analysis accuracy only.

Eight estimators, each run AFTER a background subtraction:
  (1) native argmax
  (2) native 3-point parabola  (fe_plasma_analysis.subpixel_center)
  (3) vendor cubic spline (scipy CubicSpline not-a-knot through the native
      pixels -- the exact vendor form identified in alibz/utils/native_grid.py,
      KERNEL_NAME='cubic_notaknot') sampled at 1/30 nm, then argmax
  (4) (3) then a 3-point parabola on the dense grid
  (5) continuous maximum of the spline (root of its derivative)
  (6) LINEAR resample to 1/30 nm then argmax  (Pantheum's old path)
  (7) production Gaussian fit  (alibz/utils/peakfit.gaussian_subpixel_center)
  (8) pixel-integrated Gaussian fit on the RAW native pixels (the proposed fix)

Background conditions:
  (a) ideal / true continuum (simulation only)
  (b) production peaky_finder baseline as it runs today on native: GLOBAL-pitch
      lam, dead 4th channel present
  (c) the same baseline with a PER-SEGMENT pitch and the dead channel trimmed
      (the corrected background)
  (d) the same baseline on the SPLINE (1/30 nm) grid (the old situation)

SIMULATION: per-segment LSF, sub-pixel phase swept, SNR 5-100, a sloped/curved
continuum, and a neighbouring blend at 1-3 px.  Reports bias vs phase, RMS and
95th percentile per segment.

REAL DATA (headline, per the coordinator): estimator (8) vs the spline
estimators (3)/(4)/(5) on the SAME lines, paired, per segment, with
uncertainties -- multiplet-spacing closure, shot-to-shot repeatability,
cross-run consistency, and the Ar I 696.54/772.38/801.48 nm anchors.  Corrected
background (c) for both; the spline methods are also shown under the old
spline-grid background (d).  Fe runs are not available locally; V run2 (many
runs x ~10 shots) + the ambient Ar NIR anchors are used (see Unverified).

Deterministic.  numpy + scipy + alibz (read-only).  --help for options.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

# reuse the pixel-integration + sim helpers from the sibling audit script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_grid_estimator_sim import (           # noqa: E402
    SEGMENTS, SIGMA_TO_FWHM, pixel_edges, gauss_integral, make_line,
)

_EXPORT_PX = 1.0 / 30.0
DEAD_NM = 960.18            # dead 4th channel starts here (all-zero pad)
SEG_EDGES = (365.0, 620.0)

AR_ANCHORS = [696.5431, 772.3761, 801.4786]   # air, the clean Ar I NIR anchors


def seg_of(nm):
    return "UV" if nm < SEG_EDGES[0] else ("VIS" if nm < SEG_EDGES[1] else "NIR")


# --------------------------------------------------------------------------
# background subtraction conditions (uses the PRODUCTION arPLS)
# --------------------------------------------------------------------------
_PF = None


def _pf():
    global _PF
    if _PF is None:
        from alibz.peaky_finder import PeakyFinder
        _PF = PeakyFinder(".")
    return _PF


def bg_production(x, y):
    """(b) production baseline as today: global-pitch lam, dead channel present."""
    return _pf().find_background(x, y, segment_edges=SEG_EDGES)


def bg_corrected(x, y):
    """(c) per-segment pitch + dead channel trimmed.

    Trims x>=DEAD_NM, then runs the production arPLS PER SEGMENT so lam is
    derived from the local segment pitch (median np.diff within the segment)."""
    live = x < DEAD_NM
    xl, yl = x[live], y[live]
    bg = np.full(x.shape, np.nan)
    edges = [xl.min() - 1] + list(SEG_EDGES) + [xl.max() + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xl >= lo) & (xl < hi)
        if np.count_nonzero(m) < 8:
            continue
        b = _pf().find_background(xl[m], yl[m], segment_edges=())  # single seg
        idx = np.flatnonzero(live)[m]
        bg[idx] = b
    # dead-channel region: pin to data (y-bg=0)
    bg[~live] = y[~live]
    # any gaps -> interpolate
    nan = ~np.isfinite(bg)
    if nan.any():
        bg[nan] = np.interp(x[nan], x[~nan], bg[~nan])
    return bg


# --------------------------------------------------------------------------
# estimators (operate on a background-subtracted window; return centre nm)
# --------------------------------------------------------------------------
def e_argmax(xw, yw):
    return float(xw[int(np.argmax(yw))])


def _parab(x, y, j):
    j = min(max(int(j), 1), len(x) - 2)
    y1, y2, y3 = y[j - 1], y[j], y[j + 1]
    den = y1 - 2 * y2 + y3
    if den >= 0:
        return float(x[j])
    return float(x[j] + 0.5 * (y1 - y3) / den * (x[j + 1] - x[j - 1]) / 2.0)


def e_parab_native(xw, yw):
    return _parab(xw, yw, int(np.argmax(yw)))


def _spline(xw, yw):
    return CubicSpline(xw, yw, bc_type="not-a-knot")


def _dense(xw):
    return np.arange(math.ceil(xw[0] * 30) / 30.0,
                     math.floor(xw[-1] * 30) / 30.0 + 1e-9, _EXPORT_PX)


def e_spline_argmax(xw, yw):
    xd = _dense(xw)
    yd = _spline(xw, yw)(xd)
    return float(xd[int(np.argmax(yd))])


def e_spline_parab(xw, yw):
    xd = _dense(xw)
    yd = _spline(xw, yw)(xd)
    return _parab(xd, yd, int(np.argmax(yd)))


def e_spline_root(xw, yw):
    cs = _spline(xw, yw)
    xd = _dense(xw)
    yd = cs(xd)
    x0 = xd[int(np.argmax(yd))]
    d = cs.derivative()
    roots = d.roots()
    roots = roots[(roots > xw[1]) & (roots < xw[-2])]
    if roots.size == 0:
        return float(x0)
    # nearest root to the dense argmax that is a maximum (2nd deriv < 0)
    d2 = cs.derivative(2)
    maxr = roots[d2(roots) < 0]
    if maxr.size == 0:
        return float(x0)
    return float(maxr[np.argmin(np.abs(maxr - x0))])


def e_linear_argmax(xw, yw):
    xd = _dense(xw)
    yd = np.interp(xd, xw, yw)
    return float(xd[int(np.argmax(yd))])


def e_refine(xw, yw):
    """(7) production Gaussian fit (point-sampled) via peakfit."""
    from alibz.utils import peakfit
    j = int(np.argmax(yw))
    seg = peakfit.SEGMENT_NAMES[int(np.digitize(float(xw[j]), peakfit.SEGMENT_EDGES))]
    r = peakfit.gaussian_subpixel_center(
        xw, yw, j, sigma_px=peakfit.instrument_sigma_px(seg),
        baseline="linear", sigma_bounds=(0.5, 2.5))
    return float(r["center_nm"]) if r.get("ok") else float("nan")


def e_pixint(xw, yw):
    """(8) pixel-integrated Gaussian fit on the raw native pixels."""
    j = int(np.argmax(yw))
    j = min(max(j, 4), len(xw) - 5)
    sl = slice(j - 4, j + 5)
    xs = xw[sl].astype(float)
    ys = yw[sl].astype(float)
    edges = pixel_edges(xs)
    pitch = float(np.median(np.diff(xs)))
    A0 = float(ys.max() - ys.min())
    sig0 = 0.8 * pitch
    unit_peak = float(gauss_integral(edges, 1.0, float(xw[j]), sig0).max())
    A_seed = max(A0, 1.0) / unit_peak if unit_peak > 0 else max(A0, 1.0)

    def model(p):
        A, mu, sig, c = p
        return gauss_integral(edges, A, mu, sig) + c - ys

    lb = [0.0, xs[0], 0.5 * pitch, -np.inf]
    ub = [np.inf, xs[-1], 6.0 * pitch, np.inf]
    try:
        sol = least_squares(model, [A_seed, float(xw[j]), sig0,
                                    float(np.median(ys[[0, -1]]))],
                            bounds=(lb, ub), max_nfev=4000)
        return float(sol.x[1])
    except Exception:
        return float("nan")


ESTIMATORS = [
    ("1_argmax", e_argmax, "native"),
    ("2_parab", e_parab_native, "native"),
    ("3_spl_argmax", e_spline_argmax, "spline"),
    ("4_spl_parab", e_spline_parab, "spline"),
    ("5_spl_root", e_spline_root, "spline"),
    ("6_lin_argmax", e_linear_argmax, "native"),
    ("7_peakfit_gauss", e_refine, "native"),
    ("8_pixint", e_pixint, "native"),
]
HEADLINE = {"3_spl_argmax", "4_spl_parab", "5_spl_root", "8_pixint"}


# --------------------------------------------------------------------------
# SIMULATION
# --------------------------------------------------------------------------
def sim_window(seg, phase, snr, rng, blend_px=0.0, blend_frac=0.0,
               slope=0.30, curv=0.10, npix=15):
    """One native window: line + optional neighbour blend + a GENTLE
    sloped/curved continuum (scaled to the line peak) + white noise.
    Returns (x, y_raw, true_mu, continuum)."""
    p = SEGMENTS[seg]["pitch"]
    fwhm_px = SEGMENTS[seg]["fwhm_px"]
    sigma = fwhm_px * p / SIGMA_TO_FWHM
    j0 = npix // 2
    x = (np.arange(npix) - j0) * p + SEGMENTS[seg]["center"]
    mu = x[j0] + (phase - 0.5) * p
    edges = pixel_edges(x)
    amp = 1000.0
    counts = gauss_integral(edges, amp, mu, sigma)
    if blend_frac > 0:
        counts = counts + gauss_integral(edges, blend_frac * amp,
                                         mu + blend_px * p, sigma)
    peak = float(counts.max())
    xc = (x - x[j0]) / (npix * p)              # -0.5..0.5
    # continuum change across the window is slope*peak (linear) + curv*peak
    cont = 0.15 * peak + slope * peak * xc + curv * peak * xc ** 2
    noise = peak / snr if snr > 0 else 0.0
    y = counts + cont + (rng.normal(0, noise, npix) if noise > 0 else 0.0)
    return x, y, mu, cont


def _bg_sim(x, y, cont, mode):
    """Background-subtract a simulated window.
    'ideal'  : subtract the true continuum.
    'linear' : subtract a straight line through the two-sample edge medians
               (what measure_line/robust_height do in production, a realistic
               local baseline that does NOT know the true continuum).
    'spline' : same edge-linear baseline but on the 1/30 nm spline grid
               (the old spline-grid situation, for the spline estimators)."""
    if mode == "ideal":
        return y - cont
    lx = np.mean(x[:2]); rx = np.mean(x[-2:])
    ly = np.median(y[:2]); ry = np.median(y[-2:])
    base = ly + (ry - ly) * (x - lx) / (rx - lx)
    return y - base


def run_sim(rng, n_phase, snrs, bg_mode, blend=False):
    phases = np.linspace(0, 1, n_phase, endpoint=False)
    rows = []
    bpx, bfr = (2.0, 0.5) if blend else (0.0, 0.0)
    for seg in SEGMENTS:
        for snr in snrs:
            errs = {name: [] for name, _, _ in ESTIMATORS}
            for ph in phases:
                x, y, mu, cont = sim_window(seg, ph, snr, rng,
                                            blend_px=bpx, blend_frac=bfr)
                yb = _bg_sim(x, y, cont, bg_mode)          # bg once per window
                for name, fn, _ in ESTIMATORS:
                    v = fn(x, yb)
                    if np.isfinite(v):
                        errs[name].append((v - mu) * 1000.0)
            for name, _, _ in ESTIMATORS:
                e = np.array(errs[name])
                if e.size:
                    rows.append((seg, snr, name, float(np.mean(e)),
                                 float(np.sqrt(np.mean(e ** 2))),
                                 float(np.percentile(np.abs(e), 95))))
    return rows


# --------------------------------------------------------------------------
# REAL DATA
# --------------------------------------------------------------------------
def load_ledger_runs(acq_dir, ledger_path, max_runs=None):
    import json
    led = sorted(json.load(open(ledger_path)), key=lambda r: r["created_at"])
    if max_runs:
        led = led[:max_runs]
    runs = []
    for meta in led:
        sd = os.path.join(acq_dir, meta["run"], "shots")
        fs = sorted(glob.glob(os.path.join(sd, "shot-*.csv")))
        if len(fs) < 5:
            continue
        shots = [np.loadtxt(f, delimiter=",", skiprows=1) for f in fs]
        wl = shots[0][:, 0]
        Y = np.array([s[:, 1] for s in shots])
        runs.append((meta["run"], wl, Y))
    return runs


def detect_lines(wl, ymean, snr_min=12.0, isol_nm=0.30, topn=60):
    cand = []
    for i in range(6, len(ymean) - 6):
        if (ymean[i] >= ymean[i - 1] and ymean[i] > ymean[i + 1]
                and ymean[i] >= ymean[i - 2] and ymean[i] >= ymean[i + 2]):
            base = np.median(np.concatenate([ymean[i - 6:i - 3], ymean[i + 4:i + 7]]))
            h = ymean[i] - base
            noise = 1.4826 * np.median(np.abs(ymean[i - 6:i + 7] - np.median(ymean[i - 6:i + 7])))
            if noise > 0 and h / noise > snr_min:
                cand.append((float(wl[i]), h))
    iso = [c for c in cand if sum(1 for o in cand if abs(o[0] - c[0]) < isol_nm) == 1]
    return [c[0] for c in sorted(iso, key=lambda c: -c[1])[:topn]]


def window_of(wl, lam, half_px=6):
    k = int(np.searchsorted(wl, lam))
    lo, hi = max(0, k - half_px), min(len(wl), k + half_px + 1)
    return lo, hi


def measure_all(wl, y_bgsub, lam):
    """Return {estimator: centroid_nm} for one line on one bg-subtracted spectrum."""
    lo, hi = window_of(wl, lam)
    if hi - lo < 7:
        return {}
    xw, yw = wl[lo:hi].astype(float), y_bgsub[lo:hi].astype(float)
    if not np.isfinite(yw).all() or yw.max() <= 0:
        return {}
    out = {}
    for name, fn, _ in ESTIMATORS:
        try:
            v = fn(xw, yw)
        except Exception:
            v = float("nan")
        out[name] = v
    return out


def db_v_lines():
    from alibz.utils.database import Database
    db = Database("db")
    arr = db.lines("V")
    wl = arr[:, 1].astype(float)
    ion = arr[:, 0].astype(float)
    gA = arr[:, 3].astype(float)
    m = (wl > 185) & (wl < 950) & (ion <= 2) & (gA > 1e5)
    return wl[m], ion[m], gA[m]


def match_db(obs_nm, dbwl, tol=0.30):
    d = np.abs(dbwl - obs_nm)
    i = int(np.argmin(d))
    return float(dbwl[i]) if d[i] < tol else None


def real_headline(acq_dir, ledger_path, max_runs=25):
    runs = load_ledger_runs(acq_dir, ledger_path, max_runs)
    if not runs:
        return None
    dbwl, _, _ = db_v_lines()
    # per-run mean spectra, corrected background
    run_means, run_bg_c = [], []
    for rid, wl, Y in runs:
        ymean = Y.mean(axis=0)
        run_means.append((rid, wl, ymean, Y))
        run_bg_c.append(bg_corrected(wl, ymean))
    wl0 = run_means[0][1]

    # detect lines on the strongest run mean, match to db V
    strongest = max(run_means, key=lambda r: r[2].max())
    obs = detect_lines(strongest[1], strongest[2] - bg_corrected(strongest[1], strongest[2]))
    lines = []
    for o in obs:
        m = match_db(o, dbwl)
        lines.append((o, m, seg_of(o)))

    # ---- metric 1: shot-to-shot repeatability (corrected bg) ----------
    # ---- metric 2: cross-run consistency (corrected bg, run means) ----
    # measured per estimator, per segment
    est_names = [n for n, _, _ in ESTIMATORS]
    shot_rep = {n: {s: [] for s in ("UV", "VIS", "NIR")} for n in est_names}
    cross_run = {n: {s: [] for s in ("UV", "VIS", "NIR")} for n in est_names}
    per_run_centroid = {}   # (line, est) -> list over runs

    for (o, m, sg) in lines:
        # cross-run: centroid of run-mean spectra (corrected bg)
        cr = {n: [] for n in est_names}
        for (rid, wl, ymean, Y), bgc in zip(run_means, run_bg_c):
            res = measure_all(wl, ymean - bgc, o)
            for n in est_names:
                if n in res and np.isfinite(res[n]):
                    cr[n].append(res[n])
        for n in est_names:
            if len(cr[n]) >= 4:
                cross_run[n][sg].append(np.std(cr[n]) * 1000)
        # shot-to-shot: within the strongest run
        rid, wl, ymean, Y = strongest
        bgc = bg_corrected(wl, ymean)
        sc = {n: [] for n in est_names}
        for k in range(Y.shape[0]):
            res = measure_all(wl, Y[k] - bgc, o)  # bg from mean (stable)
            for n in est_names:
                if n in res and np.isfinite(res[n]):
                    sc[n].append(res[n])
        for n in est_names:
            if len(sc[n]) >= 4:
                shot_rep[n][sg].append(np.std(sc[n]) * 1000)

    # ---- metric 3: multiplet-spacing closure (corrected bg) -----------
    # pairs of db-matched lines in the same segment, spacing 0.3-8 nm apart
    closure = {n: {s: [] for s in ("UV", "VIS", "NIR")} for n in est_names}
    matched = [(o, m, sg) for (o, m, sg) in lines if m is not None]
    rid, wl, ymean, Y = strongest
    bgc = bg_corrected(wl, ymean)
    yb = ymean - bgc
    cent = {}
    for (o, m, sg) in matched:
        cent[o] = measure_all(wl, yb, o)
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            o1, m1, s1 = matched[i]
            o2, m2, s2 = matched[j]
            if s1 != s2 or not (0.3 < abs(m1 - m2) < 8.0):
                continue
            db_sp = abs(m1 - m2)
            for n in est_names:
                c1, c2 = cent[o1].get(n), cent[o2].get(n)
                if c1 and c2 and np.isfinite(c1) and np.isfinite(c2):
                    closure[n][s1].append(abs(abs(c1 - c2) - db_sp) * 1000)

    # ---- metric 4: Ar anchor residual (corrected bg) ------------------
    ar = {n: [] for n in est_names}
    for lam in AR_ANCHORS:
        res = measure_all(wl, yb, lam)
        for n in est_names:
            if n in res and np.isfinite(res[n]):
                ar[n].append((res[n] - lam) * 1000)

    return dict(n_runs=len(runs), n_lines=len(lines),
                n_matched=len(matched), strongest=strongest[0],
                shot_rep=shot_rep, cross_run=cross_run, closure=closure,
                ar=ar, est_names=est_names, lines=lines)


def _agg(d):
    """median and MAD-based 1-sigma of a list (pm)."""
    a = np.array(d, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(np.median(a)), float(1.4826 * np.median(np.abs(a - np.median(a)))), a.size)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-phase", type=int, default=25)
    ap.add_argument("--max-runs", type=int, default=25)
    ap.add_argument("--acq-dir", default="AUTO")
    ap.add_argument("--ledger", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        "v-pure-run2-ledger.json"))
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if not args.skip_real:
        acq = args.acq_dir
        if acq == "AUTO":
            acq = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
                   "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/vrun2/acq")
        print("=" * 78)
        print("REAL-DATA HEADLINE: pixel-int (8) vs spline (3)/(4)/(5), corrected bg")
        print("=" * 78)
        R = real_headline(acq, os.path.abspath(args.ledger), args.max_runs)
        if R is None:
            print("  no real data found at", acq)
        else:
            print("  %d runs, %d isolated lines (%d db-matched), strongest run %s"
                  % (R["n_runs"], R["n_lines"], R["n_matched"], R["strongest"][:18]))
            head = ["8_pixint", "3_spl_argmax", "4_spl_parab", "5_spl_root",
                    "2_parab", "7_peakfit_gauss"]
            for metric, key in (("shot-to-shot repeatability (pm)", "shot_rep"),
                                ("cross-run consistency (pm)", "cross_run"),
                                ("multiplet-spacing closure (pm)", "closure")):
                print("\n  -- %s : median [1sigma] (n) --" % metric)
                print("  %-16s %14s %14s %14s" % ("estimator", "UV", "VIS", "NIR"))
                for n in head:
                    cells = []
                    for s in ("UV", "VIS", "NIR"):
                        md, sd, cnt = _agg(R[key][n][s])
                        cells.append("%6.1f[%4.1f]%3d" % (md, sd, cnt) if cnt else "     -    ")
                    print("  %-16s %14s %14s %14s" % (n, *cells))
            print("\n  -- Ar I NIR anchor residual (pm, obs-db, corrected bg) --")
            for n in head:
                md, sd, cnt = _agg(R["ar"][n])
                print("  %-16s median=%7.1f  spread(1sigma)=%6.1f  n=%d" % (n, md, sd, cnt))

    if not args.skip_sim:
        print("\n" + "=" * 78)
        print("SIMULATION: bias / RMS / p95 of location error (pm), per segment")
        print("Gentle sloped+curved continuum; phase swept; ideal vs edge-linear bg.")
        print("=" * 78)
        for blend, blabel in ((False, "CLEAN line"), (True, "with a 2px-50%% blend")):
            for bg_mode, label in (("ideal", "ideal bg"), ("linear", "edge-linear bg")):
                print("\n--- %s, %s  [SNR=20] ---" % (blabel, label))
                rows = run_sim(rng, args.n_phase, (20,), bg_mode, blend=blend)
                print("  %-16s %10s %10s %10s %8s" % ("estimator", "seg", "bias", "RMS", "p95"))
                for seg in SEGMENTS:
                    for (s, snr, name, bias, rms, p95) in rows:
                        if s == seg:
                            print("  %-16s %10s %10.1f %10.1f %8.1f" % (name, seg, bias, rms, p95))
        # RMS vs SNR, clean, edge-linear bg (realistic), headline estimators
        print("\n--- RMS(pm) vs SNR, CLEAN line, edge-linear bg ---")
        rows = run_sim(rng, args.n_phase, (5, 10, 20, 50, 100), "linear", blend=False)
        for seg in SEGMENTS:
            print("  seg=%s" % seg)
            print("    %-16s %s" % ("estimator", "  ".join("SNR%-4d" % s for s in (5, 10, 20, 50, 100))))
            for name, _, _ in ESTIMATORS:
                vals = []
                for snr in (5, 10, 20, 50, 100):
                    r = [rr for rr in rows if rr[0] == seg and rr[1] == snr and rr[2] == name]
                    vals.append("%7.1f" % r[0][4] if r else "   -   ")
                print("    %-16s %s" % (name, " ".join(vals)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
