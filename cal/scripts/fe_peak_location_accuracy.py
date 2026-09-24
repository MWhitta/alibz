#!/usr/bin/env python3
"""DECISIVE TEST: real-data peak-location ACCURACY on Fe (2026-09-22 native runs).

Truth = Fe I/II AIR lab wavelengths from alibz/utils/database.py (good to <1 pm).
Metric = per run and segment, the RMS of (obs - db) about a smooth CUBIC-in-lambda
calibration curve, with iterative 3xMAD outlier clipping (clipped count reported).
Each accepted line sits at its own sub-pixel phase, so phase-dependent estimator
bias shows up as SCATTER about the smooth curve.  Estimators are paired on the
SAME accepted lines and compared with bootstrap CIs on the RMS differences.

Estimators:
  2  native 3-point parabola      (fe_plasma_analysis.subpixel_center)
  4  spline-parab   (CubicSpline not-a-knot -> 1/30 nm -> parabola)
  5  spline-root    (continuous max = root of the spline derivative)
  7  peakfit Gaussian             (alibz/utils/peakfit.gaussian_subpixel_center)
  8  pixel-integrated Gaussian, FREE width
  8e pixel-integrated fit with the EMPIRICAL per-segment line shape (built by
     stacking many isolated Fe lines at their fitted sub-pixel positions,
     iterated; fits only amplitude + centre + local linear background;
     pooled per-segment across all lines/runs -- NOT strict leave-one-out, so
     8e is if anything slightly flattered; it still only ties (8)).

Matching: per run, a validated shift model (estimate_shift_model: isolated
dominant Fe I/II anchors + Ar) predicts each db line's position; a line is
accepted only if it is strength-DOMINANT in its neighbourhood (blend screen),
a single clean peak sits within ~1 px of the prediction, and it is not an
optically-thick resonance line (screened + clipped).  Each run has its own
wavelength map (a 13:39 recal split the day), so runs are treated separately.

Data: local copy of the moissanite Fe shots under the sibling scratchpad
(--acq / --ledger).  Deterministic.  numpy+scipy+alibz.  --help for options.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peak_location_spline_vs_native import (            # noqa: E402
    bg_corrected, e_parab_native, e_spline_parab, e_spline_root, e_refine,
    e_pixint, seg_of, SEG_EDGES, AR_ANCHORS)
from native_grid_estimator_sim import pixel_edges, gauss_integral  # noqa: E402

DEF_ACQ = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
           "39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/acq")
DEF_LEDGER = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
              "39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/ledger.json")

# Known optically-thick / self-reversed Fe I resonance lines to exclude a priori
# (the coordinator flagged 438.35 ~-0.32 nm displaced); clipping catches the rest.
THICK_FE = [248.327, 344.061, 358.119, 371.993, 373.486, 373.713, 374.556,
            385.991, 404.581, 438.355, 489.149, 495.760, 516.749, 526.954]
KB = 8.617333262e-5


# --------------------------------------------------------------------------
def load_fe_runs(acq_dir, ledger_path):
    led = json.load(open(ledger_path))
    runs = []
    for meta in sorted(led, key=lambda r: r["created_at"]):
        if not meta.get("run", "").startswith("run-") or "Fe" not in meta.get("sample", ""):
            continue
        sd = os.path.join(acq_dir, meta["run"], "shots")
        fs = sorted(glob.glob(os.path.join(sd, "shot-*.csv")))
        if len(fs) < 5:
            continue
        shots = [np.loadtxt(f, delimiter=",", skiprows=1) for f in fs]
        wl = shots[0][:, 0]
        Y = np.array([s[:, 1] for s in shots])
        runs.append((meta["run"], meta["created_at"], wl, Y))
    return runs


def fe_db(dbpath="db"):
    from alibz.utils.database import Database
    db = Database(dbpath)
    a = db.lines("Fe")
    ion = a[:, 0].astype(float); wl = a[:, 1].astype(float)
    gA = a[:, 3].astype(float); Ei = a[:, 4].astype(float); Ek = a[:, 5].astype(float)
    keep = (ion >= 1) & (ion <= 2) & (wl > 186) & (wl < 948) & (gA > 1e4)
    return dict(wl=wl[keep], gA=gA[keep], Ei=Ei[keep], Ek=Ek[keep], ion=ion[keep]), db


def dominant_lines(F, T=8000.0, win_nm=0.15, dom=2.5):
    """Fe lines whose Boltzmann strength dominates their +-win_nm neighbourhood
    by dom x (blend screen), excluding thick resonance lines."""
    s = F["gA"] * np.exp(-F["Ek"] / (KB * T))
    wl = F["wl"]
    order = np.argsort(wl)
    wl, s = wl[order], s[order]
    out = []
    for i in range(wl.size):
        if any(abs(wl[i] - t) < 0.03 for t in THICK_FE):
            continue
        lo = np.searchsorted(wl, wl[i] - win_nm)
        hi = np.searchsorted(wl, wl[i] + win_nm)
        nb = np.concatenate([s[lo:i], s[i + 1:hi]])
        if nb.size == 0 or s[i] >= dom * np.max(nb):
            out.append((float(wl[i]), float(s[i])))
    return out


# --------------------------------------------------------------------------
# empirical per-segment line shape (for estimator 8e)
# --------------------------------------------------------------------------
class EmpiricalShape:
    """Super-sampled sampled-response shape S(u), u=(lambda-centre)/pitch."""

    def __init__(self, ugrid=None):
        self.ugrid = ugrid if ugrid is not None else np.arange(-6, 6.01, 0.1)
        self.S = None

    def build(self, stacks, exclude=None):
        """stacks: list of (u_array, s_array) per contributing line."""
        us, ss = [], []
        for k, (u, s) in enumerate(stacks):
            if exclude is not None and k == exclude:
                continue
            us.append(u); ss.append(s)
        u = np.concatenate(us); s = np.concatenate(ss)
        S = np.full(self.ugrid.size, np.nan)
        half = 0.5 * (self.ugrid[1] - self.ugrid[0])
        for i, uc in enumerate(self.ugrid):
            m = np.abs(u - uc) <= (half + 0.15)
            if np.count_nonzero(m) >= 3:
                S[i] = np.median(s[m])
        ok = np.isfinite(S)
        S = np.interp(self.ugrid, self.ugrid[ok], S[ok])
        S = np.clip(S, 0.0, None)
        self.S = S / S.max()
        return self

    def __call__(self, u):
        return np.interp(u, self.ugrid, self.S, left=0.0, right=0.0)


def _window(wl, yb, lam, half_px=6):
    k = int(np.searchsorted(wl, lam))
    lo, hi = max(0, k - half_px), min(len(wl), k + half_px + 1)
    return wl[lo:hi].astype(float), yb[lo:hi].astype(float)


def fit_pixint_free(xw, yw):
    """estimator 8 as a standalone (returns centre nm)."""
    return e_pixint(xw, yw)


def stack_for(xw, yw, center, pitch):
    j = int(np.argmax(yw))
    base = np.median(np.concatenate([yw[:2], yw[-2:]]))
    amp = float(yw[j] - base)
    if amp <= 0:
        return None
    u = (xw - center) / pitch
    s = (yw - base) / amp
    return u, s


def fit_empirical(xw, yw, shape, pitch, c0):
    """Fit A*S((x-c)/pitch)+b0+b1*(x-c); return centre nm."""
    xc = xw - c0

    def model(p):
        A, c, b0, b1 = p
        return A * shape((xw - c) / pitch) + b0 + b1 * xc - yw
    j = int(np.argmax(yw))
    base = np.median(np.concatenate([yw[:2], yw[-2:]]))
    A0 = max(float(yw[j] - base), 1.0)
    lb = [0.0, xw[0], -np.inf, -np.inf]
    ub = [np.inf, xw[-1], np.inf, np.inf]
    try:
        sol = least_squares(model, [A0, c0, base, 0.0], bounds=(lb, ub), max_nfev=3000)
        return float(sol.x[1])
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
ESTS = ["2_parab", "4_spl_parab", "5_spl_root", "7_peakfit", "8_pixint", "8e_empirical"]
_FN = {"2_parab": e_parab_native, "4_spl_parab": e_spline_parab,
       "5_spl_root": e_spline_root, "7_peakfit": e_refine, "8_pixint": e_pixint}


def measure_run(wl, ymean, F, dom_lines, shapes, snr_min=6.0):
    """Return per-segment list of dicts: db, {est: obs}, for accepted lines."""
    from line_physics_analysis import estimate_shift_model, db_to_obs
    yb = ymean - bg_corrected(wl, ymean)
    model = estimate_shift_model(wl, ymean, _DB, "Fe")
    accepted = {s: [] for s in ("UV", "VIS", "NIR")}
    stacks = {s: [] for s in ("UV", "VIS", "NIR")}
    for lam_db, strength in dom_lines:
        sg = seg_of(lam_db)
        pred = db_to_obs(model, lam_db)
        kp = int(np.searchsorted(wl, pred))
        pitch = float(np.median(np.diff(wl[max(0, kp - 5):kp + 6])))
        shalf = max(0.12, 3.0 * pitch)         # >=3 px search half-width
        lo = np.searchsorted(wl, pred - shalf)
        hi = np.searchsorted(wl, pred + shalf)
        if hi - lo < 3:
            continue
        j = lo + int(np.argmax(yb[lo:hi]))
        # single clean peak within +-max(0.18, 2px) nm (blend screen on the data)
        bhalf = max(0.18, 2.0 * pitch)
        w0 = np.searchsorted(wl, wl[j] - bhalf); w1 = np.searchsorted(wl, wl[j] + bhalf)
        seg_y = yb[w0:w1]
        base = np.median(np.concatenate([yb[max(0, w0 - 4):w0], yb[w1:w1 + 4]])) if w0 >= 4 else 0.0
        h = float(yb[j] - base)
        noise = 1.4826 * np.median(np.abs(seg_y - np.median(seg_y)))
        if noise <= 0 or h / noise < snr_min:
            continue
        nmax = sum(1 for i in range(w0 + 1, w1 - 1)
                   if yb[i] > yb[i - 1] and yb[i] > yb[i + 1] and yb[i] > 0.4 * yb[j])
        if nmax != 1:
            continue
        if abs(float(wl[j]) - pred) > max(0.09, 0.7 * pitch):
            continue
        xw, yw = _window(wl, yb, float(wl[j]))
        if xw.size < 9:
            continue
        pitch = float(np.median(np.diff(xw)))
        obs = {}
        for name in ("2_parab", "4_spl_parab", "5_spl_root", "7_peakfit", "8_pixint"):
            try:
                obs[name] = _FN[name](xw, yw)
            except Exception:
                obs[name] = float("nan")
        c0 = obs["8_pixint"] if np.isfinite(obs["8_pixint"]) else float(wl[j])
        accepted[sg].append(dict(db=lam_db, obs=obs, xw=xw, yw=yw, pitch=pitch, c0=c0))
        st = stack_for(xw, yw, c0, pitch)
        if st is not None:
            stacks[sg].append(st)
    return accepted, stacks


def cubic_rms(db, obs, clip=3.0, iters=5):
    """RMS of (obs-db) about a cubic-in-lambda fit, iterative MAD clipping."""
    db = np.asarray(db, float); obs = np.asarray(obs, float)
    good = np.isfinite(obs) & np.isfinite(db)
    if good.sum() < 6:
        return float("nan"), int(good.sum()), 0
    x = db[good]; r = (obs - db)[good]
    keep = np.ones(x.size, bool)
    ncl = 0
    for _ in range(iters):
        deg = 3 if keep.sum() >= 8 else (2 if keep.sum() >= 5 else 1)
        c = np.polyfit(x[keep] - x.mean(), r[keep], deg)
        res = r - np.polyval(c, x - x.mean())
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        newkeep = np.abs(res) < clip * max(mad, 1e-4)
        if newkeep.sum() < 5 or np.array_equal(newkeep, keep):
            keep = newkeep; break
        keep = newkeep
    ncl = int((~keep).sum())
    rms = float(np.sqrt(np.mean(res[keep] ** 2))) * 1000.0
    return rms, int(keep.sum()), ncl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acq", default=DEF_ACQ)
    ap.add_argument("--ledger", default=DEF_LEDGER)
    ap.add_argument("--max-runs", type=int, default=0, help="0=all")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    global _DB
    F, _DB = fe_db()
    dom = dominant_lines(F)
    print("Fe dominant (blend-screened) db lines:", len(dom))
    runs = load_fe_runs(args.acq, args.ledger)
    if args.max_runs:
        runs = runs[:args.max_runs]
    print("Fe runs:", len(runs))

    # pooled empirical shape per segment (built once from all runs' stacks)
    shapes = {s: EmpiricalShape() for s in ("UV", "VIS", "NIR")}

    # collect per-run accepted lines and pooled stacks
    per_run = []
    pooled_stacks = {s: [] for s in ("UV", "VIS", "NIR")}
    for rid, ts, wl, Y in runs:
        acc, st = measure_run(wl, Y.mean(axis=0), F, dom, shapes)
        per_run.append((rid, ts, wl, Y, acc))
        for s in st:
            pooled_stacks[s].extend(st[s])
    for s in shapes:
        if len(pooled_stacks[s]) >= 8:
            shapes[s].build(pooled_stacks[s])
    print("pooled shape lines per segment:",
          {s: len(pooled_stacks[s]) for s in pooled_stacks})

    # add 8e measurement per accepted line (pooled per-segment empirical shape)
    for (rid, ts, wl, Y, acc) in per_run:
        for s in ("UV", "VIS", "NIR"):
            if shapes[s].S is None:
                continue
            for rec in acc[s]:
                rec["obs"]["8e_empirical"] = fit_empirical(
                    rec["xw"], rec["yw"], shapes[s], rec["pitch"], rec["c0"])

    # ---- per-segment RMS pooled over runs (paired) --------------------
    print("\n" + "=" * 78)
    print("Fe ACCURACY: RMS of (obs-db) about a per-run cubic-in-lambda, pooled")
    print("over runs (median [IQR] of per-run RMS, pm); paired on same lines")
    print("=" * 78)
    # aligned per-run RMS (clipped) so estimators can be paired BY RUN
    seg_run_rms = {s: {e: {} for e in ESTS} for s in ("UV", "VIS", "NIR")}
    seg_run_ncl = {s: {e: {} for e in ESTS} for s in ("UV", "VIS", "NIR")}
    seg_nlines = {s: [] for s in ("UV", "VIS", "NIR")}
    for ri, (rid, ts, wl, Y, acc) in enumerate(per_run):
        for s in ("UV", "VIS", "NIR"):
            recs = acc[s]
            if len(recs) < 6:
                continue
            db = [r["db"] for r in recs]
            seg_nlines[s].append(len(recs))
            for e in ESTS:
                obs = [r["obs"].get(e, np.nan) for r in recs]
                rms, nk, ncl = cubic_rms(db, obs)
                if np.isfinite(rms):
                    seg_run_rms[s][e][ri] = rms
                    seg_run_ncl[s][e][ri] = ncl
    print("%-14s %18s %18s %18s" % ("estimator", "UV", "VIS", "NIR"))
    for e in ESTS:
        cells = []
        for s in ("UV", "VIS", "NIR"):
            v = np.array(list(seg_run_rms[s][e].values()))
            if v.size:
                cells.append("%5.1f [%4.1f-%4.1f] %2d" %
                             (np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.size))
            else:
                cells.append("        -         ")
        print("%-14s %18s %18s %18s" % (e, *cells))
    print("\nlines/run/segment (median): " +
          ", ".join("%s=%.0f" % (s, np.median(seg_nlines[s]) if seg_nlines[s] else 0)
                    for s in ("UV", "VIS", "NIR")))
    print("median clipped/run (8_pixint): " +
          ", ".join("%s=%.0f" % (s, np.median(list(seg_run_ncl[s]["8_pixint"].values()))
                                  if seg_run_ncl[s]["8_pixint"] else 0)
                    for s in ("UV", "VIS", "NIR")))

    # ---- paired bootstrap OVER RUNS on the clipped per-run RMS ---------
    print("\n" + "=" * 78)
    print("Paired accuracy vs 8_pixint: median per-run dRMS (est - 8_pixint) with")
    print("bootstrap-over-runs 95%% CI (pm).  Positive => estimator WORSE than 8.")
    print("=" * 78)
    ref = "8_pixint"
    for s in ("UV", "VIS", "NIR"):
        runs_ref = seg_run_rms[s][ref]
        print("  seg=%s  8_pixint median RMS=%.1f pm (n_runs=%d)"
              % (s, np.median(list(runs_ref.values())) if runs_ref else float("nan"),
                 len(runs_ref)))
        for e in ESTS:
            if e == ref:
                continue
            pairs = [(seg_run_rms[s][e][ri] - runs_ref[ri])
                     for ri in runs_ref if ri in seg_run_rms[s][e]]
            if len(pairs) < 6:
                print("    %-14s (too few paired runs)" % e); continue
            pairs = np.array(pairs)
            bs = np.array([np.median(pairs[rng.integers(0, pairs.size, pairs.size)])
                           for _ in range(args.boot)])
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            tag = ("8 better" if lo > 0 else ("8 worse" if hi < 0 else "tie"))
            print("    %-14s dRMS=%+6.1f  95%%CI[%+6.1f,%+6.1f]  %s"
                  % (e, np.median(pairs), lo, hi, tag))

    # ---- 8e vs others on the V run2 Ar lines (NIR) --------------------
    print("\n" + "=" * 78)
    print("Ar NIR anchors (V run2), obs-db per estimator (pm): median [spread]")
    print("=" * 78)
    _ar_check(rng)
    return 0


def _ar_check(rng):
    vacq = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
            "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/vrun2/acq")
    runs = sorted(glob.glob(os.path.join(vacq, "run-*", "shots")))
    if not runs:
        print("  V run2 not found"); return
    # strongest run mean
    best = None
    for sd in runs:
        fs = sorted(glob.glob(os.path.join(sd, "shot-*.csv")))
        if len(fs) < 8:
            continue
        Y = np.array([np.loadtxt(f, delimiter=",", skiprows=1)[:, 1] for f in fs])
        wl = np.loadtxt(fs[0], delimiter=",", skiprows=1)[:, 0]
        ym = Y.mean(axis=0)
        if best is None or ym.max() > best[2].max():
            best = (wl, Y, ym)
    wl, Y, ym = best
    yb = ym - bg_corrected(wl, ym)
    # empirical NIR shape from V's own strong isolated NIR lines
    shp = EmpiricalShape(); stacks = []
    for i in range(6, len(yb) - 6):
        if wl[i] < 620:
            continue
        if yb[i] > yb[i - 1] and yb[i] > yb[i + 1] and yb[i] >= yb[i - 2] and yb[i] >= yb[i + 2]:
            base = np.median(np.concatenate([yb[i - 6:i - 3], yb[i + 4:i + 7]]))
            noise = 1.4826 * np.median(np.abs(yb[i - 6:i + 7] - np.median(yb[i - 6:i + 7])))
            if noise > 0 and (yb[i] - base) / noise > 20 and \
               sum(1 for o in range(6, len(yb) - 6) if abs(wl[o] - wl[i]) < 0.35 and
                   yb[o] > yb[o - 1] and yb[o] > yb[o + 1]) == 1:
                xw, yw = _window(wl, yb, float(wl[i]))
                p = float(np.median(np.diff(xw)))
                c0 = e_pixint(xw, yw)
                st = stack_for(xw, yw, c0, p)
                if st is not None:
                    stacks.append(st)
    if len(stacks) >= 6:
        shp.build(stacks)
    for name in ("2_parab", "5_spl_root", "7_peakfit", "8_pixint", "8e_empirical"):
        res = []
        for lam in AR_ANCHORS:
            xw, yw = _window(wl, yb, lam)
            if xw.size < 9 or yw.max() <= 0:
                continue
            if name == "8e_empirical":
                if shp.S is None:
                    continue
                p = float(np.median(np.diff(xw)))
                v = fit_empirical(xw, yw, shp, p, e_pixint(xw, yw))
            else:
                v = _FN[name](xw, yw)
            if np.isfinite(v):
                res.append((v - lam) * 1000)
        if res:
            print("  %-14s median=%7.1f  spread=%5.1f  n=%d"
                  % (name, np.median(res), np.std(res), len(res)))


if __name__ == "__main__":
    sys.exit(main())
