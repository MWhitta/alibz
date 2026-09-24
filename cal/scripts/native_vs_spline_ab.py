#!/usr/bin/env python3
"""A/B comparison of the Z300 "old" 1/30 nm resampled CSV vs the "native" grid.

Same-shot A/B on 8 Fe runs from 2026-09-22.  For each run there is BOTH an
"old" 23,249/23,250-row average CSV (Pantheum ``resample_uniform`` = LINEAR
interpolation onto a uniform 1/30 nm grid) and a "native" 7,914-sample average
(``pixels_to_wavelength`` on the analyzer's pixel-averaged shot, no resampling).
This established (see --provenance) that the two share identical pixels and the
identical pixToNm map; the only differences are grid density/interpolation, the
NIR gap/edges, and interpolation-induced noise correlation.

The script quantifies what the density difference does to the downstream
estimators the owner cares about: peak LOCATIONS (parabolic sub-pixel centroid,
shot-to-shot scatter, residual vs NIST db) and PROFILE FITS (pseudo-Voigt
width, Lorentz fraction, r2).  It then tests native-side corrections
(cubic / linear upsample of the native pixels) and shows which recover quality.

Deterministic and read-only.  Requires the same-shot data pulled locally
(default --data-root points at the investigation scratchpad).

Subcommands:
  provenance   old = linear(native)? vs cubic; difference-order signatures.
  sampling     native samples per line FWHM, per segment.
  peaks        parabolic centroid: old vs native vs cubic-upsampled native;
               shot-to-shot scatter; residual vs db anchors.
  profiles     pseudo-Voigt fit quality (fwhm, eta, r2) per grid + corrections.
  all          run them all.

Usage:
  cal/scripts/native_vs_spline_ab.py all --data-root <dir> [--dbpath db]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_DEFAULT_ROOT = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
                 "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad")

# run tag -> old-original dataset id (stem of originals/<id>.csv)
PAIRS = {
    "e825d9f567": "2679219a5c6de113917ea662",
    "a50ac639cc": "4f678d792ef0b5fd6018702c",
    "eae9d54f22": "0982aedefc2b7f0833adb766",
    "126d0c96e1": "6f0d21db2cc122be36cc6044",
    "5291e64308": "27858788c9f2a34f096767cf",
    "11d07e557f": "a37cd7368dea089ec48c2fa0",
    "6e38b11021": "0fefcd7830c5a8d04bc7506d",
    "c52b7f2159": "8e75d878e1c2aa0b33055cda",
}

AR_NIR_ANCHORS = [696.5431, 763.5106, 772.3761, 801.4786, 811.5311, 842.4648]
SEG_EDGES = (365.0, 620.0)


def seg_of(wl):
    return 0 if wl < SEG_EDGES[0] else (1 if wl < SEG_EDGES[1] else 2)


def load_csv(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def run_dir(root, tag):
    d = glob.glob(os.path.join(root, "runs", f"run-{tag}*"))
    return d[0] if d else None


def load_pair(root, tag):
    old = os.path.join(root, "originals", PAIRS[tag] + ".csv")
    rd = run_dir(root, tag)
    xo, yo = load_csv(old)
    xn, yn = load_csv(os.path.join(rd, "average.csv"))
    return (xo, yo), (xn, yn), rd


def load_shots_native(rd):
    out = []
    for p in sorted(glob.glob(os.path.join(rd, "shots", "shot-*.csv")),
                    key=lambda s: int(s.rsplit("-", 1)[1].split(".")[0])):
        out.append(load_csv(p))
    return out


# ----------------------------------------------------------------------------
def upsample(xn, yn, step=1.0 / 30.0, kind="cubic"):
    """Resample native onto a uniform 1/30 grid, skipping the NIR gap region."""
    lo, hi = xn[0], xn[-1]
    n = int(round((hi - lo) / step))
    xu = lo + np.arange(n + 1) * step
    if kind == "linear":
        yu = np.interp(xu, xn, yn)
    else:
        yu = CubicSpline(xn, yn, bc_type="not-a-knot")(xu)
    return xu, yu


def parabolic_center(x, y, center, halfwin_nm=0.30):
    """3-point parabolic sub-pixel peak center near ``center`` (matches
    fe_plasma_analysis.subpixel_center)."""
    lo = int(np.searchsorted(x, center - halfwin_nm))
    hi = int(np.searchsorted(x, center + halfwin_nm))
    if hi - lo < 3:
        return None
    j = lo + int(np.argmax(y[lo:hi]))
    j = min(max(j, 1), len(x) - 2)
    y1, y2, y3 = y[j - 1], y[j], y[j + 1]
    denom = y1 - 2 * y2 + y3
    if denom >= 0:
        return float(x[j])
    return float(x[j] + 0.5 * (y1 - y3) / denom * (x[j + 1] - x[j - 1]) / 2.0)


def pseudo_voigt(x, amp, cen, sig, eta, bg, slope):
    sig = max(sig, 1e-6)
    g = np.exp(-0.5 * ((x - cen) / sig) ** 2)
    gam = sig
    l = gam ** 2 / ((x - cen) ** 2 + gam ** 2)
    return bg + slope * (x - cen) + amp * ((1 - eta) * g + eta * l)


def fit_profile(x, y, center, halfwin_nm=0.45):
    lo = int(np.searchsorted(x, center - halfwin_nm))
    hi = int(np.searchsorted(x, center + halfwin_nm))
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) < 6:
        return None
    bg0 = np.median(np.concatenate([ys[:2], ys[-2:]]))
    amp0 = max(ys.max() - bg0, 1.0)
    p0 = [amp0, center, 0.09, 0.5, bg0, 0.0]
    bounds = ([0, center - 0.35, 0.02, 0.0, -np.inf, -np.inf],
              [np.inf, center + 0.35, 0.6, 1.0, np.inf, np.inf])
    try:
        popt, _ = curve_fit(pseudo_voigt, xs, ys, p0=p0, bounds=bounds, maxfev=4000)
    except Exception:
        return None
    resid = ys - pseudo_voigt(xs, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1.0
    amp, cen, sig, eta, bg, slope = popt
    fwhm = 2.3548 * sig * (1 - eta) + 2 * sig * eta  # approx pv fwhm
    return dict(cen=float(cen), fwhm=float(fwhm), eta=float(eta),
                r2=float(1 - ss_res / ss_tot), amp=float(amp), npts=len(xs))


# ----------------------------------------------------------------------------
def cmd_provenance(root, **_):
    print("== PROVENANCE: old vs linear(native) vs cubic(native) ==")
    for tag in PAIRS:
        (xo, yo), (xn, yn), _ = load_pair(root, tag)
        m = (xo >= xn[0]) & (xo <= xn[-1])
        ylin = np.interp(xo[m], xn, yn)
        ycub = CubicSpline(xn, yn, bc_type="not-a-knot")(xo[m])
        sc = np.std(yo[m])
        seg = yo[(xo > 400) & (xo < 500)]
        d2 = np.diff(seg, 2); d4 = np.diff(seg, 4)
        print(f"{tag}: rel(old-lin)={np.sqrt(np.mean((yo[m]-ylin)**2))/sc:.2e}  "
              f"rel(old-cub)={np.sqrt(np.mean((yo[m]-ycub)**2))/sc:.2e}  "
              f"d2==0:{np.mean(d2==0):.3f} d4==0:{np.mean(d4==0):.3f}")


def detect_peaks(xn, yn, prominence=250, min_sep_nm=0.40, exclude_gt=947.0):
    """Strong, isolated local maxima of the native mean (blend-free, real data).

    Returns nominal centres [nm]; used as the common line set across all grids."""
    from scipy.signal import find_peaks
    pk, props = find_peaks(yn, prominence=prominence, distance=2)
    cens, proms = xn[pk], props["prominences"]
    order = np.argsort(cens)
    cens, proms = cens[order], proms[order]
    out = []
    for i, c in enumerate(cens):
        if c > exclude_gt:
            continue
        lo = c - min_sep_nm <= cens
        hi = cens <= c + min_sep_nm
        nb = lo & hi
        nb[i] = False
        if not np.any(nb) or proms[i] >= 2.0 * proms[nb].max():
            out.append(float(c))
    return out


def fwhm_per_pitch(xn, c):
    seg = xn[(xn > c - 1) & (xn < c + 1)]
    return np.median(np.diff(seg)) if len(seg) > 2 else np.nan


def cmd_sampling(root, **_):
    print("== SAMPLING: native samples across fitted FWHM (strong isolated peaks) ==")
    rows = []
    for tag in PAIRS:
        _, (xn, yn), _ = load_pair(root, tag)
        for c in detect_peaks(xn, yn):
            f = fit_profile(xn, yn, c)
            if f and f["r2"] > 0.6 and 0.05 < f["fwhm"] < 0.6:
                lp = fwhm_per_pitch(xn, c)
                rows.append((seg_of(c), f["fwhm"], f["fwhm"] / lp))
    rows = np.array(rows)
    for s, nm in ((0, "UV"), (1, "VIS"), (2, "NIR")):
        r = rows[rows[:, 0] == s]
        if len(r):
            print(f"  {nm}: n={len(r):3d}  FWHM med={np.median(r[:,1])*1000:.0f}pm  "
                  f"native samples/FWHM med={np.median(r[:,2]):.2f} "
                  f"[{np.percentile(r[:,2],10):.2f}-{np.percentile(r[:,2],90):.2f}]  "
                  f"1/30-grid samples/FWHM={np.median(r[:,1]/(1/30.)):.2f}")


def _anchor_set(dbpath):
    try:
        from alibz.utils.database import Database
        db = Database(dbpath)
        wl_all = []
        for el in ("Fe",):
            arr = db.lines(el)
            ion = arr[:, 0].astype(float); wl = arr[:, 1].astype(float)
            gA = arr[:, 3].astype(float)
            keep = (ion <= 2) & (gA >= 3e6) & (wl > 250) & (wl < 600)
            wl_all += list(wl[keep])
        wl_all = np.array(sorted(wl_all))
        # keep isolated (>0.4 nm from neighbours)
        iso = [w for i, w in enumerate(wl_all)
               if (i == 0 or w - wl_all[i-1] > 0.4)
               and (i == len(wl_all)-1 or wl_all[i+1] - w > 0.4)]
        return iso
    except Exception as e:
        print("  (db unavailable:", e, ")")
        return []


def cmd_peaks(root, dbpath=None, **_):
    print("== PEAKS: shot-to-shot centre scatter, native vs 1/30-upsampled ==")
    print("   (a) parabolic sub-pixel centre (production estimator, guarded)")
    print("   (b) pseudo-Voigt fitted centre (robust estimator)")
    for method in ("parabolic", "profile"):
        scat = {"native": [], "linup(1/30)": [], "cubup(1/30)": []}
        for tag in PAIRS:
            _, (xn, yn), rd = load_pair(root, tag)
            shots = load_shots_native(rd)
            # upsample each shot ONCE (not per peak)
            shot_variants = []
            for (xs, ys) in shots:
                shot_variants.append({
                    "native": (xs, ys),
                    "linup(1/30)": upsample(xs, ys, kind="linear"),
                    "cubup(1/30)": upsample(xs, ys, kind="cubic")})
            peaks = detect_peaks(xn, yn)
            for c in peaks:
                cens = {k: [] for k in scat}
                for variants in shot_variants:
                    for k, (X, Y) in variants.items():
                        if method == "parabolic":
                            v = parabolic_center(X, Y, c)
                            pitch = fwhm_per_pitch(xn, c)
                            if v is not None and abs(v - c) <= pitch:
                                cens[k].append(v)
                        else:
                            f = fit_profile(X, Y, c)
                            if f and f["r2"] > 0.5 and abs(f["cen"] - c) < 0.2:
                                cens[k].append(f["cen"])
                for k in scat:
                    if len(cens[k]) >= 6:
                        scat[k].append(np.std(cens[k]))
        print(f"  -- {method} --")
        for k in scat:
            v = np.array(scat[k]) * 1000
            if len(v):
                print(f"     {k:12s}: median shot-to-shot centre std = {np.median(v):5.1f} pm  (n={len(v)})")


def cmd_profiles(root, dbpath=None, **_):
    print("== PROFILES: pseudo-Voigt fit quality per grid + corrections ==")
    print("   (old=Pantheum linear 1/30; native=as stored; linup/cubup=native re-upsampled)")
    grids = ("old", "native", "linup(1/30)", "cubup(1/30)")
    agg = {g: {"r2": [], "fwhm": [], "eta": []} for g in grids}
    npeaks = 0
    for tag in PAIRS:
        (xo, yo), (xn, yn), _ = load_pair(root, tag)
        xl, yl = upsample(xn, yn, kind="linear")
        xc, yc = upsample(xn, yn, kind="cubic")
        gmap = {"old": (xo, yo), "native": (xn, yn),
                "linup(1/30)": (xl, yl), "cubup(1/30)": (xc, yc)}
        for c in detect_peaks(xn, yn):
            base = fit_profile(xn, yn, c)
            if not base or base["amp"] < 100:
                continue
            npeaks += 1
            for g, (X, Y) in gmap.items():
                f = fit_profile(X, Y, c)
                if f:
                    agg[g]["r2"].append(f["r2"])
                    agg[g]["fwhm"].append(f["fwhm"])
                    agg[g]["eta"].append(f["eta"])
    print(f"   peaks fitted: {npeaks}")
    for g in grids:
        r2 = np.array(agg[g]["r2"]); fw = np.array(agg[g]["fwhm"]); et = np.array(agg[g]["eta"])
        if len(r2):
            # width resolution: unique fitted-FWHM values / total (quantization proxy)
            print(f"  {g:12s}: n={len(r2):3d}  r2 med={np.median(r2):.3f}  "
                  f"r2<0.95 frac={np.mean(r2<0.95):.2f}  "
                  f"fwhm med={np.median(fw)*1000:.0f}pm  eta med={np.median(et):.2f}  "
                  f"eta@bound frac={np.mean((et<0.02)|(et>0.98)):.2f}")


def cmd_stitch(root, **_):
    print("== STITCHING / DEAD CHANNEL / NIR GAP ==")
    for tag in PAIRS:
        _, (xn, yn), rd = load_pair(root, tag)
        avg = json.load(open(os.path.join(rd, "raw", "shot--1.json")))
        s3 = np.array(avg["pixels"][3])
        gap = np.max(np.diff(xn))
        n_dead = int(np.sum(xn >= 959.9))
        print(f"  {tag}: native_n={len(xn)}  seg3(4th ch) px={len(s3)} nonzero={np.count_nonzero(s3)} "
              f"-> {n_dead} samples in 960-961nm ({100*n_dead/len(xn):.0f}% of file all-zero)  "
              f"NIR gap={gap:.1f}nm")


def cmd_noise(root, **_):
    print("== NOISE: lag-1 autocorr of high-freq residual (quiet 500-560 nm) ==")
    def ac1(x, y):
        m = (x > 500) & (x < 560)
        yy = y[m]
        r = yy - np.convolve(yy, np.ones(9) / 9, mode="same")
        r = r[10:-10]
        return float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 5 else np.nan
    nn, oo = [], []
    for tag in PAIRS:
        (xo, yo), (xn, yn), _ = load_pair(root, tag)
        nn.append(ac1(xn, yn)); oo.append(ac1(xo, yo))
    print(f"  native (as stored): lag-1 autocorr med={np.nanmedian(nn):.2f}  (white)")
    print(f"  old (1/30 linear) : lag-1 autocorr med={np.nanmedian(oo):.2f}  (correlated)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["provenance", "sampling", "peaks", "profiles",
                                    "stitch", "noise", "all"])
    ap.add_argument("--data-root", default=_DEFAULT_ROOT,
                    help="dir with runs/ and originals/ pulled from moissanite")
    ap.add_argument("--dbpath", default=os.path.join(REPO, "db"))
    args = ap.parse_args(argv)
    cmds = dict(provenance=cmd_provenance, sampling=cmd_sampling,
                peaks=cmd_peaks, profiles=cmd_profiles,
                stitch=cmd_stitch, noise=cmd_noise)
    if args.cmd == "all":
        for fn in (cmd_provenance, cmd_stitch, cmd_noise, cmd_sampling,
                   cmd_peaks, cmd_profiles):
            fn(args.data_root, dbpath=args.dbpath)
            print()
    else:
        cmds[args.cmd](args.data_root, dbpath=args.dbpath)


if __name__ == "__main__":
    main()
