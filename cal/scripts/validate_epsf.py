#!/usr/bin/env python
"""Validate the Z300 empirical LSF (ePSF) against the comparison estimators.

Reads the stage pickles written by cal/scripts/build_epsf.py (--workdir) and
writes cal/data/epsf-validation-<tag>.json plus figures in
cal/figures/epsf-<tag>/ (tag from --dataset).  Deterministic (fixed bootstrap seeds).

Tests (report section in brackets):
  (a) phase uniformity of centres on the coordinator's peak set, per segment,
      KS + chi2, for spline-root / parabola / free-width pixel-int Gaussian /
      ePSF, with the effective number of distinct lines; plus the
      dither-consistency test (phase bias that survives per-run registration)
  (b) cross-validation: every reference line is fitted with the ePSF built
      WITHOUT its db line (5 folds by line)
  (c) accuracy on identified thin Fe lines (obs - db about ONE common smooth
      per-run per-segment curve), paired, bootstrap CIs; Ar I 696.54 /
      772.38(+772.42) / 801.48 nm by leave-one-out Ar curves
  (d) areas: V I 417.94/415.97 same-upper-level ratio and same-upper-level
      multiplet ratios
  (e) fit residual vs sub-pixel phase
  classes: excess Lorentzian width and self-absorption fits vs class / delay

Usage:
    .venv/bin/python cal/scripts/validate_epsf.py [--workdir DIR] [--out JSON]
        [--figdir DIR] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from epsf_common import (REPO, DATASETS, load_all, coordinator_peaks, estimators_at,  # noqa: E402
                         running_median, db_table, bgsub, spline_root, parab, pixint_nm)
from build_epsf import is_reference, fold_of, SEG_NAMES, N_FOLDS  # noqa: E402
from alibz.utils.epsf import (EPSF, fit_line, fit_pixint_gauss, phase_of,  # noqa: E402
                              dither_structure)

warnings.filterwarnings("ignore")
RNG = 20260923
EST = ("vr", "vp", "sp", "pa", "p8", "es", "ep")
EST_NAME = {"vr": "vendor cubic spline: root", "vp": "vendor cubic spline: parabola on 1/30 nm grid",
            "sp": "local 9-pt spline-root (coordinator)", "pa": "native 3-pt parabola", "p8": "pixint Gaussian (free width)",
            "es": "ePSF segment-average", "ep": "ePSF (lambda-interpolated sub-bands)"}


# ---------------------------------------------------------------------------
class Shapes:
    """Segment-average shapes ('es') and wavelength-interpolated sub-band
    shapes ('ep', recommended), each with held-out (CV) variants."""

    def __init__(self, built, ref, bj):
        self.full = {sn: EPSF.from_dict(built[f"{sn}/full"]["epsf"]) for sn in SEG_NAMES}
        self.cv = {sn: [EPSF.from_dict(built[f"{sn}/cv{f}"]["epsf"]) for f in range(N_FOLDS)]
                   for sn in SEG_NAMES}
        keys = sorted({d["line_key"] for d in ref}, key=lambda s: float(s.split("-")[-1]))
        self.fold = {k: fold_of(k, keys) for k in keys}
        self.band_wl, self.band_full, self.band_cv = {}, {}, {}
        for sn in SEG_NAMES:
            B = bj["segments"][sn]["sub_bands"]
            self.band_wl[sn] = np.array([b["wl_center_nm"] for b in B])
            self.band_full[sn] = [np.array(b["epsf"]["coef"]) for b in B]
            self.band_cv[sn] = [[np.array(c["coef"]) for c in b["cv_folds"]] for b in B]

    def for_line(self, seg, line_key=None):
        """Segment-average shape; held-out if the line was in the reference build."""
        if line_key is not None and line_key in self.fold:
            return self.cv[seg][self.fold[line_key]], True
        return self.full[seg], False

    def at(self, seg, wl, line_key=None):
        """Sub-band shapes linearly interpolated (coefficients) at db wavelength
        wl; held-out variants if the line was in the reference build."""
        held = line_key is not None and line_key in self.fold
        C = ([c[self.fold[line_key]] for c in self.band_cv[seg]] if held else self.band_full[seg])
        w = self.band_wl[seg]
        j = int(np.searchsorted(w, wl))
        if j <= 0:
            c = C[0]
        elif j >= w.size:
            c = C[-1]
        else:
            t = (wl - w[j - 1]) / (w[j] - w[j - 1]); c = (1 - t) * C[j - 1] + t * C[j]
        return EPSF(c, 0.125, 3.0), held


def u_of_nm(seg, wl):
    return float(seg.nm_to_u(wl))


def estimators_window(seg, k, ep, neighbors=None, c0=None):
    """All four centres (px) + areas for the line at local max k."""
    e = estimators_at(seg, k)
    if e is None:
        return None
    out = dict(sp=u_of_nm(seg, e["sp"]), pa=u_of_nm(seg, e["pa"]), p8=u_of_nm(seg, e["p8"]),
               vr=u_of_nm(seg, e["vr"]), vp=u_of_nm(seg, e["vp"]),
               sg8=e["sg"])
    f = fit_line(ep, seg.u, seg.y, seg.sigma, c0=float(k) if c0 is None else c0,
                 neighbors=neighbors, fallback=False)
    if not f.get("ok"):
        return None
    out.update(ep=f["center"], ep_area=f["area"], ep_chi2=f["chi2nu"], ep_err=f["center_err"])
    return out


# ---------------------------------------------------------------------------
def uniformity(ph, nb=10):
    ph = np.asarray(ph)
    h, _ = np.histogram(ph, bins=np.linspace(-0.5, 0.5, nb + 1))
    exp = ph.size / nb
    chi2 = float(np.sum((h - exp) ** 2 / exp))
    p_chi2 = float(stats.chi2.sf(chi2, nb - 1))
    ks = stats.kstest(ph + 0.5, "uniform")
    return dict(n=int(ph.size), hist=[int(x) for x in h], chi2=chi2, p_chi2=p_chi2,
                ks_D=float(ks.statistic), ks_p=float(ks.pvalue))


def cluster_uniformity(ph, groups, nb=10, n_perm=2000, seed=RNG):
    """Cluster-aware phase test: the statistic is the chi2 of the histogram
    of per-window phases; its null distribution is simulated by drawing ONE
    uniform phase per distinct line and keeping each line's observed
    within-line phase offsets (drift), so recurring lines are not counted as
    independent."""
    ph = np.asarray(ph); groups = np.asarray(groups)
    obs = uniformity(ph, nb)["chi2"]
    rng = np.random.default_rng(seed)
    G, gi = np.unique(groups, return_inverse=True)
    med = np.array([np.median(ph[gi == j]) for j in range(G.size)])
    off = ph - med[gi]
    null = []
    for _ in range(n_perm):
        base = rng.uniform(-0.5, 0.5, G.size)
        p = phase_of(base[gi] + off)
        null.append(uniformity(p, nb)["chi2"])
    null = np.array(null)
    return dict(n_distinct=int(G.size), chi2=float(obs), p_cluster=float(np.mean(null >= obs)))


def dither_test(c, groups, runs, upos, nb=8):
    """Phase bias that survives per-run registration (identification-free).

    Fits c = kappa_L + a_r + b_r u (dither_structure); residuals binned in
    phase; returns the binned median residual (px) and its peak-to-peak with a
    bootstrap-over-lines 68 % interval."""
    cst, used, info = dither_structure(c, groups, runs, upos, run_slope=True, min_per_run=5)
    ok = np.isfinite(cst)
    if ok.sum() < 30:
        return dict(ok=False)
    r = (np.asarray(c) - cst)[ok]
    ph = phase_of(np.asarray(c)[ok])
    g = np.asarray(groups)[ok]
    bins = np.linspace(-0.5, 0.5, nb + 1)

    def curve(sel):
        return np.array([np.median(r[sel & (ph >= a) & (ph < b)]) if np.sum(sel & (ph >= a) & (ph < b)) >= 3
                         else np.nan for a, b in zip(bins[:-1], bins[1:])])
    base = curve(np.ones(r.size, bool))
    rng = np.random.default_rng(RNG)
    G = np.unique(g)
    ptps = []
    for _ in range(300):
        pick = rng.choice(G, G.size, replace=True)
        w = np.zeros(r.size, bool)
        cnt = defaultdict(int)
        for p in pick:
            cnt[p] += 1
        idx = np.concatenate([np.repeat(np.where(g == k)[0], n) for k, n in cnt.items()])
        rr, pp = r[idx], ph[idx]
        cv = [np.median(rr[(pp >= a) & (pp < b)]) if np.sum((pp >= a) & (pp < b)) >= 3 else np.nan
              for a, b in zip(bins[:-1], bins[1:])]
        ptps.append(np.nanmax(cv) - np.nanmin(cv))
    return dict(ok=True, n=int(ok.sum()), n_lines=info["n_lines"], n_runs=info["n_runs"],
                resid_mad_px=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                curve_px=[None if not np.isfinite(x) else float(x) for x in base],
                ptp_px=float(np.nanmax(base) - np.nanmin(base)),
                ptp_px_boot68=[float(np.percentile(ptps, 16)), float(np.percentile(ptps, 84))])


# ---------------------------------------------------------------------------
def test_a(runs, shapes, lines, reg):
    """Coordinator's peaks: phases per estimator (+ dither test)."""
    idmap = {(d["run"], d["seg"], d["k"]): d for d in lines}
    regd = {(x["run"], x["seg"]): x for x in reg}
    rows = []
    for r in runs:
        for p in coordinator_peaks(r):
            seg = r.segs[p["seg"]]
            k = p["k"]
            d = idmap.get((r.id, p["seg"], k))
            wl_true = d["wl_db"] if d else float(seg.u_to_nm(k - regd[(r.id, p["seg"])]["du0"]))
            ep, held = shapes.at(p["seg"], wl_true, d["line_key"] if d else None)
            f = fit_line(ep, seg.u, seg.y, seg.sigma, c0=float(k),
                         neighbors=d.get("neighbors") if d else None, fallback=False)
            es, _ = shapes.for_line(p["seg"], d["line_key"] if d else None)
            fs = fit_line(es, seg.u, seg.y, seg.sigma, c0=float(k),
                          neighbors=d.get("neighbors") if d else None, fallback=False)
            if not f.get("ok") or not fs.get("ok"):
                continue
            rows.append(dict(run=r.id, family=r.family, seg=p["seg"], k=k,
                             sp=u_of_nm(seg, p["sp"]), pa=u_of_nm(seg, p["pa"]),
                             vr=u_of_nm(seg, p["vr"]), vp=u_of_nm(seg, p["vp"]),
                             p8=u_of_nm(seg, p["p8"]), ep=f["center"], es=fs["center"], sg8=p["sg"],
                             held_out=held, ident=d["line_key"] if d else None,
                             cls=line_class(d) if d else None))
    # group recurring peaks by (family, seg, position within 1.5 px): chain-link
    groups = {}
    for fam in ("Fe", "V"):
        for sn in SEG_NAMES:
            R = sorted([x for x in rows if x["family"] == fam and x["seg"] == sn], key=lambda x: x["p8"])
            gid, last = -1, -1e9
            for x in R:
                if x["p8"] - last > 1.5:
                    gid += 1
                x["group"] = f"{fam}-{sn}-{gid}"
                last = x["p8"]
    res = {}
    for sn in SEG_NAMES:
        R = [x for x in rows if x["seg"] == sn]
        out = dict(n=len(R), n_distinct=len({x["group"] for x in R}),
                   n_heldout=int(sum(x["held_out"] for x in R)))
        for e in EST:
            ph = phase_of([x[e] for x in R])
            out[e] = dict(uniform=uniformity(ph),
                          cluster=cluster_uniformity(ph, [x["group"] for x in R]),
                          dither=dither_test([x[e] for x in R], [x["group"] for x in R],
                                             [x["run"] for x in R], [float(x["k"]) for x in R]))
        res[sn] = out
    return res, rows


# ---------------------------------------------------------------------------
def line_class(d):
    if d is None:
        return None
    lvl = "res" if d["Ei"] <= 0.12 else ("low" if d["Ei"] <= 1.2 else "high")
    return dict(species=f"{d['el']}{'I' * d['ion']}", ion=d["ion"], level=lvl,
                tau_bin=("thin" if d["tau_rel"] < 0.1 else ("mid" if d["tau_rel"] < 0.3 else "thick")))


def fit_all_lines(lines, shapes, jobs):
    """Every identified isolated window: ePSF (held-out when applicable),
    free Gaussian, + lorentz and selfabs kernels, + classic estimators."""
    tasks = [d for d in lines if d["iso_ok"] and d["snr"] >= 15]
    chunks = [(tasks[i::jobs], shapes) for i in range(jobs)]
    out = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for part in ex.map(_fit_chunk, chunks):
            out += part
    return out


def _fit_chunk(args):
    tasks, shapes = args
    res = []
    for d in tasks:
        sn = d["seg"]
        ep, held = shapes.at(sn, d["wl_db"], d["line_key"])
        es, _ = shapes.for_line(sn, d["line_key"])
        u, y, s = d["u"], d["y"], d["sigma"]
        nb = d.get("neighbors")
        f0 = fit_line(ep, u, y, s, c0=float(d["uc"]), neighbors=nb, fallback=False)
        fseg = fit_line(es, u, y, s, c0=float(d["uc"]), neighbors=nb, fallback=False)
        if not f0.get("ok"):
            continue
        fl = fit_line(ep, u, y, s, c0=f0["center"], neighbors=nb, fallback=False,
                      kernel="lorentz", kernel_p0=[0.05], kernel_bounds=([0.0], [3.0]))
        fs = fit_line(ep, u, y, s, c0=f0["center"], neighbors=nb, fallback=False,
                      kernel="selfabs", kernel_p0=[1.0, 0.05], kernel_bounds=([1e-3, 0.02], [300.0, 0.02 + 1e-6]))
        g = fit_pixint_gauss(u, y, s, c0=float(d["uc"]), half=4.5)
        # classic estimators on the stored window (nm grid reconstructed from u)
        k = int(np.argmin(np.abs(u - d["k"])))
        x_nm = d["wl_k"] + (u - d["k"]) * d["pitch"]
        try:
            yb = bgsub(x_nm, y, k) if k >= 9 and k <= len(u) - 10 else y - np.median(np.r_[y[:3], y[-3:]])
            sp = d["k"] + (spline_root(x_nm, yb, k, half=min(4, k, len(u) - 1 - k)) - d["wl_k"]) / d["pitch"]
            pa = d["k"] + (parab(x_nm, yb, k) - d["wl_k"]) / d["pitch"]
        except Exception:
            sp = pa = np.nan
        m = np.abs(u - f0["center"]) <= 4.5
        dd = u[m] - f0["center"]
        rec = {kk: d[kk] for kk in ("run", "family", "seg", "k", "line_key", "el", "ion", "Ei", "Ek", "gA", "resid",
                                    "wl_db", "u_db", "du_model", "tau_rel", "snr", "delay", "period",
                                    "pitch", "s_rel")}
        rec.update(held_out=held, ref=is_reference(d), cls=line_class(d),
                   es=(fseg["center"] if fseg.get("ok") else np.nan),
                   ep=f0["center"], ep_area=f0["area"], ep_chi2=f0["chi2nu"], ep_err=f0["center_err"],
                   ep_area_err=f0["area_err"],
                   ep_resid=(f0["resid"] / f0["area"]).tolist(), ep_d=dd.tolist(),
                   lor_gamma=(fl["kernel_params"][0] if fl.get("ok") else np.nan),
                   lor_gamma_err=(fl["kernel_err"][0] if fl.get("ok") else np.nan),
                   lor_chi2=(fl["chi2nu"] if fl.get("ok") else np.nan),
                   lor_area=(fl["area"] if fl.get("ok") else np.nan),
                   sa_tau=(fs["kernel_params"][0] if fs.get("ok") else np.nan),
                   sa_chi2=(fs["chi2nu"] if fs.get("ok") else np.nan),
                   sa_area=(fs["area"] if fs.get("ok") else np.nan),
                   sa_center=(fs["center"] if fs.get("ok") else np.nan),
                   p8=(g["center"] if g.get("ok") else np.nan), p8_area=(g["area"] if g.get("ok") else np.nan),
                   p8_sigma=(g["sigma_px"] if g.get("ok") else np.nan),
                   p8_resid=None, sp=sp, pa=pa)
        if g.get("ok"):
            from alibz.utils.epsf import pixint_gauss
            mm = np.abs(u - float(d["uc"])) <= 4.5
            model = g["area"] * pixint_gauss(u[mm] - g["center"], g["sigma_px"]) + g["b0"] + g["b1"] * (u[mm] - g["center"])
            rec["p8_resid"] = ((y[mm] - model) / g["area"]).tolist()
            rec["p8_d"] = (u[mm] - g["center"]).tolist()
        res.append(rec)
    return res


# ---------------------------------------------------------------------------
def boot_ci(x, fn=np.median, n=2000, seed=RNG):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = [fn(rng.choice(x, x.size, replace=True)) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _c_rows(F, element, deg, source):
    rows = []
    for (run, sn), R in group_by(F, lambda x: (x["run"], x["seg"])).items():
        R = [x for x in R if x["family"] == element and x["el"] == element and x["cls"]["tau_bin"] != "thick"
             and x["cls"]["level"] != "res" and abs(x.get("resid", 0.0)) < 0.35
             and all(np.isfinite(x[e]) for e in EST)]
        if len(R) < deg + 4:
            continue
        u = np.array([x["u_db"] for x in R])
        D = np.array([[x[e] - x["u_db"] for e in EST] for x in R])
        ref = np.median(D, axis=1) if source == "median" else (
            D.mean(axis=1) if source == "mean" else D[:, EST.index(source)])
        keep = np.ones(len(R), bool)
        for _ in range(5):
            c = np.polyfit(u[keep] - 1000, ref[keep], deg)
            rr = ref - np.polyval(c, u - 1000)
            s = 1.4826 * np.median(np.abs(rr[keep]))
            new = np.abs(rr) <= max(3.5 * s, 0.15)
            if np.array_equal(new, keep):
                break
            keep = new
        curve = np.polyval(c, u - 1000)
        for i, x in enumerate(R):
            rows.append(dict(run=run, seg=sn, line_key=x["line_key"], kept=bool(keep[i]),
                             pitch=x["pitch"], **{e: float(D[i, j] - curve[i]) for j, e in enumerate(EST)}))
    return rows


def test_c(F, element="Fe", deg=2):
    """Accuracy on identified thin lines of ``element`` (its own runs): obs - db
    about ONE common per-run per-segment quadratic fitted to the median of all
    estimators (clip mask common).  Robustness: the same statistic with the
    curve fitted to each single estimator and to the mean."""
    out = {}
    rows = _c_rows(F, element, deg, "median")
    for sn in SEG_NAMES:
        R = [x for x in rows if x["seg"] == sn]
        if not R:
            continue
        kept = [x for x in R if x["kept"]]
        o = dict(n_lines=len(R), n_kept=len(kept), n_runs=len({x["run"] for x in R}),
                 n_distinct=len({x["line_key"] for x in R}))
        for e in EST:
            a = np.array([abs(x[e]) * x["pitch"] * 1000 for x in R])
            ak = np.array([abs(x[e]) * x["pitch"] * 1000 for x in kept])
            rk = np.array([x[e] * x["pitch"] * 1000 for x in kept])
            o[e] = dict(median_abs_pm_all=float(np.median(a)), ci95_all=boot_ci(a),
                        p90_abs_pm_all=float(np.percentile(a, 90)),
                        frac_gt_1px_all=float(np.mean([abs(x[e]) > 1 for x in R])),
                        median_abs_pm_kept=float(np.median(ak)), ci95_kept=boot_ci(ak),
                        rms_pm_kept=float(np.sqrt(np.mean(rk ** 2))),
                        rms_ci95_kept=boot_ci(rk, fn=lambda v: np.sqrt(np.mean(v ** 2))))
        for e in EST:
            if e == "ep":
                continue
            dlt = np.array([(abs(x[e]) - abs(x["ep"])) * x["pitch"] * 1000 for x in kept])
            o[f"paired_{e}_minus_ep"] = dict(median_pm=float(np.median(dlt)), ci95=boot_ci(dlt),
                                             frac_ep_better=float(np.mean(dlt > 0)))
        # vendor spline vs every other estimator (owner's question)
        for base in ("vr", "vp"):
            for e in EST:
                if e == base:
                    continue
                dlt = np.array([(abs(x[base]) - abs(x[e])) * x["pitch"] * 1000 for x in kept])
                o[f"paired_{base}_minus_{e}"] = dict(median_pm=float(np.median(dlt)), ci95=boot_ci(dlt),
                                                     frac_other_better=float(np.mean(dlt > 0)))
        out[sn] = o
    rob = {}
    for source in list(EST) + ["mean"]:
        rr = _c_rows(F, element, deg, source)
        for sn in SEG_NAMES:
            R = [x for x in rr if x["seg"] == sn]
            if R:
                rob.setdefault(sn, {})[source] = {e: float(np.median([abs(x[e]) * x["pitch"] * 1000 for x in R]))
                                                  for e in EST}
    for sn in out:
        out[sn]["curve_robustness_median_abs_pm"] = rob.get(sn)
    return out, rows


AR_TARGETS = {"696.543": 696.5430, "772.405 (772.376+772.421 blend)": 772.405, "801.479": 801.4786}


def test_c_argon(F, runs, reg, shapes):
    """Ar I 696.54 / 772.38+772.42 (blend; gA-weighted centroid at 9 kK 772.405,
    +-~5 pm with T) / 801.48, measured directly at the registered position in
    every run, scored against a linear curve through the OTHER identified
    Ar I lines of the same run (targets +-0.5 nm excluded)."""
    regd = {(x["run"], x["seg"]): x for x in reg}
    byid = {r.id: r for r in runs}
    ep = shapes.full["NIR"]
    out = defaultdict(lambda: defaultdict(list))
    for run, R in group_by([x for x in F if x["el"] == "Ar" and x["seg"] == "NIR"], lambda x: x["run"]).items():
        seg = byid[run].segs["NIR"]
        rg = regd[(run, "NIR")]
        for key, wl in AR_TARGETS.items():
            O = [x for x in R if abs(x["wl_db"] - wl) > 0.5 and all(np.isfinite(x[e]) for e in EST)]
            if len(O) < 4:
                continue
            u0 = float(seg.nm_to_u(wl)); up = u0 + np.polyval(rg["poly"], u0)
            k = int(round(up)); k = k - 2 + int(np.argmax(seg.y[k - 2:k + 3]))
            e = estimators_at(seg, k)
            epl, _ = shapes.at("NIR", wl)
            f = fit_line(epl, seg.u, seg.y, seg.sigma, c0=float(k), fallback=False)
            fs = fit_line(ep, seg.u, seg.y, seg.sigma, c0=float(k), fallback=False)
            if e is None or not f.get("ok") or not fs.get("ok"):
                continue
            meas = dict(sp=u_of_nm(seg, e["sp"]), pa=u_of_nm(seg, e["pa"]), p8=u_of_nm(seg, e["p8"]),
                        vr=u_of_nm(seg, e["vr"]), vp=u_of_nm(seg, e["vp"]),
                        es=fs["center"], ep=f["center"])
            u = np.array([x["u_db"] for x in O])
            pitch = float(seg.pitch(k))
            for est in EST:
                dv = np.array([x[est] - x["u_db"] for x in O])
                c = np.polyfit(u - 1000, dv, 1)
                rr = dv - np.polyval(c, u - 1000)
                sd = 1.4826 * np.median(np.abs(rr))
                kk = np.abs(rr) <= max(3.5 * sd, 0.15)
                c = np.polyfit(u[kk] - 1000, dv[kk], 1)
                out[key][est].append(float((meas[est] - u0 - np.polyval(c, u0 - 1000)) * pitch * 1000))
    res = {}
    for key, v in out.items():
        res[key] = {e: dict(n=len(v[e]), median_pm=float(np.median(v[e])),
                            mad_pm=float(1.4826 * np.median(np.abs(np.array(v[e]) - np.median(v[e])))),
                            median_abs_pm=float(np.median(np.abs(v[e]))), ci95_abs=boot_ci(np.abs(v[e])))
                    for e in EST}
    return res


def group_by(xs, key):
    g = defaultdict(list)
    for x in xs:
        g[key(x)].append(x)
    return g


# ---------------------------------------------------------------------------
def _forced(ep, seg, c_fix, others, half=4.5):
    """Areas at FIXED centres (target + neighbours), linear background: linear LSQ."""
    m = np.abs(seg.u - c_fix) <= half
    u, y, sg = seg.u[m], seg.y[m], seg.sigma[m]
    cols = [ep(u - c_fix), np.ones_like(u), u - c_fix] + [ep(u - c) for c in others]
    X = np.stack(cols, 1) / sg[:, None]
    sol, *_ = np.linalg.lstsq(X, y / sg, rcond=None)
    cov = np.linalg.pinv(X.T @ X)
    res = y / sg - X @ sol
    chi2nu = float(res @ res / max(1, u.size - X.shape[1]))
    return float(sol[0]), float(np.sqrt(cov[0, 0] * max(chi2nu, 1.0))), chi2nu


def test_d(runs, shapes, reg, F):
    """V I 417.94/415.97 (same upper level a6D-z2G... ; thin ratio (gA/lambda) 2.05).

    Estimators per V run: FORCED ePSF areas at the registered positions
    (registration refined from the ePSF centres of the run's identified VIS
    lines; local maxima within 2-4.5 px fitted as fixed-centre neighbours),
    free ePSF fit, free pixel-int Gaussian, trapezoid (+-0.12 nm, sidebands
    0.25-0.50 nm), and local-max height.  415.97 is weak (single-run SNR ~2-10),
    so the run-to-run scatter is noise-dominated; per-delay medians are given."""
    tab = db_table(["V"])
    wl_a, wl_b = 417.9408, 415.9680
    ia = int(np.argmin(np.abs(tab["wl"] - wl_a))); ib = int(np.argmin(np.abs(tab["wl"] - wl_b)))
    R0 = (tab["gA"][ia] / tab["wl"][ia]) / (tab["gA"][ib] / tab["wl"][ib])
    rows = []
    for r in runs:
        if r.family != "V":
            continue
        seg = r.segs["VIS"]
        Fr = [x for x in F if x["run"] == r.id and x["seg"] == "VIS" and np.isfinite(x["ep"])]
        if len(Fr) < 5:
            continue
        uu = np.array([x["u_db"] for x in Fr]); dd = np.array([x["ep"] - x["u_db"] for x in Fr])
        c = np.polyfit(uu, dd, 1)
        for _ in range(3):
            rr = dd - np.polyval(c, uu); s_ = 1.4826 * np.median(np.abs(rr)); k_ = np.abs(rr) <= max(3 * s_, 0.1)
            c = np.polyfit(uu[k_], dd[k_], 1)
        yb = seg.y - running_median(seg.y, 41)
        vals = {}
        for name, wl in (("a", wl_a), ("b", wl_b)):
            ep, _ = shapes.at("VIS", wl)
            u0 = float(seg.nm_to_u(wl)); cf = u0 + np.polyval(c, u0)
            k = int(round(cf))
            nb = [float(j) for j in range(k - 5, k + 6) if 2 <= abs(j - cf) <= 4.5 and yb[j] == yb[j - 1:j + 2].max()
                  and yb[j] > 3 * seg.sigma[j]]
            Af, Ae, chi = _forced(ep, seg, cf, nb)
            f = fit_line(ep, seg.u, seg.y, seg.sigma, c0=cf, neighbors=nb, fallback=False, center_bound=0.5)
            g = fit_pixint_gauss(seg.u, seg.y, seg.sigma, c0=cf, half=4.5)
            x = seg.wl; cc = float(seg.u_to_nm(cf))
            cen = (x >= cc - 0.12) & (x <= cc + 0.12)
            side = ((x >= cc - 0.50) & (x <= cc - 0.25)) | ((x >= cc + 0.25) & (x <= cc + 0.50))
            pb = np.polyfit(x[side] - cc, seg.y[side], 1)
            net = seg.y[cen] - np.polyval(pb, x[cen] - cc)
            kk = int(np.clip(round(cf), 1, seg.n - 2))
            vals[name] = dict(forced=Af, forced_err=Ae, forced_chi2=chi,
                              ep=f["area"] if f.get("ok") else np.nan,
                              p8=g["area"] if g.get("ok") else np.nan,
                              trapz=float(np.trapezoid(net, x[cen]) / seg.pitch(cf)),
                              height=float(max(yb[kk - 1:kk + 2])), snr=float(Af / Ae) if Ae > 0 else np.nan)
        rows.append(dict(run=r.id, delay=r.delay, period=r.period,
                         **{f"R_{e}": (float(vals["a"][e] / vals["b"][e]) if vals["b"][e] not in (0, 0.0) else np.nan) for e in ("forced", "ep", "p8", "trapz", "height")},
                         snr_a=vals["a"]["snr"], snr_b=vals["b"]["snr"],
                         A_a=vals["a"]["forced"], A_b=vals["b"]["forced"],
                         Ae_a=vals["a"]["forced_err"], Ae_b=vals["b"]["forced_err"]))
    out = dict(R0_thin=float(R0), n_runs=len(rows),
               median_snr_forced=dict(a=float(np.median([x["snr_a"] for x in rows])),
                                      b=float(np.median([x["snr_b"] for x in rows]))))
    for e in ("forced", "ep", "p8", "trapz", "height"):
        v = np.array([x[f"R_{e}"] for x in rows]); v = v[np.isfinite(v) & (v > 0)]
        lv = np.log(v)
        out[e] = dict(n=int(v.size), median=float(np.median(v)), ci95=boot_ci(v),
                      robust_sd_log=float(1.4826 * np.median(np.abs(lv - np.median(lv)))),
                      by_delay={str(dl): float(np.median([x[f"R_{e}"] for x in rows if x["delay"] == dl]))
                                for dl in sorted({x["delay"] for x in rows})})
    # inverse-variance pooled forced ratio per delay (sum of areas)
    out["forced_pooled_by_delay"] = {}
    for dl in sorted({x["delay"] for x in rows}):
        X = [x for x in rows if x["delay"] == dl]
        A = sum(x["A_a"] for x in X); B = sum(x["A_b"] for x in X)
        eA = np.sqrt(sum(x["Ae_a"] ** 2 for x in X)); eB = np.sqrt(sum(x["Ae_b"] ** 2 for x in X))
        Rr = A / B if B > 0 else np.nan
        out["forced_pooled_by_delay"][str(dl)] = dict(n_runs=len(X), R=float(Rr),
                                                      R_err=float(abs(Rr) * np.sqrt((eA / A) ** 2 + (eB / B) ** 2)) if B > 0 and A > 0 else None)
    return out, rows


def test_d_multiplets(F):
    """Same-upper-level pairs (same el/ion/Ek, same run & segment, both thin-ish,
    within 15 nm): log(R_meas/R0) scatter across runs per estimator."""
    out = defaultdict(list)
    for (run, sn), R in group_by(F, lambda x: (x["run"], x["seg"])).items():
        for i, a in enumerate(R):
            for b in R[i + 1:]:
                if a["el"] != b["el"] or a["ion"] != b["ion"] or abs(a["Ek"] - b["Ek"]) > 1e-4:
                    continue
                if abs(a["wl_db"] - b["wl_db"]) > 15 or a["snr"] < 20 or b["snr"] < 20:
                    continue
                if a["cls"]["tau_bin"] == "thick" or b["cls"]["tau_bin"] == "thick":
                    continue
                s, w = (a, b) if a["gA"] / a["wl_db"] >= b["gA"] / b["wl_db"] else (b, a)
                R0 = (s["gA"] / s["wl_db"]) / (w["gA"] / w["wl_db"])
                key = f"{s['el']}{'I' * s['ion']} {s['wl_db']:.2f}/{w['wl_db']:.2f}"
                rec = dict(run=run, R0=R0)
                for e, ae in (("ep", "ep_area"), ("p8", "p8_area"), ("sa", "sa_area"), ("lor", "lor_area")):
                    # areas in counts*px -> counts*nm for the ratio
                    rec[e] = float(np.log((s[ae] * s["pitch"]) / (w[ae] * w["pitch"]) / R0))
                out[key].append(rec)
    res = {}
    for key, v in out.items():
        if len(v) < 3:
            continue
        res[key] = dict(n_runs=len(v), R0=v[0]["R0"],
                        **{e: dict(median_log_ratio=float(np.nanmedian([x[e] for x in v])),
                                   robust_sd=float(1.4826 * np.nanmedian(np.abs(np.array([x[e] for x in v]) - np.nanmedian([x[e] for x in v])))))
                           for e in ("ep", "p8", "sa", "lor")})
    return res


# ---------------------------------------------------------------------------
def test_e(F, nb=5):
    """Mean normalised residual vs sample offset, per phase bin (held-out ePSF
    vs free pixint Gaussian), reference lines only."""
    out = {}
    for sn in SEG_NAMES:
        R = [x for x in F if x["seg"] == sn and x["ref"] and x["p8_resid"] is not None]
        res = {}
        for name, rk, dk, ck in (("ep", "ep_resid", "ep_d", "ep"), ("p8", "p8_resid", "p8_d", "p8_mine")):
            bins = np.linspace(-0.5, 0.5, nb + 1)
            dgrid = np.arange(-4, 5)
            M = np.full((nb, dgrid.size), np.nan)
            for i, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
                S = [x for x in R if a <= phase_of(x[ck]) < b]
                for j, dv in enumerate(dgrid):
                    vals = [rr for x in S for rr, d in zip(x[rk], x[dk]) if abs(d - dv) < 0.5]
                    if len(vals) >= 5:
                        M[i, j] = np.median(vals)
            peak = np.nanmedian([max(np.abs(x[rk])) for x in R]) if R else np.nan
            res[name] = dict(matrix=M.tolist(), d=dgrid.tolist(),
                             rms_of_phase_bin_means=float(np.sqrt(np.nanmean(M ** 2))),
                             max_abs=float(np.nanmax(np.abs(M))),
                             phase_dependence=float(np.sqrt(np.nanmean((M - np.nanmean(M, axis=0)) ** 2))))
            res[name]["chi2_by_phase"] = None
        out[sn] = res
        chi = {}
        for a, b in zip(np.linspace(-0.5, 0.5, nb + 1)[:-1], np.linspace(-0.5, 0.5, nb + 1)[1:]):
            chi[f"{a:+.1f}"] = float(np.median([x["ep_chi2"] for x in R if a <= phase_of(x["ep"]) < b])) if R else None
        out[sn]["ep_chi2_by_phase"] = chi
    return out


# ---------------------------------------------------------------------------
def classes(F):
    """Excess Lorentzian HWHM (px) and self-absorption preference by class.
    Significance: per-run class medians, differences paired by run."""
    out = {}
    V = [x for x in F if np.isfinite(x["lor_gamma"])]
    for sn in SEG_NAMES:
        S = [x for x in V if x["seg"] == sn]
        o = {}
        by = defaultdict(list)
        for x in S:
            c = x["cls"]
            for key in (f"ion{c['ion']}", f"level-{c['level']}", f"tau-{c['tau_bin']}", f"species-{c['species']}",
                        "reference" if x["ref"] else "non-reference"):
                by[key].append(x)
        for key, X in sorted(by.items()):
            g = np.array([x["lor_gamma"] for x in X])
            dchi = np.array([x["ep_chi2"] - x["lor_chi2"] for x in X])
            dsa = np.array([x["lor_chi2"] - x["sa_chi2"] for x in X])
            per_run = defaultdict(list)
            for x in X:
                per_run[x["run"]].append(x["lor_gamma"])
            rm = np.array([np.median(v) for v in per_run.values()])
            o[key] = dict(n=len(X), n_runs=len(per_run), n_distinct=len({x["line_key"] for x in X}),
                          gamma_median_px=float(np.median(g)), gamma_ci95=boot_ci(g),
                          run_median_sd_px=float(np.std(rm, ddof=1)) if rm.size > 1 else None,
                          frac_lorentz_improves_chi2=float(np.mean(dchi > 0.5)),
                          frac_selfabs_beats_lorentz=float(np.mean(dsa > 0.5)),
                          sa_tau_median=float(np.nanmedian([x["sa_tau"] for x in X])))
        # paired comparisons by run
        pairs = {}
        for a, b in (("ion1", "ion2"), ("tau-thin", "tau-thick"), ("tau-thin", "tau-mid"),
                     ("level-high", "level-res"), ("level-high", "level-low"), ("reference", "non-reference")):
            if a not in by or b not in by:
                continue
            ra = defaultdict(list); rb = defaultdict(list)
            for x in by[a]:
                ra[x["run"]].append(x["lor_gamma"])
            for x in by[b]:
                rb[x["run"]].append(x["lor_gamma"])
            common = sorted(set(ra) & set(rb))
            if len(common) < 3:
                continue
            dlt = np.array([np.median(rb[r]) - np.median(ra[r]) for r in common])
            t = dlt.mean() / (dlt.std(ddof=1) / np.sqrt(dlt.size)) if dlt.std(ddof=1) > 0 else np.nan
            pairs[f"{b} - {a}"] = dict(n_runs=len(common), mean_diff_px=float(dlt.mean()),
                                       sd_across_runs_px=float(dlt.std(ddof=1)), t=float(t),
                                       p=float(2 * stats.t.sf(abs(t), dlt.size - 1)) if np.isfinite(t) else None)
        o["paired_by_run"] = pairs
        out[sn] = o
    # delay trend, V run2 (per class, per delay: median of per-run medians, SD across runs)
    trend = {}
    for sn in SEG_NAMES:
        for sp in ("VI", "VII"):
            X = [x for x in V if x["seg"] == sn and x["family"] == "V" and x["cls"]["species"] == sp
                 and x["cls"]["tau_bin"] != "thick"]
            if len(X) < 20:
                continue
            rows = {}
            for dl in sorted({x["delay"] for x in X if x["delay"] is not None}):
                per_run = defaultdict(list)
                for x in X:
                    if x["delay"] == dl:
                        per_run[x["run"]].append(x["lor_gamma"])
                rm = [np.median(v) for v in per_run.values()]
                rows[str(dl)] = dict(n_runs=len(rm), gamma_px=float(np.median(rm)),
                                     sd_runs=float(np.std(rm, ddof=1)) if len(rm) > 1 else None)
            dls = np.array([x["delay"] for x in X], float); gs = np.array([x["lor_gamma"] for x in X])
            rho = stats.spearmanr(dls, gs)
            trend[f"{sn}-{sp}"] = dict(by_delay=rows, spearman_rho=float(rho.statistic), p=float(rho.pvalue))
    out["delay_trend"] = trend
    return out


def stack_profiles(F):
    """Median normalised residual (data - instrument ePSF fit) vs offset d in
    1/4-px bins, per class: a flattened/reversed core shows as a central dip."""
    out = {}
    for sn in SEG_NAMES:
        for key, sel in (("reference", lambda x: x["ref"]),
                         ("thick", lambda x: x["cls"]["tau_bin"] == "thick"),
                         ("resonance", lambda x: x["cls"]["level"] == "res"),
                         ("ion", lambda x: x["ion"] == 2), ("neutral", lambda x: x["ion"] == 1)):
            X = [x for x in F if x["seg"] == sn and sel(x)]
            if len(X) < 20:
                continue
            d = np.concatenate([x["ep_d"] for x in X]); r = np.concatenate([x["ep_resid"] for x in X])
            bins = np.arange(-4.5, 4.51, 0.25)
            med = [float(np.median(r[(d >= a) & (d < b)])) if np.sum((d >= a) & (d < b)) >= 10 else None
                   for a, b in zip(bins[:-1], bins[1:])]
            out[f"{sn}/{key}"] = dict(n=len(X), bins=bins.tolist(), median_resid=med)
    return out


# ---------------------------------------------------------------------------
def figures(figdir, shapes, A_rows, F, cls, dres, built_json):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from alibz.utils.epsf import gauss_epsf, fit_line as _fl
    os.makedirs(figdir, exist_ok=True)
    col = {"vr": "#636363", "vp": "#bdbdbd", "sp": "#8a5a00", "pa": "#6a3d9a", "p8": "#1f78b4", "es": "#fdae61", "ep": "#d7301f"}
    # 1 shapes
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, sn in enumerate(SEG_NAMES):
        e = shapes.full[sn]
        d, v = e.grid(1 / 64)
        for cv in shapes.cv[sn]:
            ax[i].plot(*cv.grid(1 / 64), color="0.75", lw=0.8)
        ax[i].plot(d, v, color=col["ep"], lw=2, label=f"ePSF FWHM {e.fwhm():.2f} px")
        # best pixel-integrated Gaussian of equal FWHM
        from scipy.optimize import brentq
        sg = brentq(lambda s: gauss_epsf(s).fwhm() - e.fwhm(), 0.1, 2.0)
        g = gauss_epsf(sg)
        ax[i].plot(*g.grid(1 / 64), color=col["p8"], ls="--", label=f"pix-int Gaussian σ={sg:.2f} px")
        ax[i].set_title(f"{sn}: n={e.meta.get('n_lines')} windows")
        ax[i].set_xlabel("d = u − c (px)"); ax[i].set_xlim(-3, 3); ax[i].legend(fontsize=7)
    ax[0].set_ylabel("Ψ(d) (per px)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "epsf_shapes.png"), dpi=130); plt.close(fig)
    # 2 sub-bands
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for sn in SEG_NAMES:
        B = built_json["segments"][sn]["sub_bands"]
        wl = [np.mean(b["wl_range_nm"]) for b in B]
        ax[0].plot(wl, [b["fwhm_px"] for b in B], "o-", label=sn)
        ax[1].plot(wl, [b["fwhm_nm"] * 1000 for b in B], "o-", label=sn)
    ax[0].set_ylabel("FWHM (px)"); ax[1].set_ylabel("FWHM (pm)")
    for a in ax:
        a.set_xlabel("wavelength (nm, band mean)"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "subband_fwhm.png"), dpi=130); plt.close(fig)
    # 3 phase histograms
    fig, ax = plt.subplots(3, len(EST), figsize=(3 * len(EST), 7.5), sharex=True)
    for i, sn in enumerate(SEG_NAMES):
        R = [x for x in A_rows if x["seg"] == sn]
        for j, e in enumerate(EST):
            ph = phase_of([x[e] for x in R])
            ax[i, j].hist(ph, bins=np.linspace(-0.5, 0.5, 11), color=col[e])
            ax[i, j].axhline(len(R) / 10, color="k", lw=0.8)
            if i == 0:
                ax[i, j].set_title(EST_NAME[e], fontsize=9)
        ax[i, 0].set_ylabel(sn)
    for a in ax[-1]:
        a.set_xlabel("sub-pixel phase")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "phase_histograms.png"), dpi=130); plt.close(fig)
    # 4 dither-test curves
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    xb = np.linspace(-0.5, 0.5, 9); xc = 0.5 * (xb[1:] + xb[:-1])
    for i, sn in enumerate(SEG_NAMES):
        R = [x for x in A_rows if x["seg"] == sn]
        for e in EST:
            t = dither_test([x[e] for x in R], [x["group"] for x in R], [x["run"] for x in R], [float(x["k"]) for x in R])
            if t.get("ok"):
                ax[i].plot(xc, [np.nan if v is None else v for v in t["curve_px"]], "o-", color=col[e], label=EST_NAME[e])
        ax[i].axhline(0, color="k", lw=0.6); ax[i].set_title(sn); ax[i].set_xlabel("phase")
    ax[0].set_ylabel("median residual after per-run registration (px)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "dither_test.png"), dpi=130); plt.close(fig)
    # 5 class gamma vs delay
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, sn in enumerate(SEG_NAMES):
        for sp, c in (("VI", "#1b9e77"), ("VII", "#d95f02")):
            t = cls["delay_trend"].get(f"{sn}-{sp}")
            if not t:
                continue
            dl = sorted(t["by_delay"], key=float)
            ax[i].errorbar([float(x) for x in dl], [t["by_delay"][x]["gamma_px"] for x in dl],
                           yerr=[t["by_delay"][x]["sd_runs"] or 0 for x in dl], fmt="o-", color=c, label=sp)
        ax[i].set_xscale("symlog", linthresh=1); ax[i].set_xlim(-0.2, 80); ax[i].set_title(sn); ax[i].set_xlabel("delay (µs)")
        ax[i].legend(fontsize=7)
    ax[0].set_ylabel("excess Lorentz HWHM γ (px)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "class_gamma_vs_delay.png"), dpi=130); plt.close(fig)
    # 6 V ratio
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rows = dres[1]
    for e, m in (("height", "s"), ("trapz", "^"), ("p8", "v"), ("ep", "o"), ("forced", "D")):
        ax.plot([x["delay"] * {"height": 0.85, "trapz": 0.92, "p8": 1.0, "ep": 1.08, "forced": 1.16}[e] for x in rows],
                [x[f"R_{e}"] for x in rows], m, ms=4, label=e, alpha=0.8)
    ax.axhline(dres[0]["R0_thin"], color="k", lw=0.8, label="thin (gA/λ)")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.2, 80); ax.set_yscale("log"); ax.set_xlabel("delay (µs)"); ax.set_ylabel("V I 417.94 / 415.97")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "v_417_415_ratio.png"), dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "alibz-epsf-work"))
    ap.add_argument("--dataset", default="corrected-20260924", choices=sorted(DATASETS))
    ap.add_argument("--out", default=None, help="default cal/data/epsf-validation-<tag>.json")
    ap.add_argument("--epsf", default=None, help="default cal/data/epsf-z300-<tag>.json")
    ap.add_argument("--figdir", default=None, help="default cal/figures/epsf-<tag>/")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tag = DATASETS[a.dataset]["tag"]
    a.out = a.out or os.path.join(REPO, "cal", "data", f"epsf-validation-{tag}.json")
    a.epsf = a.epsf or os.path.join(REPO, "cal", "data", f"epsf-z300-{tag}.json")
    a.figdir = a.figdir or os.path.join(REPO, "cal", "figures", f"epsf-{tag}")
    runs = load_all(cache=os.path.join(a.workdir, "runs.pkl"), dataset=a.dataset)
    lines, reg = pickle.load(open(os.path.join(a.workdir, "catalog.pkl"), "rb"))
    ref, built = pickle.load(open(os.path.join(a.workdir, "built.pkl"), "rb"))
    bj = json.load(open(a.epsf))
    shapes = Shapes(built, ref, bj)
    fits_cache = os.path.join(a.workdir, "fits.pkl")
    if os.path.exists(fits_cache) and os.path.getmtime(fits_cache) > os.path.getmtime(os.path.join(a.workdir, "built.pkl")):
        F = pickle.load(open(fits_cache, "rb"))
    else:
        F = fit_all_lines(lines, shapes, a.jobs)
        pickle.dump(F, open(fits_cache, "wb"))
    print(f"fitted {len(F)} identified windows")
    # classic estimators exactly as the coordinator defines them (segment arrays)
    byid = {r.id: r for r in runs}
    for x in F:
        seg = byid[x["run"]].segs[x["seg"]]
        e = estimators_at(seg, x["k"])
        if e is None:
            x["sp"] = x["pa"] = x["p8c"] = x["vr"] = x["vp"] = np.nan
            continue
        x["sp"] = u_of_nm(seg, e["sp"]); x["pa"] = u_of_nm(seg, e["pa"]); x["p8c"] = u_of_nm(seg, e["p8"])
        x["vr"] = u_of_nm(seg, e["vr"]); x["vp"] = u_of_nm(seg, e["vp"])
    for x in F:
        x["p8_mine"] = x["p8"]; x["p8"] = x["p8c"]
    A, A_rows = test_a(runs, shapes, lines, reg)
    print("test a done")
    C, C_rows = test_c(F, "Fe")
    CVv, _ = test_c(F, "V")
    CA = test_c_argon(F, runs, reg, shapes)
    D = test_d(runs, shapes, reg, F)
    DM = test_d_multiplets(F)
    E = test_e(F)
    CL = classes(F)
    SP = stack_profiles(F)
    B = {}
    for sn in SEG_NAMES:
        Rf = [x for x in F if x["seg"] == sn and x["ref"] and all(np.isfinite(x[e]) for e in EST)]
        B[sn] = dict(n_ref=int(sum(x["ref"] for x in F if x["seg"] == sn)),
                     heldout_chi2_median=float(np.median([x["ep_chi2"] for x in F if x["seg"] == sn and x["ref"]])),
                     fold_fwhm_px=[e.fwhm() for e in shapes.cv[sn]], full_fwhm_px=shapes.full[sn].fwhm(),
                     heldout_phase=uniformity(phase_of([x["ep"] for x in Rf])),
                     heldout_phase_cluster=cluster_uniformity(phase_of([x["ep"] for x in Rf]), [x["line_key"] for x in Rf]),
                     **{f"dither_{e}": dither_test([x[e] for x in Rf], [x["line_key"] for x in Rf],
                                                   [x["run"] for x in Rf], [x["u_db"] for x in Rf]) for e in EST})
    reg_check = {}
    for sn in SEG_NAMES:
        X = [x for x in reg if x["seg"] == sn]
        dp = np.array([x["du_pattern"] for x in X if x.get("du_pattern") is not None])
        du1 = np.array([np.polyval(x["poly"], 1000.0) for x in X])
        reg_check[sn] = dict(n_runs=len(X),
                             pattern_offset_px=dict(median=float(np.median(dp)) if dp.size else None,
                                                    min=float(dp.min()) if dp.size else None,
                                                    max=float(dp.max()) if dp.size else None),
                             refined_registration_px_at_u1000=dict(median=float(np.median(du1)),
                                                                   min=float(du1.min()), max=float(du1.max())),
                             manifest_axis_check_px=sorted({x.get("axis_check") for x in X if x.get("axis_check") is not None}))
    doc = dict(registration_check=reg_check, c_accuracy_V=CVv, a_phase=A, b_crossval=B, c_accuracy=C, c_argon=CA, d_v_ratio=D[0], d_v_ratio_rows=D[1],
               d_multiplets=DM, e_residual_phase=E, classes=CL, stacks=SP)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=lambda o: None if o is None else float(o))
    print("wrote", a.out)
    figures(a.figdir, shapes, A_rows, F, CL, D, bj)
    print("figures in", a.figdir)


if __name__ == "__main__":
    main()
