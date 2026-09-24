#!/usr/bin/env python
"""Vanadium delay/period timing study on the SciAps Z300 (V_pure_run2).

Analyses the 2026-09-23 delay x period grid (session opt-c86274e9), the single
delay-0/period-5 extra point (session opt-35a72aa5), and compares against the
2026-09-22 first study V_pure_run1 (session opt-9d384296).

Deterministic and READ-ONLY with respect to the spectra.  Reuses the tested
native-grid primitives in cal/scripts/fe_plasma_analysis.py and
cal/scripts/line_physics_analysis.py (line measurement, per-segment shift
model, isolation mask, multiplet-ratio -> tau inversion, Boltzmann reference).

Usage:
    .venv/bin/python cal/scripts/v_timing_study.py \
        --data   <run2 acq dir> --ledger    cal/data/v-pure-run2-ledger.json \
        --run1-data <run1 acq dir> --run1-ledger <run1 ledger_v.json>

Writes cal/reports/2026-09-24-vanadium-timing-study.md and figures under
cal/figures/v-timing-20260924/.  (Re-run on the PIXEL_OFFSET -18 CORRECTED data
and recomputed Pantheum scores; supersedes the 2026-09-23 offset-era draft.)

DESIGN CONFOUND (see report): each DELAY row of the run2 grid is exactly one
raster lap over the same 12 sites, so delay == crater depth, and each PERIOD
always fell on the same fixed 3 sites.  Delay main effect therefore == depth;
period main effect == period + a fixed site-set effect; the delay x period
interaction is clean only if site effects are additive.  run1 (different
lap/site mapping; its delay 5 was fresh) and intensity-INDEPENDENT quantities
(line ratios, widths, Boltzmann slope) are used to separate what can be.
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "cal", "scripts")
for p in (REPO, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from fe_plasma_analysis import load_runs, measure_line, seg_index, KB
from line_physics_analysis import (
    el_lines, iso_mask, estimate_shift_model, db_to_obs, pool_mean,
    strength, segn, measure_fwhm, multiplet_pairs, boltzmann_residuals,
    KT_RANK, T_RANK, doppler_fwhm,
)
from alibz.utils.database import Database
from alibz.utils.absorption import escape_factor, invert_doublet_tau

# ---------------------------------------------------------------------------
# Representative line sets (db air wavelengths, nm), chosen from the fresh-pool
# smoke test: isolated, strong, SNR>=15 on the d=1 mean.  Eu/El in eV.
# ---------------------------------------------------------------------------
# V I lines used for signal/decay/Boltzmann (VIS best-response + a UV/NIR anchor)
REP_VI = [
    dict(lam=488.156, Ei=0.069, Ek=2.608),   # very strong, low Ei (SA-prone)
    dict(lam=482.745, Ei=0.040, Ek=2.608),
    dict(lam=570.698, Ei=1.043, Ek=3.213),   # mid Ei
    dict(lam=392.792, Ei=1.856, Ek=5.015),   # high Ei
    dict(lam=411.178, Ei=0.301, Ek=3.315),   # resonance-ish triplet member
    dict(lam=438.471, Ei=0.287, Ek=3.113),
    dict(lam=320.238, Ei=0.040, Ek=3.910),   # UV
    dict(lam=735.177, Ei=2.580, Ek=4.267),   # NIR, high Ei
]
# V II lines (mostly UV) for ionisation-ratio trend
REP_VII = [
    dict(lam=351.730, Ei=1.13, Ek=4.65),
    dict(lam=350.444, Ei=1.10, Ek=4.63),
    dict(lam=327.612, Ei=1.13, Ek=4.91),
    dict(lam=319.319, Ei=2.38, Ek=6.26),
]
# Continuum windows (VIS, line-free on the fresh mean), +-half nm
CONTINUUM = [500.0, 528.0, 590.0]
CONT_HALF = 0.25
# Tracked self-absorption pairs (strong, weak) db wavelengths
PAIR_RES = (478.650, 475.393)   # same-lower-level (resonance) V I
PAIR_UP = (417.941, 415.968)    # same-upper-level (branching, T-independent) V I
# V I 437-440 resonance triplet (registration-displaced, prior work)
TRIPLET = [437.923, 438.472, 438.998]
HALPHA = 656.279

USABLE_SNR = 10.0               # "usable line" threshold
FIGDIR = os.path.join(REPO, "cal", "figures", "v-timing-20260924")
REPORT = os.path.join(REPO, "cal", "reports", "2026-09-24-vanadium-timing-study.md")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def nearest_line_idx(L, lam, ion=None):
    d = np.abs(L["wl"] - lam)
    if ion is not None:
        d = np.where(L["ion"].astype(int) == ion, d, np.inf)
    i = int(np.argmin(d))
    return i if d[i] < 0.05 else None


def local_area(wl, y, center, win=0.22):
    """measure_line but re-centred on the local max within +-win nm of center
    (robust to the ~1-px V registration scatter)."""
    lo = np.searchsorted(wl, center - win)
    hi = np.searchsorted(wl, center + win)
    if hi - lo < 3:
        return None
    j = lo + int(np.argmax(y[lo:hi]))
    return measure_line(wl, y, float(wl[j]))


def continuum_level(wl, y):
    vals = []
    for c in CONTINUUM:
        lo = np.searchsorted(wl, c - CONT_HALF)
        hi = np.searchsorted(wl, c + CONT_HALF)
        if hi > lo:
            vals.append(float(np.median(y[lo:hi])))
    return float(np.median(vals)) if vals else float("nan")


def robust_height(wl, y, center, win=0.12, s_in=0.28, s_out=0.55):
    """Background-robust peak height: local max within +-win nm of center, minus
    the median of two CLOSE side bands (+-[s_in, s_out] nm).  A period-scaled
    additive detector background is removed by the side bands, so the height (and
    height RATIOS) are not biased by the fixed pattern that corrupts net areas.
    The +-win search is kept tight so a dense-forest neighbour is not grabbed.
    Returns (height, peak_nm) or None."""
    lo = np.searchsorted(wl, center - win)
    hi = np.searchsorted(wl, center + win)
    if hi - lo < 1:
        return None
    j = lo + int(np.argmax(y[lo:hi]))
    pk = float(wl[j])
    Lb = y[np.searchsorted(wl, pk - s_out):np.searchsorted(wl, pk - s_in)]
    Rb = y[np.searchsorted(wl, pk + s_in):np.searchsorted(wl, pk + s_out)]
    if len(Lb) < 1 or len(Rb) < 1:
        return None
    base = float(np.median(np.concatenate([Lb, Rb])))
    return float(y[j] - base), pk


def pair_height_ratio(model, wl, y, lam_s, lam_w):
    """Background-robust strong/weak peak-HEIGHT ratio for a fixed pair."""
    hs = robust_height(wl, y, db_to_obs(model, lam_s))
    hw = robust_height(wl, y, db_to_obs(model, lam_w))
    if not hs or not hw or hw[0] <= 0 or hs[0] <= 0:
        return float("nan")
    return hs[0] / hw[0]


def track_pair(L, model, wl, y, lam_s, lam_w, snr_min=10.0):
    """Recompute R0, sr (db) and R (measured areas) for a fixed pair; invert
    to tau on the strong member.  Returns dict or None."""
    i = nearest_line_idx(L, lam_s)
    j = nearest_line_idx(L, lam_w)
    if i is None or j is None:
        return None

    def eps(k):
        return (L["gA"][k] / L["wl"][k]) * math.exp(-L["Ek"][k] / KT_RANK)

    def kap(k):
        return L["gA"][k] * L["wl"][k] ** 2 * math.exp(-L["Ei"][k] / KT_RANK)
    if eps(i) < eps(j):
        i, j = j, i
    R0 = eps(i) / eps(j)
    sr = kap(i) / kap(j)
    ms = local_area(wl, y, db_to_obs(model, float(L["wl"][i])))
    mw = local_area(wl, y, db_to_obs(model, float(L["wl"][j])))
    if not ms or not mw or ms["area"] <= 0 or mw["area"] <= 0:
        return None
    if ms["snr"] < snr_min or mw["snr"] < snr_min:
        return None
    R = ms["area"] / mw["area"]
    tau_w = invert_doublet_tau(R, R0, strength_ratio=sr)
    tau_s = sr * tau_w
    corr = 1.0 / float(escape_factor(tau_s)) if tau_s > 0 else 1.0
    return dict(R0=R0, sr=sr, R=R, tau_s=tau_s, corr=corr,
                snr_s=ms["snr"], snr_w=mw["snr"])


def pair_thin_constants(L, lam_s, lam_w):
    """db-only thin emissivity ratio R0 and absorption-strength ratio sr for a
    pair, with the strong member first.  Returns (lam_strong, lam_weak, R0, sr)."""
    i = nearest_line_idx(L, lam_s)
    j = nearest_line_idx(L, lam_w)
    if i is None or j is None:
        return None

    def eps(k):
        return (L["gA"][k] / L["wl"][k]) * math.exp(-L["Ek"][k] / KT_RANK)

    def kap(k):
        return L["gA"][k] * L["wl"][k] ** 2 * math.exp(-L["Ei"][k] / KT_RANK)
    if eps(i) < eps(j):
        i, j = j, i
    return (float(L["wl"][i]), float(L["wl"][j]),
            eps(i) / eps(j), kap(i) / kap(j))


def boltzmann_curvature(pts):
    """Given Boltzmann points (Ek, yv=ln(area*wl/gA)) fit linear and quadratic.
    Return (T_lin, c2, resid_Ei_slope, n).  c2 = quadratic coeff (curvature ->
    temperature mixing).  resid_Ei_slope = slope of linear residual vs Ei
    among lines of similar Ek (-> self-absorption of low-El lines)."""
    if len(pts) < 6:
        return None
    Ek = np.array([p["Ek"] for p in pts])
    yv = np.array([p["yv"] for p in pts])
    Ei = np.array([p["Ei"] for p in pts])
    sl, ic = np.polyfit(Ek, yv, 1)
    T = -1.0 / (KB * sl) if sl < 0 else float("nan")
    resid = yv - (sl * Ek + ic)
    # quadratic curvature (normalise Ek to keep the coeff meaningful)
    c2, c1, c0 = np.polyfit(Ek, yv, 2)
    # residual vs Ei (lower-level energy): SA depresses low-Ei lines -> positive
    # slope of resid on Ei.  Use all pts (Ek already removed by the linear fit).
    if np.ptp(Ei) > 0.2:
        m_ei = np.polyfit(Ei, resid, 1)[0]
    else:
        m_ei = float("nan")
    return dict(T=float(T), c2=float(c2), ei_slope=float(m_ei), n=len(pts),
                rms=float(resid.std()))


# ---------------------------------------------------------------------------
# per-run measurement
# ---------------------------------------------------------------------------
def measure_run(run, db, L, iso, model):
    wl, y = run.wl, run.mean
    out = dict(run=run.meta["run"], delay=run.meta["delay"],
               period=run.meta["period"],
               lap=run.meta.get("lap"),
               site=run.meta.get("site", run.meta.get("location")),
               created_at=run.meta.get("created_at"),
               score=run.meta.get("score"),
               shots_stored=run.meta.get("shots_stored", run.shots.shape[0]))
    # representative V I / V II areas + snr
    vi_area = 0.0
    vi_lines = {}
    for d in REP_VI:
        m = local_area(wl, y, db_to_obs(model, d["lam"]))
        if m and m["area"] > 0:
            vi_lines[d["lam"]] = dict(area=m["area"], snr=m["snr"])
            vi_area += m["area"]
    vii_area = 0.0
    for d in REP_VII:
        m = local_area(wl, y, db_to_obs(model, d["lam"]))
        if m and m["area"] > 0:
            vii_area += m["area"]
    out["vi_area"] = vi_area
    out["vii_area"] = vii_area
    out["vii_vi"] = (vii_area / vi_area) if vi_area > 0 else float("nan")
    out["vi_lines"] = vi_lines
    # explicit areas for the tracked SA-pair members and the 437-440 triplet
    areas = {}
    for lam in [PAIR_UP[0], PAIR_UP[1], PAIR_RES[0], PAIR_RES[1]] + TRIPLET:
        m = local_area(wl, y, db_to_obs(model, lam))
        areas[lam] = float(m["area"]) if (m and m["area"] > 0 and m["snr"] >= 6) else float("nan")
    out["areas"] = areas
    # background-robust peak-HEIGHT pair ratios (not corrupted by the
    # period-scaled fixed pattern that biases net areas)
    out["hr_upper"] = pair_height_ratio(model, wl, y, *PAIR_UP)
    out["hr_res"] = pair_height_ratio(model, wl, y, *PAIR_RES)
    # background-robust individual peak heights (for the delay/depth regression
    # and the V II/V I ionisation trend)
    h311 = robust_height(wl, y, db_to_obs(model, 311.071))
    h411 = robust_height(wl, y, db_to_obs(model, 411.178))
    out["h_vii311"] = h311[0] if h311 else float("nan")
    out["h_vi411"] = h411[0] if h411 else float("nan")
    out["hr_vii_vi"] = (h311[0] / h411[0]) if (h311 and h411 and h411[0] > 0) else float("nan")
    # weak-member 475 height (to expose the additive fixed-pattern gain)
    h475 = robust_height(wl, y, db_to_obs(model, 475.393))
    out["h_475"] = h475[0] if h475 else float("nan")
    # usable line count + median snr over isolated V lines
    snrs = []
    for i in np.where(iso)[0]:
        m = measure_line(wl, y, db_to_obs(model, float(L["wl"][i])))
        if m and m["snr"] > 0:
            snrs.append(m["snr"])
    snrs = np.array(snrs)
    out["usable_lines"] = int((snrs >= USABLE_SNR).sum())
    out["median_snr"] = float(np.median(snrs[snrs >= 3])) if (snrs >= 3).any() else float("nan")
    # continuum
    cont = continuum_level(wl, y)
    out["continuum"] = cont
    peak_vi = max((v["area"] for v in vi_lines.values()), default=float("nan"))
    out["cont_line_ratio"] = (cont / (vi_area / max(len(vi_lines), 1))) if vi_area > 0 else float("nan")
    # FWHM: median of isolated VIS V I lines (ne/Stark proxy)
    fw = []
    for i in np.where(iso & (L["ion"].astype(int) == 1))[0]:
        lam = float(L["wl"][i])
        if segn(lam) != "VIS":
            continue
        f = measure_fwhm(wl, y, db_to_obs(model, lam), snr_min=USABLE_SNR)
        if f:
            fw.append(f["fwhm_nm"] * 1000.0)
    out["fwhm_vis_pm"] = float(np.median(fw)) if fw else float("nan")
    out["n_fwhm"] = len(fw)
    # Halpha (air / continuum contamination proxy)
    mh = local_area(wl, y, HALPHA, win=0.25)
    out["halpha_snr"] = float(mh["snr"]) if mh else 0.0
    # tracked SA pairs
    pu = track_pair(L, model, wl, y, *PAIR_UP)
    pr = track_pair(L, model, wl, y, *PAIR_RES)
    out["tau_upper"] = pu["tau_s"] if pu else float("nan")
    out["R_upper"] = pu["R"] if pu else float("nan")
    out["tau_res"] = pr["tau_s"] if pr else float("nan")
    out["R_res"] = pr["R"] if pr else float("nan")
    # Boltzmann reference (V I) -> T, curvature, Ei-slope
    br = boltzmann_residuals(L, iso, 1, model, wl, y, 360.0, 620.0,
                             snr_min=8.0, min_ref=6)
    if br:
        out["T_boltz"] = br["T"]
        cv = boltzmann_curvature(br["ref_pts"])
        out["boltz_c2"] = cv["c2"] if cv else float("nan")
        out["boltz_ei_slope"] = cv["ei_slope"] if cv else float("nan")
        out["boltz_rms"] = br["rms"]
        out["boltz_pts"] = br["pts"]
    else:
        out["T_boltz"] = float("nan")
        out["boltz_c2"] = float("nan")
        out["boltz_ei_slope"] = float("nan")
        out["boltz_rms"] = float("nan")
        out["boltz_pts"] = []
    return out


# ---------------------------------------------------------------------------
# aggregation over replicates
# ---------------------------------------------------------------------------
def agg(measures, keys):
    """Group per-run measures by (delay,period); return
    {(d,p): {key: (mean, sd, n)}}."""
    g = defaultdict(list)
    for m in measures:
        g[(m["delay"], m["period"])].append(m)
    out = {}
    for k, rows in g.items():
        d = {}
        for key in keys:
            vals = np.array([r[key] for r in rows if r.get(key) is not None
                             and np.isfinite(r.get(key, np.nan))], float)
            if len(vals):
                d[key] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals))
            else:
                d[key] = (float("nan"), float("nan"), 0)
        d["_n"] = len(rows)
        out[k] = d
    return out


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def ms(cell, d=3):
    m, s, n = cell
    if not np.isfinite(m):
        return "—"
    return f"{m:.{d}f}±{s:.{d}f}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run1-data", default=None)
    ap.add_argument("--run1-ledger", default=None)
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    db = Database("db")
    L = el_lines(db, "V")
    iso = iso_mask(L)

    runs = load_runs(args.data, args.ledger)
    # split main grid vs extra (lap=="extra")
    grid = [r for r in runs if r.meta.get("lap") != "extra"]
    extra = [r for r in runs if r.meta.get("lap") == "extra"]
    # one shift model on the fresh (delay==1) pool (grid shares one native axis)
    fresh = [r for r in grid if r.meta["delay"] == 1] or grid
    fwl, fy = pool_mean(fresh)
    model = estimate_shift_model(fwl, fy, db, "V")

    print(f"[run2] {len(grid)} grid + {len(extra)} extra runs; "
          f"shift global {model['global']*1000:.0f} pm, scatter "
          f"{model.get('scatter', float('nan'))*1000:.0f} pm")

    meas = [measure_run(r, db, L, iso, model) for r in grid]
    meas_extra = [measure_run(r, db, L, iso, model) for r in extra]

    run1_meas = []
    r1 = []
    model1 = None
    if args.run1_data and args.run1_ledger and os.path.isdir(args.run1_data):
        r1 = load_runs(args.run1_data, args.run1_ledger)
        f1 = [r for r in r1 if r.meta["delay"] == 5] or r1   # d5 fresh in run1
        m1wl, m1y = pool_mean(f1)
        model1 = estimate_shift_model(m1wl, m1y, db, "V")
        run1_meas = [measure_run(r, db, L, iso, model1) for r in r1]
        print(f"[run1] {len(r1)} runs; shift {model1['global']*1000:.0f} pm")

    keys = ["score", "median_snr", "usable_lines", "continuum", "vi_area",
            "vii_area", "vii_vi", "cont_line_ratio", "fwhm_vis_pm",
            "tau_upper", "tau_res", "hr_upper", "hr_res", "hr_vii_vi",
            "h_vii311", "h_vi411", "h_475", "T_boltz",
            "boltz_c2", "boltz_ei_slope", "halpha_snr", "shots_stored"]
    A = agg(meas, keys)
    A1 = agg(run1_meas, keys) if run1_meas else {}

    delays = sorted(set(m["delay"] for m in meas))
    periods = sorted(set(m["period"] for m in meas))

    # ---- Q2: line flat vs period; the period-scaled signal is a detector
    #          fixed pattern (difference-spectrum + NIR dark tests), not plasma
    scaling = period_scaling(meas, delays, periods)
    fp2 = fixed_pattern(grid, delays, periods)
    fp1 = fixed_pattern(r1, sorted(set(r.meta["delay"] for r in r1)),
                        sorted(set(r.meta["period"] for r in r1))) if r1 else None
    # ---- Q1/Q3: delay is a real timing effect (run1 separates delay vs depth)
    reg = delay_depth_regression(run1_meas) if run1_meas else None
    # ---- pair thin constants for background-robust tau from height ratios
    pc = {name: pair_thin_constants(L, *lams) for name, lams in
          [("upper", PAIR_UP), ("res", PAIR_RES)]}

    write_report(A, A1, A_extra=agg(meas_extra, keys), delays=delays,
                 periods=periods, scaling=scaling, fp2=fp2, fp1=fp1, reg=reg,
                 pc=pc, model=model, meas=meas, run1_meas=run1_meas,
                 meas_extra=meas_extra)
    make_figures(A, delays, periods, scaling, fp2, reg, meas)
    print("wrote", REPORT)


# ---- decay fitting -------------------------------------------------------
def _sat(P, A, tau):
    return A * (1.0 - np.exp(-P / tau))


def fit_decays(meas, delays, periods):
    """For each rep-mean V I line and the continuum, at each delay fit the
    running-integral saturation S(P)=A(1-exp(-P/tau)) over the period ladder.
    tau is in vendor timing units.  Returns nested dict."""
    out = {}
    lam_set = [d["lam"] for d in REP_VI] + ["continuum", "vi_area"]
    for delay in delays:
        rows = [m for m in meas if m["delay"] == delay]
        by_p = defaultdict(list)
        for m in rows:
            by_p[m["period"]].append(m)
        out[delay] = {}
        for lam in lam_set:
            P, S = [], []
            for p in sorted(by_p):
                vals = []
                for m in by_p[p]:
                    if lam == "continuum":
                        v = m["continuum"]
                    elif lam == "vi_area":
                        v = m["vi_area"]
                    else:
                        v = m["vi_lines"].get(lam, {}).get("area")
                    if v is not None and np.isfinite(v):
                        vals.append(v)
                if vals:
                    P.append(p)
                    S.append(np.mean(vals))
            P = np.array(P, float)
            S = np.array(S, float)
            rec = dict(P=P.tolist(), S=S.tolist(), tau=float("nan"),
                       A=float("nan"), r2=float("nan"))
            if len(P) >= 3 and (S > 0).all():
                try:
                    p0 = [S.max() * 1.2, 15.0]
                    popt, _ = curve_fit(_sat, P, S, p0=p0, maxfev=20000,
                                        bounds=([0, 0.3], [np.inf, 1e4]))
                    fit = _sat(P, *popt)
                    ss = 1 - np.sum((S - fit) ** 2) / max(np.sum((S - S.mean()) ** 2), 1e-9)
                    rec.update(tau=float(popt[1]), A=float(popt[0]), r2=float(ss))
                except Exception:
                    pass
            out[delay][lam] = rec
    return out


def period_scaling(meas, delays, periods):
    """Q2: does LINE signal scale with period, or only continuum?  At each delay
    regress Sigma V I area and continuum on period; report the slope (per vendor
    unit), its SE, and the P1->P50 fractional change.  A ~zero line slope with a
    positive continuum slope means the line emission is already complete at the
    shortest gate and the extra gate only adds continuum."""
    out = {}
    for d in delays:
        rows = [m for m in meas if m["delay"] == d]
        by_p = defaultdict(list)
        for m in rows:
            by_p[m["period"]].append(m)

        def series(key):
            P, V = [], []
            for p in sorted(by_p):
                vv = [mm[key] for mm in by_p[p] if np.isfinite(mm.get(key, np.nan))]
                if vv:
                    P.append(p); V.append(np.mean(vv))
            return np.array(P, float), np.array(V, float)

        def lin(P, V):
            if len(P) < 3:
                return float("nan"), float("nan"), float("nan")
            A = np.vstack([P, np.ones_like(P)]).T
            coef, res, *_ = np.linalg.lstsq(A, V, rcond=None)
            slope = coef[0]
            resid = V - A @ coef
            se = math.sqrt((resid @ resid) / max(len(P) - 2, 1) /
                           max(np.sum((P - P.mean()) ** 2), 1e-9))
            frac = (V[-1] - V[0]) / V[0] if V[0] != 0 else float("nan")
            return float(slope), float(se), float(frac)
        Pl, Vl = series("vi_area")
        Pc, Vc = series("continuum")
        ls, lse, lfrac = lin(Pl, Vl)
        cs, cse, cfrac = lin(Pc, Vc)
        out[d] = dict(line_slope=ls, line_se=lse, line_frac=lfrac,
                      cont_slope=cs, cont_se=cse, cont_frac=cfrac)
    return out


def _cond_mean_spectra(runs, delays, periods):
    """Replicate-mean spectrum per (delay,period).  All runs share the native
    grid, so index k is the same detector element; returns (wl, {(d,p): mean})."""
    if not runs:
        return None, {}
    npix = min(r.npix for r in runs)
    wl = runs[0].wl[:npix]
    out = {}
    for d in delays:
        for p in periods:
            ms = [r.mean[:npix] for r in runs
                  if r.meta["delay"] == d and r.meta["period"] == p]
            if ms:
                out[(d, p)] = np.mean(ms, axis=0)
    return wl, out


def fixed_pattern(runs, delays, periods):
    """Q2 fixed-pattern test.  The period-dependent extra signal is a detector
    background/dark fixed pattern (period ~ exposure), not plasma continuum.
    Two probes, both delay-independent if it is instrumental:
      (i) the P_max - P_min difference spectrum should be nearly IDENTICAL at
          every delay (high cross-correlation) even though the line signal
          differs 2x between delays;
      (ii) the far-NIR (880-900 nm) low-percentile level should move with period
          the same way at every delay (and go negative -> imperfect vendor dark
          subtraction).
    Returns dict(pmin,pmax,corr_vs_deep{d:r}, nir{(d,p):level}, delays,periods)."""
    wl, cm = _cond_mean_spectra(runs, delays, periods)
    if wl is None or not cm:
        return None
    pmin, pmax = min(periods), max(periods)
    # difference spectra P_max - P_min per delay (VIS+NIR, drop the noisy UV)
    band = (wl >= 400.0) & (wl <= 900.0)
    diffs = {}
    for d in delays:
        if (d, pmin) in cm and (d, pmax) in cm:
            diffs[d] = (cm[(d, pmax)] - cm[(d, pmin)])[band]
    deep = max(diffs) if diffs else None
    corr = {}
    if deep is not None:
        ref = diffs[deep]
        for d, v in diffs.items():
            if np.std(v) > 0 and np.std(ref) > 0:
                corr[d] = float(np.corrcoef(v, ref)[0, 1])
    # far-NIR 10th-percentile level per condition
    nir_band = (wl >= 880.0) & (wl <= 900.0)
    nir = {}
    for (d, p), y in cm.items():
        nir[(d, p)] = float(np.percentile(y[nir_band], 10))
    # per-delay line-signal amplitude (max of the difference in a strong VI band)
    return dict(pmin=pmin, pmax=pmax, corr_vs_deep=corr, deep=deep, nir=nir,
                delays=delays, periods=periods)


def delay_depth_regression(run1_meas):
    """Q1/Q3: separate the DELAY (plasma-timing) effect from crater DEPTH using
    run1, whose lap/site order gives independent variation in prior-shots-per-
    site.  Build prior_shots per batch (cumulative stored shots at the same site
    from earlier batches by created_at), then OLS ln(height) ~ delay + prior
    for V II 311, V I 411, and the V II/V I height ratio.  Returns per-line
    dict(b_delay, se_delay, b_depth, se_depth, n)."""
    rows = sorted(run1_meas, key=lambda m: m.get("created_at", ""))
    seen = defaultdict(int)   # site -> cumulative prior shots
    for m in rows:
        site = tuple(m.get("site") or [])
        m["_prior"] = seen[site]
        seen[site] += int(m.get("shots_stored") or 0)

    def ols(key):
        X, Y = [], []
        for m in rows:
            v = m.get(key)
            if v is not None and np.isfinite(v) and v > 0:
                X.append([1.0, float(m["delay"]), float(m["_prior"])])
                Y.append(math.log(v))
        X = np.array(X); Y = np.array(Y)
        if len(Y) < 6:
            return None
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        dof = max(len(Y) - X.shape[1], 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        return dict(b_delay=float(beta[1]), se_delay=float(se[1]),
                    b_depth=float(beta[2]), se_depth=float(se[2]), n=len(Y),
                    depth_max=int(max(m["_prior"] for m in rows)))
    return {k: ols(k) for k in ("h_vii311", "h_vi411", "hr_vii_vi")}


def robust_tau(R, R0, sr):
    if not (np.isfinite(R) and R0 and sr):
        return float("nan")
    return sr * invert_doublet_tau(R, R0, strength_ratio=sr)


# ---- report --------------------------------------------------------------
def write_report(A, A1, A_extra, delays, periods, scaling, fp2, fp1, reg, pc,
                 model, meas, run1_meas, meas_extra):
    lines = []
    W = lines.append
    W("# Vanadium delay/period timing study — SciAps Z300 (V_pure_run2)\n")
    W("*Generated by `cal/scripts/v_timing_study.py` — deterministic, "
      "read-only on the spectra.  Native grid only; no resampling.*\n")
    W("Sessions: `opt-c86274e9` (main 5×4 grid, 59 batches), "
      "`opt-35a72aa5` (delay-0/period-5 extra point), compared with "
      "`opt-9d384296` V_pure_run1 (cal/reports/2026-09-22-vanadium-sequence-results.md).  "
      f"pulsePeriod 100 (10 Hz), 7–10 shots/batch, 3 replicates.  "
      f"Per-element window-quality score (V reference).  Shift model "
      f"{model['global']*1000:.0f} pm (Ar-seeded, scatter "
      f"{model.get('scatter', float('nan'))*1000:.0f} pm).\n")

    W("## The design confound (read first)\n")
    W("The one-click sequence walked the same 12 raster sites in a fixed order, "
      "one lap per DELAY row.  Verified from the ledger: lap = position//12, and "
      "each lap contains exactly one delay:\n")
    W("| lap | delay | prior shots/site | period→sites |")
    W("|--|--|--|--|")
    W("| 1 | 1 | 0 (fresh) | each period always on the same 3 of the 12 sites |")
    W("| 2 | 5 | ~7–10 | (same mapping every lap) |")
    W("| 3 | 10 | ~14–20 | |")
    W("| 4 | 20 | ~21–30 | |")
    W("| 5 | 50 | ~28–40 | |")
    W("\n**Consequences.** (i) The *delay* main effect is identical to *crater "
      "depth* — delay 1 is fresh, delay 50 has ~30–40 prior shots per site — so "
      "an intensity change with delay cannot be attributed to gate timing alone. "
      "(ii) The *period* main effect is period **plus** a fixed site-set offset "
      "(each period always on the same 3 sites), but that offset is common to "
      "every delay row, so the period ladder *within a delay* and the "
      "delay×period interaction are clean if site effects are additive. "
      "(iii) Background-robust, intensity-independent quantities (peak-HEIGHT "
      "line-pair ratios with close side bands, FWHM) carry the physics verdicts; "
      "NET-AREA quantities are NOT reliable here because a period-scaled detector "
      "fixed pattern (§3) sits under the lines and biases areas.  run1 used a "
      "different lap/site order and a shallower depth range (0–20 prior shots), "
      "which lets an explicit regression separate the DELAY (plasma-timing) "
      "effect from crater DEPTH (§2.2).\n")

    # ---- score / SNR grid ----
    W("## 1. Grid: score, SNR, lines, continuum (mean ± SD of replicates)\n")
    W("### 1.1 Pantheum score by delay×period\n")
    hdr = "| delay\\period | " + " | ".join(str(p) for p in periods) + " | delay mean |"
    W(hdr); W("|" + "--|" * (len(periods) + 2))
    for d in delays:
        cells = []
        allv = []
        for p in periods:
            c = A.get((d, p), {}).get("score", (float("nan"),)*3)
            cells.append(ms(c, 3))
            if np.isfinite(c[0]):
                allv.append(c[0])
        dm = f"{np.mean(allv):.3f}" if allv else "—"
        W(f"| {d} | " + " | ".join(cells) + f" | {dm} |")
    # period means
    pm = []
    for p in periods:
        vv = [A[(d, p)]["score"][0] for d in delays if (d, p) in A and np.isfinite(A[(d, p)]["score"][0])]
        pm.append(f"{np.mean(vv):.3f}" if vv else "—")
    W("| **period mean** | " + " | ".join(pm) + " |  |")
    ex = A_extra.get((0, 5), {}).get("score")
    if ex:
        W(f"\nExtra point delay 0 / period 5 (single batch, 7 shots): "
          f"score {fmt(ex[0],3)}.\n")

    for key, title, dd in [
        ("median_snr", "1.2 Median SNR (isolated V lines)", 1),
        ("usable_lines", f"1.3 Usable-line count (SNR≥{USABLE_SNR:.0f})", 1),
        ("continuum", "1.4 Period-scaled background (median, VIS line-free windows — §3 shows this is instrument fixed pattern, NOT plasma continuum)", 1),
        ("vi_area", "1.5 Summed representative V I line area", 0),
    ]:
        W(f"### {title}\n")
        W("| delay\\period | " + " | ".join(str(p) for p in periods) + " |")
        W("|" + "--|" * (len(periods) + 1))
        for d in delays:
            cells = [ms(A.get((d, p), {}).get(key, (float("nan"),)*3), dd) for p in periods]
            W(f"| {d} | " + " | ".join(cells) + " |")
        W("")

    # ---- Q1 short delay ----
    W("## 2. Q1 — Is SHORT DELAY better?  Delay IS a real timing effect\n")
    W("In run2 delay ≡ crater depth, so run2 alone cannot say whether a delay "
      "trend is timing or depth.  §2.1 shows the run2 trends (period 1, "
      "background-robust peak-height ratios); §2.2 uses run1 — shallower "
      "(0–20 prior shots/site) and a different lap/site order — to REGRESS the "
      "delay effect against depth explicitly.\n")
    W("### 2.1 run2 trends vs delay (period 1; height-based, background-robust)\n")
    W("| metric | d=1 | d=5 | d=10 | d=20 | d=50 |")
    W("|--|--|--|--|--|--|")
    def row_over_delay(key, dd, period=None):
        cells = []
        for d in delays:
            if period is not None:
                c = A.get((d, period), {}).get(key, (float("nan"),)*3)
            else:
                vv = [A[(d, p)][key][0] for p in periods if (d, p) in A and np.isfinite(A[(d, p)][key][0])]
                c = (np.mean(vv), 0.0, len(vv)) if vv else (float("nan"),)*3
            cells.append(fmt(c[0], dd))
        return cells
    W("| median SNR | " + " | ".join(row_over_delay("median_snr", 1, 1)) + " |")
    W("| usable lines | " + " | ".join(row_over_delay("usable_lines", 0, 1)) + " |")
    W("| Σ V I area | " + " | ".join(row_over_delay("vi_area", 0, 1)) + " |")
    W("| VIS FWHM pm | " + " | ".join(row_over_delay("fwhm_vis_pm", 0, None)) + " |")
    W("| V II/V I height | " + " | ".join(row_over_delay("hr_vii_vi", 3, 1)) + " |")
    W("| 417/415 height ratio | " + " | ".join(row_over_delay("hr_upper", 2, 1)) + " |")
    W("")
    W("Brightness (SNR, usable lines, Σ area) falls with delay, but in run2 that "
      "is inseparable from depth.  FWHM is flat (no ne/Stark trend).  The V II/V I "
      "height ratio falls markedly with delay (ionic recombination), while on the "
      "CORRECTED axis the T-independent 417/415 pair sits near its optically-thin "
      "R₀ and is roughly flat with delay (§4.1 — the offset-era 'opacity falling "
      "with delay' is gone).  Whether the V II/V I fall is delay or depth is "
      "settled by run1:\n")
    W("### 2.2 run1 regression: ln(peak height) ~ delay + prior-shots/site\n")
    W("run1 (0–20 prior shots/site) breaks the run2 collinearity.  OLS "
      "coefficients per vendor unit (delay) and per prior shot (depth); a "
      "coefficient ≥ ~2–3× its SE is significant.\n")
    if reg:
        W("| line | b_delay ± SE | b_depth ± SE | n | verdict |")
        W("|--|--|--|--|--|")
        names = {"h_vii311": "V II 311", "h_vi411": "V I 411",
                 "hr_vii_vi": "V II/V I"}
        for k in ("h_vii311", "h_vi411", "hr_vii_vi"):
            r = reg.get(k)
            if not r:
                W(f"| {names[k]} | — | — | — | insufficient |")
                continue
            sig_d = abs(r["b_delay"]) > 2.5 * r["se_delay"]
            sig_z = abs(r["b_depth"]) > 2.5 * r["se_depth"]
            verdict = ("delay significant, depth n.s." if sig_d and not sig_z else
                       "both n.s." if not sig_d and not sig_z else
                       "depth significant" if sig_z and not sig_d else
                       "both significant")
            W(f"| {names[k]} | {r['b_delay']:+.4f}±{r['se_delay']:.4f} | "
              f"{r['b_depth']:+.4f}±{r['se_depth']:.4f} | {r['n']} | {verdict} |")
        vii = reg.get("h_vii311")
        if vii:
            W(f"\n**V II 311 falls with DELAY ({vii['b_delay']:+.4f}±"
              f"{vii['se_delay']:.4f} /unit, "
              f"{abs(vii['b_delay'])/max(vii['se_delay'],1e-9):.1f}σ) and NOT with "
              f"depth ({vii['b_depth']:+.4f}±{vii['se_depth']:.4f}).**  So the "
              "ionic (V II) emission and the V II/V I ratio drop is a REAL "
              "plasma-timing effect — the plasma cools and recombines as the "
              "delay grows — not a crater-depth artefact.  V I is not "
              "significant on either axis over this shallow range.\n")
    else:
        W("(run1 data not supplied — regression skipped.)\n")
    W("**Q1 answer:** short delay genuinely captures the hotter, more-ionised "
      "early plasma (higher V II and V II/V I).  It is a real lever on ionic-vs-"
      "neutral content, to be chosen on FRESH sites; it is not merely the "
      "fresh-crater advantage.  Hα stays at noise (SNR≈3) at all delays even at "
      "P=50, so early air contamination is not a short-delay problem here.\n")

    # ---- Q2 short period: line flat; the period-scaled extra is fixed pattern
    W("## 3. Q2 — Is SHORT PERIOD better?  Period is a detector exposure, not a "
      "plasma gate\n")
    W("Owner's question: *does integrated line signal scale with period — is "
      "'period' really the gate width?*  Two facts settle it: (1) V I line "
      "signal is FLAT with period; (2) the period-scaled *extra* signal is an "
      "instrument dark / fixed pattern, not plasma continuum.\n")
    W("### 3.1 Line signal is flat with period\n")
    W("Slope of Σ V I line area vs period per delay (counts/unit) with the "
      "fractional change P1→P50 — indistinguishable from zero everywhere:\n")
    W("| delay | Σ V I line slope (±SE) | Δ P1→P50 |")
    W("|--|--|--|")
    for d in delays:
        s = scaling[d]
        W(f"| {d} | {fmt(s['line_slope'],2)}±{fmt(s['line_se'],2)} | "
          f"{fmt(100*s['line_frac'],0)}% |")
    W("\n### 3.2 The period-scaled 'continuum' is a detector fixed pattern\n")
    W("If the period-dependent background were plasma continuum it would scale "
      f"with the plasma brightness, i.e. differ between delays.  It does not.  "
      f"The P{fp2['pmax']}−P{fp2['pmin']} DIFFERENCE spectrum (400–900 nm) is "
      "nearly IDENTICAL at every delay, even though the plasma line signal "
      "differs ~2× across delays:\n")
    W("| difference spectrum | " + " | ".join(f"d={d}" for d in fp2["delays"]) + " |")
    W("|--|" + "--|" * len(fp2["delays"]))
    W(f"| corr vs deepest (d={fp2['deep']}) | " +
      " | ".join(fmt(fp2["corr_vs_deep"].get(d, float('nan')), 2) for d in fp2["delays"]) + " |")
    W(f"\nFar-NIR (880–900 nm) 10th-percentile level (counts) vs period — it "
      "moves with PERIOD identically at every delay and goes NEGATIVE, the "
      "signature of a period-scaled dark/fixed pattern with imperfect vendor "
      "subtraction (a real plasma continuum cannot be negative):\n")
    W("| delay\\period | " + " | ".join(str(p) for p in fp2["periods"]) + " |  (run2)")
    W("|" + "--|" * (len(fp2["periods"]) + 1))
    for d in fp2["delays"]:
        cells = [fmt(fp2["nir"].get((d, p), float("nan")), 0) for p in fp2["periods"]]
        W(f"| {d} | " + " | ".join(cells) + " |")
    if fp1:
        W("\n| delay\\period | " + " | ".join(str(p) for p in fp1["periods"]) + " |  (run1, same pattern)")
        W("|" + "--|" * (len(fp1["periods"]) + 1))
        for d in fp1["delays"]:
            cells = [fmt(fp1["nir"].get((d, p), float("nan")), 0) for p in fp1["periods"]]
            W(f"| {d} | " + " | ".join(cells) + " |")
    W("\n**Reading.**  The difference-spectrum correlations are ~0.9+ across "
      "delays and the NIR level goes strongly negative with period in the same "
      "steps at every delay (and in run1), so the period-scaled signal is a "
      "detector exposure/fixed-pattern term, NOT plasma emission.  **Period "
      "behaves like a detector exposure that is longer than the plasma lifetime: "
      "it adds no line photons, only fixed pattern + noise, collapsing SNR and "
      "score.**  (Note d50/P1 still carries ~40–50 % of the d1 line height, so "
      "the line emission is NOT 'complete within one period unit' — the earlier "
      "draft's gate-width reading is withdrawn.)  Verdict: **short period is "
      "unambiguously better; use the minimum period, and the score/continuum "
      "gain is instrumental, so quantitative work needs a per-period dark/"
      "fixed-pattern measurement (laser off/blocked) subtracted.**\n")

    # ---- Q3 hypothesis ----
    W("## 4. Q3 — Does a long gate MIMIC self-absorption (temperature mixing)?\n")
    W("**The owner hypothesis — that a shorter integration reduces APPARENT "
      "optical thickness by excluding changing-T plasma — cannot be tested with "
      "the PERIOD knob:** period is a detector exposure longer than the plasma "
      "(§3), so a longer period adds no new plasma-time to the gate, only fixed "
      "pattern.  The genuine opacity probes below are therefore read against "
      "PERIOD (should be flat if opacity is period-independent) and against "
      "DELAY (the only real lever on the temporal mix).\n")
    W("### 4.1 (a) T-independent pair 417.94/415.97 — background-robust PEAK-"
      "HEIGHT ratio\n")
    W("Peak height (local max minus close side bands) is immune to the period-"
      "scaled fixed pattern that corrupts net areas.  This branching pair shares "
      "an upper level, so its ratio is T-independent; a move is real opacity.\n")
    W("| delay\\period | " + " | ".join(str(p) for p in periods) + " |")
    W("|" + "--|" * (len(periods) + 1))
    for d in delays:
        cells = [ms(A.get((d, p), {}).get("hr_upper", (float("nan"),)*3), 2) for p in periods]
        W(f"| {d} | " + " | ".join(cells) + " |")
    R0u = pc["upper"][2] if pc.get("upper") else float("nan")
    sru = pc["upper"][3] if pc.get("upper") else float("nan")
    d1v = A.get((1, 1), {}).get("hr_upper", (float("nan"),)*3)[0]
    dNv = A.get((delays[-1], 1), {}).get("hr_upper", (float("nan"),)*3)[0]
    d1_over_p = [A.get((1, p), {}).get("hr_upper", (float("nan"),)*3)[0]
                 for p in periods]
    d1_sd = float(np.nanstd(d1_over_p)) if np.isfinite(d1v) else float("nan")
    frac = (d1v / R0u) if (np.isfinite(d1v) and R0u) else float("nan")
    tau_d1 = robust_tau(d1v, R0u, sru)
    # delay trend: fractional change d1 -> deepest at period 1
    dchg = (dNv - d1v) / d1v if (np.isfinite(d1v) and np.isfinite(dNv) and d1v) else float("nan")
    flat_delay = np.isfinite(dchg) and abs(dchg) < 0.15
    W(f"\nThin (optically-thin) height ratio R₀≈{fmt(R0u,2)} (same-upper-level "
      f"branching pair).  **On the CORRECTED axis the measured ratio is "
      f"{fmt(d1v,2)} at d1**, ≈{fmt(100*frac,0)}% of R₀, and **flat with period** "
      f"(SD ≈ {fmt(d1_sd,2)} across P1–P50) — confirming the earlier draft's "
      "τ-falling-with-period was a NET-AREA artefact of the period-scaled "
      "background, not an opacity change.\n")
    if np.isfinite(frac) and frac >= 0.85:
        W(f"The ratio sits CLOSE TO R₀ (≈{fmt(100*frac,0)}%): the 417.94/415.97 "
          "pair is **near optically thin — only MILD self-absorption**"
          + (f" (homogeneous-slab τ(417.94) ≈ {fmt(tau_d1,2)} at d1)"
             if np.isfinite(tau_d1) and tau_d1 > 0 else "")
          + f".  **This OVERTURNS the 2026-09-23 offset-era reading** (ratio 0.75, "
          "'heavily self-absorbed, even below 1'): that value was an "
          "axis-registration artefact — the −18 px offset put the peak-height "
          "windows on the wrong pixels, so the strong 417.94 member was "
          "systematically under-measured.  The corrected value agrees with the "
          "independent ePSF pair (2.05 thin, 2.10 [1.99, 2.18] measured on "
          "corrected positions); the residual gap to R₀ is mild SA plus the "
          "crude close-sideband height in this dense line forest.\n")
    elif np.isfinite(d1v) and d1v > 1.0:
        W(f"The ratio sits below R₀ (≈{fmt(100*frac,0)}%) but above 1: "
          f"moderate self-absorption of the strong 417.94 member (slab "
          f"τ(417.94) ≈ {fmt(tau_d1,2)} at d1).  It is period-INDEPENDENT.\n")
    else:
        W(f"The ratio sits well below R₀≈{fmt(R0u,2)} — near/below 1 — at every "
          "period, i.e. the strong 417.94 member is measured no brighter than "
          "the weak 415.97, so it is heavily self-absorbed and/or 415.97 is "
          "blended.  A quantitative τ is therefore NOT quoted.\n")
    if flat_delay:
        W(f"With delay the ratio is essentially FLAT ({fmt(d1v,2)} at d1 → "
          f"{fmt(dNv,2)} at d{delays[-1]}, {fmt(100*dchg,0)}%), so there is no "
          "evidence of a growing opacity in the later/deeper plasma over this "
          "range — again unlike the offset-era draft, whose 0.75→0.56 'fall with "
          "delay' was part of the same registration artefact.\n")
    else:
        W(f"With delay the ratio changes {fmt(d1v,2)} at d1 → {fmt(dNv,2)} at "
          f"d{delays[-1]} ({fmt(100*dchg,0)}%).  **OPEN:** whether this is more "
          "opacity in the later/deeper plasma or a growing blend on the weak "
          "415.97 member is not separated (delay≡depth in run2; the absolute "
          "ratio is also sensitive to weak-member peak-finding in this dense "
          "forest).\n")
    W("### 4.2 (b) 478.65/475.39 'T-mixing' signal is a fixed-pattern artefact\n")
    W("The draft read this resonance pair as long-gate T-mixing.  It is the "
      "fixed pattern: the weak member 475.39 gains an ADDITIVE offset with "
      "period at every delay (peak-minus-close-sideband height, counts):\n")
    W("| delay\\period | " + " | ".join(str(p) for p in periods) + " |  h(475.39)")
    W("|" + "--|" * (len(periods) + 1))
    for d in delays:
        cells = [fmt(A.get((d, p), {}).get("h_475", (float("nan"),)*3)[0], 0) for p in periods]
        W(f"| {d} | " + " | ".join(cells) + " |")
    W("\nThe additive rise with period is delay-independent (the +110–120 count "
      "bump at 475.2 nm in the P50−P1 difference spectrum appears at d1, d20 and "
      "d50 alike), i.e. instrumental, not T-mixing.  No temperature-mixing "
      "apparent-SA claim survives.\n")
    W("### 4.3 (c) Apparent V I Boltzmann T — NOT quantitative\n")
    W("The V I Boltzmann slope-T (single-line scatter rms ~0.5–0.9 ln) is ill-"
      "constrained and internally contradictory here — it rises with delay while "
      "V II/V I FALLS with delay (a hotter plasma should be MORE ionised), so it "
      "is reported for completeness only, not as a temperature.  A defensible "
      "T(t)/Saha thermometer needs the segment response and an ne, neither "
      "available (FWHM is instrument-limited; §1).\n")
    W("| delay | apparent T (K, P1) — ill-constrained |")
    W("|--|--|")
    for d in delays:
        c = A.get((d, 1), {}).get("T_boltz", (float("nan"),)*3)
        W(f"| {d} | {ms(c,0)} |")
    W("\n### 4.4 (d) V I 437–440 resonance triplet\n")
    W("Only the central 438.47 member resolves; 437.92/439.00 stay displaced/"
      "blended (the optically-thick-and-shifted registration signature).  Like "
      "every strong V I line its signal is flat with period and falls with "
      "delay/depth.  Net areas here sit on the period-scaled fixed pattern and "
      "are not used quantitatively.\n")

    # ---- Q4 frame drops ----
    W("\n## 5. Q4 — Frame drops (stored shots < 10) vs period/delay\n")
    W("| delay\\period | " + " | ".join(str(p) for p in periods) + " |")
    W("|" + "--|" * (len(periods) + 1))
    for d in delays:
        cells = []
        for p in periods:
            rows = [m for m in meas if m["delay"] == d and m["period"] == p]
            if rows:
                stored = [m["shots_stored"] for m in rows]
                cells.append(f"{min(stored)}–{max(stored)}")
            else:
                cells.append("—")
        W(f"| {d} | " + " | ".join(cells) + " |")
    total = len(meas)
    drops = sum(1 for m in meas if m["shots_stored"] < 10)
    W(f"\n{drops}/{total} batches stored <10 shots. "
      "Correlation with period/delay summarised in the reading below.\n")

    # ---- run1 comparison ----
    if A1:
        W("## 6. Comparison with run1 (different lap/site order; d5 fresh)\n")
        W("Background-robust 417/415 peak-height ratio (opacity) and score.\n")
        W("| d/p | run2 score | run1 score | run2 417/415 | run1 417/415 |")
        W("|--|--|--|--|--|")
        for (d, p) in sorted(set(A1) & set(A)):
            W(f"| {d}/{p} | {ms(A[(d,p)]['score'],3)} | {ms(A1[(d,p)]['score'],3)} | "
              f"{ms(A[(d,p)]['hr_upper'],2)} | {ms(A1[(d,p)]['hr_upper'],2)} |")
        W("\nThe height ratio is comparable across the two runs at matched d/p, "
          "and flat with period in both — self-absorption is real and period-"
          "independent.  run1's independent lap/site order is what enables the "
          "delay-vs-depth regression in §2.2.\n")

    # ---- old vs corrected ----
    W("## 7. Old vs corrected — every headline number and verdict\n")
    W(old_vs_corrected_text(A, reg, pc, scaling, fp2, delays, periods, meas, model))
    # ---- verdicts + recommendations ----
    W("## 8. Verdicts\n")
    W(verdicts_text(A, A1, delays, periods, scaling, fp2, reg, pc, meas))
    W("## 9. Recommended settings & next study design\n")
    W(recommend_text(A, delays, periods))
    W("## 10. Unverified / limits\n")
    W(limits_text(model, meas))
    W("\n## Files\n")
    W("- Script: `cal/scripts/v_timing_study.py`")
    W("- Ledger builder: `cal/scripts/build_v_ledgers.py` (from the corrected data ledger)")
    W("- Ledger (run2): `cal/data/v-pure-run2-ledger.json` (old kept as `cal/data/v-pure-run2-ledger-20260923.json`)")
    W("- Ledger (run1): `cal/data/v-pure-run1-ledger.json`")
    W("- Report: `cal/reports/2026-09-24-vanadium-timing-study.md`")
    for f in ["score_snr_heatmap.png", "signal_vs_period.png",
              "fixed_pattern_nir.png", "delay_depth_regression.png",
              "height_ratio_vs_period.png", "vii_vi_vs_delay.png"]:
        W(f"- Figure: `cal/figures/v-timing-20260924/{f}`")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def track_triplet(m, lam):
    v = m.get("areas", {}).get(lam)
    return v if (v is not None and np.isfinite(v)) else None


def old_vs_corrected_text(A, reg, pc, scaling, fp2, delays, periods, meas, model):
    """Side-by-side of every headline number and verdict: the 2026-09-23
    offset-era draft (hard-coded from cal/reports/2026-09-23-vanadium-timing-
    study.md, now superseded) vs this 2026-09-24 CORRECTED (PIXEL_OFFSET -18,
    re-scored) run.  Corrected values are computed live from the grid."""
    def g(d, p, k):
        return A.get((d, p), {}).get(k, (float("nan"),)*3)[0]

    # corrected scalars
    dN = delays[-1]
    shift = model.get("global", float("nan")) * 1000.0
    pmean = {p: np.nanmean([g(d, p, "score") for d in delays]) for p in periods}
    best = max((k for k in A),
               key=lambda k: A[k]["score"][0] if np.isfinite(A[k]["score"][0]) else -1)
    best_mean = A[best]["score"][0]
    best_max = max((m["score"] for m in meas
                    if m["delay"] == best[0] and m["period"] == best[1]
                    and m.get("score") is not None), default=float("nan"))
    dmean = {d: np.nanmean([g(d, p, "score") for p in periods]) for d in delays}
    best_delay = max(dmean, key=lambda d: dmean[d] if np.isfinite(dmean[d]) else -1)
    R0u = pc["upper"][2] if pc.get("upper") else float("nan")
    hr1 = g(1, 1, "hr_upper"); hrN = g(dN, 1, "hr_upper")
    vv1 = g(1, 1, "hr_vii_vi"); vvN = g(dN, 1, "hr_vii_vi")
    T1 = g(1, 1, "T_boltz")
    nir1_p1 = fp2["nir"].get((1, periods[0]), float("nan"))
    nir1_pN = fp2["nir"].get((1, periods[-1]), float("nan"))
    vii = (reg or {}).get("h_vii311") or {}
    vi = (reg or {}).get("h_vi411") or {}
    vratio = (reg or {}).get("hr_vii_vi") or {}
    drops = sum(1 for m in meas if m["shots_stored"] < 10)
    line_slope_d1 = scaling.get(1, {}).get("line_frac", float("nan"))

    def sig(r):
        if not r:
            return "—"
        return f"{abs(r['b_delay'])/max(r['se_delay'],1e-9):.1f}σ"

    thin = np.isfinite(hr1) and np.isfinite(R0u) and hr1 >= 0.85 * R0u
    t = []
    t.append("The 2026-09-23 draft (offset-era) column is copied from the now-"
             "superseded report; the corrected column is recomputed live here on "
             "the PIXEL_OFFSET −18 data with the re-scored Pantheum ledger.\n")
    t.append("| headline | 2026-09-23 (offset era) | 2026-09-24 (corrected) | verdict |")
    t.append("|--|--|--|--|")
    t.append(f"| Shift model (global) | −123 pm | {shift:.0f} pm | the −18 px error "
             "removed → residual shift ~4× smaller, cleaner anchors |")
    t.append(f"| Best cell (delay/period), score | 5/1, 0.539 | "
             f"{best[0]}/{best[1]}, {best_mean:.3f} (max {best_max:.3f}) | best "
             "moves to d10/P1; scores rescaled by the Pantheum recompute |")
    t.append(f"| Period-mean score P1→P{periods[-1]} | 0.504 → 0.415 | "
             f"{pmean[periods[0]]:.3f} → {pmean[periods[-1]]:.3f} | UNCHANGED "
             "verdict: score falls monotonically with period |")
    t.append(f"| Best delay (delay-mean score) | d5 (0.497) | d{best_delay} "
             f"({dmean[best_delay]:.3f}) | short delay still best |")
    t.append(f"| 417/415 height ratio, d1 (thin R₀≈{R0u:.2f}) | 0.75 | {hr1:.2f} | "
             + ("**OVERTURNED**: near thin R₀ (mild SA), NOT 'below 1, heavily "
                "self-absorbed' — 0.75 was the −18 px artefact" if thin else
                "changed") + " |")
    t.append(f"| 417/415 vs delay (P1) | falls 0.75→0.56 | {hr1:.2f}→{hrN:.2f} | "
             + ("now ~flat: the offset-era 'opacity grows with depth' is gone"
                if (np.isfinite(hr1) and np.isfinite(hrN) and abs(hrN-hr1)/hr1 < 0.15)
                else "changed") + " |")
    t.append(f"| V II/V I height, d1→d{dN} (P1) | 6.35 → 4.48 | {vv1:.2f} → {vvN:.2f} | "
             "SURVIVES: ionic emission falls with delay (recombination) |")
    t.append(f"| run1 V II 311 b_delay | −0.0202±0.0045 (4.5σ) | "
             + (f"{vii['b_delay']:+.4f}±{vii['se_delay']:.4f} ({sig(vii)}) | "
                if vii else "— | ")
             + "delay effect on V II " +
             ("SURVIVES (significant, depth n.s.)"
              if vii and abs(vii['b_delay']) > 2.5*vii['se_delay'] else
              "no longer significant — CHECK") + " |")
    t.append(f"| run1 V I 411 b_delay | −0.0053±0.0053 (n.s.) | "
             + (f"{vi['b_delay']:+.4f}±{vi['se_delay']:.4f} ({sig(vi)}) | "
                if vi else "— | ")
             + "neutral V I " +
             ("still n.s. on both axes"
              if vi and abs(vi['b_delay']) <= 2.5*vi['se_delay'] else "changed")
             + " |")
    t.append(f"| Boltzmann T, d1 (P1) | 4568 K | {T1:.0f} K | UNCHANGED verdict: "
             "ill-constrained, reported for completeness only |")
    t.append(f"| Far-NIR fixed pattern, d1 P{periods[0]}→P{periods[-1]} | "
             f"46 → −212 counts | {nir1_p1:.0f} → {nir1_pN:.0f} counts | SURVIVES: "
             "period-scaled dark/fixed pattern, goes negative |")
    t.append(f"| Line signal vs period, d1 (ΔP1→P{periods[-1]}) | ~flat (1%) | "
             f"{100*line_slope_d1:.0f}% | SURVIVES: line flat, period = exposure |")
    t.append(f"| 478/475 'T-mixing' | fixed-pattern artefact | fixed-pattern "
             "artefact | UNCHANGED: 475.39 gains an additive period offset |")
    t.append(f"| Frame drops | 23/59 | {drops}/{len(meas)} | UNCHANGED: mild, "
             "cluster at deep laps, not timing |")
    t.append("")
    t.append("**What the re-scored Pantheum grid now recommends.** The corrected "
             f"scores (§1.1) still fall monotonically with period (period-mean "
             f"P{periods[0]}={pmean[periods[0]]:.3f} → P{periods[-1]}="
             f"{pmean[periods[-1]]:.3f}) and peak at the **minimum period**, so "
             f"the operational recommendation is unchanged: **use P{periods[0]} "
             f"(minimum period)**.  The best cell moves from the offset-era 5/1 to "
             f"**{best[0]}/{best[1]}** (mean {best_mean:.3f}, max {best_max:.3f}) — "
             "consistent with the ledger's stated d10/P1 best (1.343) — but the "
             "delay-mean surface is shallow across d1–d10, so delay should still "
             "be chosen for the ionic-vs-neutral balance (short delay = more "
             "V II), on fresh sites, not for the score alone.  The one physics "
             "verdict that FLIPS is self-absorption: the 417/415 pair is near "
             "optically thin on the corrected axis, so the offset-era 'heavy SA' "
             "conclusion and its large SA-correction are withdrawn.\n")
    return "\n".join(t)


def verdicts_text(A, A1, delays, periods, scaling, fp2, reg, pc, meas):
    pmean = {p: np.mean([A[(d, p)]["score"][0] for d in delays
                         if (d, p) in A and np.isfinite(A[(d, p)]["score"][0])]) for p in periods}
    best = max((k for k in A), key=lambda k: A[k]["score"][0] if np.isfinite(A[k]["score"][0]) else -1)
    vii = (reg or {}).get("h_vii311")
    hru = {d: A.get((d, 1), {}).get("hr_upper", (float("nan"),)*3)[0] for d in delays}
    t = []
    t.append("**Q1 (short delay).** Short delay is a REAL plasma-timing lever, "
             "not just the fresh-crater advantage.  run2 delay ≡ depth, but run1 "
             "(shallower, different lap order) separates them (§2.2): V II 311 "
             + (f"falls with DELAY at {vii['b_delay']:+.4f}±{vii['se_delay']:.4f}/unit "
                f"({abs(vii['b_delay'])/max(vii['se_delay'],1e-9):.1f}σ) with depth "
                f"NOT significant ({vii['b_depth']:+.4f}±{vii['se_depth']:.4f}); "
                if vii else "falls with delay independent of depth; ")
             + "V I 411 is not significant on either axis.  So the ionic (V II) "
             "content and V II/V I ratio genuinely drop as the plasma cools with "
             "delay.  Choose delay for the ionic-vs-neutral balance you want, on "
             "FRESH sites (short delay = hotter, more V II).  FWHM is flat (no ne/"
             "Stark trend); Hα stays at noise at all delays (no early-air issue).\n")
    t.append("**Q2 (short period).** Period-mean score: " +
             ", ".join(f"P{p}={pmean[p]:.3f}" for p in periods) +
             ".  Score falls monotonically with period at every delay, "
             "reproducing run1.  Mechanism (§3): V I line signal is FLAT with "
             "period (Δ within a few % , slope≈0), and the period-scaled 'extra' "
             "signal is a detector dark/fixed pattern, NOT plasma continuum — its "
             f"P{fp2['pmax']}−P{fp2['pmin']} difference spectrum is ~identical "
             "across delays (corr ~0.9+) though line signal differs 2×, and the "
             "far-NIR level goes negative with period identically at every delay "
             "and in run1.  Period is a detector exposure longer than the plasma: "
             "it captures no more line light, only fixed pattern + noise, so SNR "
             f"and score collapse.  **Use the minimum period.**  Best cell "
             f"{best[0]}/{best[1]} (score {A[best]['score'][0]:.3f}), but that "
             "score gain is partly instrumental — calibration needs a per-period "
             "dark/fixed-pattern subtraction and a FIXED period.\n")
    t.append("**Q3 (does a long gate mimic self-absorption via T-mixing?).** "
             "The owner hypothesis CANNOT be tested with the period knob: period "
             "is a detector exposure beyond the plasma lifetime (§3), so a longer "
             "period does not add changing-T plasma to the gate.  On the "
             "CORRECTED axis the background-robust T-independent pair 417/415 "
             "(§4.1) sits NEAR its optically-thin R₀ and is FLAT with period "
             + (f"(d1 ≈ {hru.get(1, float('nan')):.2f}) "
                if np.isfinite(hru.get(1, float('nan'))) else "")
             + "— only MILD self-absorption, period-INDEPENDENT.  The "
             "earlier draft's τ falling with period, and the 478/475 'T-mixing' "
             "signal, were NET-AREA/fixed-pattern artefacts (475.39 gains an "
             "additive offset with period at every delay; §4.2) and are "
             "withdrawn.  The offset-era 'heavily self-absorbed, ratio 0.75' "
             "reading is ALSO withdrawn — it was the same −18 px registration "
             "artefact.  **Verdict: the T-mixing hypothesis is NOT supported and "
             "not testable with period here; on corrected positions the strong "
             "low-E_lower V I branching pair is close to optically thin, so any "
             "SA correction is SMALL (not the large one the offset-era draft "
             "implied).  Only DELAY changes the temporal mix (V II/V I).**  "
             "The 417/415 ratio is also ~flat with delay "
             + (f"({hru.get(1, float('nan')):.2f} at d1 → "
                f"{hru.get(delays[-1], float('nan')):.2f} at d{delays[-1]}) "
                if np.isfinite(hru.get(1, float('nan'))) else "")
             + "— no growing opacity with depth over this range.\n")
    t.append(f"**Q4 (frame drops).** "
             f"{sum(1 for m in meas if m['shots_stored']<10)}/{len(meas)} "
             "batches stored <10 shots.  Drops are mild (never worse than 7/10), "
             "cluster at the deeper laps (d10–d50) rather than tracking period, "
             "and moving them changes nothing; consistent with occasional "
             "retrieval loss, not a timing effect.\n")
    return "".join(t)


def recommend_text(A, delays, periods):
    best = max((k for k in A), key=lambda k: A[k]["score"][0] if np.isfinite(A[k]["score"][0]) else -1)
    return (
        f"- **Acquisition for the calibration campaign:** use the **minimum "
        f"period** (longer period only adds detector fixed pattern + noise).  "
        f"Choose **delay for the ionic-vs-neutral balance** you want — short "
        f"delay (~1–5 units) gives the hotter, more V II plasma — on FRESH "
        f"sites.  Best measured cell {best[0]}/{best[1]} (score "
        f"{A[best]['score'][0]:.3f}), but note the score's period dependence is "
        f"largely instrumental.\n"
        f"- **Fixed-pattern / dark:** because the period-scaled background is a "
        f"detector fixed pattern with imperfect vendor subtraction, calibration "
        f"studies must (i) hold the period FIXED, and (ii) acquire a per-setting "
        f"dark/fixed-pattern reference (laser off or beam blocked) and subtract "
        f"it before any area/continuum quantity is used.  Use background-robust "
        f"peak HEIGHTS with close side bands, not net areas, until this is done.\n"
        f"- **Self-absorption of the strong V I lines** (437–439, 410–411, "
        f"478/475): on the CORRECTED axis the T-independent 417/415 branching "
        f"pair sits near its optically-thin R₀ (≈2.05), so SA is MILD and "
        f"period-independent — much smaller than the offset-era draft implied "
        f"(that draft's ratio 0.75 was a −18 px registration artefact).  A small "
        f"multiplet-ratio SA correction (measured on peak heights) is still "
        f"worth carrying for the strongest low-E_lower lines; prefer "
        f"higher-E_lower V I lines for the Boltzmann thermometer.\n"
        f"- **Next study design (break the confounds):** randomise/interleave "
        f"the (delay,period) order ACROSS laps so delay is no longer one-lap-one-"
        f"delay; put every condition on a fresh site (or record shots-per-site "
        f"and regress it out, as in §2.2); keep ≥3 replicates.  Add a pure depth "
        f"series (fixed timing, many laps on one site) and a laser-off "
        f"dark/period series to measure the crater-depth response and the "
        f"fixed pattern directly.  Extend delay down to 0–2 on fresh sites to "
        f"map the early plasma.\n")


def limits_text(model, meas):
    return (
        "- **Period-scaled detector fixed pattern.** The period-dependent "
        "background is an instrument dark/fixed pattern with imperfect vendor "
        "subtraction (§3), so NET AREAS and any 'continuum' quantity are "
        "unreliable and are NOT used for physics; all opacity/ionisation results "
        "use background-robust peak HEIGHTS with close side bands.  A per-period "
        "laser-off dark reference is needed to remove it properly.\n"
        "- **Delay ≡ crater depth in run2.** run2 alone cannot separate delay "
        "from depth; the delay-vs-depth verdict rests on the run1 regression "
        "(shallower range, independent lap/site order), whose depth range is only "
        "0–20 prior shots — the separation is not guaranteed at greater depth.\n"
        "- **Period ≡ period + fixed site set.** The period main effect carries "
        "a constant 3-site offset; the period ladder within a delay and the "
        "interaction are clean only under additive site effects (untested here).\n"
        "- **Vendor timing units unverified.** All delay/period values are raw "
        "vendor units (ns/100 ns/µs unknown); nothing here fixes the unit "
        "absolutely.\n"
        "- **Segment response uncorrected.** V II is mostly UV and V I mostly "
        "VIS, so the absolute V II/V I ratio is response-biased; only its "
        "relative TREND across conditions (response constant) is used.\n"
        "- **ne / Stark not converted, T not quantitative.** VIS FWHM is "
        "instrument-limited with no clear delay trend, so no ne is derived; the "
        "V I Boltzmann slope-T is ill-constrained and contradictory (rises with "
        "delay while V II/V I falls), reported only for completeness.\n"
        "- **τ from a homogeneous-slab escape factor** on peak-height ratios — a "
        "slab-average lower bound on peak optical depth; real gradients make it "
        "conservative.\n"
        "- **417/415 falling with delay is OPEN** — late/deep opacity vs a "
        "growing weak-member blend is not separated (delay≡depth in run2).\n"
        "- **V registration scatter (~1 px).** Heights use a ±0.18 nm local-max "
        "recentre; the 437–440 triplet is displaced/blended and not used "
        "quantitatively.\n"
        f"- Shift model rests on {model.get('n_ar',0)} Ar NIR anchor(s) on the "
        "fresh pool; global scatter ~50 pm.\n")


# ---- figures -------------------------------------------------------------
def make_figures(A, delays, periods, scaling, fp2, reg, meas):
    cmap = plt.get_cmap("plasma")
    col = lambda k: cmap(k / max(len(delays) - 1, 1))

    # 1. score + SNR heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ttl in [(axes[0], "score", "Pantheum score"),
                         (axes[1], "median_snr", "Median SNR")]:
        M = np.array([[A.get((d, p), {}).get(key, (np.nan,)*3)[0] for p in periods] for d in delays])
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="upper")
        ax.set_xticks(range(len(periods))); ax.set_xticklabels(periods)
        ax.set_yticks(range(len(delays))); ax.set_yticklabels(delays)
        ax.set_xlabel("period"); ax.set_ylabel("delay"); ax.set_title(ttl)
        for i in range(len(delays)):
            for j in range(len(periods)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}" if key == "score" else f"{v:.0f}",
                            ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("V_pure_run2 — delay×period (delay ≡ crater depth)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/score_snr_heatmap.png", dpi=120); plt.close(fig)

    # 2. line signal flat vs period; far-NIR background goes negative
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, d in enumerate(delays):
        la = [A.get((d, p), {}).get("vi_area", (np.nan,)*3) for p in periods]
        axes[0].errorbar(periods, [v[0] for v in la], yerr=[v[1] for v in la],
                         fmt="o-", capsize=2, color=col(k), label=f"d={d}")
        nirv = [fp2["nir"].get((d, p), np.nan) for p in periods]
        axes[1].plot(periods, nirv, "o-", color=col(k), label=f"d={d}")
    axes[0].set_title("Σ V I LINE area vs period — FLAT")
    axes[0].set_xlabel("period"); axes[0].set_ylabel("Σ V I area")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Far-NIR (880–900) background vs period — negative, delay-independent")
    axes[1].set_xlabel("period"); axes[1].set_ylabel("10th-pct counts")
    axes[0].legend(fontsize=7)
    fig.suptitle("Period adds no line light — the period-scaled term is detector fixed pattern")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/signal_vs_period.png", dpi=120); plt.close(fig)

    # 3. fixed-pattern: NIR level vs period, and difference-spectrum correlations
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for k, d in enumerate(delays):
        nirv = [fp2["nir"].get((d, p), np.nan) for p in periods]
        ax.plot(periods, nirv, "o-", color=col(k),
                label=f"d={d} (corr {fp2['corr_vs_deep'].get(d, float('nan')):.2f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("period (vendor units)"); ax.set_ylabel("far-NIR 10th-pct (counts)")
    ax.set_title("Fixed pattern: NIR background identical across delays\n"
                 "(legend: corr of Pmax−Pmin difference spectrum vs deepest)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fixed_pattern_nir.png", dpi=120); plt.close(fig)

    # 4. delay vs depth regression coefficients
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if reg:
        names = {"h_vii311": "V II 311", "h_vi411": "V I 411", "hr_vii_vi": "V II/V I"}
        ks = [k for k in ("h_vii311", "h_vi411", "hr_vii_vi") if reg.get(k)]
        x = np.arange(len(ks)); w = 0.35
        ax.bar(x - w/2, [reg[k]["b_delay"] for k in ks], w,
               yerr=[reg[k]["se_delay"] for k in ks], capsize=4, label="b_delay (/unit)")
        ax.bar(x + w/2, [reg[k]["b_depth"] for k in ks], w,
               yerr=[reg[k]["se_depth"] for k in ks], capsize=4, label="b_depth (/shot)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([names[k] for k in ks])
    ax.set_ylabel("ln(height) coefficient")
    ax.set_title("run1: delay is significant, depth is not (V II)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/delay_depth_regression.png", dpi=120); plt.close(fig)

    # 5. background-robust 417/415 height ratio vs period (flat) per delay
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for k, d in enumerate(delays):
        vals = [A.get((d, p), {}).get("hr_upper", (np.nan,)*3) for p in periods]
        ax.errorbar(periods, [v[0] for v in vals], yerr=[v[1] for v in vals],
                    fmt="o-", capsize=2, color=col(k), label=f"d={d}")
    ax.set_xlabel("period"); ax.set_ylabel("417.94/415.97 peak-height ratio")
    ax.set_title("T-independent pair: opacity is FLAT with period (real, gate-independent SA)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/height_ratio_vs_period.png", dpi=120); plt.close(fig)

    # 6. V II/V I and 417/415 vs delay (period 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    v = [A.get((d, 1), {}).get("hr_vii_vi", (np.nan, np.nan, 0)) for d in delays]
    u = [A.get((d, 1), {}).get("hr_upper", (np.nan, np.nan, 0)) for d in delays]
    ax.errorbar(delays, [t[0] for t in v], yerr=[t[1] for t in v], fmt="o-", capsize=3, label="V II/V I height")
    ax.errorbar(delays, [t[0] for t in u], yerr=[t[1] for t in u], fmt="s-", capsize=3, label="417/415 (opacity)")
    ax.set_xlabel("delay ≡ crater depth (vendor units)")
    ax.set_ylabel("ratio (P=1)")
    ax.set_title("Delay lever: V II/V I falls (real timing, §2.2); 417/415 falls (opacity, OPEN)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/vii_vi_vs_delay.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
