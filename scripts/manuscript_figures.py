#!/usr/bin/env python3
"""Manuscript figures from real spectra via the production pipeline.

Subcommands
-----------
    analyze [--keys ...] [--workers N]   fill manuscript/figures/cache/<key>.pkl + .json
    fig F1|F2|F3|F4|F5 ... | all         render manuscript/figures/F<n>-<slug>.{pdf,png}

Rendering reads the cache only (``--from-cache`` is the default); it never
recomputes.  Every analysis is run with the ``analyze_spectrum`` adopted
defaults (n_calls 40, grid search, gp_seed 42, overlap "conserve",
partition "barklem2016", ipd "debye") and ``capture_stages=True``.  The Fe
runs reuse ``scripts.fit_quality_panel.load_fe`` (18-px corrected native grid)
through ``scripts.showcase_data.load`` and pass no extra keywords (the Fe panel
passes only n_calls/draws).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CACHE = REPO / "manuscript" / "figures" / "cache"
FIGDIR = REPO / "manuscript" / "figures"
DB = str(REPO / "db")
SEGMENT_EDGES = (365.0, 620.0)

KEYS = ["user_spec1", "mw2_1000", "jc_mdd017_24_15_2nd_1", "jc_alaskire",
        "fe_fresh_d5p25", "fe_reused_fail"]

# ----------------------------------------------------------------- analysis


def _run_one(key: str) -> dict:
    from scripts.showcase_data import load
    from alibz.pipeline import analyze_spectrum
    x, y, entry = load(key)
    t0 = time.time()
    analysis = analyze_spectrum(np.asarray(x, float), np.asarray(y, float),
                                DB, capture_stages=True)
    rt = time.time() - t0
    payload = {"key": key, "x": np.asarray(x, float), "y": np.asarray(y, float),
               "entry": entry, "runtime_s": rt, "analysis": analysis}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{key}.pkl", "wb") as fh:
        pickle.dump(payload, fh)
    (CACHE / f"{key}.json").write_text(json.dumps(summarize(payload), indent=2) + "\n")
    return {"key": key, "runtime_s": rt}


def _run_one_safe(key: str) -> dict:
    try:
        return _run_one(key)
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"key": key, "error": f"{exc}", "tb": traceback.format_exc()}


def cmd_analyze(args) -> int:
    keys = args.keys or KEYS
    workers = max(1, min(args.workers, len(keys)))
    print(f"analyzing {keys} with {workers} worker(s)")
    results = []
    if workers == 1:
        for k in keys:
            print(f"  start {k}")
            results.append(_run_one_safe(k))
            print(f"  done  {results[-1]}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for r in pool.imap_unordered(_run_one_safe, keys):
                print(f"  done  {r}")
                results.append(r)
    err = [r for r in results if "error" in r]
    for r in err:
        print(f"ERROR {r['key']}: {r['error']}", file=sys.stderr)
        print(r.get("tb", ""), file=sys.stderr)
    return 1 if err else 0


# ------------------------------------------------------------- cache access


def load_cache(key: str) -> dict:
    with open(CACHE / f"{key}.pkl", "rb") as fh:
        return pickle.load(fh)


def stages_of(payload: dict) -> list:
    return payload["analysis"].get("stages", [])


def stage_map(payload: dict) -> dict:
    """stage-name -> snapshot (deepen collapsed to a list under 'deepen')."""
    out = {}
    for s in stages_of(payload):
        if s["stage"] == "deepen":
            out.setdefault("deepen", []).append(s)
        else:
            out[s["stage"]] = s
    return out


def _top_fracs(frac: dict, thresh: float = 0.01) -> dict:
    return {k: round(float(v), 4) for k, v in
            sorted(frac.items(), key=lambda kv: -kv[1]) if v >= thresh}


def summarize(payload: dict) -> dict:
    a = payload["analysis"]
    sm = stage_map(payload)
    res = a["result"]
    solves = []
    for name in ("pass1", "pass2"):
        s = sm.get(name)
        if s:
            solves.append(dict(stage=name, label=s["label"],
                               solve_index=s.get("solve_index"),
                               T=round(float(s["T"]), 1),
                               log_ne=round(float(s["log_ne"]), 4),
                               r_squared=round(float(s.get("r_squared", float("nan"))), 4),
                               n_peaks=int(s["n_peaks"]),
                               fractions=_top_fracs(s["element_fractions"])))
    for s in sm.get("deepen", []):
        solves.append(dict(stage="deepen", bar=s.get("bar"),
                           solve_index=s.get("solve_index"),
                           T=round(float(s["T"]), 1),
                           log_ne=round(float(s["log_ne"]), 4),
                           accepted=bool(s.get("accepted")),
                           collapsed=bool(s.get("collapsed")),
                           n_peaks=int(s["n_peaks"]),
                           fractions=_top_fracs(s["element_fractions"])))
    for name in ("deepen_final", "sa_recovery"):
        s = sm.get(name)
        if s:
            d = dict(stage=name, label=s.get("label"),
                     T=round(float(s["T"]), 1) if s.get("T") is not None else None,
                     log_ne=round(float(s["log_ne"]), 4) if s.get("log_ne") is not None else None,
                     n_peaks=int(s["n_peaks"]),
                     fractions=_top_fracs(s["element_fractions"]))
            solves.append(d)

    def dec_counts(stage_name):
        s = sm.get(stage_name)
        if not s:
            return {}
        c = {}
        for d in s.get("decisions", []):
            c[d.get("action", "?")] = c.get(d.get("action", "?"), 0) + 1
        return c

    dets = []
    for d in a.get("detections", []) or []:
        dets.append({k: (round(float(d[k]), 4) if isinstance(d.get(k), (int, float)) and d.get(k) is not None else d.get(k))
                     for k in ("element", "status", "fraction", "fraction_resolved",
                               "unc", "z", "n_lines", "clear_lines",
                               "strongest_peak_nm", "confounder", "stage_disagreement")})
    sa = sm.get("sa_recovery", {})
    extra = {}
    if payload["key"] == "mw2_1000":
        try:
            extra["f3_refinement"] = _f3_numbers(payload)
        except Exception as exc:  # noqa: BLE001
            extra["f3_refinement"] = {"error": str(exc)}
        try:
            extra["f2_walkthrough"] = _f2_numbers(payload)
        except Exception as exc:  # noqa: BLE001
            extra["f2_walkthrough"] = {"error": str(exc)}
    if payload["key"].startswith("fe_"):
        try:
            extra["f5_boltzmann"] = _f5_numbers(payload)
        except Exception as exc:  # noqa: BLE001
            extra["f5_boltzmann"] = {"error": str(exc)}
    if payload["key"] == "jc_mdd017_29_2c_1":
        try:
            extra["f4_survey"] = _f4_numbers()
        except Exception as exc:  # noqa: BLE001
            extra["f4_survey"] = {"error": str(exc)}
    return dict(
        key=payload["key"], runtime_s=round(float(payload["runtime_s"]), 1),
        n_points=int(payload["x"].size),
        wl_range_nm=[round(float(payload["x"].min()), 2), round(float(payload["x"].max()), 2)],
        final=dict(T=round(float(res.temperature), 1), log_ne=round(float(res.ne), 4),
                   sigma=round(float(res.sigma), 5), gamma=round(float(res.gamma), 5),
                   r_squared=round(float(res.r_squared), 4),
                   fractions=_top_fracs(res.element_fractions)),
        solves=solves,
        n_peaks_per_stage={s["stage"] + (f"[{s.get('bar')}]" if s["stage"] == "deepen" else ""):
                           int(s["n_peaks"]) for s in stages_of(payload)},
        decisions=dict(refine_data=dec_counts("refine_data"),
                       refine_physics=dec_counts("refine_physics")),
        detections=dets,
        resolved_fractions=_top_fracs(a.get("resolved_fractions", {}) or {}),
        sa_recovery=dict(sa_used=sa.get("sa_used") if isinstance(sa, dict) else None,
                         n_sa_records=len(sa.get("sa_records", []) if isinstance(sa, dict) else [])),
        **extra,
    )


#: F3 windows (shared by the figure and the cache numbers)
F3_WINDOWS = (("K_I_766.49", 765.0, 771.0, 766.49),
              ("Li_I_670.78", 669.5, 672.0, 670.78))


def _f3_numbers(payload):
    from alibz.refinement import sa_voigt
    from alibz.utils.voigt import multi_voigt
    a = payload["analysis"]; x = payload["x"]; y = payload["y"]
    fit = a["final"]; bg = _bg(fit, y); ybg = y - bg
    model_full = _model(fit, x)
    out = {}
    for name, lo, hi, tgt in F3_WINDOWS:
        m = (x >= lo) & (x <= hi)
        decs = _decs_in(a, lo, hi)
        yS = np.zeros(int(m.sum())); yA = np.zeros(int(m.sum()))
        for st, d in decs:
            yS = yS + multi_voigt(x[m], np.asarray(d["params_single"], float)[:4])
            yA = yA + sa_voigt(x[m], *np.asarray(d["params_asym"], float)[:6])
        yF = model_full[m]
        st0, d0 = min(decs, key=lambda sd: abs(sd[1]["center"] - tgt))
        bic = {k: round(float(v), 1) for k, v in d0["bic"].items()}
        out[name] = dict(stage=st0, center=round(float(d0["center"]), 3),
                         action=d0["action"], verdict=d0["verdict"], bic=bic,
                         delta_bic_S_minus_A=round(bic["S"] - bic["A"], 1),
                         n_decisions_in_window=len(decs),
                         rms_sym=round(_rms(ybg[m] - yS), 1),
                         rms_asym=round(_rms(ybg[m] - yA), 1),
                         rms_final=round(_rms(ybg[m] - yF), 1),
                         window=[round(lo, 2), round(hi, 2)])
    return out


JC_SURVEY = REPO / "data" / "jchristensen_20261104"
F4_SAMPLE = "MDD017-29-2c #1 Kaleb"


def _f4_numbers():
    import csv
    det = list(csv.DictReader(open(JC_SURVEY / "detections.csv")))
    summ = list(csv.DictReader(open(JC_SURVEY / "summary.csv")))

    def fl(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    mn = [r for r in det if r["element"] == "Mn"]
    r2s = [fl(r.get("r_squared")) for r in summ]
    r2s = [v for v in r2s if v == v]
    survey = dict(n_spectra=len({r["sample"] for r in det}),
                  mn_confounded=sum(1 for r in mn if r["status"] == "confounded"),
                  mn_upper_limit=sum(1 for r in mn if r["status"] == "upper-limit"),
                  samples_any_confounded=len({r["sample"] for r in det
                                              if r["status"] == "confounded"}),
                  median_r2=round(float(np.median(r2s)), 3) if r2s else None)
    mnmg = {}
    for el in ("Mn", "Mg"):
        rr = [r for r in det if r["sample"] == F4_SAMPLE and r["element"] == el]
        if rr:
            r = rr[0]
            mnmg[el] = dict(status=r["status"], fraction=round(fl(r["fraction"]), 4),
                            fraction_resolved=round(fl(r["fraction_resolved"]), 4),
                            contested_share=round(fl(r["contested_share"]), 3),
                            confounder=r["confounder"] or None)
    return dict(sample=F4_SAMPLE, survey=survey, mn_mg=mnmg)


def _f2_numbers(payload):
    a = payload["analysis"]
    sm = {}
    for s in a["stages"]:
        sm.setdefault(s["stage"], []).append(s)
    snaps = _f2_solve_snaps(sm)
    final_fr = a["result"].element_fractions
    els = [e for e, f in final_fr.items() if f >= 0.01]
    worst_el, worst = None, 0.0
    for e in els:
        vals = [float(s[1]["element_fractions"].get(e, 0.0)) for s in snaps]
        d = max(vals) - min(vals)
        if d > worst:
            worst, worst_el = d, e
    chosen, fallback, scored = _select_windows(a, payload["x"], payload["y"])
    return dict(solve_labels=[s[0] for s in snaps],
                max_abs_frac_change=round(float(worst), 4), max_change_element=worst_el,
                windows=[{k: c[k] for k in ("lo", "hi", "score", "na", "nsm",
                                            "rms", "rel_change")} for c in chosen],
                used_fallback=bool(fallback))


def _f5_numbers(payload):
    pts, T, log_ne, kB = _fe_boltzmann_points(payload)
    ions = pts[:, 0].astype(int) if pts.size else np.zeros(0, int)
    n_drop = int(np.sum(pts[:, 2] < -12.0)) if pts.size else 0
    out = dict(n_fe_peaks=int(pts.shape[0]) if pts.size else 0,
               n_fe_i=int(np.sum(ions == 1)), n_fe_ii=int(np.sum(ions == 2)),
               T=round(float(T), 1), log_ne=round(float(log_ne), 4),
               slope_per_eV=round(-1.0 / (kB * T), 4),
               n_dropped_below_m12=n_drop)
    ff = _f5_feI_fit(payload)
    out["feI_vis"] = dict(resolved=bool(ff.get("resolved")),
                          n_lines=int(ff.get("n", 0)),
                          n_candidates=int(ff.get("n_candidates", 0)),
                          T_B=round(float(ff["T"]), 1) if not ff.get("bad") else None,
                          sigT=round(float(ff["sigT"]), 1) if not ff.get("bad") else None,
                          r2=round(float(ff.get("r2", float("nan"))), 3) if not ff.get("bad") else None,
                          span_eV=round(float(ff.get("span", 0.0)), 2),
                          shift_vis_nm=round(float(ff.get("shift_vis_nm", 0.0)), 5),
                          reason=ff.get("reason"))
    return out


def cmd_resummarize(args) -> int:
    keys = args.keys or KEYS
    for k in keys:
        pkl = CACHE / f"{k}.pkl"
        if not pkl.is_file():
            print(f"  skip {k} (no pkl)"); continue
        payload = load_cache(k)
        (CACHE / f"{k}.json").write_text(json.dumps(summarize(payload), indent=2) + "\n")
        print(f"  resummarized {k}")
    return 0


# --------------------------------------------------------------- rendering

FIG_FUNCS = {}
_SEG_LABELS = (("UV", 180, 365), ("VIS", 365, 620), ("NIR", 620, 961))


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _setup_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "figure.dpi": 200, "savefig.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "alibz",
        "axes.linewidth": 0.6, "lines.linewidth": 0.8, "legend.frameon": False,
    })


def _model(fit, xs):
    from alibz.utils.voigt import multi_voigt
    pk = np.atleast_2d(np.asarray(fit["sorted_parameter_array"], float))
    if pk.size == 0:
        return np.zeros_like(xs)
    return multi_voigt(xs, np.ravel(pk[:, :4]))


def _bg(fit, y):
    return np.asarray(fit.get("background", np.zeros_like(y)), float)


def _noise(ybg):
    from alibz.peaky_finder import PeakyFinder
    return PeakyFinder._noise_scale_local(np.asarray(ybg, float))


def _peaks(fit):
    return np.atleast_2d(np.asarray(fit["sorted_parameter_array"], float))


def _seglines(ax, labels=False, y=None):
    for e in SEGMENT_EDGES:
        ax.axvline(e, ls="--", color="0.45", lw=0.6, zorder=0)
    if labels and y is not None:
        for name, lo, hi in _SEG_LABELS:
            ax.text(0.5 * (lo + hi), y, name, ha="center", va="top",
                    fontsize=6.5, color="0.35")


def _save(fig, name):
    plt = _plt()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def _jnum(key):
    return json.loads((CACHE / f"{key}.json").read_text())


# ------------------------------------------------------------------- F1

def fig_F1():
    plt = _plt(); _setup_rc()
    from alibz.utils.colors import spectral_line, wavelength_to_rgb
    p = load_cache("user_spec1"); a = p["analysis"]
    x, y = p["x"], p["y"]; final = a["final"]; res = a["result"]
    bg = _bg(final, y); ybg = y - bg; model = _model(final, x)
    noise = _noise(ybg); pk = _peaks(final)
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 6.2), sharex=True,
                            gridspec_kw={"height_ratios": [2, 3, 1.3]})
    spectral_line(axs[0], x, y, lw=0.4)
    axs[0].plot(x, bg, color="0.2", lw=0.8, label="fitted background")
    axs[0].set_ylabel("raw counts"); axs[0].legend(loc="lower right", frameon=True, framealpha=0.9)
    _seglines(axs[0], labels=True, y=0.98 * axs[0].get_ylim()[1])

    spectral_line(axs[1], x, ybg, lw=0.4)
    axs[1].plot(x, model, color="0.3", lw=0.7, alpha=0.9, label="final model")
    base = axs[1].get_ylim()[0]
    span = ybg.max() - base
    axs[1].vlines(pk[:, 1], base, base + 0.035 * span,
                  colors=wavelength_to_rgb(pk[:, 1]), lw=0.5, zorder=3)
    axs[1].set_ylabel("counts - background"); axs[1].legend(loc="upper left")
    _seglines(axs[1])
    txt = (f"{pk.shape[0]} fitted peaks\n$T$ = {res.temperature:.0f} K\n"
           f"$\\log n_e$ = {res.ne:.2f}\n$r^2$ = {res.r_squared:.3f}")
    axs[1].text(0.985, 0.95, txt, transform=axs[1].transAxes, ha="right",
                va="top", fontsize=6.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", lw=0.5, alpha=0.9))

    resid = (ybg - model) / np.maximum(noise, 1e-12)
    axs[2].plot(x, resid, "k-", lw=0.4)
    axs[2].fill_between(x, -2, 2, color="g", alpha=0.15,
                        label=r"$\pm 2\sigma$ local noise")
    axs[2].axhline(0, color="0.5", lw=0.5)
    axs[2].set_ylim(-8, 8)
    axs[2].set_ylabel(r"residual [$\sigma$]")
    axs[2].set_xlabel("wavelength [nm]"); axs[2].legend(loc="upper right")
    _seglines(axs[2])
    axs[0].set_xlim(x.min(), x.max())
    fig.align_ylabels(axs)
    _save(fig, "F1-overview")


FIG_FUNCS["F1"] = fig_F1


# ------------------------------------------------------------------- F2

def _peak_centres(fit):
    pk = _peaks(fit)
    return pk[:, 1] if pk.size else np.zeros(0)


def _peak_heights(fit):
    from scipy.special import voigt_profile
    pk = _peaks(fit)
    if pk.size == 0:
        return np.zeros((0, 2))
    h = pk[:, 0] * voigt_profile(0.0, np.maximum(pk[:, 2], 1e-6),
                                 np.maximum(pk[:, 3], 1e-6))
    return np.column_stack([pk[:, 1], h])


def _added_centres(fit_prev, fit_cur, tol=0.06):
    prev = _peak_centres(fit_prev); cur = _peak_centres(fit_cur)
    if cur.size == 0:
        return np.zeros(0)
    if prev.size == 0:
        return cur
    return np.array([c for c in cur if np.min(np.abs(prev - c)) > tol])


def _resid_sigma_rms(fit, x, y, lo, hi):
    bg = _bg(fit, y); ybg = y - bg
    nz = np.maximum(_noise(ybg), 1e-12)
    m = (x >= lo) & (x <= hi)
    return _rms(((ybg - _model(fit, x)) / nz)[m])


def _f2_solve_snaps(sm):
    snaps = [("P1", sm["pass1"][0]), ("P2", sm["pass2"][0])]
    for s in sm.get("deepen", []):
        if s.get("accepted"):
            snaps.append((f"D{s.get('bar')}", s))
    snaps.append(("SA", sm["sa_recovery"][0]))
    return snaps


def _select_windows(a, x, y, width=5.0):
    """Score 5-nm candidate windows by visible change; reject windows whose
    per-column residual RMS spread is < 5%.  Returns (chosen, used_fallback,
    scored)."""
    sm = {}
    for s in a["stages"]:
        sm.setdefault(s["stage"], []).append(s)
    blind = sm["blind_fit"][0]["fit"]
    phys = sm["refine_physics"][0]["fit"]
    final = a["final"]
    noise = np.maximum(_noise(y - _bg(final, y)), 1e-12)
    added = _added_centres(phys, final)
    hts = _peak_heights(final)
    splitmerge = [float(d["center"]) for d in sm["refine_data"][0].get("decisions", [])
                  if d.get("action") in ("split", "merge")]
    centers = np.arange(x.min() + width, x.max() - width, width / 2)
    scored = []
    for c in centers:
        lo, hi = c - width / 2, c + width / 2
        na = 0
        for ac in added:
            if lo <= ac <= hi:
                si = noise[min(len(noise) - 1, int(np.searchsorted(x, ac)))]
                hi_ = hts[np.argmin(np.abs(hts[:, 0] - ac)), 1] if hts.size else 0.0
                if hi_ > 5 * si:
                    na += 1
        nsm = sum(1 for s in splitmerge if lo <= s <= hi)
        rmss = [_resid_sigma_rms(f, x, y, lo, hi) for f in (blind, phys, final)]
        rel = (max(rmss) - min(rmss)) / max(np.mean(rmss), 1e-9)
        scored.append(dict(c=float(c), lo=float(lo), hi=float(hi),
                           score=na + nsm, na=na, nsm=nsm,
                           rms=[round(r, 2) for r in rmss], rel_change=round(float(rel), 3)))
    passing = [s for s in scored if s["rel_change"] >= 0.05 and s["score"] > 0]
    pool = sorted(passing or scored, key=lambda s: (-s["score"], -s["rel_change"]))
    chosen = []
    for s in pool:
        if all(abs(s["c"] - ch["c"]) > width for ch in chosen):
            chosen.append(s)
        if len(chosen) == 2:
            break
    chosen.sort(key=lambda s: s["c"])
    return chosen, (not passing), scored


def fig_F2():
    plt = _plt(); _setup_rc()
    from alibz.utils.colors import spectral_line
    from alibz.elements import element_color, element_sort_key
    p = load_cache("mw2_1000"); a = p["analysis"]; x, y = p["x"], p["y"]
    sm = {}
    for s in a["stages"]:
        sm.setdefault(s["stage"], []).append(s)
    stages = [("blind fit", sm["blind_fit"][0]["fit"]),
              ("3a+3b refinement", sm["refine_physics"][0]["fit"]),
              ("final", a["final"])]
    chosen, fallback, _scored = _select_windows(a, x, y)
    windows = [(c["lo"], c["hi"]) for c in chosen]
    print(f"  F2 windows: " + ", ".join(
        f"{c['lo']:.1f}-{c['hi']:.1f} (score {c['score']}, rel {c['rel_change']})"
        for c in chosen) + (f"  [FALLBACK: no window passed]" if fallback else ""))
    col_colors = ["#1b9e77", "#d95f02", "#7570b3"]

    fig = plt.figure(figsize=(6.8, 7.8), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.5], wspace=0.06)
    left = outer[0, 0].subgridspec(4, 3, height_ratios=[3, 1, 3, 1],
                                   hspace=0.06, wspace=0.08)
    for wi, (lo, hi) in enumerate(windows):
        m = (x >= lo) & (x <= hi)
        ymax = max(1.0, float(np.max((y - _bg(a["final"], y))[m])) * 1.18)
        for ci, (lab, fit) in enumerate(stages):
            ad = fig.add_subplot(left[2 * wi, ci])
            ar = fig.add_subplot(left[2 * wi + 1, ci], sharex=ad)
            bg = _bg(fit, y); ybg = y - bg; mdl = _model(fit, x)
            nz = np.maximum(_noise(ybg), 1e-12)
            ad.plot(x[m], ybg[m], "-", color="0.62", lw=2.0, alpha=0.9,
                    solid_capstyle="round")
            ad.plot(x[m], mdl[m], color=col_colors[ci], lw=0.9)
            ad.set_ylim(min(0, float(np.min(ybg[m]))), ymax)
            # ticks for peaks new in this column vs the previous column
            if ci > 0:
                new = _added_centres(stages[ci - 1][1], fit)
                new = new[(new >= lo) & (new <= hi)]
                ad.vlines(new, 0.90 * ymax, 0.99 * ymax, color="#e7298a", lw=1.0)
            rr = ((ybg - mdl) / nz)[m]
            ar.plot(x[m], rr, "k-", lw=0.5)
            ar.axhspan(-2, 2, color="g", alpha=0.12); ar.axhline(0, color="0.5", lw=0.5)
            ar.set_ylim(-6, 6); ar.set_xlim(lo, hi)
            ar.text(0.03, 0.12, f"RMS {_rms(rr):.1f}$\\sigma$", transform=ar.transAxes,
                    fontsize=5.2, color="0.2")
            if wi == 0:
                ad.set_title(lab, fontsize=7)
            if ci == 0:
                ad.set_ylabel(f"{lo:.0f}-{hi:.0f} nm\ncounts$-$bg", fontsize=6.3)
                ar.set_ylabel(r"resid [$\sigma$]", fontsize=6.0)
            else:
                ad.tick_params(labelleft=False); ar.tick_params(labelleft=False)
            ad.tick_params(labelbottom=False)
            # x labels only on the very bottom row (wi==1 residual strips)
            ar.tick_params(labelbottom=True, labelsize=5.5)   # each window has its own range
            if wi == 1:
                ar.set_xlabel("nm", fontsize=6.3)

    # right column: composition / plasma / n-peaks
    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1.2, 1, 0.8], hspace=0.5)
    snaps = _f2_solve_snaps(sm)
    xpos = np.arange(len(snaps))
    final_fr = a["result"].element_fractions
    els = sorted([e for e, f in final_fr.items() if f >= 0.01], key=element_sort_key)
    axc = fig.add_subplot(right[0]); bottoms = np.zeros(len(snaps)); handles = []
    for e in els:
        vals = np.array([float(s[1]["element_fractions"].get(e, 0.0)) for s in snaps])
        h = axc.bar(xpos, vals, bottom=bottoms, width=0.8, color=element_color(e),
                    edgecolor="white", linewidth=0.2)
        handles.append(h); bottoms += vals
    axc.set_xticks(xpos); axc.set_xticklabels([s[0] for s in snaps], fontsize=6)
    axc.set_ylabel("composition / solve", fontsize=6.3); axc.set_ylim(0, 1)
    axc.legend(handles, els, ncol=min(len(els), 5), fontsize=5.0,
               loc="lower center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=0.7, handlelength=0.8, handletextpad=0.3)

    axt = fig.add_subplot(right[1])
    Ts = np.array([float(s[1]["T"]) if s[1].get("T") is not None else np.nan for s in snaps])
    nes = np.array([float(s[1]["log_ne"]) if s[1].get("log_ne") is not None else np.nan for s in snaps])
    r2s = [s[1].get("r_squared") for s in snaps]
    axt.plot(xpos, Ts / 1e3, "o-", color="#d62728", ms=3)
    axt.set_ylabel("T [kK]", color="#d62728", fontsize=6.3)
    axt.tick_params(axis="y", colors="#d62728", labelsize=5.5)
    Tm = np.nanmean(Ts) / 1e3
    axt.set_ylim(Tm - 1.0, Tm + 1.0)
    for xi, Tv in zip(xpos, Ts / 1e3):
        if np.isfinite(Tv):
            axt.annotate(f"{Tv*1e3:.0f}", (xi, Tv), fontsize=4.6, color="#d62728",
                         textcoords="offset points", xytext=(0, 4), ha="center")
    axn = axt.twinx()
    axn.plot(xpos, nes, "s--", color="#1f77b4", ms=3)
    axn.set_ylabel(r"$\log n_e$", color="#1f77b4", fontsize=6.3)
    axn.tick_params(axis="y", colors="#1f77b4", labelsize=5.5)
    nem = np.nanmean(nes)
    axn.set_ylim(nem - 0.5, nem + 0.5)
    axt.set_xticks(xpos); axt.set_xticklabels([s[0] for s in snaps], fontsize=6)
    axt.set_title("plasma state / solve", fontsize=6.8)
    r2txt = "  ".join(f"{s[0]} r²={v:.2f}" for s, v in zip(snaps, r2s) if v is not None)
    axt.text(0.0, -0.42, r2txt, transform=axt.transAxes, fontsize=4.8, color="0.25")

    axp = fig.add_subplot(right[2])
    order = ["blind_fit", "refine_data", "pass1", "refine_physics", "seed",
             "recover", "deblend", "pass2", "deepen", "deepen_final",
             "sa_recovery", "detections"]
    npk = {}
    for s in a["stages"]:
        npk[s["stage"]] = s["n_peaks"]  # last deepen bar wins for 'deepen'
    seq = [(k, npk[k]) for k in order if k in npk]
    axp.step(range(len(seq)), [v for _, v in seq], where="mid", color="0.3", lw=1.0)
    axp.scatter(range(len(seq)), [v for _, v in seq], s=8, color="0.3")
    axp.set_xticks(range(len(seq)))
    axp.set_xticklabels([k.replace("_", " ") for k, _ in seq], fontsize=4.4,
                        rotation=90, va="top")
    axp.set_ylabel("n peaks", fontsize=6.3)
    axp.tick_params(axis="y", labelsize=5.5)
    axp.set_title("peaks per stage", fontsize=6.8)
    _save(fig, "F2-walkthrough")


FIG_FUNCS["F2"] = fig_F2


# ------------------------------------------------------------------- F3

def _find_dec(a, target, stages=("refine_physics", "refine_data")):
    sm = {}
    for s in a["stages"]:
        sm.setdefault(s["stage"], []).append(s)
    best = None
    for st in stages:
        for d in sm.get(st, [{}])[0].get("decisions", []):
            if d.get("params_single") is None:
                continue
            if best is None or abs(d["center"] - target) < abs(best[1]["center"] - target):
                best = (st, d)
        if best is not None and abs(best[1]["center"] - target) < 0.5:
            break
    return best


def _rms(res):
    res = np.asarray(res, float)
    return float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan")


def _decs_in(a, lo, hi):
    """sa-tag/asymmetric decisions (with asym params) whose centre is in the window."""
    sm = {}
    for s in a["stages"]:
        sm.setdefault(s["stage"], []).append(s)
    out = []
    for st in ("refine_physics", "refine_data"):
        for d in sm.get(st, [{}])[0].get("decisions", []):
            if (lo <= d.get("center", -1) <= hi and d.get("params_single") is not None
                    and d.get("params_asym") is not None):
                out.append((st, d))
    # one record per centre (prefer refine_physics, listed first)
    seen, uniq = set(), []
    for st, d in out:
        key = round(float(d["center"]), 2)
        if key not in seen:
            seen.add(key); uniq.append((st, d))
    return uniq


def fig_F3():
    plt = _plt(); _setup_rc()
    from alibz.refinement import sa_voigt
    from alibz.utils.voigt import multi_voigt
    p = load_cache("mw2_1000"); a = p["analysis"]; x, y = p["x"], p["y"]
    fit = a["final"]; bg = _bg(fit, y); ybg = y - bg
    model_full = _model(fit, x)
    # K window widened to show the 769.90 partner; Li window tight on 670.78
    targets = [("K I 766.49 / 769.90 nm", 765.0, 771.0, 766.49),
               ("Li I 670.78 nm", 669.5, 672.0, 670.78)]
    fig, axall = plt.subplots(2, 2, figsize=(6.5, 5.2),
                              gridspec_kw={"height_ratios": [3, 1.2]},
                              constrained_layout=True)
    for ci, (title, lo, hi, tgt) in enumerate(targets):
        m = (x >= lo) & (x <= hi)
        decs = _decs_in(a, lo, hi)
        yS = np.zeros(int(m.sum())); yA = np.zeros(int(m.sum()))
        for st, d in decs:
            yS = yS + multi_voigt(x[m], np.asarray(d["params_single"], float)[:4])
            yA = yA + sa_voigt(x[m], *np.asarray(d["params_asym"], float)[:6])
        yF = model_full[m]
        ax, axr = axall[0, ci], axall[1, ci]
        ax.plot(x[m], ybg[m], "-", color="0.6", lw=2.4, alpha=0.9, label="data$-$bg",
                solid_capstyle="round")
        ax.plot(x[m], yS, color="#1f77b4", lw=1.1, label="symmetric (S)")
        ax.plot(x[m], yA, color="#d62728", lw=1.1, ls="--", label="self-absorbed (A)")
        ax.plot(x[m], yF, color="k", lw=0.9, alpha=0.85, label="final model")
        ax.set_title(title, fontsize=7)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=5.6, loc="upper right")
        # primary decision for the printed numbers
        st0, d0 = min(decs, key=lambda sd: abs(sd[1]["center"] - tgt))
        bic = d0["bic"]; dbic = float(bic["S"]) - float(bic["A"])
        rms_s, rms_a, rms_f = _rms(ybg[m] - yS), _rms(ybg[m] - yA), _rms(ybg[m] - yF)
        info = (f"{st0.split('_')[-1]}: {d0['action']}, {d0['verdict']}\n"
                f"$\\Delta$BIC(S$-$A)={dbic:.0f} (S {bic['S']:.0f}/A {bic['A']:.0f}/B {bic['B']:.0f})\n"
                f"window RMS: S={rms_s:.1f}, A={rms_a:.1f}, final={rms_f:.1f}")
        ax.text(0.02, 0.60, info, transform=ax.transAxes, va="top", ha="left",
                fontsize=5.5, bbox=dict(boxstyle="round", fc="white", ec="0.6",
                                        lw=0.4, alpha=0.92))
        if ci == 0:
            ax.set_ylabel("counts $-$ bg", fontsize=6.5)
            axr.set_ylabel("data $-$ model\n[counts]", fontsize=6.3)
        axr.plot(x[m], ybg[m] - yS, color="#1f77b4", lw=0.7, label="S")
        axr.plot(x[m], ybg[m] - yA, color="#d62728", lw=0.7, ls="--", label="A")
        axr.plot(x[m], ybg[m] - yF, color="k", lw=0.7, label="final")
        axr.axhline(0, color="0.5", lw=0.5)
        axr.set_xlim(lo, hi)
        axr.set_xlabel("wavelength [nm]", fontsize=6.5)
        if ci == 0:
            axr.legend(fontsize=5.2, loc="upper right", ncol=3, columnspacing=0.8,
                       handlelength=1.2)
    _save(fig, "F3-refinement")


FIG_FUNCS["F3"] = fig_F3


# ------------------------------------------------------------------- F4

def fig_F4():
    plt = _plt(); _setup_rc()
    from scipy.special import voigt_profile
    from alibz.elements import element_color, element_sort_key
    p = load_cache("jc_mdd017_29_2c_1"); a = p["analysis"]
    x, y = p["x"], p["y"]; fit = a["final"]; res = a["result"]
    bg = _bg(fit, y); ybg = y - bg; model = _model(fit, x)
    pk = _peaks(fit)
    lo, hi = 278.5, 286.0
    m = (x >= lo) & (x <= hi)

    def area_to_height(wl, area):
        j = int(np.argmin(np.abs(pk[:, 1] - wl)))
        f0 = voigt_profile(0.0, max(pk[j, 2], 1e-6), max(pk[j, 3], 1e-6))
        return float(area * f0)

    fig = plt.figure(figsize=(6.5, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x[m], ybg[m], "-", color="0.6", lw=1.6, alpha=0.9, label="data$-$bg")
    ax.plot(x[m], model[m], color="k", lw=0.9, alpha=0.85, label="final model")
    support = a.get("support", {}) or {}
    dx = {"Mn": -0.05, "Mg": +0.05}
    seen = set()
    for el in ("Mn", "Mg"):
        col = element_color(el)
        for pred, wl, obs in support.get(el, []):
            if not (lo <= wl <= hi):
                continue
            ph, oh = area_to_height(wl, pred), area_to_height(wl, obs)
            lp = f"{el} predicted" if ("p" + el) not in seen else None
            lo_ = f"{el} observed" if ("o" + el) not in seen else None
            seen.update({"p" + el, "o" + el})
            ax.vlines(wl + dx[el], 0, ph, color=col, lw=2.2, label=lp)
            ax.plot(wl + dx[el], oh, marker="_", color=col, ms=9, mew=1.4, label=lo_)
    ax.annotate("Mg II 279.55/280.27", (280.0, ax.get_ylim()[1] * 0.96),
                fontsize=5.8, ha="center", color="0.3")
    ax.annotate("Mg I 285.2", (285.2, ax.get_ylim()[1] * 0.42),
                fontsize=5.8, ha="center", color="0.3")
    ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("counts $-$ bg")
    ax.legend(fontsize=5.4, loc="upper right", ncol=2, columnspacing=0.8)
    ax.set_xlim(lo, hi)

    axb = fig.add_subplot(gs[0, 1])
    asfit = res.element_fractions
    resolved = a.get("resolved_fractions", {}) or {}
    contested = a.get("contested", {}) or {}
    els = sorted(set(["Mn", "Mg"]) | {e for e, f in asfit.items() if f >= 0.01},
                 key=element_sort_key)
    xp = np.arange(len(els))
    axb.bar(xp - 0.2, [asfit.get(e, 0.0) for e in els], 0.4, label="as-fit",
            color="#9467bd")
    axb.bar(xp + 0.2, [resolved.get(e, 0.0) for e in els], 0.4, label="resolved",
            color="#2ca02c")
    top = max([asfit.get(e, 0.0) for e in els] + [0.1])
    for i, e in enumerate(els):
        cs = contested.get(e, {}).get("contested_share")
        if cs is not None:
            axb.text(i, max(asfit.get(e, 0.0), resolved.get(e, 0.0)) + 0.03 * top,
                     f"cs {cs:.2f}", ha="center", fontsize=4.6, color="0.3", rotation=90)
    axb.set_xticks(xp); axb.set_xticklabels(els, fontsize=6)
    axb.set_ylabel("element fraction", fontsize=6.5)
    axb.set_ylim(0, top * 1.35)
    axb.legend(fontsize=6, loc="upper right")
    dmap = {d["element"]: d for d in a.get("detections", [])}
    lines = []
    for e in ("Mn", "Mg"):
        if e in dmap:
            dd = dmap[e]
            cf = f" (conf {dd['confounder']})" if dd.get("confounder") else ""
            lines.append(f"{e}: {dd['status']}{cf}")
    axb.set_title("\n".join(lines), fontsize=5.6, color="0.25")
    _save(fig, "F4-confounder")


FIG_FUNCS["F4"] = fig_F4


# ------------------------------------------------------------------- F5

def _fe_boltzmann_points(payload):
    """Per-Fe-peak reduced Boltzmann quantity at the fitted (T, log ne).

    Y_j = ln[ A_obs,j * 4pi * lambda_j * Z_s(T) / ( h c * gA_j * f_s(T,ne) ) ]
    using the indexer's line_emissivity convention; both ion stages land on
    the single line Y = ln C - E_k/(k_B T) because f_s divides out the Saha
    stage fraction.  Peaks are mapped to the strongest Fe line of the
    assigned ion within 0.15 nm of the peak centre (registration-shifted).
    """
    from alibz.utils.database import Database
    from alibz.pipeline import _get_sb
    from alibz.utils.sahaboltzmann import line_emissivity
    from alibz.utils.constants import BOLTZMANN, PLANCK, SPEED_OF_LIGHT
    from alibz.peaky_indexer_v3 import (_COL_ION, _COL_WAVELENGTH, _COL_GA,
                                        _COL_EI, _COL_EK)
    from alibz.utils.wavelength import shift_at
    a = payload["analysis"]; res = a["result"]; fit = a["final"]
    pk = _peaks(fit)
    T = float(res.temperature); log_ne = float(res.ne)
    db = Database(DB)
    sb = _get_sb(DB, "barklem2016", "debye")
    fe = db.lines("Fe")[:, [_COL_ION, _COL_WAVELENGTH, _COL_GA, _COL_EI, _COL_EK]].astype(float)
    shift = a.get("shift")
    pts = []
    Tarr = np.array([T])
    ci, _ = sb.ionization_distribution("Fe", Tarr, log_ne)
    for pa in res.peak_assignments:
        if pa["element"] != "Fe" or pa["total_predicted"] <= 0:
            continue
        ion = int(pa["ion"]); wl = float(pk[pa["peak_idx"], 1])
        try:
            s_loc = float(shift_at(shift, wl)) if shift is not None else 0.0
        except Exception:
            s_loc = 0.0
        rows = fe[fe[:, 0] == ion]
        if rows.size == 0:
            continue
        dobs = np.abs((rows[:, 1] + s_loc) - wl)
        near = dobs <= 0.15
        if not np.any(near):
            continue
        sub = rows[near]
        j = int(np.argmax(sub[:, 2]))      # strongest gA line in window
        lam, gA, Ei, Ek = sub[j, 1], sub[j, 2], sub[j, 3], sub[j, 4]
        Z = float(sb.stage_partition("Fe", Tarr, ion=ion)[0])
        f = float(ci[0, ion - 1])
        if gA <= 0 or f <= 0 or Z <= 0:
            continue
        obs_fe = pa["observed"] * (pa["contribution"] / pa["total_predicted"])
        if obs_fe <= 0:
            continue
        emiss_full = float(line_emissivity(lam, gA, Ek, T, Z, f))
        emiss_noexp = emiss_full * np.exp(Ek / (BOLTZMANN * T))
        if emiss_noexp <= 0:
            continue
        pts.append((ion, float(Ek), float(np.log(obs_fe / emiss_noexp)), float(wl)))
    return np.array(pts, float), T, log_ne, BOLTZMANN


# -- Fe I visible-line Boltzmann thermometer, copied verbatim (2026-11-04) from
#    cal/scripts/fe_plasma_analysis.py (build_thermo_set/measure_thermo/
#    robust_boltzmann/fit_resolved) so the figure needs no argparse/heavy import.
KB = None  # set from alibz.utils.constants at call time
T0_THERMO = 9000.0


def build_thermo_set(db, stage, wl_lo, wl_hi, ei_min, topn, isolate_frac=0.15,
                     iso_win=0.25, t0=T0_THERMO):
    kt0 = KB * t0
    fe = db.lines("Fe")
    ion = fe[:, 0].astype(float); wl = fe[:, 1].astype(float)
    gA = fe[:, 3].astype(float); Ei = fe[:, 4].astype(float)
    Ek = fe[:, 5].astype(float); gk = fe[:, 13].astype(float)
    fe12 = ion <= 2
    w12 = wl[fe12]; s12 = gA[fe12] * np.exp(-Ek[fe12] / kt0)
    sel = (ion == stage) & (wl >= wl_lo) & (wl < wl_hi) & (Ei > ei_min) & (gA > 0)
    cand = []
    for i in np.where(sel)[0]:
        own = gA[i] * np.exp(-Ek[i] / kt0)
        near = (np.abs(w12 - wl[i]) < iso_win) & (np.abs(w12 - wl[i]) > 1e-6)
        if s12[near].sum() < isolate_frac * own:
            cand.append(dict(strength=float(own), wl=float(wl[i]),
                             Ek=float(Ek[i]), Ei=float(Ei[i]),
                             gA=float(gA[i]), gk=float(gk[i]), ion=stage))
    cand.sort(key=lambda c: -c["strength"])
    return cand[:topn]


def measure_thermo(wl, y, center, search=0.30, area_half=0.32,
                   side_lo=0.45, side_hi=0.90):
    lo = int(np.searchsorted(wl, center - search))
    hi = int(np.searchsorted(wl, center + search))
    if hi <= lo:
        return None
    j = lo + int(np.argmax(y[lo:hi])); pk = wl[j]
    L = list(range(int(np.searchsorted(wl, pk - side_hi)),
                   int(np.searchsorted(wl, pk - side_lo))))
    R = list(range(int(np.searchsorted(wl, pk + side_lo)),
                   int(np.searchsorted(wl, pk + side_hi))))
    if len(L) < 3 or len(R) < 3:
        return None
    lx, rx = wl[L].mean(), wl[R].mean()
    ly, ry = np.median(y[L]), np.median(y[R])
    if rx == lx:
        return None
    bg = lambda xx: ly + (ry - ly) * (xx - lx) / (rx - lx)
    side = L + R
    res = y[side] - bg(wl[side])
    noise = max(1.0, 1.4826 * np.median(np.abs(res - np.median(res))))
    C = list(range(int(np.searchsorted(wl, pk - area_half)),
                   int(np.searchsorted(wl, pk + area_half))))
    if len(C) < 3:
        return None
    h = y[C] - bg(wl[C])
    return dict(area=float(np.trapezoid(h, wl[C])), height=float(h.max()),
                snr=float(h.max() / noise), offset=float(pk - center))


def robust_boltzmann(wl, y, cand, shift, minlines=8, snr_min=5.0, clip=1.6):
    pts = []
    for c in cand:
        m = measure_thermo(wl, y, c["wl"] + shift)
        if not m or m["snr"] < snr_min or m["area"] <= 0:
            continue
        if abs(m["offset"] - shift) > 0.2:
            continue
        pts.append([c["Ek"], math.log(m["area"] * c["wl"] / c["gA"]),
                    c["wl"], m["snr"]])
    P = np.array(pts, dtype=float)
    if len(P) < minlines:
        return dict(bad=True, n=len(P), reason="too_few_lines")
    dropped = []
    for _ in range(40):
        xx, yv = P[:, 0], P[:, 1]
        sl, ic = np.polyfit(xx, yv, 1)
        r = yv - (sl * xx + ic)
        rms = r.std()
        w = int(np.argmax(np.abs(r)))
        if abs(r[w]) > clip * rms and len(P) - 1 >= minlines:
            dropped.append(float(P[w, 2]))
            P = np.delete(P, w, axis=0)
            continue
        break
    xx, yv = P[:, 0], P[:, 1]
    (sl, ic), cov = np.polyfit(xx, yv, 1, cov=True)
    span = float(xx.max() - xx.min())
    if sl >= 0:
        return dict(bad=True, n=len(P), span=span, reason="positive_slope")
    T = -1.0 / (KB * sl)
    sigT = abs(1.0 / (KB * sl ** 2)) * math.sqrt(cov[0, 0])
    yh = sl * xx + ic
    r2 = 1 - np.sum((yv - yh) ** 2) / np.sum((yv - yv.mean()) ** 2)
    return dict(T=float(T), sigT=float(sigT), n=int(len(P)), span=span,
                r2=float(r2), rms=float((yv - yh).std()), slope=float(sl),
                intr=float(ic), bad=False, x=xx, y=yv,
                wls=[float(v) for v in P[:, 2]], n_dropped=len(dropped))


def fit_resolved(f, span_min=2.4, r2_min=0.5, rel_sig=0.5, n_min=8):
    return bool(f and not f.get("bad") and f.get("T", float("nan")) == f.get("T", float("nan"))
               and f.get("T", 0) > 0 and f.get("n", 0) >= n_min
               and f.get("span", 0) >= span_min and f.get("r2", 0) >= r2_min
               and f.get("sigT", 1e9) / f.get("T", 1) < rel_sig)


def _f5_feI_fit(payload):
    """Line-selected Fe I visible Boltzmann fit on the same Fe spectrum."""
    global KB
    from alibz.utils.database import Database
    from alibz.utils.constants import BOLTZMANN
    from alibz.utils.wavelength import shift_at
    KB = BOLTZMANN
    a = payload["analysis"]; wl, y = payload["x"], payload["y"]
    db = Database(DB)
    shift = shift_at(a.get("shift"), 520.0) if a.get("shift") is not None else 0.0
    cand = build_thermo_set(db, 1, 420, 620, 0.8, 60)
    fit = robust_boltzmann(np.asarray(wl, float), np.asarray(y, float), cand, float(shift))
    fit["shift_vis_nm"] = float(shift)
    fit["resolved"] = fit_resolved(fit)
    fit["n_candidates"] = len(cand)
    return fit


def fig_F5():
    plt = _plt(); _setup_rc()
    from alibz.elements import element_color, element_sort_key
    p = load_cache("fe_fresh_d5p25")
    pts, T, log_ne, kB = _fe_boltzmann_points(p)
    npts = int(pts.shape[0]) if pts.size else 0
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                  gridspec_kw={"width_ratios": [1.6, 1.0]},
                                  constrained_layout=True)
    YLO, YHI = -12.0, 2.0
    seg_col = {"UV": "#7570b3", "VIS": "#1b9e77", "NIR": "#d95f02"}
    n_drop = 0; Y = Ek = None
    if pts.size:
        wlp = pts[:, 3]; Ek = pts[:, 1]; Y = pts[:, 2]
        for name, lo, hi in _SEG_LABELS:
            mm = (wlp >= lo) & (wlp < hi) & (Y >= YLO)
            if np.any(mm):
                ax.scatter(Ek[mm], Y[mm], s=13, color=seg_col[name],
                           label=f"Fe II {name}", edgecolors="0.25", linewidths=0.3)
        n_drop = int(np.sum(Y < YLO))
        slope = -1.0 / (kB * T)
        keep = Y >= YLO
        intercept = float(np.median(Y[keep] - slope * Ek[keep]))
        xs = np.array([Ek.min(), Ek.max()])
        ax.plot(xs, slope * xs + intercept, "k-", lw=1.0,
                label=f"pipeline $T$={T/1e3:.1f} kK")
    ff = _f5_feI_fit(p)
    med_pipe = float(np.median(Y[Y >= YLO])) if pts.size else 0.0
    if not ff.get("bad"):
        fx, fy = np.asarray(ff["x"]), np.asarray(ff["y"])
        off = med_pipe - float(np.median(fy))
        ax.scatter(fx, fy + off, s=18, marker="^", facecolors="none",
                   edgecolors="#e7298a", linewidths=0.8,
                   label=f"Fe I VIS ({ff['n']} lines)")
        xs2 = np.array([fx.min(), fx.max()])
        ax.plot(xs2, ff["slope"] * xs2 + ff["intr"] + off, color="#e7298a",
                lw=1.1, ls="--",
                label=f"Fe I $T_B$={ff['T']/1e3:.1f}$\\pm${ff['sigT']/1e3:.1f} kK")
    elif pts.size:
        sl_ref = -1.0 / (kB * 10805.0)
        xs2 = np.array([1.0, 6.0])
        icref = med_pipe - sl_ref * float(np.median(Ek[Ek < 6.5]))
        ax.plot(xs2, sl_ref * xs2 + icref, color="#e7298a", lw=1.1, ls=":",
                label=f"Fe I unresolved; campaign 10.8 kK")
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(r"upper-level energy $E_k$ [eV]")
    ax.set_ylabel(r"$\ln[\,A_{\rm obs}\,\lambda Z_s/(h c\, gA\, f_s)\,]$  (Fe I offset)",
                  fontsize=6.0)
    ax.legend(fontsize=5.0, loc="lower left")
    ax.set_title(f"two thermometers, {npts} Fe II peaks", fontsize=7)

    # (b) composition: consistent colours, fresh solid / reused hatched
    res = p["analysis"]["result"]
    fr = res.element_fractions
    pf = load_cache("fe_reused_fail")["analysis"]["result"].element_fractions
    zmap = {d["element"]: d.get("z") for d in _jnum("fe_fresh_d5p25")["detections"]}
    els = sorted({e for e, f in fr.items() if f >= 0.005}
                 | {e for e, f in pf.items() if f >= 0.05}, key=element_sort_key)
    xp = np.arange(len(els))
    cols = [element_color(e) for e in els]
    axb.bar(xp - 0.2, [fr.get(e, 0.0) for e in els], 0.4, color=cols,
            label="fresh site")
    axb.bar(xp + 0.2, [pf.get(e, 0.0) for e in els], 0.4, color=cols,
            hatch="////", edgecolor="white", linewidth=0.3, alpha=0.6,
            label="reused (fail)")
    for i, e in enumerate(els):
        z = zmap.get(e)
        axb.text(i - 0.2, fr.get(e, 0.0) + 0.015,
                 f"{fr.get(e,0.0):.2f}" + (f"\n$z${z:.0f}" if z is not None else ""),
                 ha="center", fontsize=5.0)
    axb.set_xticks(xp); axb.set_xticklabels(els, fontsize=6)
    axb.set_ylabel("element fraction", fontsize=6.5); axb.set_ylim(0, 1.14)
    axb.legend(fontsize=5.6, loc="upper center")
    axb.set_title("composition: fresh vs reused", fontsize=7)
    _save(fig, "F5-standard")


FIG_FUNCS["F5"] = fig_F5


def cmd_fig(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    names = args.names
    if names == ["all"]:
        names = ["F1", "F2", "F3", "F4", "F5"]
    rc = 0
    for n in names:
        fn = FIG_FUNCS.get(n)
        if fn is None:
            print(f"no such figure {n}", file=sys.stderr)
            rc = 1
            continue
        print(f"rendering {n} ...")
        fn()
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--keys", nargs="*", default=None)
    a.add_argument("--workers", type=int, default=6)
    a.set_defaults(func=cmd_analyze)
    r = sub.add_parser("resummarize")
    r.add_argument("--keys", nargs="*", default=None)
    r.set_defaults(func=cmd_resummarize)
    f = sub.add_parser("fig")
    f.add_argument("names", nargs="+")
    f.add_argument("--from-cache", action="store_true", default=True)
    f.set_defaults(func=cmd_fig)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
