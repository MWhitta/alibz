"""Shared inverse-model benchmark for CF-LIBS (Objective 2).

Generates synthetic LIBS spectra with KNOWN ground truth (T, Ne, composition)
from the canonical NIST-backed forward model (:mod:`bench.nist_forward`), then
scores BOTH inverse engines on the same data so a head-to-head decides which
goes to production (the user's "benchmark both, then choose"):

* Inverse-B  — the standalone full-spectrum VarPro engine
               (``cflibs.inversion.CFLIBSInverter``): given the candidate
               element set, solves T, Ne, self-absorption and composition
               jointly from the whole spectrum.
* Inverse-A  — the integrated peak-vector engine
               (``alibz.PeakyFinder`` -> ``alibz.PeakyIndexer``): fits Voigt
               peaks, then matches them against the FULL 92-element NIST db and
               solves T, Ne, composition by Bayesian-opt + NNLS.

Design asymmetry worth remembering: Inverse-B is handed the candidate element
set; Inverse-A discovers elements from all 92 in the database. The scorer
therefore reports both true-element recovery AND false positives.

Metrics per case: element-ID precision/recall/F1 (at a detection threshold),
composition mean-absolute-error over the element union, temperature % error,
and electron-density error in dex.

Run a quick smoke config locally; run the full config on moissanite (64 cores).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bench.nist_forward import build_nist_forward, default_rock_elements, STARK_NE_REF_CM3

# cflibs prototype (path set up by nist_forward import)
from cflibs.inversion import CFLIBSInverter          # noqa: E402
from cflibs.synthetic import generate_synthetic_observation, estimate_noise_sigma  # noqa: E402


# ----------------------------------------------------------------------
# Ground-truth sampling + synthetic generation
# ----------------------------------------------------------------------

def sample_composition(elements: Sequence[str], rng: np.random.Generator,
                       n_active_range: Tuple[int, int] = (3, 7),
                       exclude_from_active: Sequence[str] = ("H",)) -> np.ndarray:
    """Random sparse composition over ``elements`` (Dirichlet on a random active
    subset), returned as fractions summing to 1. ``H`` is kept out of the active
    majors by default (it is a trace Ne-diagnostic species, not a matrix element).
    """
    idx = [i for i, e in enumerate(elements) if e not in exclude_from_active]
    lo, hi = n_active_range
    n_active = int(rng.integers(lo, min(hi, len(idx)) + 1))
    active = rng.choice(idx, size=n_active, replace=False)
    fr = np.zeros(len(elements))
    fr[active] = rng.dirichlet(np.ones(n_active) * 1.5)
    return fr


def sample_theta(rng: np.random.Generator) -> dict:
    """Random plasma/instrument parameters in the physically-typical LIBS range."""
    T = float(rng.uniform(7000.0, 13000.0))
    logNe = float(rng.uniform(16.0, 17.3))
    logNcol = float(rng.uniform(13.0, 14.3))
    return dict(
        T=T, Ne_cm3=10.0 ** logNe, logNcol=logNcol, Ncol_m2=(10.0 ** logNcol) * 1.0e4,
        lam_shift_nm=float(rng.uniform(-0.01, 0.01)),
        sigma_inst_nm=float(rng.uniform(0.02, 0.06)),
        stark_scale=1.0,
    )


@dataclass
class Case:
    grid: np.ndarray
    observed: np.ndarray
    sigma: np.ndarray
    clean: np.ndarray
    c_true: np.ndarray            # per-element fractions over `elements`
    theta_true: dict
    elements: List[str]


def generate_case(fm, elements, rng, peak_counts=3.0e4) -> Case:
    theta = sample_theta(rng)
    c = sample_composition(elements, rng)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    grid, observed, sigma, clean, _diag, _gain = generate_synthetic_observation(
        fm, theta, c, peak_counts=peak_counts, seed=seed
    )
    return Case(grid=grid, observed=observed, sigma=sigma, clean=clean,
                c_true=c, theta_true=theta, elements=list(elements))


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def _frac_dict(elements: Sequence[str], c: np.ndarray) -> Dict[str, float]:
    s = float(np.sum(c))
    c = c / s if s > 0 else c
    return {e: float(v) for e, v in zip(elements, c) if v > 0}


def score(truth: Dict[str, float], pred: Dict[str, float], det_thresh: float = 0.03) -> dict:
    true_set = {e for e, v in truth.items() if v >= det_thresh}
    pred_set = {e for e, v in pred.items() if v >= det_thresh}
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = set(truth) | set(pred)
    mae = float(np.mean([abs(truth.get(e, 0.0) - pred.get(e, 0.0)) for e in union])) if union else 0.0
    # relative error on the true majors only
    maj = [e for e, v in truth.items() if v >= det_thresh]
    rel = float(np.mean([abs(truth[e] - pred.get(e, 0.0)) / truth[e] for e in maj])) if maj else 0.0
    return dict(precision=precision, recall=recall, f1=f1, comp_mae=mae,
                major_rel_err=rel, false_positives=sorted(pred_set - true_set),
                missed=sorted(true_set - pred_set))


# ----------------------------------------------------------------------
# Engine adapters
# ----------------------------------------------------------------------

def run_cflibs(fm, case: Case, de_maxiter: int, de_popsize: int = 12,
               verbose: bool = False) -> Optional[dict]:
    norm = np.percentile(case.observed, 99.5)
    norm = norm if norm > 0 else 1.0
    inv = CFLIBSInverter(fm, baseline_order=5)
    bounds = dict(T=(6000.0, 15000.0), logNe=(15.5, 17.6), logNcol=(12.5, 14.5),
                  lam_shift_nm=(-0.03, 0.03), sigma_inst_nm=(0.01, 0.08),
                  stark_scale=(0.3, 3.0))
    t0 = time.time()
    try:
        fr = inv.fit(case.observed / norm, case.sigma / norm, bounds,
                     de_popsize=de_popsize, de_maxiter=de_maxiter,
                     polish_maxiter=120, seed=0, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        return dict(engine="cflibs", error=repr(exc), seconds=time.time() - t0)
    pred = _frac_dict(fm.elements, fr.c)
    return dict(engine="cflibs", pred=pred, T=float(fr.theta["T"]),
                logNe=float(np.log10(fr.theta["Ne_cm3"])),
                chi2_reduced=float(fr.chi2_reduced), seconds=time.time() - t0)


def run_indexer(case: Case, n_calls: int, max_ion_stage: int = 2,
                verbose: bool = False) -> Optional[dict]:
    from alibz import PeakyFinder, PeakyIndexer

    finder = PeakyFinder.__new__(PeakyFinder)
    t0 = time.time()
    try:
        fit = finder.fit_spectrum(case.grid, case.observed, subtract_background=True,
                                  plot=False, n_sigma=1)
        peaks = fit["sorted_parameter_array"]
        if peaks.shape[0] == 0:
            return dict(engine="indexer", error="no_peaks", seconds=time.time() - t0)
        res = PeakyIndexer(peaks).run(max_ion_stage=max_ion_stage, n_calls=n_calls,
                                      verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        return dict(engine="indexer", error=repr(exc), seconds=time.time() - t0)
    # Per-element composition straight from FitResult: ion stages are
    # independent estimates of the total element density and are combined
    # by the indexer's tied re-solve (never summed - summing double-counts
    # under the physical-concentration semantics).
    pred = {e: float(v) for e, v in res.element_fractions.items()}
    return dict(engine="indexer", pred=pred, T=float(res.temperature),
                logNe=float(res.ne), r_squared=float(res.r_squared),
                n_peaks=int(peaks.shape[0]), seconds=time.time() - t0)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_benchmark(elements=None, wl_range=(370.0, 600.0), grid_step=0.02, n_cases=3,
                  seed=0, de_maxiter=20, indexer_calls=20, engines=("cflibs", "indexer"),
                  verbose=True) -> dict:
    elements = list(elements) if elements is not None else default_rock_elements()
    grid = np.arange(wl_range[0], wl_range[1], grid_step)
    if verbose:
        print(f"Building forward model: {len(elements)} elements, "
              f"{wl_range[0]}-{wl_range[1]} nm ({len(grid)} px)...")
    fm, meta = build_nist_forward(elements, grid, max_stage=2)
    if verbose:
        print(f"  {meta['_total_lines']} in-band lines")

    rng = np.random.default_rng(seed)
    results = []
    for ci in range(n_cases):
        case = generate_case(fm, elements, rng)
        truth = _frac_dict(elements, case.c_true)
        row = dict(case=ci, truth=truth,
                   theta_true=dict(T=case.theta_true["T"],
                                   logNe=float(np.log10(case.theta_true["Ne_cm3"]))))
        if verbose:
            majors = {e: round(v, 3) for e, v in sorted(truth.items(), key=lambda x: -x[1])}
            print(f"\n[case {ci}] T={case.theta_true['T']:.0f} "
                  f"logNe={np.log10(case.theta_true['Ne_cm3']):.2f}  truth={majors}")
        for eng in engines:
            out = run_cflibs(fm, case, de_maxiter) if eng == "cflibs" else run_indexer(case, indexer_calls)
            if "error" in out:
                row[eng] = dict(error=out["error"], seconds=round(out["seconds"], 1))
                if verbose:
                    print(f"  {eng:8}: ERROR {out['error'][:80]} ({out['seconds']:.0f}s)")
                continue
            sc = score(truth, out["pred"])
            T_err = 100.0 * (out["T"] - case.theta_true["T"]) / case.theta_true["T"]
            Ne_err = out["logNe"] - np.log10(case.theta_true["Ne_cm3"])
            row[eng] = dict(pred={k: round(v, 3) for k, v in sorted(out["pred"].items(), key=lambda x: -x[1])},
                            T=round(out["T"]), logNe=round(out["logNe"], 2),
                            T_err_pct=round(T_err, 1), Ne_err_dex=round(Ne_err, 2),
                            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()},
                            seconds=round(out["seconds"], 1))
            if verbose:
                print(f"  {eng:8}: F1={sc['f1']:.2f} comp_MAE={sc['comp_mae']:.3f} "
                      f"majRelErr={sc['major_rel_err']:.2f} T_err={T_err:+.1f}% "
                      f"Ne_err={Ne_err:+.2f}dex ({out['seconds']:.0f}s)  FP={sc['false_positives']}")
        results.append(row)

    summary = _aggregate(results, engines)
    if verbose:
        print("\n=== SUMMARY ===")
        for eng, agg in summary.items():
            print(f"  {eng:8}: {agg}")
    return dict(config=dict(elements=elements, wl_range=wl_range, n_cases=n_cases,
                            de_maxiter=de_maxiter, indexer_calls=indexer_calls),
                results=results, summary=summary)


def _aggregate(results, engines) -> dict:
    out = {}
    for eng in engines:
        rows = [r[eng] for r in results if eng in r and "error" not in r[eng]]
        n_err = sum(1 for r in results if eng in r and "error" in r[eng])
        if not rows:
            out[eng] = dict(n_ok=0, n_err=n_err)
            continue
        out[eng] = dict(
            n_ok=len(rows), n_err=n_err,
            mean_f1=round(float(np.mean([r["f1"] for r in rows])), 3),
            mean_comp_mae=round(float(np.mean([r["comp_mae"] for r in rows])), 3),
            mean_major_rel_err=round(float(np.mean([r["major_rel_err"] for r in rows])), 3),
            med_T_err_pct=round(float(np.median([abs(r["T_err_pct"]) for r in rows])), 1),
            med_Ne_err_dex=round(float(np.median([abs(r["Ne_err_dex"]) for r in rows])), 2),
            mean_seconds=round(float(np.mean([r["seconds"] for r in rows])), 1),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="CF-LIBS shared inverse benchmark")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--wl-lo", type=float, default=370.0)
    ap.add_argument("--wl-hi", type=float, default=600.0)
    ap.add_argument("--grid-step", type=float, default=0.02)
    ap.add_argument("--de-maxiter", type=int, default=20)
    ap.add_argument("--indexer-calls", type=int, default=20)
    ap.add_argument("--engines", nargs="+", default=["cflibs", "indexer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    report = run_benchmark(wl_range=(args.wl_lo, args.wl_hi), grid_step=args.grid_step,
                           n_cases=args.n_cases, seed=args.seed, de_maxiter=args.de_maxiter,
                           indexer_calls=args.indexer_calls, engines=tuple(args.engines))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
