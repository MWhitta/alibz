"""Fit-quality A/B panel: fixed synthetic + real + Fe items, run and compare.

``fit-quality-panel run --variant NAME [--config-json '{...}']`` scores every
panel item with :func:`alibz.pipeline.analyze_spectrum` and writes one JSON;
``fit-quality-panel compare BASE.json VAR.json`` prints a paired markdown
table with better/worse by each metric's sign convention.
"""

from __future__ import annotations

import os

# Single-threaded BLAS before numpy is imported anywhere (workers inherit).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import multiprocessing as mp
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
FE_NPZ = "data/triage-real/fe_aesar_9998_means_20260922.npz"
REAL_GLOB = "data/remote_samples/*.csv"
FE_T_PLAUSIBLE = 10_000.0
PANELS = ("synthetic", "real", "fe")

# metric -> +1 higher is better, -1 lower is better
SIGN = {
    "median_rel_err": -1, "max_rel_err": -1, "spurious_mass": -1,
    "abs_T_rel_err": -1, "n_majors_missed": -1, "dominant_correct": +1,
    "r_squared": +1, "median_stage_disagreement": -1,
    "max_stage_disagreement": -1, "ne_at_bound": -1,
    "fe_fraction": +1, "non_fe_mass": -1, "n_non_fe": -1,
    "abs_T_dev_10kK": -1,
}
HEADLINE = {
    "synthetic": ["median_rel_err", "spurious_mass", "abs_T_rel_err",
                  "n_majors_missed", "dominant_correct"],
    "real": ["r_squared", "median_stage_disagreement", "n_detected",
             "ne_at_bound"],
    "fe": ["fe_fraction", "non_fe_mass", "n_non_fe", "abs_T_dev_10kK"],
}
SIGN["n_detected"] = 0  # informative, no direction
UNCHANGED_TOL = 1e-9


# ----------------------------------------------------------------- helpers
def _f(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _top(fr: dict, n: int = 5) -> dict:
    items = sorted(((k, float(v)) for k, v in fr.items() if v and v > 0),
                   key=lambda kv: -kv[1])[:n]
    return {k: round(v, 5) for k, v in items}


def _stage_stats(res, elements=None) -> dict:
    sd = getattr(res, "stage_disagreement", None) or {}
    vals = [_f(v) for k, v in sd.items() if elements is None or k in elements]
    vals = [v for v in vals if v is not None]
    return {"median_stage_disagreement": statistics.median(vals) if vals else None,
            "max_stage_disagreement": max(vals) if vals else None}


def _plasma(analysis: dict) -> dict:
    res = analysis["result"]
    out = {"partition_extrapolated": analysis.get("partition_extrapolated"),
           "T_rec": _f(getattr(res, "temperature", None)),
           "log_ne_rec": _f(getattr(res, "ne", None)),
           "r_squared": _f(getattr(res, "r_squared", None)),
           "ne_bounds_source": analysis.get("ne_bounds_source")}
    nb = analysis.get("ne_bounds")
    ne = out["log_ne_rec"]
    at = None
    if nb is not None and ne is not None:
        try:
            at = bool(min(abs(ne - float(nb[0])), abs(ne - float(nb[1]))) < 0.05)
        except Exception:
            at = None
    out["ne_at_bound"] = None if at is None else float(at)
    ci = getattr(res, "convergence_info", None) or {}
    if "stage_temperature_offset_k" in ci:
        out["stage_temperature_offset_k"] = _f(ci["stage_temperature_offset_k"])
    if "lte_flagged_species" in ci:
        out["lte_flagged_species"] = sorted(ci["lte_flagged_species"])
    bi = analysis.get("basin_info") or {}
    out["basin_ambiguous"] = bi.get("basin_ambiguous")
    out["delta_bic_runner_up"] = _f(bi.get("delta_bic_runner_up"))
    return out


def _row_extras(path: str, analysis: dict) -> dict:
    """flags / qc_status via the pipeline's own summary-row builder."""
    try:
        from alibz.pipeline import _summary_row
        row = _summary_row(path, analysis)
        return {"flags": [s for s in str(row.get("flags", "")).split(";") if s],
                "qc_status": row.get("qc_status"),
                "qc_reasons": row.get("qc_reasons")}
    except Exception as exc:  # fakes in tests, or a pipeline change
        return {"flags": [], "qc_status": None,
                "qc_reasons": f"summary_row_unavailable:{type(exc).__name__}"}


# ------------------------------------------------------------- item scoring
def score_synthetic(name: str, db: str, kw: dict, seed: int) -> dict:
    from bench.synth_cases import CASES, render_case, score_recovery
    from alibz.pipeline import analyze_spectrum
    spec = dict(CASES[name])
    comp = spec.pop("composition")
    kw = dict(kw)
    render = kw.pop("_render", None) or {}  # --render-config-json (generator options)
    x, y, scene = render_case(comp, seed=seed, dbpath=db, **spec, **render)
    t0 = time.time()
    a = analyze_spectrum(x, y, db, **kw)
    rt = time.time() - t0
    res = a["result"]
    sc = score_recovery(comp, res.element_fractions)
    md = getattr(scene, "metadata", {}) or {}
    m = _plasma(a)
    m.update(
        majors_missed=list(sc["majors_missed"]),
        n_majors_missed=len(sc["majors_missed"]),
        median_rel_err=_f(sc["median_rel_err"]), max_rel_err=_f(sc["max_rel_err"]),
        spurious_mass=_f(sc["spurious_mass"]),
        dominant_correct=float(bool(sc["dominant_correct"])),
        T_truth=_f(md.get("truth_temperature_k")),
        log_ne_truth=_f(md.get("truth_log_ne_cm3")),
        top5=_top(res.element_fractions), runtime_s=round(rt, 2))
    if m["T_rec"] is not None and m["T_truth"]:
        m["T_rel_err"] = (m["T_rec"] - m["T_truth"]) / m["T_truth"]
        m["abs_T_rel_err"] = abs(m["T_rel_err"])
    else:
        m["T_rel_err"] = m["abs_T_rel_err"] = None
    return m


def score_real(path: str, db: str, kw: dict) -> dict:
    from alibz.pipeline import analyze_spectrum, load_spectrum_csv
    x, y = load_spectrum_csv(path)
    t0 = time.time()
    a = analyze_spectrum(x, y, db, **kw)
    rt = time.time() - t0
    res = a["result"]
    fr = {k: float(v) for k, v in res.element_fractions.items() if v > 0}
    m = _plasma(a)
    m.update(_row_extras(path, a))
    m.update(_stage_stats(res, set(fr)))
    m.update(n_detected=len(fr), top5=_top(fr), runtime_s=round(rt, 2))
    return m


RENDER_KEYS = ("partition_source", "ipd")  # config keys the synthetic generator understands
FE_PIXEL_OFFSET = 18  # module default; set from --fe-pixel-offset


def load_fe(run_id: str, root: Path = REPO, pixel_offset: Optional[int] = None):
    import numpy as np
    d = np.load(root / FE_NPZ, allow_pickle=True)
    ids = [str(r) for r in d["run_ids"]]
    i = ids.index(run_id)
    wl = np.asarray(d["wavelength_nm"][i], float)
    y = np.asarray(d["intensity_counts"][i], float)
    # This npz (2026-09-22) predates the Z300 PIXEL_OFFSET -18 fix: its axis is
    # ~18 px too blue (UV -1.6, VIS -2.3 nm).  Re-derive the corrected axis by
    # taking the label of sample i+18 for sample i (and dropping the last 18
    # samples).  Approximation: across the 365/620 nm segment seams the pitch
    # changes, so ~18 samples at each seam get a slightly wrong label (<1 nm).
    po = FE_PIXEL_OFFSET if pixel_offset is None else pixel_offset
    if po:
        wl, y = wl[po:], y[:len(y) - po]
    # Drop the dead 4th channel: samples after the large (>5 nm) wavelength
    # jump past the last live channel (947.9 -> 960.2 nm; indices 5848..7913,
    # all zero but one stray count at the last sample), then any trailing zeros.
    end = len(y)
    gaps = np.diff(wl)
    if gaps.size and gaps.max() > 5.0:
        end = int(np.argmax(gaps)) + 1
    nz = np.nonzero(y[:end] != 0)[0]
    end = int(nz[-1]) + 1 if nz.size else end
    return wl[:end], y[:end]


def fe_run_ids(root: Path = REPO) -> List[str]:
    import numpy as np
    d = np.load(root / FE_NPZ, allow_pickle=True)
    return [str(r) for r in d["run_ids"]]


def score_fe(run_id: str, db: str, kw: dict, pixel_offset: Optional[int] = None) -> dict:
    from alibz.pipeline import analyze_spectrum
    wl, y = load_fe(run_id, pixel_offset=pixel_offset)
    t0 = time.time()
    a = analyze_spectrum(wl, y, db, **kw)
    rt = time.time() - t0
    res = a["result"]
    fr = {k: float(v) for k, v in res.element_fractions.items() if v > 0}
    fe = fr.get("Fe", 0.0)
    nonfe = {k: v for k, v in fr.items() if k != "Fe"}
    m = _plasma(a)
    m.update(_row_extras(run_id, a))
    sd = getattr(res, "stage_disagreement", {}) or {}
    m.update(fe_fraction=fe, non_fe_mass=1.0 - fe,
             non_fe_above_0p5pct={k: round(v, 4) for k, v in nonfe.items()
                                  if v > 0.005},
             n_non_fe=sum(1 for v in nonfe.values() if v > 0.005),
             n_detected=len(fr), fe_stage_disagreement=_f(sd.get("Fe")),
             top5=_top(fr), n_samples=int(len(y)), runtime_s=round(rt, 2))
    m["abs_T_dev_10kK"] = (abs(m["T_rec"] - FE_T_PLAUSIBLE) / FE_T_PLAUSIBLE
                           if m["T_rec"] is not None else None)
    return m


def run_item(item: dict, db: str, kw: dict, seed: int) -> dict:
    """Score one panel item.  Raises on failure (caller records error row)."""
    if item["panel"] != "synthetic":
        kw = {k: v for k, v in kw.items() if k != "_render"}
    if item["panel"] == "synthetic":
        return score_synthetic(item["id"], db, kw, seed)
    if item["panel"] == "real":
        return score_real(item["path"], db, kw)
    return score_fe(item["id"], db, kw, item.get("fe_pixel_offset"))


def build_items(panels, only=None, root: Path = REPO) -> List[dict]:
    items: List[dict] = []
    if "synthetic" in panels:
        from bench.synth_cases import CASES
        items += [{"panel": "synthetic", "id": n} for n in CASES]
    if "real" in panels:
        for p in sorted(glob.glob(str(root / REAL_GLOB))):
            items.append({"panel": "real", "id": Path(p).stem, "path": p})
    if "fe" in panels:
        items += [{"panel": "fe", "id": r} for r in fe_run_ids(root)]
    if only:
        items = [i for i in items if i["id"] in set(only)]
    return items


# --------------------------------------------------------------- execution
class _Timeout(Exception):
    pass


def _guarded(item, db, kw, seed, timeout_s) -> dict:
    """Run one item; always return a result row (never raises)."""
    row = {"panel": item["panel"], "id": item["id"], "status": "ok",
           "metrics": {}, "error": None}
    use_alarm = hasattr(signal, "SIGALRM") and timeout_s
    if use_alarm:
        def _alarm(*_):
            raise _Timeout(f"timeout after {timeout_s}s")
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s), 5.0)
    try:
        row["metrics"] = run_item(item, db, kw, seed)
    except BaseException as exc:  # noqa: BLE001 - recorded, never aborts
        tb = traceback.format_exc().strip().splitlines()
        row["status"], row["error"] = "error", f"{type(exc).__name__}: {exc} | {tb[-1]}"[:500]
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)
    return row


def _child(item, db, kw, seed, timeout_s, conn):
    conn.send(_guarded(item, db, kw, seed, timeout_s))
    conn.close()


def execute(items, db, kw, seed, workers, timeout_s) -> List[dict]:
    """workers<=1 -> inline (used by tests); else one process per item with a
    hard wall-clock kill at timeout_s+60 so a hang/segfault is an error row."""
    if workers <= 1:
        return [_guarded(i, db, kw, seed, timeout_s) for i in items]
    ctx = mp.get_context("spawn")
    pending, running, done = list(items), [], {}
    while pending or running:
        while pending and len(running) < workers:
            it = pending.pop(0)
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(it, db, kw, seed, timeout_s, w))
            p.start(); w.close()
            running.append((it, p, r, time.time()))
        for rec in list(running):
            it, p, r, t0 = rec
            row = None
            if r.poll(0.2):
                try:
                    row = r.recv()
                except EOFError:
                    row = None
            if row is None and not p.is_alive() and not r.poll(0):
                row = {"panel": it["panel"], "id": it["id"], "status": "error",
                       "metrics": {}, "error": f"worker died (exitcode {p.exitcode})"}
            elif row is None and timeout_s and time.time() - t0 > timeout_s + 60:
                p.kill()
                row = {"panel": it["panel"], "id": it["id"], "status": "error",
                       "metrics": {}, "error": f"hard timeout {timeout_s}s+60"}
            if row is not None:
                p.join(5)
                running.remove(rec)
                done[(it["panel"], it["id"])] = row
                print(f"[{len(done)}/{len(items)}] {it['panel']}/{it['id']} "
                      f"{row['status']} {row['error'] or ''}", flush=True)
    return [done[(i["panel"], i["id"])] for i in items]


# --------------------------------------------------------------- aggregate
def _vals(rows, key):
    return [v for r in rows if r["status"] == "ok"
            for v in [_f(r["metrics"].get(key))] if v is not None]


def aggregate(rows: List[dict]) -> dict:
    out: Dict[str, Any] = {}
    for panel in PANELS:
        pr = [r for r in rows if r["panel"] == panel]
        if not pr:
            continue
        agg = {"n_items": len(pr),
               "n_ok": sum(r["status"] == "ok" for r in pr),
               "n_error": sum(r["status"] != "ok" for r in pr)}
        for k in HEADLINE[panel]:
            v = _vals(pr, k)
            agg[k] = {"mean": statistics.fmean(v) if v else None,
                      "median": statistics.median(v) if v else None,
                      "n": len(v)}
        out[panel] = agg
    return out


# ------------------------------------------------------------------ compare
def _verdict_one(metric: str, b, a):
    if b is None or a is None:
        return "n/a"
    d = a - b
    if abs(d) <= UNCHANGED_TOL * max(1.0, abs(b)):
        return "same"
    s = SIGN.get(metric, 0)
    if s == 0:
        return "changed"
    return "better" if d * s > 0 else "worse"


def _fmt(v):
    if v is None:
        return "-"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def compare_docs(base: dict, var: dict) -> str:
    bi = {(r["panel"], r["id"]): r for r in base["items"]}
    vi = {(r["panel"], r["id"]): r for r in var["items"]}
    keys = [k for k in bi if k in vi]
    lines = [f"# Fit-quality comparison: {base.get('variant')} -> {var.get('variant')}", "",
             f"baseline config: `{json.dumps(base.get('config'))}`  ",
             f"variant config: `{json.dumps(var.get('config'))}`", ""]
    missing = ([f"only in baseline: {p}/{i}" for (p, i) in bi if (p, i) not in vi]
               + [f"only in variant: {p}/{i}" for (p, i) in vi if (p, i) not in bi])
    if missing:
        lines += ["## Items not in both files", ""] + [f"- {m}" for m in missing] + [""]
    tally: Dict[str, Dict[str, Dict[str, int]]] = {}
    lines += ["## Per-item", "", "| panel | item | metric | before | after | delta | verdict |",
              "|---|---|---|---|---|---|---|"]
    for k in keys:
        b, a = bi[k], vi[k]
        if b["status"] != "ok" or a["status"] != "ok":
            lines.append(f"| {k[0]} | {k[1]} | status | {b['status']} | {a['status']} | - | "
                         f"{'worse' if a['status'] != 'ok' and b['status'] == 'ok' else 'better' if b['status'] != 'ok' and a['status'] == 'ok' else 'n/a'} |")
            continue
        for m in HEADLINE[k[0]]:
            bv, av = _f(b["metrics"].get(m)), _f(a["metrics"].get(m))
            v = _verdict_one(m, bv, av)
            d = None if bv is None or av is None else av - bv
            lines.append(f"| {k[0]} | {k[1]} | {m} | {_fmt(bv)} | {_fmt(av)} | "
                         f"{_fmt(d)} | {v} |")
            t = tally.setdefault(k[0], {}).setdefault(
                m, {"better": 0, "worse": 0, "same": 0, "n/a": 0, "changed": 0})
            t[v] += 1
    lines += ["", "## Aggregate (items better / worse / unchanged per headline metric)", "",
              "| panel | metric | better | worse | unchanged | n/a | base mean | var mean |",
              "|---|---|---|---|---|---|---|---|"]
    ab, av_ = aggregate(base["items"]), aggregate(var["items"])
    verdicts = []
    for panel in PANELS:
        if panel not in tally:
            continue
        nb = nw = 0
        for m, t in tally[panel].items():
            lines.append(
                f"| {panel} | {m} | {t['better']} | {t['worse']} | "
                f"{t['same'] + t['changed']} | {t['n/a']} | "
                f"{_fmt(ab[panel][m]['mean'])} | {_fmt(av_.get(panel, {}).get(m, {}).get('mean'))} |")
            nb += t["better"]; nw += t["worse"]
        verdict = "improved" if nb > nw else "regressed" if nw > nb else "no net change"
        verdicts.append(f"- **{panel}**: {verdict} ({nb} better vs {nw} worse metric-comparisons)")
    lines += ["", "## Verdict", ""] + (verdicts or ["- no comparable items"])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- CLI
def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return None


def cmd_run(args) -> int:
    kw = {"n_calls": args.n_calls, "draws": args.draws}
    cfg = json.loads(args.config_json) if args.config_json else {}
    kw.update(cfg)
    render = (json.loads(args.render_config_json) if args.render_config_json
              else {k: cfg[k] for k in RENDER_KEYS if k in cfg})
    if render:
        kw["_render"] = render
    panels = [p.strip() for p in args.panel.split(",") if p.strip()]
    only = [s.strip() for s in args.items.split(",")] if args.items else None
    items = build_items(panels, only)
    for it in items:
        it["fe_pixel_offset"] = args.fe_pixel_offset
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else REPO / "reports/fit-panel" / f"{args.variant}-{stamp}.json"
    t0 = time.time()
    rows = execute(items, args.db, kw, args.seed, args.workers, args.timeout_s)
    doc = {"variant": args.variant, "created": datetime.now().isoformat(timespec="seconds"),
           "git_head": _git_head(), "config": cfg, "render_config": render,
           "settings": {"n_calls": args.n_calls, "draws": args.draws, "seed": args.seed,
                        "panels": panels, "db": args.db, "workers": args.workers,
                        "timeout_s": args.timeout_s,
                        "fe_pixel_offset": args.fe_pixel_offset, "wall_s": round(time.time() - t0, 1)},
           "items": rows, "aggregate": aggregate(rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str))
    print(f"wrote {out}  ({sum(r['status'] != 'ok' for r in rows)} error rows)")
    return 0


def cmd_compare(args) -> int:
    md = compare_docs(json.loads(Path(args.baseline).read_text()),
                      json.loads(Path(args.variant).read_text()))
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fit-quality-panel", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True)
    r.add_argument("--out")
    r.add_argument("--panel", default="synthetic,real,fe")
    r.add_argument("--items")
    r.add_argument("--n-calls", type=int, default=8)
    r.add_argument("--draws", type=int, default=2)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--config-json")
    r.add_argument("--render-config-json", help="generator options for the synthetic "
                   "panel (partition_source, ipd); default: those keys of --config-json, "
                   "so the scene is rendered AND inverted with the same physics")
    r.add_argument("--db", default="db")
    r.add_argument("--fe-pixel-offset", type=int, default=FE_PIXEL_OFFSET,
                   help="samples to shift the Fe npz wavelength axis (the stored "
                        "axis lacks Z300 PIXEL_OFFSET -18); 0 = raw stored axis")
    r.add_argument("--timeout-s", type=int, default=900)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("baseline"); c.add_argument("variant"); c.add_argument("--md")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
