"""End-to-end production pipeline: directory of spectra -> composition table.

Orchestrates the full analysis chain on every spectrum in a directory and
writes two artifacts INTO that directory:

1. ``summary.csv`` — one row per spectrum with plasma parameters and
   quantitative element abundances (atom fraction of detected emitters)
   plus a per-element uncertainty;
2. ``fit_inspection.ipynb`` — a ready-to-run notebook that reproduces the
   full analysis on any single spectrum in the directory with the standard
   inspection plots (`plot_spectrum_overview`, refinement decisions,
   seeded minor lines, composition chart).

Per-spectrum chain (the same sequence validated interactively on MW2-112):

   load CSV -> PeakyFinder.fit_spectrum        (blind fit)
            -> estimate_wavelength_shift        (global shift, db frame)
            -> refine_fit                       (blends vs self-absorption)
            -> PeakyIndexerV3.fit  [pass 1]     (whole-pattern, sa_doublets)
            -> seed_minor_lines                 (elements from pass 1)
            -> PeakyIndexerV3.fit  [pass 2]     (warm-started at pass-1 T, ne)
            -> uncertainty resampling           (see below)

Electron density is initialised per shot from the H-alpha Lorentzian width
(``halpha_log_ne``) when the line is present.

Uncertainty semantics
---------------------
``<El>_unc`` is the 1-sigma STATISTICAL uncertainty from propagating the
fitted peak-area uncertainties (``estimate_peak_uncertainties``, joint-GLS
blend-group errors) through the concentration solve: the observed peak
amplitudes are resampled ``draws`` times at the best-fit plasma state
(T, n_e fixed), the linear concentration solve and element aggregation are
re-run per draw, and the spread of the resulting element fractions is
reported.  It does NOT include systematic error from the plasma model
(LTE, single-T), the atomic database, or self-absorption corrections; the
per-element ``stage_disagreement`` diagnostic (reported in the notebook,
and flagged in the CSV ``flags`` column when > 0.5) is the first-order
indicator of those systematics.
"""

import csv
import glob
import json
import os
import re
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PATTERN = "*.csv"
DEFAULT_N_CALLS = 40
DEFAULT_DRAWS = 32
DEFAULT_TIMEOUT_S = 900
#: pass-1 element fraction above which an element is treated as
#: "established" and eligible to seed minor lines.
ESTABLISHED_MIN_FRACTION = 0.002
#: stage_disagreement above which an element is flagged in the CSV.
STAGE_FLAG_THRESHOLD = 0.5

_SAMPLE_SUFFIX_RE = re.compile(
    r"_\d{8}_\d{6}_(?:AM|PM)_AverageSpectrum$", re.IGNORECASE
)


def resolve_dbpath(dbpath: Optional[str] = None) -> str:
    """Resolve the atomic-database directory.

    Order: explicit argument, ``ALIBZ_DB`` environment variable, ``./db``,
    the repository-root ``db`` next to the installed package.  An explicit
    argument (or set env var) that does not exist raises immediately —
    silently falling back to a different database than the one requested
    would be worse than failing.
    """
    for explicit in (dbpath, os.environ.get("ALIBZ_DB")):
        if explicit:
            if os.path.isdir(explicit):
                return os.path.abspath(explicit)
            raise FileNotFoundError(
                f"atomic database directory not found: {explicit!r}")
    candidates = ["db", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "db")]
    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "atomic database directory not found; searched "
        + ", ".join(repr(c) for c in candidates)
        + ". Pass --db or set ALIBZ_DB."
    )


def sample_name(path: str) -> str:
    """Human sample name from a spectrometer export filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return _SAMPLE_SUFFIX_RE.sub("", stem)


def load_spectrum_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column ``wavelength,intensity`` CSV (header optional)."""
    wl, inten = [], []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                w, v = float(row[0]), float(row[1])
            except ValueError:
                continue  # header or junk line
            wl.append(w)
            inten.append(v)
    if not wl:
        raise ValueError(f"no numeric wavelength,intensity rows in {path}")
    x = np.asarray(wl, dtype=float)
    y = np.asarray(inten, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


# ---------------------------------------------------------------------------
# Single-spectrum analysis
# ---------------------------------------------------------------------------

_DB_CACHE: dict = {}


def _get_db(dbpath: str):
    """Per-process Database singleton (the pickle load is expensive)."""
    from alibz.utils.database import Database
    db = _DB_CACHE.get(dbpath)
    if db is None:
        db = Database(dbpath)
        _DB_CACHE[dbpath] = db
    return db


def _halpha_ne(peak_array: np.ndarray):
    """Per-shot (ne_init, ne_bounds) from the H-alpha Stark width.

    Returns ``(None, None)`` when no usable H-alpha line is present; the
    indexer then falls back to its defaults.
    """
    from alibz.utils.stark import halpha_ne_bounds
    bounds = halpha_ne_bounds(peak_array)
    if bounds is None:
        return None, None
    return 0.5 * (bounds[0] + bounds[1]), bounds


def element_uncertainties(
    indexer,
    result,
    area_sigma: np.ndarray,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> Dict[str, float]:
    """1-sigma element-fraction uncertainties by amplitude resampling.

    Perturbs the observed peak amplitudes with the fitted per-peak area
    uncertainties, re-runs ONLY the linear concentration solve and element
    aggregation at the best-fit (T, n_e), and returns the standard
    deviation of each element's fraction over the draws.  The nonlinear
    plasma parameters are held fixed, so this measures how the composition
    responds to measurement noise at the accepted plasma state.
    """
    rng = np.random.default_rng(seed)
    amp0 = indexer._obs_amp.copy()
    sig = np.asarray(area_sigma, dtype=float).copy()
    # degenerate/pinned covariances report nan; fall back to 10% of the
    # amplitude so those peaks still carry a nonzero, conservative error
    bad = ~np.isfinite(sig)
    sig[bad] = 0.1 * np.abs(amp0[bad])
    samples: Dict[str, List[float]] = {}
    elements = list(result.element_fractions)
    try:
        for _ in range(int(draws)):
            indexer._obs_amp = np.clip(
                amp0 + rng.standard_normal(amp0.size) * sig, 0.0, None
            )
            c, _cost = indexer._solve_concentrations(
                result.temperature, result.ne
            )
            _conc, fracs, _dis = indexer._aggregate_elements(
                c, indexer._last_A
            )
            for el in set(elements) | set(fracs):
                samples.setdefault(el, []).append(float(fracs.get(el, 0.0)))
    finally:
        indexer._obs_amp = amp0
    out = {}
    for el in elements:
        vals = samples.get(el, [])
        out[el] = float(np.std(vals)) if len(vals) > 1 else float("nan")
    return out


def analyze_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    dbpath: str,
    n_calls: int = DEFAULT_N_CALLS,
    draws: int = DEFAULT_DRAWS,
    seed_minor: bool = True,
    verbose: bool = False,
) -> dict:
    """Run the full chain on one spectrum.

    Returns a dict with the intermediate fits (``fit``, ``refined``,
    ``final``, ``decisions``, ``records``, ``shift``), the pass-2
    ``result`` (:class:`FitResult`), ``element_uncertainty``, and the
    ``established`` element list.  Raises on hard failures; the directory
    driver converts those into an error row.
    """
    from alibz import PeakyFinder, refine_fit, seed_minor_lines
    from alibz.inspection import estimate_peak_uncertainties
    from alibz.peaky_indexer_v3 import PeakyIndexerV3
    from alibz.utils.wavelength import estimate_wavelength_shift

    db = _get_db(dbpath)
    finder = PeakyFinder.__new__(PeakyFinder)  # fit_spectrum needs no data dir
    fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False,
                              n_sigma=0)
    peaks = fit["sorted_parameter_array"]
    if peaks.size == 0:
        raise ValueError("blind fit found no peaks")

    shift, n_anchor = estimate_wavelength_shift(peaks, db)
    refined, decisions = refine_fit(x, y, fit, db=db, shift_nm=shift)
    rpeaks = refined["sorted_parameter_array"]

    ne_init, ne_bounds = _halpha_ne(rpeaks)
    idx_kwargs = dict(dbpath=dbpath)
    run_kwargs = dict(sa_doublets=True, n_calls=n_calls, verbose=verbose)
    if ne_init is not None:
        idx_kwargs["ne_init"] = ne_init
        run_kwargs["ne_bounds"] = ne_bounds

    def _db_frame(peaks: np.ndarray) -> np.ndarray:
        # indexer matches peak centers against db positions within its
        # shift_tolerance; remove the measured instrument shift first
        out = peaks.copy()
        out[:, 1] -= shift
        return out

    # pass 1: establish elements
    idx1 = PeakyIndexerV3(_db_frame(rpeaks), **idx_kwargs)
    res1 = idx1.run(**run_kwargs)
    established = sorted(
        el for el, f in res1.element_fractions.items()
        if f >= ESTABLISHED_MIN_FRACTION
    )

    final, records = refined, []
    if seed_minor and established:
        final, records = seed_minor_lines(x, y, refined, db, established,
                                          shift_nm=shift)
    fpeaks = final["sorted_parameter_array"]

    # pass 2: final composition, warm-started at the pass-1 plasma state
    idx2 = PeakyIndexerV3(_db_frame(fpeaks), dbpath=dbpath,
                          temperature_init=res1.temperature,
                          ne_init=res1.ne)
    result = idx2.run(**run_kwargs)

    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    area_sigma = estimate_peak_uncertainties(x, y - bg, fpeaks)[:, 0]
    unc = element_uncertainties(idx2, result, area_sigma, draws=draws)

    return dict(
        fit=fit, refined=refined, final=final, decisions=decisions,
        records=records, shift=shift, n_anchor=n_anchor, ne_init=ne_init,
        result=result, element_uncertainty=unc, established=established,
    )


# ---------------------------------------------------------------------------
# Directory driver
# ---------------------------------------------------------------------------

def _summary_row(path: str, analysis: dict) -> dict:
    res = analysis["result"]
    info = res.convergence_info or {}
    flags = [
        f"{el}:stage_spread"
        for el, d in sorted(res.stage_disagreement.items())
        if np.isfinite(d) and d > STAGE_FLAG_THRESHOLD
        and res.element_fractions.get(el, 0.0) > 0
    ]
    return dict(
        file=os.path.basename(path),
        sample=sample_name(path),
        status="ok",
        n_peaks=int(analysis["final"]["sorted_parameter_array"].shape[0]),
        shift_pm=round(1000.0 * analysis["shift"], 1),
        T_K=round(float(res.temperature), 0),
        log_ne=round(float(res.ne), 2),
        r_squared=round(float(res.r_squared), 4),
        sa_converged=info.get("sa_converged"),
        flags=";".join(flags),
        fractions={el: float(f) for el, f in res.element_fractions.items()
                   if f > 0},
        uncertainties=analysis["element_uncertainty"],
    )


def _error_row(path: str, message: str) -> dict:
    return dict(
        file=os.path.basename(path), path=path, sample=sample_name(path),
        status=f"error: {message}"[:200], n_peaks=0, shift_pm="",
        T_K="", log_ne="", r_squared="", sa_converged="", flags="",
        fractions={}, uncertainties={},
    )


_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _worker_init():
    # Belt-and-suspenders: the effective single-threading comes from the
    # env vars the parent sets BEFORE spawning (children inherit them
    # ahead of their numpy import); this only covers a fork start method.
    for var in _BLAS_THREAD_VARS:
        os.environ.setdefault(var, "1")


def _analyze_file(args) -> dict:
    """Analyze one file; NEVER raises — every failure becomes an error row.

    The timeout alarm is confined to the analysis call and disarmed before
    any exception handling runs, so an alarm that fires during teardown
    cannot escape as a stray ``_Timeout``.
    """
    path, dbpath, n_calls, draws, timeout_s = args
    use_alarm = bool(timeout_s) and hasattr(signal, "SIGALRM")
    try:
        x, y = load_spectrum_csv(path)
        old = None
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(int(timeout_s))
        try:
            analysis = analyze_spectrum(x, y, dbpath, n_calls=n_calls,
                                        draws=draws)
        finally:
            if use_alarm:
                signal.alarm(0)                       # disarm FIRST
                signal.signal(signal.SIGALRM, old)
        return _summary_row(path, analysis)
    except _Timeout:
        return _error_row(path, f"timeout after {timeout_s}s")
    except BaseException as exc:  # noqa: BLE001 - isolate every failure
        if isinstance(exc, KeyboardInterrupt):
            raise
        traceback.print_exc(file=sys.stderr)
        return _error_row(path, f"{type(exc).__name__}: {exc}")


def analyze_directory(
    data_dir: str,
    pattern: str = DEFAULT_PATTERN,
    dbpath: Optional[str] = None,
    workers: int = 1,
    n_calls: int = DEFAULT_N_CALLS,
    draws: int = DEFAULT_DRAWS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    limit: Optional[int] = None,
    exclude: Sequence[str] = ("summary.csv",),
    progress=print,
) -> List[dict]:
    """Analyze every spectrum matching ``pattern`` in ``data_dir``.

    Returns summary rows (see :func:`_summary_row`) in filename order.
    Failures are captured as error rows, never raised.  ``exclude`` lists
    basenames to skip — by default the tool's own ``summary.csv``, so a
    re-run in the same directory does not try to analyze its previous
    output as a spectrum.
    """
    dbpath = resolve_dbpath(dbpath)
    files = sorted(f for f in glob.glob(os.path.join(data_dir, pattern))
                   if os.path.basename(f) not in set(exclude))
    if limit:
        files = files[:int(limit)]
    if not files:
        raise FileNotFoundError(
            f"no files matching {pattern!r} in {data_dir!r}")

    def _emit(msg):
        # progress must be visible even when stdout is redirected to a file
        # (block-buffered off a tty), so flush every line
        progress(msg)
        try:
            sys.stdout.flush()
        except (ValueError, OSError):
            pass

    jobs = {f: (f, dbpath, n_calls, draws, timeout_s) for f in files}
    rows: Dict[str, dict] = {}          # keyed by full path (basenames collide)
    t0 = time.time()
    n = len(files)

    def _record(i, path, row):
        rows[path] = row
        _emit(f"[{i}/{n}] {row['sample']}: {row['status']}"
              f"  ({time.time() - t0:.0f}s elapsed)")

    if workers <= 1:
        _worker_init()
        try:
            for i, path in enumerate(files, 1):
                _record(i, path, _analyze_file(jobs[path]))
        except KeyboardInterrupt:
            _emit("interrupted; writing partial results")
    else:
        # spawn children inherit the parent env, so set BLAS threads to 1
        # HERE (before the pool is created and before the child imports
        # numpy) to avoid workers x cores thread oversubscription.
        saved = {v: os.environ.get(v) for v in _BLAS_THREAD_VARS}
        for v in _BLAS_THREAD_VARS:
            os.environ[v] = "1"
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_worker_init) as pool:
                futures = {pool.submit(_analyze_file, job): path
                           for path, job in jobs.items()}
                try:
                    for i, fut in enumerate(as_completed(futures), 1):
                        path = futures[fut]
                        try:
                            row = fut.result()
                        except Exception as exc:  # worker died / pool broke
                            row = _error_row(path, f"worker failed: {exc}")
                        _record(i, path, row)
                except KeyboardInterrupt:
                    _emit("interrupted; cancelling and writing partial results")
                    for fut in futures:
                        fut.cancel()
        finally:
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

    # files with no row (interrupted before completion) become error rows
    return [rows.get(f, _error_row(f, "not analyzed (interrupted)"))
            for f in files]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_summary_csv(rows: Sequence[dict], path: str) -> List[str]:
    """Write the wide summary table; returns the element column order.

    Elements are ordered by their median abundance across samples;
    each contributes ``<El>`` (atom fraction of detected emitters) and
    ``<El>_unc`` (1-sigma statistical; see module docstring) columns.
    """
    all_el: Dict[str, List[float]] = {}
    for row in rows:
        for el, f in row["fractions"].items():
            all_el.setdefault(el, []).append(f)
    elements = sorted(all_el, key=lambda el: -float(np.median(all_el[el])))

    meta = ["file", "sample", "status", "n_peaks", "shift_pm", "T_K",
            "log_ne", "r_squared", "sa_converged", "flags"]
    header = meta + [c for el in elements for c in (el, f"{el}_unc")]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            rec = [row.get(k, "") for k in meta]
            for el in elements:
                f = row["fractions"].get(el)
                u = row["uncertainties"].get(el)
                rec.append(f"{f:.5f}" if f is not None else "")
                rec.append(f"{u:.5f}" if u is not None
                           and np.isfinite(u) else "")
            w.writerow(rec)
    return elements


def _nb_cell(cell_type: str, source: str) -> dict:
    import uuid
    cell = {
        "id": uuid.uuid4().hex[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_inspection_notebook(
    data_dir: str,
    dbpath: str,
    pattern: str = DEFAULT_PATTERN,
    summary_name: str = "summary.csv",
    n_calls: int = DEFAULT_N_CALLS,
) -> dict:
    """Notebook (nbformat-4.5 JSON dict) that inspects this directory.

    Reads ``summary.csv`` for the composition overview and re-runs the
    full pipeline live on one selectable spectrum for the standard fit
    inspection views.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M")
    md_title = f"""# LIBS fit inspection — {os.path.basename(os.path.abspath(data_dir))}

Generated by `alibz-analyze` on {stamp}.

- **Data directory:** `{data_dir}`
- **Summary table:** [`{summary_name}`]({summary_name}) — element abundances
  are atom fractions of detected emitters; `<El>_unc` is the 1-sigma
  *statistical* uncertainty from propagating fitted peak-area errors through
  the concentration solve at the best-fit plasma state. Systematics (LTE,
  database, self-absorption model) are not included; the
  `stage_disagreement` diagnostic below is their first-order indicator.
- Set `SPECTRUM_FILE` below to inspect any spectrum in the directory.
"""
    code_setup = f"""import csv, glob, os
import numpy as np
import matplotlib.pyplot as plt

from alibz import plot_spectrum_overview
from alibz.pipeline import analyze_spectrum, load_spectrum_csv, sample_name

DATA_DIR = {data_dir!r}
DB_PATH  = {dbpath!r}
FILES = sorted(f for f in glob.glob(os.path.join(DATA_DIR, {pattern!r}))
               if os.path.basename(f) != {summary_name!r})
print(f"{{len(FILES)}} spectra")
with open(os.path.join(DATA_DIR, {summary_name!r})) as fh:
    SUMMARY = list(csv.DictReader(fh))
ELEMENTS = [c for c in SUMMARY[0] if c + '_unc' in SUMMARY[0]]
print("elements:", ELEMENTS)"""

    code_overview = """# composition overview: per-element abundance across all samples
ok = [r for r in SUMMARY if r['status'] == 'ok']
fig, ax = plt.subplots(figsize=(12, 5))
xpos = np.arange(len(ok))
bottom = np.zeros(len(ok))
for el in ELEMENTS:
    vals = np.array([float(r[el]) if r[el] else 0.0 for r in ok])
    if vals.max() <= 0:
        continue
    ax.bar(xpos, vals, bottom=bottom, label=el)
    bottom += vals
ax.set_xticks(xpos)
ax.set_xticklabels([r['sample'][:18] for r in ok], rotation=75, fontsize=7)
ax.set_ylabel('atom fraction of detected emitters')
ax.legend(ncol=6, fontsize=8)
ax.set_title('Composition by sample')
fig.tight_layout()"""

    code_run = f"""SPECTRUM_FILE = FILES[0]   # <-- change to inspect another spectrum
print(sample_name(SPECTRUM_FILE))
x, y = load_spectrum_csv(SPECTRUM_FILE)
a = analyze_spectrum(x, y, DB_PATH, n_calls={n_calls}, draws=16)
res = a['result']
print(f"T = {{res.temperature:.0f}} K   log ne = {{res.ne:.2f}}   "
      f"r^2 = {{res.r_squared:.3f}}   peaks = {{a['final']['sorted_parameter_array'].shape[0]}}")"""

    code_composition = """# composition +- uncertainty for this spectrum
els = sorted(res.element_fractions, key=lambda e: -res.element_fractions[e])
els = [e for e in els if res.element_fractions[e] > 0]
fr  = [res.element_fractions[e] for e in els]
un  = [a['element_uncertainty'].get(e, float('nan')) for e in els]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(els)), fr, yerr=un, capsize=3)
ax.set_xticks(range(len(els))); ax.set_xticklabels(els)
ax.set_yscale('log'); ax.set_ylabel('atom fraction')
ax.set_title('Composition with 1-sigma statistical uncertainty')
fig.tight_layout()
print(f"{'el':>4} {'fraction':>10} {'unc':>10} {'stage_dis':>10}")
for e in els:
    d = res.stage_disagreement.get(e, float('nan'))
    print(f"{e:>4} {res.element_fractions[e]:10.5f} "
          f"{a['element_uncertainty'].get(e, float('nan')):10.5f} {d:10.2f}")"""

    code_fitplot = """# full-span fit overview of the final model (blind -> refined -> seeded)
fig, axs = plot_spectrum_overview(x, y, a['final'])"""

    code_decisions = """# refinement decisions: blends split, self-absorbed lines merged
for d in sorted(a['decisions'], key=lambda d: d['center']):
    if d['action'] == 'none':
        continue
    extra = ''
    if 'tau_a' in d:
        extra = (f"  tau={d['tau_a']:.2f} delta={1000*d['delta_nm']:+.0f} pm"
                 f"  emission={d['emission_area']:.3g}"
                 f" observed={d['observed_area']:.3g}")
    print(f"{d['center']:9.3f}  {d['kind']:8s} {d['verdict']:18s}"
          f" {d['action']:5s}{extra}")"""

    code_minor = """# prior-seeded minor lines accepted for this spectrum
added = [r for r in a['records'] if r['action'] == 'added']
print(f"established elements: {a['established']}")
print(f"{len(added)} minor lines added")
for r in sorted(added, key=lambda r: r['wavelength_db']):
    print(f"  {r['element']:2s} {'I' if r['stage']==1 else 'II':3s}"
          f" {r['wavelength_db']:9.3f}  fitted={r['area']:8.1f}"
          f"  snr={r['snr']:5.1f}")"""

    md_notes = """## Reading the results

- **`stage_disagreement`** ~ relative spread between the independent ion-stage
  estimates of an element (0 = consistent single-plasma LTE). Values > 0.5 are
  flagged in `summary.csv`; they indicate non-LTE, a wrong plasma state, or a
  phase-heterogeneous target where the element ionises differently per host
  mineral.
- **Self-absorbed resonance lines** (flags in the decision table above) carry
  their reconstructed unattenuated emission areas in the decision records; the
  peak table stores observed (attenuated) areas.
- For methodology see `docs/fit_pipeline.md` in the alibz repository.
"""
    cells = [
        _nb_cell("markdown", md_title),
        _nb_cell("code", code_setup),
        _nb_cell("markdown", "## Composition across all samples"),
        _nb_cell("code", code_overview),
        _nb_cell("markdown",
                 "## Single-spectrum inspection\n\nRe-runs the full pipeline "
                 "live (~1–3 min) on `SPECTRUM_FILE`."),
        _nb_cell("code", code_run),
        _nb_cell("code", code_composition),
        _nb_cell("code", code_fitplot),
        _nb_cell("markdown", "### Refinement decisions"),
        _nb_cell("code", code_decisions),
        _nb_cell("markdown", "### Seeded minor lines"),
        _nb_cell("code", code_minor),
        _nb_cell("markdown", md_notes),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_inspection_notebook(nb: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def execute_notebook(path: str, timeout_s: int = 1800) -> Tuple[bool, str]:
    """Execute a notebook in place.

    Returns ``(ok, message)``.  Never raises: a missing dependency
    (nbclient/nbformat/ipykernel), an absent kernel, or a cell that errors
    (e.g. the live-spectrum cell on a pathological file) leaves the
    already-written notebook on disk unexecuted and reports the reason,
    rather than crashing the whole CLI after ``summary.csv`` succeeded.
    Whatever cells did execute before a failure are persisted.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        return False, (f"notebook execution skipped ({exc}); "
                       "pip install alibz[notebook]")
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as exc:  # noqa: BLE001 - unreadable notebook
        return False, f"unreadable notebook: {type(exc).__name__}: {exc}"
    client = NotebookClient(nb, timeout=timeout_s, kernel_name="python3")
    try:
        client.execute()
        return True, "executed"
    except Exception as exc:  # noqa: BLE001 - execution is best-effort
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            nbformat.write(nb, path)  # persist any cells that did run
        except Exception:  # noqa: BLE001
            pass
