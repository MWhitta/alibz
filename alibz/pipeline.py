"""End-to-end production pipeline: directory of spectra -> composition table.

Orchestrates the full analysis chain on every spectrum in a directory and
writes two artifacts INTO that directory:

1. ``summary.csv`` — one row per spectrum with plasma parameters and
   quantitative element abundances (atom fraction of detected emitters)
   plus a per-element uncertainty;
2. ``detections.csv`` — long-format per-(sample, element) detection report
   with the classification status, z-score, line support, and upper limits
   (see :func:`classify_detections`) — the self-consistent record for
   borderline elements near the limit of detection;
3. ``fit_inspection.ipynb`` — a ready-to-run notebook that reproduces the
   full analysis on any single spectrum in the directory with the standard
   inspection plots (`plot_spectrum_overview`, refinement decisions,
   seeded minor lines, borderline-element line evidence, composition chart).

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
indicator of those systematics.  Measured scale of what is excluded: on
samples with an active alkali-SA degeneracy (typically the flagged ones),
perturbing the SA model moved K/Li/Na fractions by 30-60% while r-squared
stayed flat — far beyond ``<El>_unc`` — so treat flagged samples' alkali
values as model-limited, not noise-limited.
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

# Element metadata and the detection/confounder analysis live in dedicated
# modules; re-exported here so existing importers (and the generated
# notebook, which imports these from alibz.pipeline) keep working.
from alibz.elements import (  # noqa: F401
    ELEMENT_COLORS,
    ELEMENTS_BY_ATOMIC_NUMBER,
    ELEMENT_PERIODIC_BLOCK,
    PERIODIC_BLOCK_COLORS,
    element_block_color,
    element_color,
    element_periodic_block,
    element_sort_key,
)
from alibz.detections import (  # noqa: F401
    DEFAULT_DRAWS,
    DETECT_Z,
    MARGINAL_Z,
    MIN_SUPPORT_FRACTION,
    analyze_detections,
    classify_detections,
    confounder_catalog,
    contested_support,
    element_uncertainties,
    element_uncertainty_stats,
    merge_contests,
)

DEFAULT_PATTERN = "*.csv"
DEFAULT_N_CALLS = 40
DEFAULT_TIMEOUT_S = 900
#: pass-1 element fraction above which an element is treated as
#: "established" and eligible to seed minor lines.
ESTABLISHED_MIN_FRACTION = 0.002
#: stage_disagreement above which an element is flagged in the CSV.
STAGE_FLAG_THRESHOLD = 0.5
#: apply the stimulated-emission factor to optical depths.  A/B-tested on
#: 38 real spectra (2026-07-04) and REFUSED as default: fit accuracy and
#: stage consistency were neutral (median dr2 = 0.0000) while alkali/Si
#: compositions swung >20% on 14/38 samples (physics-free optimizer
#: control: 3/38, median 0%) — the factor perturbs the near-degenerate
#: alkali-SA balance when only the doublet-anchored channel carries it.
#: The physics is correct in isolation; revisit together with the global
#: SA channel (see docs/development_guide.md).
DEFAULT_STIMULATED_EMISSION = False
#: long-format per-(sample, element) detection report filename.
DETECTIONS_NAME = "detections.csv"

_SAMPLE_SUFFIX_RE = re.compile(
    r"_\d{8}_\d{6}_(?:AM|PM)_AverageSpectrum$", re.IGNORECASE
)


def resolve_dbpath(dbpath: Optional[str] = None) -> str:
    """Resolve the atomic-database directory.

    A non-default explicit path must exist or this raises immediately —
    silently falling back to a different database than the one requested
    would be worse than failing.  The literal values ``"db"``/``"./db"``
    (and ``None``) are treated as the DEFAULT request, resolved as:
    ``ALIBZ_DB`` env var (must exist if set), the working-directory
    ``db``, the source-checkout ``db``, then the installed
    ``share/alibz/db`` data directory.
    """
    from alibz.utils.database import Database

    return str(Database._resolve_dbpath(dbpath))


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


def analyze_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    dbpath: str,
    n_calls: int = DEFAULT_N_CALLS,
    draws: int = DEFAULT_DRAWS,
    seed_minor: bool = True,
    stimulated_emission: bool = DEFAULT_STIMULATED_EMISSION,
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
    from alibz.minor_lines import recover_residual_lines
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
    run_kwargs = dict(sa_doublets=True, n_calls=n_calls, verbose=verbose,
                      sa_stimulated_emission=bool(stimulated_emission))
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
        [el for el, f in res1.element_fractions.items()
         if f >= ESTABLISHED_MIN_FRACTION],
        key=element_sort_key,
    )

    final, records = refined, []
    if seed_minor and established:
        final, records = seed_minor_lines(x, y, refined, db, established,
                                          shift_nm=shift)
    # element-agnostic recovery: significant positive residual peaks are
    # real lines the seeder could not predict (e.g. Fe lines when the Fe
    # stage scale fails the Boltzmann trust gate) — fit them from the
    # data alone; the pass-2 indexer then identifies them.  The
    # asymmetric-merged self-absorbed lines are EXCLUDED: their symmetric
    # table proxy leaves a core-shaped residual by design (see
    # refine_fit), and fitting components there re-splits the merge.
    from alibz.utils.voigt import voigt_width as _vw
    sa_zones = []
    for dec in decisions:
        if (dec.get("action") == "merge"
                and str(dec.get("verdict", "")).startswith("asymmetric")
                and dec.get("params_asym") is not None):
            pA = dec["params_asym"]
            halfw = 1.5 * max(float(_vw(max(pA[2], 1e-6),
                                        max(pA[3], 1e-6))), 0.15)
            sa_zones.append((float(pA[1]), halfw))
    final, recovered = recover_residual_lines(x, y, final,
                                              exclude=tuple(sa_zones))
    fpeaks = final["sorted_parameter_array"]

    # pass 2: final composition, warm-started at the pass-1 plasma state
    idx2 = PeakyIndexerV3(_db_frame(fpeaks), dbpath=dbpath,
                          temperature_init=res1.temperature,
                          ne_init=res1.ne)
    result = idx2.run(**run_kwargs)

    # detection report + confounder (true-negative rival) analysis
    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    area_sigma = estimate_peak_uncertainties(x, y - bg, fpeaks)[:, 0]
    det = analyze_detections(idx2, result, area_sigma, shift=shift,
                             dbpath=dbpath, draws=draws)

    return dict(
        fit=fit, refined=refined, final=final, decisions=decisions,
        records=records, recovered=recovered, shift=shift,
        n_anchor=n_anchor, ne_init=ne_init,
        result=result, established=established,
        element_uncertainty=det["element_uncertainty"],
        detections=det["detections"], support=det["support"],
        contested=det["contested"],
    )


# ---------------------------------------------------------------------------
# Directory driver
# ---------------------------------------------------------------------------

def _summary_row(path: str, analysis: dict) -> dict:
    res = analysis["result"]
    info = res.convergence_info or {}
    flags = [
        f"{el}:stage_spread"
        for el, d in sorted(res.stage_disagreement.items(),
                            key=lambda item: element_sort_key(item[0]))
        if np.isfinite(d) and d > STAGE_FLAG_THRESHOLD
        and res.element_fractions.get(el, 0.0) > 0
    ]
    flags += [
        f"{d['element']}:confounded({d['confounder']})"
        for d in analysis.get("detections", [])
        if d["status"] == "confounded" and d.get("confounder")
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
        detections=analysis.get("detections", []),
    )


def _error_row(path: str, message: str) -> dict:
    return dict(
        file=os.path.basename(path), path=path, sample=sample_name(path),
        status=f"error: {message}"[:200], n_peaks=0, shift_pm="",
        T_K="", log_ne="", r_squared="", sa_converged="", flags="",
        fractions={}, uncertainties={}, detections=[],
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
    path, dbpath, n_calls, draws, timeout_s, stim = args
    use_alarm = bool(timeout_s) and hasattr(signal, "SIGALRM")
    try:
        x, y = load_spectrum_csv(path)
        old = None
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(int(timeout_s))
        try:
            analysis = analyze_spectrum(x, y, dbpath, n_calls=n_calls,
                                        draws=draws,
                                        stimulated_emission=stim)
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
    stimulated_emission: bool = DEFAULT_STIMULATED_EMISSION,
    exclude: Sequence[str] = ("summary.csv", DETECTIONS_NAME),
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

    jobs = {f: (f, dbpath, n_calls, draws, timeout_s,
                bool(stimulated_emission)) for f in files}
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

    Elements are ordered by atomic number;
    each contributes ``<El>`` (atom fraction of detected emitters) and
    ``<El>_unc`` (1-sigma statistical; see module docstring) columns.
    """
    all_el = set()
    for row in rows:
        all_el.update(row["fractions"])
    elements = sorted(all_el, key=element_sort_key)

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


def write_detections_csv(rows: Sequence[dict], path: str) -> int:
    """Write the long-format per-(sample, element) detection report.

    One row per element per spectrum, INCLUDING near-detection-limit
    evidence: the classification status (see :func:`classify_detections`),
    z-score, number of supporting lines, the strongest matched line, and
    upper limits for elements the fit zeroed.  This is the
    self-consistency companion to ``summary.csv`` — borderline elements
    (e.g. a Hg or Mo resting on one strong line) are reported with the
    evidence needed to judge them rather than silently included or
    dropped.  Returns the number of detection rows written.
    """
    header = ["sample", "element", "status", "fraction", "unc", "z",
              "n_lines", "clear_lines", "contested_share", "confounder",
              "strongest_peak_nm", "strongest_obs",
              "upper_limit", "stage_disagreement"]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            if row.get("status", "") != "ok" and not row.get("detections"):
                # errored samples must be VISIBLE here, not silently absent
                w.writerow([row["sample"], "", row.get("status", "")]
                           + [""] * (len(header) - 3))
                n += 1
                continue
            for d in row.get("detections", []):
                w.writerow([
                    row["sample"], d["element"], d["status"],
                    f"{d['fraction']:.4g}" if d["fraction"] else "",
                    f"{d['unc']:.3g}" if d.get("unc") is not None else "",
                    d.get("z", ""),
                    d.get("n_lines", ""),
                    (d["clear_lines"]
                     if d.get("clear_lines") is not None else ""),
                    (d["contested_share"]
                     if d.get("contested_share") is not None else ""),
                    d.get("confounder") or "",
                    d.get("strongest_peak_nm") or "",
                    d.get("strongest_obs") or "",
                    (f"{d['upper_limit']:.3g}"
                     if d.get("upper_limit") is not None else ""),
                    (d["stage_disagreement"]
                     if d.get("stage_disagreement") is not None else ""),
                ])
                n += 1
    return n


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
    stimulated_emission: bool = DEFAULT_STIMULATED_EMISSION,
) -> dict:
    """Notebook (nbformat-4.5 JSON dict) that inspects this directory.

    Reads ``summary.csv`` for the composition overview and re-runs the
    full pipeline live on one selectable spectrum for the standard fit
    inspection views.  ``stimulated_emission`` is baked into the live
    cell so the notebook reproduces the SAME configuration that produced
    ``summary.csv`` (a notebook silently disagreeing with the batch would
    be worse than either alone).
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
    code_setup = f"""import csv, glob, importlib, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from alibz import plot_spectrum_overview
import alibz.pipeline as alibz_pipeline

alibz_pipeline = importlib.reload(alibz_pipeline)
analyze_spectrum = alibz_pipeline.analyze_spectrum
element_color = alibz_pipeline.element_color
element_periodic_block = alibz_pipeline.element_periodic_block
element_sort_key = alibz_pipeline.element_sort_key
confounder_catalog = alibz_pipeline.confounder_catalog
load_spectrum_csv = alibz_pipeline.load_spectrum_csv
sample_name = alibz_pipeline.sample_name

DATA_DIR = {data_dir!r}
DB_PATH  = {dbpath!r}
FILES = sorted(f for f in glob.glob(os.path.join(DATA_DIR, {pattern!r}))
               if os.path.basename(f) not in ({summary_name!r},
                                              {DETECTIONS_NAME!r}))
print(f"{{len(FILES)}} spectra")
with open(os.path.join(DATA_DIR, {summary_name!r})) as fh:
    SUMMARY = list(csv.DictReader(fh))
try:
    with open(os.path.join(DATA_DIR, {DETECTIONS_NAME!r})) as fh:
        DETECTIONS = list(csv.DictReader(fh))
except FileNotFoundError:
    DETECTIONS = []
ELEMENTS = sorted([c for c in SUMMARY[0] if c + '_unc' in SUMMARY[0]],
                  key=element_sort_key)
print("elements:", ELEMENTS)
print("periodic blocks:", sorted({{element_periodic_block(e) for e in ELEMENTS}}))"""

    code_overview = """# composition overview: per-element abundance across all samples
ok = [r for r in SUMMARY if r['status'] == 'ok']
fig, ax = plt.subplots(figsize=(12, 5))
xpos = np.arange(len(ok))
bottom = np.zeros(len(ok))
for el in ELEMENTS:
    vals = np.array([float(r[el]) if r[el] else 0.0 for r in ok])
    if vals.max() <= 0:
        continue
    ax.bar(xpos, vals, bottom=bottom, color=element_color(el),
           edgecolor='white', linewidth=0.3, label=el)
    bottom += vals
ax.set_xticks(xpos)
ax.set_xticklabels([r['sample'][:18] for r in ok], rotation=75, fontsize=7)
ax.set_ylabel('atom fraction of detected emitters')
seen_elements = []
for el in ELEMENTS:
    if any(float(r[el]) if r[el] else 0.0 for r in ok):
        seen_elements.append(el)
handles = [Patch(facecolor=element_color(el), edgecolor='0.4',
                 label=f"{el} ({element_periodic_block(el)})")
           for el in seen_elements]
ax.legend(handles=handles, title='element (periodic block)', ncol=4, fontsize=7,
          title_fontsize=8)
ax.set_title('Composition by sample')
fig.tight_layout()"""

    code_run = f"""SPECTRUM_FILE = FILES[0]   # <-- change to inspect another spectrum
print(sample_name(SPECTRUM_FILE))
x, y = load_spectrum_csv(SPECTRUM_FILE)
a = analyze_spectrum(x, y, DB_PATH, n_calls={n_calls}, draws=16,
                     stimulated_emission={stimulated_emission!r})
res = a['result']
print(f"T = {{res.temperature:.0f}} K   log ne = {{res.ne:.2f}}   "
      f"r^2 = {{res.r_squared:.3f}}   peaks = {{a['final']['sorted_parameter_array'].shape[0]}}")"""

    code_composition = """# composition +- uncertainty for this spectrum
els = sorted([e for e in res.element_fractions
              if res.element_fractions[e] > 0], key=element_sort_key)
fr  = [res.element_fractions[e] for e in els]
un  = [a['element_uncertainty'].get(e, float('nan')) for e in els]
colors = [element_color(e) for e in els]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(els)), fr, yerr=un, capsize=3, color=colors,
       edgecolor='0.25', linewidth=0.6)
ax.set_xticks(range(len(els))); ax.set_xticklabels(els)
ax.set_yscale('log'); ax.set_ylabel('atom fraction')
ax.set_title('Composition with 1-sigma statistical uncertainty')
handles = [Patch(facecolor=element_color(e), edgecolor='0.4',
                 label=f"{e} ({element_periodic_block(e)})")
           for e in els]
ax.legend(handles=handles, title='element (periodic block)', ncol=4, fontsize=7,
          title_fontsize=8)
fig.tight_layout()
print(f"{'el':>4} {'periodic block':>20} {'fraction':>10} {'unc':>10} {'stage_dis':>10}")
for e in els:
    d = res.stage_disagreement.get(e, float('nan'))
    print(f"{e:>4} {element_periodic_block(e):>20} "
          f"{res.element_fractions[e]:10.5f} "
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

    md_recovered = """### Residual-recovered lines

Positive residual peaks that survived the blind fit, refinement, AND the
Boltzmann seeder are usually real lines the seeder could not predict —
most often a line-rich element (Fe) whose per-stage Boltzmann scale fails
the trust gate on real rock. The element-agnostic recovery pass fits each
significant positive residual maximum (> 4σ local noise) as a new
component from the data alone; the whole-pattern indexer then identifies
it, so recovered lines flow into the composition and `detections.csv`
through the normal channels. Anything still listed `rejected` below
remains visible in the overview residual — genuinely unexplained flux.
"""
    code_recovered = """# element-agnostic residual recovery for this spectrum
rec_added = [r for r in a['recovered'] if r['action'] == 'added']
rej = [r for r in a['recovered'] if r['action'] == 'rejected']
print(f"{len(rec_added)} residual lines recovered, {len(rej)} rejected")
for r in sorted(rec_added, key=lambda r: r['center']):
    print(f"  {r['center']:9.3f}  area={r['area']:8.1f}"
          f"  snr={r['snr']:6.1f}  dBIC={r['delta_bic']:7.1f}")
if rej:
    print('rejected (still unexplained):')
    for r in sorted(rej, key=lambda r: -r.get('resid0', 0)):
        print(f"  {r['center0']:9.3f}  resid={r['resid0']:7.0f}"
              f"  snr0={r['snr0']:5.1f}")"""

    md_borderline = """### Borderline elements: the evidence, not just the number

Near the limit of detection an abundance value alone is not a claim. The
table and zooms below show what each borderline call actually rests on:

- **detected** — z ≥ 3 with ≥ 2 supporting lines;
- **single-line** — statistically strong but resting on ONE line (a lone
  coincidence is possible; judge the zoom: is the line at the right
  wavelength, with the right width, and are its confirmatory siblings
  plausibly below noise?);
- **blended-only** — z ≥ 3 but no peak is dominated by this element
  (all fitted flux hides under other species' peaks) — maximum suspicion;
- **confounded** — every supporting peak could equally be the named
  `confounder` element's line, and the confounder's own predicted lines
  check out elsewhere in the spectrum: the abundance is an attribution
  choice, not a measurement (archetype: Mn "detected" at 50% purely from
  the Mg II 279.5/280.3 nm region — genuine Mn at that level would light
  its 403 nm triplet, which is absent);
- **marginal** (2 ≤ z < 3) / **weak** (z < 2, consistent with zero);
- **upper-limit** — the fit zeroed the element; the value is how much
  could hide below the noise (mean + 2σ of the resampled fraction).
"""
    code_borderline = """# detection report for this spectrum + line-evidence zooms
from alibz import plot_peak_zoom

print(f"{'el':>4} {'status':>12} {'fraction':>9} {'unc':>8} {'z':>6}"
      f" {'lines':>5} {'clear':>5} {'confounder':>10}"
      f"  {'strongest [nm]':>14} {'upper_lim':>9}")
for d in sorted(a['detections'], key=lambda d: -(d['fraction'] or 0)):
    print(f"{d['element']:>4} {d['status']:>12}"
          f" {d['fraction']:9.5f}"
          f" {d['unc'] if d['unc'] is not None else float('nan'):8.5f}"
          f" {d['z']:6.1f} {d['n_lines']:5d}"
          f" {d['clear_lines'] if d.get('clear_lines') is not None else '':>5}"
          f" {d.get('confounder') or '':>10}"
          f"  {d['strongest_peak_nm'] or '':>14}"
          f" {d['upper_limit'] if d['upper_limit'] is not None else '':>9}")

borderline = [d for d in a['detections']
              if d['status'] in ('single-line', 'blended-only',
                                 'confounded', 'marginal', 'weak')
              and d.get('strongest_peak_nm')]
for d in borderline[:6]:
    fig, axs = plot_peak_zoom(x, y, a['final'], d['strongest_peak_nm'],
                              span_nm=1.5)
    axs[0].set_title(f"{d['element']} ({d['status']}, z={d['z']}, "
                     f"{d['n_lines']} line(s)) — strongest matched line\\n"
                     + axs[0].get_title(), fontsize=9)"""

    md_notes = """## Reading the results

- **`detections.csv`** (written alongside `summary.csv`) is the long-format
  per-(sample, element) report with the detection status, z-score, line
  support, and upper limits for every sample — use it, not the bare
  abundance columns, when deciding whether a trace element is real.
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
    md_confounders = """## Confounders across the corpus

Some abundances rest on peaks a *rival* element could equally explain. An
element flagged **`confounded`** in `detections.csv` has EVERY supporting
peak coverable by the named rival at a concentration that element's own
true negatives allow, scanned over the corpus plasma range — its number
is an attribution choice, not a measurement. The catalog below is the
operative confounder set for THIS corpus and instrument; treat
`confounded` fractions (and any totals renormalised around them) as
upper bounds. See `docs/development_guide.md` for the method."""
    code_confounders = """# corpus confounder catalog + the confounded detections
if DETECTIONS:
    cat = confounder_catalog(DETECTIONS)   # accepts CSV-dict rows
    print("confounder pairs (element <- rival), by frequency:")
    for (el, rival), n in cat.most_common():
        print(f"  {el:>3s} <- {rival:<3s}  x{n}")
    conf = [r for r in DETECTIONS if r['status'] == 'confounded']
    print(f"\\n{len(conf)} confounded detections across {len(FILES)} spectra:")
    for r in sorted(conf, key=lambda r: -float(r['fraction'] or 0))[:20]:
        print(f"  {r['sample'][:26]:26s} {r['element']:>3s}={r['fraction']:>8s}"
              f"  <- {r['confounder']}  (contested {r['contested_share']})")
else:
    print("no detections.csv found next to summary.csv")"""

    cells = [
        _nb_cell("markdown", md_title),
        _nb_cell("code", code_setup),
        _nb_cell("markdown", "## Composition across all samples"),
        _nb_cell("code", code_overview),
        _nb_cell("markdown", md_confounders),
        _nb_cell("code", code_confounders),
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
        _nb_cell("markdown", md_recovered),
        _nb_cell("code", code_recovered),
        _nb_cell("markdown", md_borderline),
        _nb_cell("code", code_borderline),
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
