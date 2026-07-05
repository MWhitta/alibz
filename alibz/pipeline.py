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

import colorsys
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
#: apply the stimulated-emission factor to optical depths.  A/B-tested on
#: 38 real spectra (2026-07-04) and REFUSED as default: fit accuracy and
#: stage consistency were neutral (median dr2 = 0.0000) while alkali/Si
#: compositions swung >20% on 14/38 samples (physics-free optimizer
#: control: 3/38, median 0%) — the factor perturbs the near-degenerate
#: alkali-SA balance when only the doublet-anchored channel carries it.
#: The physics is correct in isolation; revisit together with the global
#: SA channel (see docs/development_guide.md).
DEFAULT_STIMULATED_EMISSION = False
#: detection classification thresholds (z = fraction / 1-sigma unc).
DETECT_Z = 3.0
MARGINAL_Z = 2.0
#: a peak "supports" an element when that element's species is the peak's
#: dominant assignment and contributes at least this share of the
#: observed amplitude.
MIN_SUPPORT_FRACTION = 0.3
#: long-format per-(sample, element) detection report filename.
DETECTIONS_NAME = "detections.csv"

ELEMENTS_BY_ATOMIC_NUMBER = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
)
ATOMIC_NUMBER = {el: i + 1 for i, el in enumerate(ELEMENTS_BY_ATOMIC_NUMBER)}

PERIODIC_BLOCK_MEMBERS = {
    "reactive nonmetal": ("H", "C", "N", "O", "P", "S", "Se"),
    "group 1": ("Li", "Na", "K", "Rb", "Cs", "Fr"),
    "group 2": ("Be", "Mg", "Ca", "Sr", "Ba", "Ra"),
    "3d-block": ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                 "Cu", "Zn"),
    "4d-block": ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd"),
    "5d-block": ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
                 "Hg"),
    "4f-block": ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                 "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"),
    "5f-block": ("Ac", "Th", "Pa", "U"),
    "post-transition metal": ("Al", "Ga", "In", "Sn", "Tl", "Pb",
                              "Bi"),
    "metalloid": ("B", "Si", "Ge", "As", "Sb", "Te", "Po"),
    "halogen": ("F", "Cl", "Br", "I", "At"),
    "noble gas": ("He", "Ne", "Ar", "Kr", "Xe", "Rn"),
}
ELEMENT_PERIODIC_BLOCK = {
    el: block
    for block, elements in PERIODIC_BLOCK_MEMBERS.items()
    for el in elements
}
PERIODIC_BLOCK_COLORS = {
    "reactive nonmetal": "#eef8ff",
    "group 1": "#dff1fc",
    "group 2": "#cfe9f8",
    "3d-block": "#bde0f1",
    "4d-block": "#aad6e9",
    "5d-block": "#96cbe0",
    "4f-block": "#82bfd5",
    "5f-block": "#6eb2c9",
    "post-transition metal": "#5aa5bd",
    "metalloid": "#4697b0",
    "halogen": "#32889f",
    "noble gas": "#1e7790",
    "other": "#0b5f78",
}


def _hex_to_rgb01(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(v * 255.0)))) for v in rgb]
    return "#{:02x}{:02x}{:02x}".format(*vals)


def _build_element_colors() -> Dict[str, str]:
    """Unique same-hue shades, grouped by periodic block/family."""
    ordered_elements = [
        el
        for _block, members in PERIODIC_BLOCK_MEMBERS.items()
        for el in sorted(members, key=lambda el: (ATOMIC_NUMBER[el], el))
    ]
    colors = {}
    n = len(ordered_elements)
    hue = 202.0 / 360.0
    saturation = 0.72
    light_min, light_max = 0.30, 0.92
    for i, el in enumerate(ordered_elements):
        frac = i / (n - 1)
        lightness = light_max - (light_max - light_min) * frac
        colors[el] = _rgb01_to_hex(colorsys.hls_to_rgb(
            hue, lightness, saturation
        ))
    return colors

ELEMENT_COLORS = _build_element_colors()

_SAMPLE_SUFFIX_RE = re.compile(
    r"_\d{8}_\d{6}_(?:AM|PM)_AverageSpectrum$", re.IGNORECASE
)


def element_sort_key(element: str) -> Tuple[int, str]:
    """Sort key for periodic-table order, with unknown labels last."""
    return ATOMIC_NUMBER.get(element, 10_000), element


def element_periodic_block(element: str) -> str:
    """Periodic-table block/family used for inspection-notebook coloring."""
    return ELEMENT_PERIODIC_BLOCK.get(element, "other")


def element_block_color(element: str) -> str:
    """Unique same-hue shade assigned to an element's periodic block."""
    return PERIODIC_BLOCK_COLORS[element_periodic_block(element)]


def element_color(element: str) -> str:
    """Unique same-hue shade assigned to an individual element."""
    return ELEMENT_COLORS.get(element, PERIODIC_BLOCK_COLORS["other"])


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


def element_uncertainty_stats(
    indexer,
    result,
    area_sigma: np.ndarray,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Per-element fraction statistics by amplitude resampling.

    Perturbs the observed peak amplitudes with the fitted per-peak area
    uncertainties, re-runs ONLY the linear concentration solve and element
    aggregation at the best-fit (T, n_e), and returns per-element
    ``{"mean": .., "std": ..}`` over the draws.  The nonlinear plasma
    parameters are held fixed, so this measures how the composition
    responds to measurement noise at the accepted plasma state.

    Statistics cover EVERY element that appears in ANY draw — including
    elements the best fit zeroed, whose draw distribution is the basis of
    the near-detection-limit upper bound reported downstream.  An element
    missing from a given draw contributed exactly 0 there (that draw's
    NNLS zeroed it), and those zeros count in the statistics.
    """
    draws = max(2, int(draws))   # a single draw has no spread: the whole
    #                                downstream z/upper-limit chain needs a std
    rng = np.random.default_rng(seed)
    amp0 = indexer._obs_amp.copy()
    sig = np.asarray(area_sigma, dtype=float).copy()
    # degenerate/pinned covariances report nan; fall back to 10% of the
    # amplitude so those peaks still carry a nonzero, conservative error
    bad = ~np.isfinite(sig)
    sig[bad] = 0.1 * np.abs(amp0[bad])
    per_draw: List[Dict[str, float]] = []
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
            per_draw.append({el: float(f) for el, f in fracs.items()})
    finally:
        indexer._obs_amp = amp0
    union = set(result.element_fractions)
    for d in per_draw:
        union |= set(d)
    out: Dict[str, Dict[str, float]] = {}
    for el in union:
        vals = np.array([d.get(el, 0.0) for d in per_draw], dtype=float)
        if vals.size > 1:
            out[el] = {"mean": float(np.mean(vals)),
                       "std": float(np.std(vals)),
                       "p95": float(np.percentile(vals, 95))}
        else:
            out[el] = {"mean": float(vals[0]) if vals.size else 0.0,
                       "std": float("nan"), "p95": float("nan")}
    return out


def element_uncertainties(
    indexer,
    result,
    area_sigma: np.ndarray,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> Dict[str, float]:
    """1-sigma fraction uncertainty per best-fit element.

    Thin wrapper over :func:`element_uncertainty_stats` (see there for
    mechanics and semantics).
    """
    stats = element_uncertainty_stats(indexer, result, area_sigma,
                                      draws=draws, seed=seed)
    return {el: (stats[el]["std"] if el in stats else float("nan"))
            for el in result.element_fractions}


def classify_detections(
    result,
    stats: Dict[str, Dict[str, float]],
    support: Dict[str, List[Tuple[float, float, float]]],
) -> List[dict]:
    """Per-element detection records for the long-format report.

    Near the limit of detection an abundance number alone is not a claim;
    each element is therefore reported WITH its evidence so borderline
    cases (single strong lines, marginal statistics) are visible instead
    of silently included or dropped:

    - ``detected``     z >= 3 and >= 2 supporting lines;
    - ``single-line``  z >= 3 but only one supporting line — statistically
      strong yet spectroscopically thin (a lone coincidence is possible);
      confirm against the line-evidence zoom in the notebook;
    - ``blended-only`` z >= 3 with NO peak dominated by this element (all
      fitted flux sits under peaks assigned to other species) — maximum
      suspicion;
    - ``marginal``     2 <= z < 3;
    - ``weak``         z < 2 (consistent with zero at ~95%);
    - ``upper-limit``  the best fit zeroed the element but its candidate
      lines are in the design: ``upper_limit`` = mean + 2 std of the
      resampled fraction is how much could hide below the noise.

    ``z = fraction / (1-sigma statistical uncertainty)``; ``support`` maps
    element -> [(contribution, wavelength_nm, observed_amp), ...] for
    peaks whose dominant assignment is that element.

    Caveats: ``strongest_peak_nm`` is the FITTED observed-frame peak
    center of the strongest supporting peak (a self-absorbed or blended
    line sits displaced from its database wavelength — that is physics,
    not an error).  Near the detection limit the resampled draws are
    clipped at zero, so the spread is mildly truncated and z mildly
    optimistic; treat 2 < z < 4 as soft rather than sharp.
    """
    detections = []
    for el in sorted(set(stats) | set(result.element_fractions),
                     key=element_sort_key):
        frac = float(result.element_fractions.get(el, 0.0))
        st = stats.get(el, {})
        std = float(st.get("std", float("nan")))
        lines = sorted(support.get(el, []), reverse=True)
        strongest = lines[0] if lines else None
        upper = None
        if frac > 0:
            if np.isfinite(std) and std > 0:
                z = frac / std
            elif float(st.get("mean", 0.0)) >= 0.5 * frac:
                # zero spread AND the draws reproduce the best fit:
                # genuinely rigid
                z = float("inf")
            else:
                # zero/undefined spread because the draws ZEROED the
                # element the best fit reports — maximal fragility, the
                # opposite of confidence
                z = 0.0
            if z >= DETECT_Z and len(lines) >= 2:
                status = "detected"
            elif z >= DETECT_Z and len(lines) == 1:
                status = "single-line"
            elif z >= DETECT_Z:
                # statistically strong yet NO peak is dominated by this
                # element: all of its fitted flux hides under peaks
                # assigned to other species — treat with maximum suspicion
                status = "blended-only"
            elif z >= MARGINAL_Z:
                status = "marginal"
            else:
                status = "weak"
        else:
            if not st or not np.isfinite(std):
                continue  # not in the candidate design at all
            # empirical 95th percentile of the resampled fraction: the
            # draw distribution near the LOD is a spike at zero plus a
            # tail, where mean + 2*std is ill-calibrated
            p95 = float(st.get("p95", float("nan")))
            upper = p95 if np.isfinite(p95) and p95 > 0 else (
                float(st.get("mean", 0.0)) + 2.0 * std)
            if upper <= 0:
                continue  # never activated in any draw: no design support
            status, z = "upper-limit", 0.0
        d = result.stage_disagreement.get(el, float("nan"))
        detections.append(dict(
            element=el, status=status, fraction=frac,
            unc=(std if np.isfinite(std) else None),
            z=(round(min(z, 999.0), 1) if np.isfinite(z) else 999.0),
            n_lines=len(lines),
            strongest_peak_nm=(round(strongest[1], 3) if strongest else None),
            strongest_obs=(round(strongest[2], 1) if strongest else None),
            upper_limit=upper,
            stage_disagreement=(round(float(d), 2) if np.isfinite(d)
                                else None),
        ))
    return detections


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
    fpeaks = final["sorted_parameter_array"]

    # pass 2: final composition, warm-started at the pass-1 plasma state
    idx2 = PeakyIndexerV3(_db_frame(fpeaks), dbpath=dbpath,
                          temperature_init=res1.temperature,
                          ne_init=res1.ne)
    result = idx2.run(**run_kwargs)

    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    area_sigma = estimate_peak_uncertainties(x, y - bg, fpeaks)[:, 0]
    stats = element_uncertainty_stats(idx2, result, area_sigma, draws=draws)
    unc = {el: (stats[el]["std"] if el in stats else float("nan"))
           for el in result.element_fractions}

    # Per-element line support, aggregated at the ELEMENT level: a peak
    # supports an element when that element's summed contribution (across
    # its ion stages) dominates the peak and covers >= MIN_SUPPORT_FRACTION
    # of the observed amplitude.  Aggregating first fixes the stage-split
    # undercount (Ca I 0.30 + Ca II 0.28 losing the per-species argmax to
    # an Fe I 0.35).  Support entries within one resolution element of
    # each other are merged (strongest kept) so a phantom-split strong
    # line cannot count as two independent lines.
    support: Dict[str, List[Tuple[float, float, float]]] = {}
    A = getattr(idx2, "_last_A", None)
    if A is not None and result.concentrations.size:
        n_pk = len(idx2._obs_wl)
        contrib = np.asarray(A)[:n_pk] * result.concentrations
        el_names = sorted({sp.element for sp in result.species})
        cols = {el: [j for j, sp in enumerate(result.species)
                     if sp.element == el] for el in el_names}
        E = np.stack([contrib[:, cols[el]].sum(axis=1) for el in el_names],
                     axis=1)                       # (n_peaks, n_elements)
        obs = np.asarray(idx2._obs_amp[:n_pk], dtype=float)
        dom = np.argmax(E, axis=1)
        for i in range(n_pk):
            el = el_names[dom[i]]
            con = float(E[i, dom[i]])
            if obs[i] > 0 and con >= MIN_SUPPORT_FRACTION * obs[i]:
                wl_obs = float(idx2._obs_wl[i]) + shift
                support.setdefault(el, []).append((con, wl_obs,
                                                   float(obs[i])))
        for el, lines in support.items():
            lines.sort(reverse=True)
            merged: List[Tuple[float, float, float]] = []
            for ln in lines:
                if all(abs(ln[1] - m[1]) > 0.15 for m in merged):
                    merged.append(ln)
            support[el] = merged
    detections = classify_detections(result, stats, support)

    return dict(
        fit=fit, refined=refined, final=final, decisions=decisions,
        records=records, shift=shift, n_anchor=n_anchor, ne_init=ne_init,
        result=result, element_uncertainty=unc, established=established,
        detections=detections,
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
              "n_lines", "strongest_peak_nm", "strongest_obs",
              "upper_limit", "stage_disagreement"]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            if row.get("status", "") != "ok" and not row.get("detections"):
                # errored samples must be VISIBLE here, not silently absent
                w.writerow([row["sample"], "", row.get("status", ""),
                            "", "", "", "", "", "", "", ""])
                n += 1
                continue
            for d in row.get("detections", []):
                w.writerow([
                    row["sample"], d["element"], d["status"],
                    f"{d['fraction']:.4g}" if d["fraction"] else "",
                    f"{d['unc']:.3g}" if d.get("unc") is not None else "",
                    d.get("z", ""),
                    d.get("n_lines", ""),
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
    code_setup = f"""import csv, glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from alibz import plot_spectrum_overview
from alibz.pipeline import (analyze_spectrum, element_color,
                            element_periodic_block, element_sort_key,
                            load_spectrum_csv, sample_name)

DATA_DIR = {data_dir!r}
DB_PATH  = {dbpath!r}
FILES = sorted(f for f in glob.glob(os.path.join(DATA_DIR, {pattern!r}))
               if os.path.basename(f) not in ({summary_name!r},
                                              {DETECTIONS_NAME!r}))
print(f"{{len(FILES)}} spectra")
with open(os.path.join(DATA_DIR, {summary_name!r})) as fh:
    SUMMARY = list(csv.DictReader(fh))
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
- **marginal** (2 ≤ z < 3) / **weak** (z < 2, consistent with zero);
- **upper-limit** — the fit zeroed the element; the value is how much
  could hide below the noise (mean + 2σ of the resampled fraction).
"""
    code_borderline = """# detection report for this spectrum + line-evidence zooms
from alibz import plot_peak_zoom

print(f"{'el':>4} {'status':>12} {'fraction':>9} {'unc':>8} {'z':>6}"
      f" {'lines':>5}  {'strongest [nm]':>14} {'upper_lim':>9}")
for d in sorted(a['detections'], key=lambda d: -(d['fraction'] or 0)):
    print(f"{d['element']:>4} {d['status']:>12}"
          f" {d['fraction']:9.5f}"
          f" {d['unc'] if d['unc'] is not None else float('nan'):8.5f}"
          f" {d['z']:6.1f} {d['n_lines']:5d}"
          f"  {d['strongest_peak_nm'] or '':>14}"
          f" {d['upper_limit'] if d['upper_limit'] is not None else '':>9}")

borderline = [d for d in a['detections']
              if d['status'] in ('single-line', 'blended-only',
                                 'marginal', 'weak')
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
