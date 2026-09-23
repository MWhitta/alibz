"""End-to-end production pipeline: directory of spectra -> composition table.

Orchestrates the full analysis chain on every spectrum in a directory and
writes two artifacts INTO that directory:

1. ``summary.csv`` — one row per spectrum with plasma parameters and
   quantitative element abundances (atom fraction of detected emitters)
   plus a per-element uncertainty;
2. ``detections.csv`` — long-format per-(sample, element) detection report
   with the classification status, z-score, line support, upper limits
   (see :func:`classify_detections`), and the true-negative confounder
   analysis: ``fraction_resolved`` credits a ``confounded`` element only
   its uncontested flux and reattributes the rest to the ``confounder``
   (see :func:`resolve_confounded`), so it is the defensible quantification
   where ``fraction`` (the raw NNLS vertex) is confounder-inflated;
3. ``fit_inspection.ipynb`` — a ready-to-run notebook that reproduces the
   full analysis on any single spectrum in the directory with the standard
   inspection plots (`plot_spectrum_overview`, refinement decisions,
   seeded minor lines, borderline-element line evidence, composition chart).

Per-spectrum chain (the same sequence validated interactively on MW2-112):

   load CSV -> PeakyFinder.fit_spectrum        (blind fit)
            -> estimate_wavelength_shift_segments (per-detector-segment
                                                 instrument shift, db frame)
            -> refine_fit [3a, data-only]       (blend splits/single-merges
                                                 from model evidence; the
                                                 asymmetric family DEFERRED)
            -> PeakyIndexerV3.fit  [pass 1]     (whole-pattern, sa_doublets;
                                                 provisional posterior)
            -> refine_fit [3b, physics]         (asymmetric merges
                                                 adjudicated with resonance
                                                 gates conditioned on the
                                                 retained candidate species)
            -> seed_minor_lines                 (elements from pass 1;
                                                 merge zones excluded)
            -> recover_residual_lines           (element-agnostic residuals)
            -> deblend_shoulders                (split shoulder-flagged
                                                 peaks: one-sided flank bump
                                                 = unresolved overlap)
            -> PeakyIndexerV3.fit  [pass 2]     (confirms elements present)
            -> iterative deepening              (for ions quantified from
               (seed + guarded recover +         intense peaks: seed their
                solve_at, 3->2 sigma)            weak lines AND recover faint
                                                 residuals near their own db
                                                 lines at progressively lower
                                                 bars; each round re-solves at
                                                 the FIXED pass-2 plasma state
                                                 (no basin drift), rejected if
                                                 it newly collapses -- see
                                                 COLLAPSE_TOP_FRACTION)
            -> alibz.profiles                   (per-segment per-peak shape
                                                 QC of each element support)
            -> recover_sa_areas                 (growth-curve emission areas
                                                 for sa-like peaks the
                                                 doublet channel does not
                                                 anchor, PLUS the refinement
                                                 merges' pre-measured
                                                 emission/observed ratios for
                                                 unanchored species; linear
                                                 re-solve at the fitted T, ne)
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
from dataclasses import dataclass, field, asdict, replace  # noqa: F401
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
    resolve_confounded,
)
from alibz import telemetry
from alibz.peaky_indexer_v3 import STAGE_CONSISTENCY_WEIGHT

DEFAULT_PATTERN = "*.csv"
DEFAULT_N_CALLS = 40
#: Bayesian-optimisation budget of the PROVISIONAL pass-1 indexer (pass 2
#: uses the full ``n_calls``); see analyze_spectrum.
PASS1_N_CALLS = 24
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
#: per-file telemetry profile (one JSON line per analyzed spectrum).
PROFILE_NAME = "profile.jsonl"
#: basin guard for the corroborated (pass-3) re-index: seeding dozens of
#: weak low-excitation lines can let the re-optimizer drift into a low-T
#: basin where ONE element's tiny Saha-Boltzmann response explains
#: everything (measured on JChristensen: Hg 1.000 from a single 194 nm
#: line, Fe 0.991 at T~5200 K — while r-squared even improved, because a
#: line-rich element fits anything).  The corroborated composition is
#: accepted only when it does NOT newly collapse onto a single element:
#: rejected when its top fraction reaches COLLAPSE_TOP_FRACTION and grew
#: by at least COLLAPSE_JUMP over the pass-2 top fraction.
COLLAPSE_TOP_FRACTION = 0.9
COLLAPSE_JUMP = 0.2
#: an ion is "confident" — quantified from its intense peaks, and so
#: trustworthy enough to license a lowered recovery bar on its weak lines
#: — when some ion stage carries at least this many clean reference lines
#: (measured on MW2-112 #1000: Fe 60/46, Si 16, Ti 17/20 vs marginal Li 3).
CONFIDENT_MIN_REFS = 4
#: iterative deepening: after the confident ions are quantified from their
#: intense peaks, their weak lines are seeded and recovered at these
#: progressively lower local-noise bars (the 4 sigma one-shot recovery has
#: already run pre-identification; 2.0 is the noise floor below which peaks
#: are not distinguishable from noise).  Each round re-solves the
#: composition at the FIXED pass-2 plasma state (no re-optimisation, so no
#: basin drift) and is rejected wholesale if it collapses.
DEEPEN_BARS = (3.0, 2.0)
#: gA floor for a confident ion's database line to mark "coverage" for the
#: guarded low-bar agnostic recovery (a bump near a strong line of a
#: present ion is very likely that ion's faint line, not noise).
SUPPORT_GA_FLOOR = 1.0e6
SUPPORT_TOL_NM = 0.06
KB_EV_K = 8.617333262e-5
FIT_R2_FAIL = 0.30
FIT_R2_WARN = 0.50
PLASMA_T_BOUNDS = (4000.0, 25000.0)
PLASMA_LOG_NE_BOUNDS = (14.0, 19.0)
#: Default log10(ne) prior (centre, sigma in dex) when no H-alpha Stark
#: width is available.  The plasma-state search is bounded to +/-
#: NE_PRIOR_BOUND_SIGMAS of it: the amplitude objective is exactly FLAT
#: in ne at fixed T (identical costs from 1e14 to 1e18 on every spectrum
#: scanned 2026-09-21), so the bound costs no data fidelity, while the
#: stage-consistency thermometer sees ne only through the Saha (T, ne)
#: degeneracy ridge -- unbounded, it slid to the cold end of that ridge at
#: ne ~ 1e15 (7 kK on the 10 kK synthetic scenes).  Typical LIBS plasmas
#: in the emission window are 1e16-1e18 cm^-3.  The bound is +/- 1 sigma
#: (log ne 16.5-17.5): two-element thermometry separates T from ne only
#: weakly along the ridge and the amplitude data leans cold, so the
#: search sits at the LOW edge of whatever box it is given; measured on
#: the synthetic scenes (grid search, 2026-09-21) the 1-sigma box gives
#: Ca 0.52 / Mg 0.48 at 8.9 kK (truth 0.6 / 0.4, 10 kK) where the
#: 2-sigma box gave Mg 0.53 / Ca 0.47 at 8.2 kK, and the feldspar scene
#: 9.2 kK / r2 0.85 against 9.7 kK / r2 0.67.  A result at the edge is
#: flagged ``electron-density-at-bound`` (the bounds actually searched).
NE_PRIOR_DEFAULT = (17.0, 0.5)
NE_PRIOR_BOUND_SIGMAS = 1.0
#: outer search mode for the indexer passes ("gp" or "grid"); the value is
#: an opaque string validated inside the engine (PeakyIndexerV3.fit), so
#: new engines/modes need no pipeline change.  "grid" (a 15 x 7 profile
#: scan over (T, log ne) seeding the GP) became the default with the
#: stage-consistency thermometer (2026-09-21): the objective now has a
#: narrow valley along the Saha (T, ne) ridge that the cold-start GP
#: missed on every spectrum tested (synthetic Ca/Mg: 13.4 kK / 16.2
#: against the 10 kK / 17.0 truth; the grid seeds land on the ridge), at
#: a lower total cost on all five real samples and ~6 s more per pass.
DEFAULT_SEARCH = "grid"
#: seed for the GP optimiser; exposed so the sensitivity harness can
#: measure (and regress) seed dependence of the composition.
DEFAULT_GP_SEED = 42
#: counts per photoelectron-equivalent for the weighted solve's
#: shot-noise term (sigma^2 += amp/gain).  1.0 treats export counts as
#: photon events — exact for the synthetic generator, a provisional
#: assumption for real SciAps exports until the corpus export-kernel
#: calibration pins the true scale.
POISSON_GAIN_COUNTS = 1.0


@dataclass(frozen=True)
class AnalysisConfig:
    """Everything that parameterizes one spectrum analysis.

    One instance travels from the CLI through :func:`analyze_directory`
    to each worker's :func:`_analyze_file` — replacing the former
    positional job tuple, whose index-based unpacking made every new
    knob a fragile, error-prone change.  ``asdict(cfg)`` is the exact
    config snapshot the run manifest records, so what ran and what is
    recorded cannot drift apart.
    """

    dbpath: str
    n_calls: int = DEFAULT_N_CALLS
    draws: int = DEFAULT_DRAWS
    timeout_s: int = DEFAULT_TIMEOUT_S
    stimulated_emission: bool = DEFAULT_STIMULATED_EMISSION
    search: str = DEFAULT_SEARCH
    gp_seed: int = DEFAULT_GP_SEED
    #: opt-in WNNLS (see PeakyIndexerV3 weighted_solve) — accurate at the
    #: true plasma state but its chi-squared still misranks T; off until
    #: that bias is resolved against suite S.
    weighted_solve: bool = False
    #: stage-consistency thermometer weight (PeakyIndexerV3._stage_tie_cost)
    stage_consistency_weight: float = STAGE_CONSISTENCY_WEIGHT
    physical_triage: str = "off"
    gas_wavelength_calibration: str = "apply"
    #: independent wavelength registration (:mod:`alibz.wavelength_registration`):
    #: "off" | "report" | "ambient" | "apply_element".  DEFAULT "ambient" applies
    #: only the composition-INDEPENDENT ambient (Ar/O/N/H) registration to the
    #: NIR segment and keeps the legacy anchor shift for UV/VIS; the golden-line
    #: and vote-mode element registrations are recorded as diagnostics but not
    #: applied.  "apply_element" additionally applies the element registration
    #: where a segment passes quality (element UV/VIS is not yet validated: the
    #: strongest pure-metal lines are optically thick / displaced).  "report"
    #: records everything and applies nothing.
    wavelength_registration: str = "ambient"
    #: sub-pixel line-centre method for the registration and the peak table:
    #: "gaussian" (instrumental-profile fit, :mod:`alibz.utils.peakfit`) or
    #: "parabolic" (legacy three-point interpolation).
    subpixel: str = "gaussian"
    segment_response_fallback_ratio: Optional[float] = None
    segment_response_fallback_source: Optional[str] = None
    segment_response_fallback_uncertainty: Optional[float] = None
    segment_shift_offsets_nm: Optional[Tuple[float, ...]] = None
    segment_shift_prior_nm: Optional[Tuple[float, ...]] = None

    @staticmethod
    def _tuple_or_none(seq: Optional[Sequence[float]]
                       ) -> Optional[Tuple[float, ...]]:
        return None if seq is None else tuple(float(v) for v in seq)



def _warm_start_temperature(temperature: float,
                            bounds: Tuple[float, float] = PLASMA_T_BOUNDS,
                            default: float = 10_000.0) -> float:
    """Temperature to warm-start the next indexer pass from.

    A pass whose fitted temperature sits ON a search bound has not
    measured T (measured: the cold pass-1 GP rails to the 4000 K floor in
    the wide-kernel degenerate basin on single-shot spectra).  Handing
    that value to the next pass as ``temperature_init`` used to prune the
    ion-stage species from its candidate table (see
    ``PREFILTER_TEMPERATURES_K``); the prefilters are now ladder-robust,
    but the pseudo-observation selection, Stark assignments and doublet
    anchoring still evaluate at the init state, so a railed T is replaced
    by the default here.
    """
    t = float(temperature)
    if not np.isfinite(t) or t <= bounds[0] + 1.0 or t >= bounds[1] - 1.0:
        return float(default)
    return t

def composition_collapsed(fr_before: dict, fr_after: dict) -> bool:
    """Basin-guard criterion for the corroborated (pass-3) re-index.

    True when the re-indexed composition NEWLY collapses onto a single
    element: its top fraction reaches ``COLLAPSE_TOP_FRACTION`` and either
    grew by ``COLLAPSE_JUMP`` over the before-state or belongs to a
    DIFFERENT element than before (measured failures: K 0.54 -> Hg 1.00,
    Si 0.35 -> Fe 0.99).  A composition that was already dominated by the
    same element stays accepted.
    """
    top_b = max(fr_before.values(), default=0.0)
    top_a = max(fr_after.values(), default=0.0)
    if top_a < COLLAPSE_TOP_FRACTION:
        return False
    el_b = max(fr_before, key=fr_before.get) if fr_before else None
    el_a = max(fr_after, key=fr_after.get) if fr_after else None
    return (top_a - top_b) >= COLLAPSE_JUMP or el_a != el_b

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


def load_spectrum_metadata(path: str) -> dict:
    """Wavelength-calibration provenance for a spectrum, if recorded.

    Looks for a sibling ``test.json`` (as written by pantheum
    ``_finish_dataset``) next to ``path`` or in its parent directory, and
    returns the fields the thermal drift model consumes -- ``wl_calibration_time``,
    ``collected_at``, ``warmup_minutes``, ``analyzer_temperature_c``,
    ``wl_calibration_coefficients`` -- or ``{}`` when none is found.  Never
    raises; a missing or malformed file yields an empty dict.
    """
    import json as _json
    keys = ("wl_calibration_time", "wl_calibration_time_raw", "collected_at",
            "warmup_minutes", "analyzer_temperature_c",
            "wl_calibration_coefficients")
    here = os.path.dirname(os.path.abspath(path))
    for candidate in (os.path.join(here, "test.json"),
                      os.path.join(os.path.dirname(here), "test.json")):
        try:
            with open(candidate) as fh:
                meta = _json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            return {k: meta[k] for k in keys if k in meta}
    return {}


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


def _get_sb(dbpath: str):
    """Per-process SahaBoltzmann singleton (level caches are expensive)."""
    from alibz.utils.sahaboltzmann import SahaBoltzmann
    key = (dbpath, "sb")
    sb = _DB_CACHE.get(key)
    if sb is None:
        sb = SahaBoltzmann(dbpath)
        _DB_CACHE[key] = sb
    return sb


def _coarse_composition(peaks: np.ndarray, db, *, max_elements=8,
                        min_gA=1.0e6, tol_nm=0.12, min_hits=2):
    """Rough element list from the brightest refined peaks.

    :func:`alibz.wavelength_registration.element_registration` needs a
    composition, but the pipeline only discovers one after indexing.  This is a
    cheap bootstrap: match the strongest peaks to bright (gA>=``min_gA``,
    ion<=2) database lines and keep the elements with at least ``min_hits``
    unambiguous hits.  Used only to SELECT registration lines; it never enters
    the composition result.
    """
    peaks = np.atleast_2d(np.asarray(peaks, dtype=float))
    if peaks.size == 0:
        return []
    wl_all, el_all = [], []
    for el in getattr(db, "supported_elements", db.elements):
        if el in getattr(db, "no_lines", ()) or el in getattr(db, "analysis_excluded_elements", ()):
            continue
        arr = np.asarray(db.lines(el))
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        keep = (ion <= 2) & (gA >= min_gA) & (wl > 180) & (wl < 1000)
        wl_all.append(wl[keep])
        el_all.append(np.full(int(np.sum(keep)), el))
    if not wl_all:
        return []
    wl_all = np.concatenate(wl_all)
    el_all = np.concatenate(el_all)
    order = np.argsort(wl_all)
    wl_all, el_all = wl_all[order], el_all[order]
    strongest = np.argsort(peaks[:, 0])[::-1][:60]
    tally: dict = {}
    for mu in peaks[strongest, 1]:
        j = np.searchsorted(wl_all, mu)
        lo, hi = max(j - 3, 0), min(j + 3, wl_all.size)
        if lo >= hi:
            continue
        d = np.abs(wl_all[lo:hi] - mu)
        k = int(np.argmin(d))
        if d[k] <= tol_nm:
            tally[el_all[lo + k]] = tally.get(el_all[lo + k], 0) + 1
    ranked = sorted((e for e, c in tally.items() if c >= min_hits),
                    key=lambda e: tally[e], reverse=True)
    return ranked[:max_elements]


def _ar_cross_check(gas_ar, ambient_nir, *, sigma_floor_nm=0.02, flag_n_sigma=3.0):
    """Consistency record between the two argon-based NIR estimators.

    ``gas_ar`` is :func:`alibz.gas_calibration.calibrate_background_gases`
    ``["Ar"]`` (regional, peak-table based) and ``ambient_nir`` is the NIR
    segment of :func:`alibz.wavelength_registration.ambient_registration`.
    They share the Ar I lines but not the peak finder, gates or aggregation,
    so agreement is a consistency check, never an independent confirmation.
    Returns None unless both produced an offset.  ``sigma_floor_nm`` is the
    single-line repeatability measured on the 2026-09-22 Fe run means (Ar I
    696.5 nm: 11 pm MAD over 25 runs), applied because a one-line segment
    reports sigma 0.
    """
    if not gas_ar or not ambient_nir:
        return None
    g = gas_ar.get("offset_nm")
    a = ambient_nir.get("shift_nm")
    if g is None or a is None:
        return None
    gs = float(gas_ar.get("uncertainty_nm") or 0.0)
    as_ = float(ambient_nir.get("sigma_nm") or 0.0)
    sigma = float(np.hypot(max(gs, sigma_floor_nm), max(as_, sigma_floor_nm)))
    diff = float(g) - float(a)
    return {
        "gas_ar_offset_nm": float(g), "gas_ar_uncertainty_nm": gs,
        "gas_ar_status": gas_ar.get("status"),
        "ambient_nir_shift_nm": float(a), "ambient_nir_sigma_nm": as_,
        "ambient_nir_n_lines": int(ambient_nir.get("n_lines") or 0),
        "difference_nm": diff, "combined_sigma_nm": sigma,
        "n_sigma": abs(diff) / sigma, "flagged": bool(abs(diff) > flag_n_sigma * sigma),
    }


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
    search: str = DEFAULT_SEARCH,
    gp_seed: int = DEFAULT_GP_SEED,
    weighted_solve: bool = False,
    stage_consistency_weight: float = STAGE_CONSISTENCY_WEIGHT,
    segment_response_fallback_ratio: Optional[float] = None,
    segment_response_fallback_source: Optional[str] = None,
    segment_response_fallback_uncertainty: Optional[float] = None,
    segment_shift_offsets_nm: Optional[Sequence[float]] = None,
    segment_shift_prior_nm: Optional[Sequence[float]] = None,
    verbose: bool = False,
    physical_triage: str = "off",
    gas_wavelength_calibration: str = "apply",
    wavelength_registration: str = "ambient",
    subpixel: str = "gaussian",
) -> dict:
    """Run the full chain on one spectrum.

    Blind fit -> data-only refinement (3a: blend/single actions; the
    asymmetric family deferred) -> pass-1 indexer (rough id) -> physics
    adjudication (3b: asymmetric merges with resonance gates conditioned
    on the retained candidate species) -> seeding + element-agnostic
    residual recovery -> pass-2 indexer (confirms the elements present) ->
    ITERATIVE DEEPENING: for the ions quantified from their intense peaks
    (>= ``CONFIDENT_MIN_REFS`` clean reference lines), seed their weak
    lines and recover faint residuals near their own database lines at
    progressively lower local-noise bars (``DEEPEN_BARS``, 3 -> 2 sigma),
    re-solving the composition at the FIXED pass-2 plasma state each round
    (:meth:`PeakyIndexerV3.solve_at` -- no re-optimisation, so no basin
    drift), stopping if a round newly collapses onto one element (basin
    guard; see ``COLLAPSE_TOP_FRACTION``) -> detection report + per-peak
    shape QC (:mod:`alibz.profiles`).

    Returns a dict with the intermediate fits (``fit``, ``refined``,
    ``final``, ``decisions``, ``records``, ``shift``), the final
    ``result`` (:class:`FitResult`), ``element_uncertainty``, the
    ``established`` element list (the pass-2 confirmed elements when
    deepening ran; the pass-1 list otherwise),
    ``profiles``/``shape_quality`` (per-peak shape physics
    and per-element support QC), and ``corroboration`` (the deepening
    summary: confident ions, per-round seeded/recovered counts, and why it
    stopped).  Raises on hard failures; the directory driver converts
    those into an error row.
    """
    from alibz import PeakyFinder, refine_fit, seed_minor_lines
    from alibz.inspection import estimate_peak_uncertainties
    from alibz.minor_lines import recover_residual_lines
    from alibz.peaky_indexer_v3 import PeakyIndexerV3
    from alibz.utils.wavelength import (estimate_wavelength_shift,
                                        estimate_wavelength_shift_segments,
                                        shift_at)

    if physical_triage not in {"off", "report", "prune"}:
        raise ValueError("physical_triage must be 'off', 'report', or 'prune'")
    if gas_wavelength_calibration not in {"off", "report", "apply"}:
        raise ValueError("gas_wavelength_calibration must be off, report, or apply")
    if wavelength_registration not in {"off", "report", "ambient", "apply_element", "apply"}:
        raise ValueError("wavelength_registration must be off, report, ambient, "
                         "or apply_element")
    if subpixel not in {"gaussian", "parabolic"}:
        raise ValueError("subpixel must be 'gaussian' or 'parabolic'")
    telemetry.reset()
    db = _get_db(dbpath)
    finder = PeakyFinder.__new__(PeakyFinder)  # fit_spectrum needs no data dir
    with telemetry.stage("blind_fit"):
        fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False,
                                  n_sigma=0)
    peaks = fit["sorted_parameter_array"]
    if peaks.size == 0:
        raise ValueError("blind fit found no peaks")

    # Independent references: raw instrument-axis samples and original observed
    # peaks, before any mixed-element shift or Fe-informed refinement. Gas
    # offsets are absolute residuals, never additions to another calibration.
    from alibz.wavelength_calibration import apply_gas_calibrations
    gas_calibrations = {}
    if gas_wavelength_calibration != "off":
        from alibz.gas_calibration import calibrate_background_gases
        with telemetry.stage("gas_wavelength_calibration"):
            gas_calibrations = calibrate_background_gases(
                x, y, db, peak_array=peaks)

    # pooled global shift from the blind table (robust median) — enough
    # for stage 3a's coarse db evidence windows
    shift0, n_anchor = estimate_wavelength_shift(peaks, db)

    # stage 3a — DATA-ONLY refinement: blend splits and single-merges are
    # pure model evidence; the asymmetric (self-absorption) family needs
    # resonance physics of elements actually PRESENT, so those verdicts
    # are recorded but deferred until pass 1 provides an element posterior
    with telemetry.stage("refine_data"):
        refined, dec_data = refine_fit(x, y, fit, db=db, shift_nm=shift0,
                                       asymmetric="defer")
    rpeaks = refined["sorted_parameter_array"]

    # Per-peak Gaussian instrumental-profile diagnostics for the refined table
    # (fitted sigma in px, chi^2/nu, blend flag).  This is ADDITIVE: it records
    # the peakfit quality of every accepted peak without changing the peak
    # centres the physics inversion consumes (those come from the blind Voigt
    # fit).  Re-centring the production table by default is a larger,
    # unvalidated change to the inversion and is intentionally NOT done here;
    # the Gaussian centre is used where it is validated -- the wavelength
    # registration -- and this table lets a caller (or the indexer) see the
    # per-peak fit quality and blends.  See the report's "Limits" section.
    peak_refinement = None
    if subpixel == "gaussian" and rpeaks.size:
        from alibz.utils.peakfit import refine_peaks
        idxs = [int(np.argmin(np.abs(x - c))) for c in np.atleast_2d(rpeaks)[:, 1]]
        recs = refine_peaks(x, y, idxs)
        peak_refinement = {
            "method": "gaussian", "version": "1.0",
            "sigma_px": [r.get("sigma_px") if r.get("ok") else None for r in recs],
            "chi2_nu": [r.get("chi2_nu") if r.get("ok") else None for r in recs],
            "blend": [bool(r.get("blend")) if r.get("ok") else None for r in recs],
            "n_blends": int(sum(1 for r in recs if r.get("ok") and r.get("blend"))),
            "n_ok": int(sum(1 for r in recs if r.get("ok"))),
        }

    # per-detector-segment shifts, from the REFINED table: the three
    # segments drift independently (measured ~25-35 pm apart on MW2-112),
    # but blind centers of split/merged bright lines are displaced by up
    # to ~150 pm, so only the refined table's medians are clean enough
    # for the estimator's significance gate to separate genuine drift
    # from fit noise (segments failing the gate keep the global shift)
    with telemetry.stage("segment_shift"):
        shift, n_shift_anchor = estimate_wavelength_shift_segments(rpeaks, db)
    shift_prior_applied = (False,) * len(shift.shifts)
    if (segment_shift_offsets_nm is not None
            and segment_shift_prior_nm is not None):
        raise ValueError("provide either shift offsets or an absolute shift "
                         "prior, not both")
    if segment_shift_offsets_nm is not None:
        offsets = np.asarray(segment_shift_offsets_nm, dtype=float)
        if offsets.shape != shift.shifts.shape:
            raise ValueError("segment_shift_offsets_nm must have one value "
                             "for each detector segment")
        use_prior = np.logical_not(np.asarray(shift.applied, dtype=bool))
        # The shared calibration is expressed as a segment offset relative to
        # this shot's pooled shift, so per-shot thermal drift is retained while
        # weak segments borrow the stable detector geometry.
        prior_shift = shift.global_shift + offsets
        shifts = np.where(use_prior, prior_shift, shift.shifts)
        from alibz.utils.wavelength import SegmentShift
        shift = SegmentShift(shift.edges, shifts, shift.global_shift,
                             shift.n_matches, shift.applied)
        shift_prior_applied = tuple(bool(v) for v in use_prior)
    elif segment_shift_prior_nm is not None:
        prior = np.asarray(segment_shift_prior_nm, dtype=float)
        if prior.shape != shift.shifts.shape:
            raise ValueError("segment_shift_prior_nm must have one value "
                             "for each detector segment")
        use_prior = (np.logical_not(np.asarray(shift.applied, dtype=bool))
                     & np.isfinite(prior))
        shifts = np.where(use_prior, prior, shift.shifts)
        from alibz.utils.wavelength import SegmentShift
        shift = SegmentShift(shift.edges, shifts, shift.global_shift,
                             shift.n_matches, shift.applied)
        shift_prior_applied = tuple(bool(v) for v in use_prior)

    # Independent wavelength registration from ambient (Ar/O) lines and the
    # sample's own strong isolated lines (see alibz.wavelength_registration).
    # Composition is bootstrapped from the refined bright peaks.  When enabled
    # and a segment's registration quality passes, its per-segment correction
    # REPLACES the legacy anchor shift for that segment; otherwise the legacy
    # shift is kept (logged in analysis['wavelength_registration']['applied']).
    wl_registration = None
    if wavelength_registration != "off":
        from alibz import wavelength_registration as _wr
        from alibz.utils.wavelength import SegmentShift
        reg_cfg = {"subpixel": subpixel, "attach_vote_diagnostic": True}
        with telemetry.stage("wavelength_registration"):
            comp = _coarse_composition(rpeaks, db)
            # the bootstrapped composition only REMOVES ambient anchors that a
            # sample line could be mistaken for; it never supplies anchors
            reg_amb = _wr.ambient_registration(x, y, db, composition=comp,
                                               config=reg_cfg)
            reg_ele = (_wr.element_registration(x, y, db, comp, config=reg_cfg)
                       if comp else dict(reg_amb))
            reg_comb = _wr.combined_registration(reg_amb, reg_ele)
        # DEFAULT ("ambient"): apply the composition-INDEPENDENT ambient (Ar I /
        # O I / N I / Halpha) registration to the NIR segment only, and keep the
        # legacy anchor shift for UV/VIS.  The golden-line and vote-mode element
        # registrations are RECORDED as diagnostics but NOT applied unless
        # "apply_element" is requested: the strongest lines of a pure-metal
        # plasma are optically thick / displaced (see docs), so element UV/VIS
        # registration is not yet validated.
        applied_segments = []
        new_shifts = list(np.asarray(shift.shifts, dtype=float))
        for si, name in enumerate(_wr.SEGMENT_NAMES[:len(new_shifts)]):
            lo = 200.0 if si == 0 else shift.edges[si - 1]
            hi = 1000.0 if si >= len(shift.edges) else shift.edges[si]
            mid = 0.5 * (lo + hi)
            if wavelength_registration == "ambient" and name == "NIR":
                a_seg = reg_amb["segments"].get(name, {})
                if (a_seg.get("model") and a_seg.get("n_lines", 0) >= 3
                        and a_seg.get("quality") in ("ok", "weak")):
                    new_shifts[si] = float(_wr._eval_model(a_seg["model"], mid))
                    applied_segments.append(name)
            elif wavelength_registration in ("apply_element", "apply"):
                seg = reg_comb["segments"].get(name, {})
                if seg.get("source") in ("element", "ambient") and seg.get("quality") == "ok":
                    new_shifts[si] = float(
                        _wr.registration_shift(reg_comb).at(mid))
                    applied_segments.append(name)
        if applied_segments:
            shift = SegmentShift(shift.edges, new_shifts, shift.global_shift,
                                 shift.n_matches, shift.applied)
        wl_registration = {
            "mode": wavelength_registration, "subpixel": subpixel,
            "applied": bool(applied_segments),
            "applied_segments": applied_segments,
            "composition_bootstrap": comp,
            "ambient": _wr.registration_to_json(reg_amb),
            "element": _wr.registration_to_json(reg_ele),
            "combined": _wr.registration_to_json(reg_comb),
            "diagnostic": _wr.registration_to_json(reg_ele.get("diagnostic")),
            "method_version": _wr.REGISTRATION_VERSION,
            # consistency with the independent Ar I engine (gas_calibration)
            "gas_cross_check": _ar_cross_check(
                gas_calibrations.get("Ar") if gas_calibrations else None,
                reg_amb["segments"].get("NIR")),
        }
        if verbose:
            print(f"wavelength_registration[{wavelength_registration}]: "
                  f"applied={applied_segments} composition={comp}")

    baseline_shift = shift
    shift, wavelength_calibration = apply_gas_calibrations(
        baseline_shift, gas_calibrations, mode=gas_wavelength_calibration)

    ne_init, ne_bounds = _halpha_ne(rpeaks)
    # physical ne prior: H-alpha Stark anchor when present, else the
    # instrument-level default.  The data cost is flat in ne while the
    # composition is not (Saha-carrying aggregation), so in weighted mode
    # this prior — not an arbitrary optimiser pick — must set ne.
    ne_prior = ((float(ne_init), 0.15) if ne_init is not None
                else NE_PRIOR_DEFAULT)
    if ne_bounds is None:
        # no H-alpha: bound the search by the default prior (see
        # NE_PRIOR_DEFAULT) so the thermometer resolves T along the Saha
        # ridge at a physical electron density
        ne_bounds = (max(PLASMA_LOG_NE_BOUNDS[0],
                         ne_prior[0] - NE_PRIOR_BOUND_SIGMAS * ne_prior[1]),
                     min(PLASMA_LOG_NE_BOUNDS[1],
                         ne_prior[0] + NE_PRIOR_BOUND_SIGMAS * ne_prior[1]))
        ne_bounds_source = "prior"
    else:
        ne_bounds_source = "halpha"
    sb = _get_sb(dbpath)
    idx_kwargs = dict(dbpath=dbpath, db=db, sb=sb,
                      weighted_solve=bool(weighted_solve),
                      stage_consistency_weight=float(stage_consistency_weight),
                      ne_prior=ne_prior,
                      wavelength_registration=(wl_registration["combined"]
                                               if wl_registration else None))
    # amp_sigma_floor is attached after the noise scale is measured below
    run_kwargs = dict(sa_doublets=True, n_calls=n_calls, verbose=verbose,
                      sa_stimulated_emission=bool(stimulated_emission),
                      search=search, random_state=gp_seed)
    idx_kwargs["ne_init"] = (float(ne_init) if ne_init is not None
                             else float(ne_prior[0]))
    run_kwargs["ne_bounds"] = tuple(float(v) for v in ne_bounds)

    bg0 = np.asarray(fit.get("background", np.zeros_like(y)), dtype=float)

    # detector segment throughput: the CCD is split at 365 and 620 nm and the
    # 620 nm junction carries a genuine ~5x gain step (SciAps).  A smooth
    # plasma continuum cannot jump by ~5x within a few nm, so the background
    # step there is a throughput ratio that multiplies LINE amplitudes and
    # distorts exactly the cross-segment relative intensities Saha-Boltzmann
    # inference relies on.  Estimate it from the continuum (noise-gated so a
    # weak continuum yields no correction) and divide it out of amplitudes AND
    # their uncertainties before every indexer solve.  The 365 nm step is
    # additive (background subtraction's job), hence edges=(620.0,) only.
    from alibz.detector import (correct_segment_response,
                                estimate_segment_response,
                                segment_response_fallback)
    _resid = y - bg0
    _noise = 1.4826 * float(np.median(np.abs(_resid - np.median(_resid))))
    # the weighted concentration solve may not trust any fitted area below
    # the spectrum's own noise scale (the GLS sigma collapses to ~0 where
    # the local window is quiet), and every peak carries photon-counting
    # noise of its own amplitude that quiet-region noise estimates miss
    idx_kwargs["amp_sigma_floor"] = _noise
    idx_kwargs["amp_sigma_poisson_gain"] = POISSON_GAIN_COUNTS
    fallback, fallback_meta = segment_response_fallback(
        edges=(620.0,), return_metadata=True)
    fallback_unc = [rec.get("uncertainty") for rec in fallback_meta]
    if segment_response_fallback_ratio is not None:
        fallback = [float(segment_response_fallback_ratio)]
        fallback_unc = [segment_response_fallback_uncertainty]
    with telemetry.stage("segment_response"):
        seg_response, seg_response_meta = estimate_segment_response(
            x, bg0, edges=(620.0,), noise_scale=_noise,
            fallback=fallback, fallback_uncertainty=fallback_unc,
            return_metadata=True)
    for rec, configured in zip(seg_response_meta, fallback_meta):
        if rec["source"] == "fallback":
            rec["source"] = (
                segment_response_fallback_source or "fallback_override"
                if segment_response_fallback_ratio is not None
                else "fallback")
            rec["fallback_q25"] = configured.get("q25")
            rec["fallback_q75"] = configured.get("q75")
            rec["fallback_n_spectra"] = configured.get("n_spectra")
    _seg_edges = np.asarray((620.0,), dtype=float)

    def _db_frame(peaks: np.ndarray) -> np.ndarray:
        # indexer matches peak centers against db positions within its
        # shift_tolerance; remove each peak's SEGMENT shift first, then divide
        # amplitudes by the detector segment response so cross-segment
        # relative intensities are physical before the solve
        out = peaks.copy()
        out[:, 1] -= shift_at(shift, out[:, 1])
        return correct_segment_response(out, seg_response, edges=(620.0,))

    _amp_sigma_memo: dict = {}

    def _amp_sigma(peaks_obs: np.ndarray) -> np.ndarray:
        # per-peak area (amplitude) noise, aligned with the peak order, so the
        # indexer can gate elements on detection significance rather than a
        # fraction of the brightest peak.  Scaled by the SAME segment response
        # as the amplitudes, so detection SNR stays gain-invariant.
        # Memoized on the exact peak table: x and y - bg0 are fixed for this
        # spectrum, and the final detections call repeats the last table
        # verbatim (as may deepening rounds that added nothing).
        peaks_obs = np.asarray(peaks_obs)
        memo_key = peaks_obs.tobytes()
        cached = _amp_sigma_memo.get(memo_key)
        if cached is not None:
            telemetry.count("amp_sigma_memo_hits")
            return cached.copy()
        with telemetry.stage("amp_sigma"):
            sig = estimate_peak_uncertainties(x, y - bg0, peaks_obs)[:, 0]
        seg = np.searchsorted(_seg_edges, np.atleast_2d(peaks_obs)[:, 1])
        out = sig / seg_response[seg]
        _amp_sigma_memo[memo_key] = out.copy()
        return out

    # pass 1: establish elements (a PROVISIONAL posterior — its basin can
    # be wrong; its outputs only license seeding and condition stage 3b)
    from alibz.gas_detection import detect_background_gases
    with telemetry.stage("background_gas_detection"):
        background_gases = detect_background_gases(
            x, y, db, shift_nm=shift, peak_array=_db_frame(rpeaks))
    gas_supported = tuple(el for el, evidence in background_gases.items()
                          if evidence["status"] == "detected")
    # A provisional rejection must not prohibit later residual recovery:
    # only pass 1 uses triage; pass 2/deepening rebuild from their own peaks.
    triage_coverage = None
    if physical_triage != "off":
        cuts = np.flatnonzero(np.diff(x) > max(0.5, 5 * np.median(np.diff(x)))) + 1
        pieces = np.split(np.asarray(x), cuts)
        triage_coverage = [(float(p[0] - shift_at(shift, p[0])),
                            float(p[-1] - shift_at(shift, p[-1])))
                           for p in pieces if len(p) >= 2]
    with telemetry.stage("indexer_pass1"):
        idx1 = PeakyIndexerV3(_db_frame(rpeaks), amp_sigma=_amp_sigma(rpeaks),
                              **idx_kwargs)
        # pass 1 is provisional (it licenses seeding and the 3b gates and
        # is re-done in pass 2 on the final peak table): a shorter search
        # costs nothing downstream and ~30 % of the indexer time
        res1 = idx1.run(**dict(run_kwargs, n_calls=min(n_calls, PASS1_N_CALLS),
                              physical_triage=physical_triage,
                              triage_coverage=triage_coverage,
                              triage_protected_elements=gas_supported))
    established = sorted(
        [el for el, f in res1.element_fractions.items()
         if f >= ESTABLISHED_MIN_FRACTION],
        key=element_sort_key,
    )

    # stage 3b — PHYSICS adjudication of the deferred asymmetric features,
    # now that an element posterior exists.  The posterior is the
    # candidate-species set the whole-pattern solve RETAINED (its
    # evidence prefilter already removed elements with no plausible line
    # pattern) rather than the established list: a wrong pass-1 basin
    # must not veto a real resonance line's merge, but conditioning on
    # retained candidates still replaces "any line in the periodic
    # table" with "species plausibly in this plasma".
    posterior = sorted({sp.element for sp in res1.species})
    with telemetry.stage("refine_physics"):
        refined, dec_phys = refine_fit(x, y, refined, db=db,
                                       elements=posterior or None,
                                       shift_nm=shift, asymmetric="only")
    decisions = dec_data + dec_phys
    rpeaks = refined["sorted_parameter_array"]

    # asymmetric-merge zones + their measured emission/observed ratios,
    # computed BEFORE any seeding so every downstream fitter (seeder,
    # residual recovery, deblending, SA recovery) respects them: the
    # merged rows' symmetric table proxy leaves a core-shaped residual by
    # design (see refine_fit), and fitting components there re-splits the
    # merge (measured 21-93% area erosion when the seeder lacked this).
    from alibz.utils.voigt import voigt_width as _vw
    sa_zones, sa_merges = [], []
    for dec in decisions:
        if (dec.get("action") == "sa-tag"
                and str(dec.get("verdict", "")).startswith("asymmetric")
                and dec.get("params_asym") is not None):
            # the SA tag now stores a faithful SYMMETRIC fit (params_single)
            # in the table, so the zone + premeasured record key on THAT
            # component (not the narrow SA-model core), so recover_sa_areas
            # matches the right row
            pS = dec.get("params_single")
            if pS is None:
                pS = dec["params_asym"]
            halfw = 1.5 * max(float(_vw(max(pS[2], 1e-6),
                                        max(pS[3], 1e-6))), 0.15)
            sa_zones.append((float(pS[1]), halfw))
            # the SA tag's measured emission/observed ratio: the ONLY
            # correction channel for tagged lines of species the doublet
            # anchors do not cover (recover_sa_areas skips the zones)
            obs = float(dec.get("observed_area") or 0.0)
            if obs > 0.0 and dec.get("emission_area"):
                sa_merges.append(dict(
                    center_nm=float(pS[1]),
                    factor=float(dec["emission_area"]) / obs,
                    tau_a=float(dec.get("tau_a", 0.0)),
                    observed_area=obs,
                    emission_area=float(dec["emission_area"])))

    final, records = refined, []
    if seed_minor and established:
        with telemetry.stage("seed_minor"):
            final, records = seed_minor_lines(x, y, refined, db, established,
                                              kT_ev=KB_EV_K * res1.temperature,
                                              shift_nm=shift,
                                              segment_edges=(620.0,),
                                              exclude=tuple(sa_zones))
    # element-agnostic recovery: significant positive residual peaks are
    # real lines the seeder could not predict (e.g. Fe lines when the Fe
    # stage scale fails the Boltzmann trust gate) — fit them from the
    # data alone; the pass-2 indexer then identifies them.
    with telemetry.stage("recover_residual"):
        final, recovered = recover_residual_lines(x, y, final,
                                                  exclude=tuple(sa_zones))

    # shoulder-triggered deblends: peaks whose profile shows a one-sided
    # flank bump (an unresolved overlapping line contaminating the fitted
    # area) are split into two components BEFORE identification, so the
    # pass-2/3 indexers see the decontaminated areas.  The refinement's
    # asymmetric-merge zones are excluded (their core residual is
    # deliberate)
    from alibz.profiles import (analyze_peak_profiles, deblend_shoulders,
                                element_shape_quality, recover_sa_areas)
    with telemetry.stage("deblend"):
        prof_pre = analyze_peak_profiles(x, y, final)
        final, deblends = deblend_shoulders(x, y, final, prof_pre,
                                            exclude=tuple(sa_zones))
    fpeaks = final["sorted_parameter_array"]

    # pass 2: identify elements + plasma state, warm-started at pass-1 state
    with telemetry.stage("indexer_pass2"):
        idx2 = PeakyIndexerV3(_db_frame(fpeaks), dbpath=dbpath,
                              db=db, sb=sb,
                              amp_sigma=_amp_sigma(fpeaks),
                              amp_sigma_floor=_noise,
                              amp_sigma_poisson_gain=POISSON_GAIN_COUNTS,
                              weighted_solve=bool(weighted_solve),
                              stage_consistency_weight=float(
                                  stage_consistency_weight),
                              ne_prior=ne_prior,
                              temperature_init=_warm_start_temperature(
                                  res1.temperature),
                              ne_init=res1.ne)
        res2 = idx2.run(**run_kwargs)

    # Basin-selection diagnostics from the pass-2 fit: deepening replaces
    # `result` with fixed-state solve_at results whose convergence_info
    # has no basin/posterior keys, so capture them here or they are lost
    # to the report.
    _info2 = res2.convergence_info or {}
    basin_info = {k: _info2[k] for k in
                  ("basin_bic", "basin_cost", "basin_k",
                   "delta_bic_runner_up", "basin_ambiguous",
                   "n_basins_considered", "posterior_fractions",
                   "posterior_spread", "n_posterior_nodes")
                  if k in _info2}

    # ITERATIVE DEEPENING: pass 2 has now CONFIRMED which elements are
    # present and quantified the confident ones from their intense peaks.
    # Each confident ion's weak lines are then seeded (Boltzmann prior) AND
    # recovered from the data (element-agnostic, but with the local-noise
    # bar lowered ONLY near that ion's own database lines) at progressively
    # lower bars -- so refinement can progress through the faint lines the
    # one-shot 4 sigma recovery leaves behind, without re-admitting the
    # chance-coincidence noise a globally-lowered bar would.  Each round
    # RE-SOLVES the composition at the FIXED pass-2 plasma state
    # (PeakyIndexerV3.solve_at -- no re-optimisation, so no basin drift;
    # the pass-2 T, ne came from the intense lines and the weak lines only
    # corroborate) and is rejected wholesale if it newly collapses.
    fidx, result = idx2, res2
    corroboration = dict(used=False, added=0, reason="seed_minor disabled")
    if seed_minor:
        from alibz.minor_lines import match_and_scale
        confirmed = sorted(
            [el for el, f in res2.element_fractions.items()
             if f >= ESTABLISHED_MIN_FRACTION],
            key=element_sort_key)
        corroboration = dict(used=False, added=0,
                             reason="no confirmed elements")
        if confirmed:
            established = confirmed
            # confident ions = quantified from intense peaks (>= CONFIDENT_
            # MIN_REFS clean reference lines in some stage)
            scales, _ = match_and_scale(
                fpeaks, db, confirmed, kT_ev=KB_EV_K * res2.temperature,
                shift_nm=shift, segment_edges=(620.0,))
            confident = sorted(
                {el for (el, _stg), info in scales.items()
                 if info["n_ref"] >= CONFIDENT_MIN_REFS},
                key=element_sort_key)
            # instrument-frame db lines of confident ions -> coverage map
            sup = []
            for el in confident:
                if el in db.no_lines:
                    continue
                arr = db.lines(el)
                if arr.size == 0:
                    continue
                mk = ((arr[:, 0].astype(float) <= 2)
                      & (arr[:, 3].astype(float) >= SUPPORT_GA_FLOOR))
                wl = arr[mk, 1].astype(float)
                if wl.size:
                    sup.append(wl + shift_at(shift, wl, frame="database"))
            supported = (np.concatenate(sup) if sup
                         else np.empty(0, dtype=float))

            work = final
            n_seed_tot = n_rec_tot = 0
            rounds = []
            collapsed_at = None
            for bar in DEEPEN_BARS:
                if not confident:
                    break
                telemetry.count("deepen_rounds")
                with telemetry.stage("deepen_seed"):
                    work, corr = seed_minor_lines(
                        x, y, work, db, confident,
                        kT_ev=KB_EV_K * res2.temperature, shift_nm=shift,
                        accept_snr=bar, min_expected_snr=bar,
                        robust_elements=set(confident),
                        segment_edges=(620.0,),
                        exclude=tuple(sa_zones))
                n_seed = sum(1 for r in corr if r.get("action") == "added")
                with telemetry.stage("deepen_recover"):
                    work, rec = recover_residual_lines(
                        x, y, work, exclude=tuple(sa_zones),
                        supported_lines=supported,
                        snr_min_supported=bar, accept_snr_supported=bar,
                        support_tol_nm=SUPPORT_TOL_NM)
                n_rec = sum(1 for r in rec if r.get("action") == "added")
                if n_seed + n_rec == 0:
                    rounds.append(dict(bar=bar, seeded=0, recovered=0,
                                       used=False))
                    continue
                # basin-safe fixed re-solve on the grown peak table
                with telemetry.stage("deepen_solve"):
                    idxN = PeakyIndexerV3(
                        _db_frame(work["sorted_parameter_array"]),
                        dbpath=dbpath, db=db, sb=sb,
                        amp_sigma=_amp_sigma(work["sorted_parameter_array"]),
                        amp_sigma_floor=_noise,
                        amp_sigma_poisson_gain=POISSON_GAIN_COUNTS,
                        weighted_solve=bool(weighted_solve),
                        stage_consistency_weight=float(
                            stage_consistency_weight),
                        ne_prior=ne_prior,
                        temperature_init=res2.temperature, ne_init=res2.ne)
                    idxN.build_candidate_matrix(
                        sa_doublets=True,
                        sa_stimulated_emission=bool(stimulated_emission))
                    resN = idxN.solve_at(res2.temperature, res2.ne,
                                         res2.sigma, res2.gamma)
                # guard against the TRUSTED pass-2 baseline (catches slow
                # cumulative drift, not just round-to-round)
                if composition_collapsed(res2.element_fractions,
                                         resN.element_fractions):
                    collapsed_at = bar
                    rounds.append(dict(bar=bar, seeded=n_seed,
                                       recovered=n_rec, used=False,
                                       reason="collapse"))
                    break
                records = records + corr + rec
                n_seed_tot += n_seed
                n_rec_tot += n_rec
                final, fpeaks, fidx, result = (
                    work, work["sorted_parameter_array"], idxN, resN)
                rounds.append(dict(bar=bar, seeded=n_seed, recovered=n_rec,
                                   used=True))
            total = n_seed_tot + n_rec_tot
            reason = ""
            if total == 0:
                reason = "no corroborating lines added"
            elif collapsed_at is not None:
                reason = f"deepening collapse at {collapsed_at} sigma"
            corroboration = dict(
                used=total > 0, added=total, seeded=n_seed_tot,
                recovered=n_rec_tot, confident=confident, rounds=rounds,
                top_before=round(float(max(res2.element_fractions.values(),
                                           default=0.0)), 3),
                top_after=round(float(max(result.element_fractions.values(),
                                          default=0.0)), 3),
                reason=reason)

    # per-segment, per-peak shape physics (alibz.profiles) on the FINAL
    # fit, then growth-curve area recovery: sa-like peaks of species NOT
    # already anchored by the indexer's doublet channel are refit with the
    # self-absorption model; accepted emission areas correct the observed
    # amplitudes and the composition is re-solved LINEARLY at the fitted
    # plasma state (no new Bayesian pass -> no basin risk; a corrected
    # composition that newly collapses is rejected wholesale)
    with telemetry.stage("sa_recovery"):
        profiles = analyze_peak_profiles(x, y, final)
        result, sa_records, sa_used = recover_sa_areas(
            fidx, result, x, y, final, profiles, exclude=tuple(sa_zones),
            premeasured=tuple(sa_merges))

    # detection report + confounder (true-negative rival) analysis; when
    # SA recovery was applied, detections see the SAME corrected
    # amplitudes the re-solved result came from
    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    area_sigma = _amp_sigma(fpeaks)
    amp_stash = None
    if sa_used:
        amp_stash = fidx._obs_amp.copy()
        for r in sa_records:
            if r["action"] == "sa-recovered":
                idx = int(r["index"])
                factor = float(r["factor"])
                fidx._obs_amp[idx] *= factor
                area_sigma[idx] *= factor
    try:
        with telemetry.stage("detections"):
            det = analyze_detections(fidx, result, area_sigma, shift=shift,
                                     dbpath=dbpath, draws=draws)
    finally:
        if amp_stash is not None:
            fidx._obs_amp = amp_stash

    # QC each element's supporting flux -- an element whose abundance
    # rests on saturated (sa-like) or overlap-contaminated (shoulder)
    # peaks is flagged rather than trusted
    shape_quality = element_shape_quality(det.get("support_idx", {}),
                                          profiles)
    for d in det["detections"]:
        q = shape_quality.get(d["element"])
        if q:
            d["sa_share"] = round(float(q["sa_share"]), 3)
            d["shoulder_share"] = round(float(q["shoulder_share"]), 3)
            d["clean_anchors"] = int(q["clean_anchors"])

    return dict(
        fit=fit, refined=refined, final=final, decisions=decisions,
        records=records, recovered=recovered, shift=shift,
        baseline_shift=baseline_shift,
        wavelength_calibration=wavelength_calibration,
        wavelength_registration=wl_registration,
        peak_refinement=peak_refinement,
        segment_response=seg_response, segment_response_edges=(620.0,),
        segment_response_metadata=seg_response_meta,
        shift_prior_applied=shift_prior_applied,
        n_anchor=n_anchor, n_shift_anchor=n_shift_anchor, ne_init=ne_init,
        ne_bounds=tuple(ne_bounds), ne_bounds_source=ne_bounds_source,
        result=result, established=established,
        physical_triage=getattr(idx1, "_triage_report", None),
        background_gases=background_gases,
        element_uncertainty=det["element_uncertainty"],
        detections=det["detections"], support=det["support"],
        contested=det["contested"],
        resolved_fractions=det["resolved_fractions"],
        profiles=profiles, shape_quality=shape_quality,
        corroboration=corroboration,
        basin_info=basin_info,
        shape_refit=dict(deblends=deblends, sa=sa_records, sa_used=sa_used),
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
    # shape-QC flag: a DOMINANT element whose supporting peaks are mostly
    # saturated (sa-like) or lack clean anchors is a model choice, not a
    # measurement (archetype: Ca II 393.3 resonance carrying 99% Ca)
    fr = res.element_fractions
    if fr:
        top_el = max(fr, key=fr.get)
        if fr[top_el] >= 0.5:
            dom = next((d for d in analysis.get("detections", [])
                        if d["element"] == top_el), None)
            # no shape entry at all = the dominant element has NO supporting
            # peaks of its own -- the weakest possible support
            weak = (dom is None or dom.get("clean_anchors") is None
                    or dom["clean_anchors"] < 2
                    or (dom.get("sa_share") or 0.0) > 0.5)
            if weak:
                flags.append(f"{top_el}:dominant-weak-shape")
    basin = analysis.get("basin_info") or {}
    if basin.get("basin_ambiguous"):
        d = basin.get("delta_bic_runner_up")
        flags.append(f"basin-ambiguous(dBIC={d if d is not None else 'inf'})")
    corro = analysis.get("corroboration") or {}
    if "collapse" in (corro.get("reason") or ""):
        flags.append("deepening-stopped(collapse)")
    n_deep = int(corro.get("added") or 0)
    if n_deep:
        flags.append(f"deepened({n_deep})")
    sr = analysis.get("shape_refit") or {}
    n_deb = sum(1 for r in sr.get("deblends", [])
                if r.get("action") == "deblended")
    n_sa = sum(1 for r in sr.get("sa", [])
               if r.get("action") == "sa-recovered")
    if n_deb:
        flags.append(f"deblended({n_deb})")
    if sr.get("sa_used") and n_sa:
        flags.append(f"sa-area-recovered({n_sa})")
    response_meta = analysis.get("segment_response_metadata") or [{}]
    response_620 = response_meta[0]
    # Historical shift columns retain baseline calibration semantics. Independent
    # gas corrections and their exact application ranges are exported separately.
    shift = analysis.get("baseline_shift", analysis["shift"])
    shift_segments = getattr(shift, "shifts", (float(shift),))
    shift_counts = getattr(shift, "n_matches", (analysis.get("n_anchor", 0),))
    shift_applied = getattr(shift, "applied", (False,) * len(shift_segments))
    shift_prior_applied = analysis.get(
        "shift_prior_applied", (False,) * len(shift_segments))
    calibration = analysis.get("wavelength_calibration", {})
    gas_cals = calibration.get("gases", {})
    def gas_pm(element, key):
        value = gas_cals.get(element, {}).get(key)
        return round(1000 * float(value), 2) if value is not None else ""
    qc_fail, qc_warn = [], []
    r_squared = float(res.r_squared)
    if r_squared < FIT_R2_FAIL:
        qc_fail.append("physical-pattern-r2")
    elif r_squared < FIT_R2_WARN:
        qc_warn.append("physical-pattern-r2")
    if (res.temperature <= PLASMA_T_BOUNDS[0] + 1.0
            or res.temperature >= PLASMA_T_BOUNDS[1] - 1.0):
        qc_fail.append("temperature-at-bound")
    ne_bounds = tuple(analysis.get("ne_bounds") or PLASMA_LOG_NE_BOUNDS)
    if (res.ne <= ne_bounds[0] + 0.01
            or res.ne >= ne_bounds[1] - 0.01):
        qc_fail.append("electron-density-at-bound")
    if max(fr.values(), default=0.0) >= COLLAPSE_TOP_FRACTION:
        qc_fail.append("composition-collapse")
    if any(flag.endswith("dominant-weak-shape") for flag in flags):
        qc_fail.append("dominant-weak-shape")
    if response_620.get("source") == "invalid":
        qc_fail.append("detector-response-invalid")
    elif response_620.get("source") != "measured":
        qc_warn.append("detector-response-fallback")
    if (not any(shift_applied) and not any(shift_prior_applied)
            and not calibration.get("applied", False)):
        qc_warn.append("global-only-wavelength-shift")
    if calibration.get("conflicts"):
        qc_warn.append("gas-wavelength-disagreement")
    registration = analysis.get("wavelength_registration") or {}
    cross_check = registration.get("gas_cross_check") or {}
    if cross_check.get("flagged"):
        qc_warn.append("ar-registration-disagreement")
    ambient_nir = ((registration.get("ambient") or {}).get("segments") or {}).get("NIR") or {}
    qc_status = "fail" if qc_fail else "warn" if qc_warn else "pass"
    qc_reasons = ";".join(dict.fromkeys(qc_fail + qc_warn))
    # guard_triggered: the reactive-guard vocabulary, regularized into its
    # own column so guard-fire rates are directly measurable.  qc_fail
    # entries already use controlled names; deepening's collapse guard is
    # promoted from the free-form flags string.
    guards = list(qc_fail)
    if any(f == "deepening-stopped(collapse)" for f in flags):
        guards.append("deepening-collapse")
    return dict(
        file=os.path.basename(path),
        sample=sample_name(path),
        status="ok",
        physical_triage=analysis.get("physical_triage"),
        background_gases=analysis.get("background_gases", {}),
        wavelength_calibration=calibration,
        gas_calibration_applied=bool(calibration.get("applied", False)),
        gas_calibration_conflicts=len(calibration.get("conflicts", [])),
        Ar_calibration_status=gas_cals.get("Ar", {}).get("status", ""),
        Ar_calibration_shift_pm=gas_pm("Ar", "offset_nm"),
        Ar_calibration_uncertainty_pm=gas_pm("Ar", "uncertainty_nm"),
        O_calibration_status=gas_cals.get("O", {}).get("status", ""),
        O_calibration_shift_pm=gas_pm("O", "offset_nm"),
        O_calibration_uncertainty_pm=gas_pm("O", "uncertainty_nm"),
        Ar_status=analysis.get("background_gases", {}).get("Ar", {}).get("status", ""),
        O_status=analysis.get("background_gases", {}).get("O", {}).get("status", ""),
        wavelength_registration_mode=registration.get("mode", ""),
        wavelength_registration_applied=";".join(registration.get("applied_segments") or []),
        ambient_nir_shift_pm=(round(1000.0 * float(ambient_nir["shift_nm"]), 1)
                              if ambient_nir.get("shift_nm") is not None else ""),
        ambient_nir_n_lines=int(ambient_nir.get("n_lines") or 0),
        ar_registration_n_sigma=(round(float(cross_check["n_sigma"]), 2)
                                 if cross_check.get("n_sigma") is not None else ""),
        n_peaks=int(analysis["final"]["sorted_parameter_array"].shape[0]),
        shift_pm=round(1000.0 * float(shift), 1),
        shift_segments_pm=";".join(f"{1000.0 * float(v):.1f}"
                                   for v in shift_segments),
        shift_anchor_counts=";".join(str(int(v)) for v in shift_counts),
        shift_segment_applied=";".join("1" if v else "0"
                                       for v in shift_applied),
        shift_prior_applied=";".join("1" if v else "0"
                                     for v in shift_prior_applied),
        response_620=round(float(response_620.get("ratio", np.nan)), 5),
        response_620_unc=(round(float(response_620["ratio_uncertainty"]), 5)
                          if response_620.get("ratio_uncertainty") is not None
                          else ""),
        response_620_source=response_620.get("source", ""),
        qc_status=qc_status,
        qc_reasons=qc_reasons,
        T_K=round(float(res.temperature), 0),
        log_ne=round(float(res.ne), 2),
        r_squared=round(float(res.r_squared), 4),
        sa_converged=info.get("sa_converged"),
        flags=";".join(flags),
        failure_stage="",
        failure_reason="",
        guard_triggered=";".join(dict.fromkeys(guards)),
        fractions={el: float(f) for el, f in res.element_fractions.items()
                   if f > 0},
        concentrations={el: float(value) for el, value in
                        res.element_concentrations.items() if value > 0},
        uncertainties=analysis["element_uncertainty"],
        detections=analysis.get("detections", []),
    )


def _error_row(path: str, message: str, failure_reason: str = "error",
               telemetry_snapshot: Optional[dict] = None) -> dict:
    """Failure row with taxonomy: ``failure_reason`` is a controlled
    vocabulary (``timeout``, ``exception:<Type>``, ``no-peaks``,
    ``worker-died``, ``not-analyzed``, ``error``) and ``failure_stage``
    is the telemetry stage the failure escaped from — together they turn
    "fails often" into a per-stage, per-cause rate.
    """
    snap = telemetry_snapshot or {}
    return dict(
        file=os.path.basename(path), path=path, sample=sample_name(path),
        status=f"error: {message}"[:200], n_peaks=0, shift_pm="",
        shift_segments_pm="", shift_anchor_counts="",
        shift_segment_applied="", shift_prior_applied="",
        response_620="", response_620_unc="",
        response_620_source="",
        qc_status="fail", qc_reasons="analysis-error",
        T_K="", log_ne="", r_squared="", sa_converged="", flags="",
        t_total_s=snap.get("t_total_s", ""),
        failure_stage=(snap.get("failure_stage")
                       or snap.get("open_stage") or ""),
        failure_reason=failure_reason,
        guard_triggered="",
        telemetry=snap or None,
        fractions={}, concentrations={}, uncertainties={}, detections=[],
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


def _analyze_file(job) -> dict:
    """Analyze one file; NEVER raises — every failure becomes an error row.

    ``job`` is ``(path, AnalysisConfig)``.  The timeout alarm is confined
    to the analysis call and disarmed before any exception handling runs,
    so an alarm that fires during teardown cannot escape as a stray
    ``_Timeout``.
    """
    path, cfg = job
    telemetry.reset()
    use_alarm = bool(cfg.timeout_s) and hasattr(signal, "SIGALRM")
    try:
        x, y = load_spectrum_csv(path)
        old = None
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            # REPEATING timer, not a one-shot alarm: an exception raised
            # by a signal handler can be silently eaten when delivery
            # lands inside C code that clears pending errors (measured:
            # numpy's string->float astype fallback in db_lines_in ate
            # the _Timeout, and the spectrum ran to completion past its
            # deadline).  A 5 s repeat interval keeps re-raising until
            # one delivery lands in a propagating context.
            signal.setitimer(signal.ITIMER_REAL, float(cfg.timeout_s), 5.0)
        try:
            analysis = analyze_spectrum(
                x, y, cfg.dbpath, n_calls=cfg.n_calls,
                draws=cfg.draws,
                stimulated_emission=cfg.stimulated_emission,
                search=cfg.search,
                gp_seed=cfg.gp_seed,
                weighted_solve=cfg.weighted_solve,
                stage_consistency_weight=cfg.stage_consistency_weight,
                physical_triage=cfg.physical_triage,
                gas_wavelength_calibration=cfg.gas_wavelength_calibration,
                wavelength_registration=cfg.wavelength_registration,
                subpixel=cfg.subpixel,
                segment_response_fallback_ratio=
                cfg.segment_response_fallback_ratio,
                segment_response_fallback_source=
                cfg.segment_response_fallback_source,
                segment_response_fallback_uncertainty=
                cfg.segment_response_fallback_uncertainty,
                segment_shift_offsets_nm=cfg.segment_shift_offsets_nm,
                segment_shift_prior_nm=cfg.segment_shift_prior_nm)
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0.0)  # disarm FIRST
                signal.signal(signal.SIGALRM, old)
        row = _summary_row(path, analysis)
        snap = telemetry.snapshot()
        row["t_total_s"] = snap["t_total_s"]
        row["telemetry"] = snap
        return row
    except _Timeout:
        return _error_row(path, f"timeout after {cfg.timeout_s}s",
                          failure_reason="timeout",
                          telemetry_snapshot=telemetry.snapshot())
    except BaseException as exc:  # noqa: BLE001 - isolate every failure
        if isinstance(exc, KeyboardInterrupt):
            raise
        traceback.print_exc(file=sys.stderr)
        reason = ("no-peaks"
                  if isinstance(exc, ValueError)
                  and "found no peaks" in str(exc)
                  else f"exception:{type(exc).__name__}")
        return _error_row(path, f"{type(exc).__name__}: {exc}",
                          failure_reason=reason,
                          telemetry_snapshot=telemetry.snapshot())


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
    search: str = DEFAULT_SEARCH,
    gp_seed: int = DEFAULT_GP_SEED,
    weighted_solve: bool = False,
    stage_consistency_weight: float = STAGE_CONSISTENCY_WEIGHT,
    segment_response_fallback_ratio: Optional[float] = None,
    segment_response_fallback_source: Optional[str] = None,
    segment_response_fallback_uncertainty: Optional[float] = None,
    segment_shift_offsets_nm: Optional[Sequence[float]] = None,
    segment_shift_prior_nm: Optional[Sequence[float]] = None,
    exclude: Sequence[str] = ("summary.csv", DETECTIONS_NAME),
    provenance: bool = True,
    strict_provenance: bool = False,
    progress=print,
    physical_triage: str = "off",
    gas_wavelength_calibration: str = "apply",
    wavelength_registration: str = "ambient",
    subpixel: str = "gaussian",
) -> List[dict]:
    """Analyze every spectrum matching ``pattern`` in ``data_dir``.

    Returns summary rows (see :func:`_summary_row`) in filename order.
    Failures are captured as error rows, never raised.  ``exclude`` lists
    basenames to skip — by default the tool's own ``summary.csv``, so a
    re-run in the same directory does not try to analyze its previous
    output as a spectrum.

    Unless ``provenance=False``, a ``run_manifest.json`` (git state,
    config snapshot, input hashes) is written into ``data_dir``; a dirty
    worktree is captured under ``data_dir/provenance/`` — or refused
    outright when ``strict_provenance`` is set — so no result can again
    outlive the exact code that produced it unrecorded.
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

    cfg = AnalysisConfig(
        dbpath=dbpath, n_calls=n_calls, draws=draws, timeout_s=timeout_s,
        stimulated_emission=bool(stimulated_emission),
        search=search, gp_seed=gp_seed,
        weighted_solve=bool(weighted_solve),
        stage_consistency_weight=float(stage_consistency_weight),
        physical_triage=physical_triage,
        gas_wavelength_calibration=gas_wavelength_calibration,
        wavelength_registration=wavelength_registration,
        subpixel=subpixel,
        segment_response_fallback_ratio=segment_response_fallback_ratio,
        segment_response_fallback_source=segment_response_fallback_source,
        segment_response_fallback_uncertainty=
        segment_response_fallback_uncertainty,
        segment_shift_offsets_nm=
        AnalysisConfig._tuple_or_none(segment_shift_offsets_nm),
        segment_shift_prior_nm=
        AnalysisConfig._tuple_or_none(segment_shift_prior_nm),
    )
    jobs = {f: (f, cfg) for f in files}

    manifest = None
    if provenance:
        from alibz.provenance import run_manifest, DirtyWorktreeError
        try:
            manifest = run_manifest(data_dir, asdict(cfg), files,
                                    dbpath=dbpath,
                                    strict=strict_provenance,
                                    progress=progress)
        except DirtyWorktreeError:
            raise
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            progress(f"provenance manifest failed (continuing): {exc}")
    rows: Dict[str, dict] = {}          # keyed by full path (basenames collide)
    t0 = time.time()
    n = len(files)

    # per-file stage/counter profile, one JSON line per completed row —
    # the measurement layer every optimization is judged against
    profile_path = os.path.join(data_dir, PROFILE_NAME)
    try:
        open(profile_path, "w").close()     # truncate any previous run
    except OSError:
        profile_path = None

    def _record(i, path, row):
        rows[path] = row
        if profile_path is not None:
            line = dict(row.get("telemetry") or {})
            line.update(file=row.get("file"), sample=row.get("sample"),
                        status=row.get("status"),
                        failure_stage=row.get("failure_stage", ""),
                        failure_reason=row.get("failure_reason", ""),
                        guard_triggered=row.get("guard_triggered", ""))
            try:
                with open(profile_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
            except (OSError, TypeError, ValueError):
                pass                        # telemetry must never kill a run
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
                            row = _error_row(path, f"worker failed: {exc}",
                                             failure_reason="worker-died")
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
    out = [rows.get(f, _error_row(f, "not analyzed (interrupted)",
                                  failure_reason="not-analyzed"))
           for f in files]
    if manifest is not None:
        from alibz.provenance import finalize_manifest
        try:
            finalize_manifest(data_dir, manifest, out)
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            progress(f"provenance finalize failed: {exc}")
    return out


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

    meta = ["test_id", "height_um", "file", "sample", "status", "n_peaks", "shift_pm",
            "shift_segments_pm", "shift_anchor_counts",
            "shift_segment_applied", "shift_prior_applied",
            "response_620", "response_620_unc",
            "response_620_source", "T_K", "log_ne", "r_squared",
            "sa_converged", "qc_status", "qc_reasons", "flags",
            "t_total_s", "failure_stage", "failure_reason",
            "guard_triggered", "Ar_status", "O_status",
            "gas_calibration_applied", "gas_calibration_conflicts",
            "Ar_calibration_status", "Ar_calibration_shift_pm",
            "Ar_calibration_uncertainty_pm", "O_calibration_status",
            "O_calibration_shift_pm", "O_calibration_uncertainty_pm",
            "wavelength_registration_mode", "wavelength_registration_applied",
            "ambient_nir_shift_pm", "ambient_nir_n_lines",
            "ar_registration_n_sigma"]
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
    header = ["sample", "test_id", "height_um", "file",
              "element", "status", "fraction", "fraction_resolved",
              "fraction_hi", "unc", "z",
              "n_lines", "clear_lines", "contested_share", "confounder",
              "strongest_peak_nm", "strongest_obs",
              "upper_limit", "stage_disagreement",
              "sa_share", "shoulder_share", "clean_anchors"]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            if row.get("status", "") != "ok" and not row.get("detections"):
                # errored samples must be VISIBLE here, not silently absent
                w.writerow([row["sample"], row.get("test_id", ""),
                            row.get("height_um", ""), row.get("file", ""),
                            "", row.get("status", "")]
                           + [""] * (len(header) - 6))
                n += 1
                continue
            for d in row.get("detections", []):
                fr = d.get("fraction_resolved", d.get("fraction"))
                fhi = d.get("fraction_hi", d.get("fraction"))
                # fr/fhi may be a legitimate 0.0 (an element resolved away
                # by the confounder): render it as "0", not blank
                w.writerow([
                    row["sample"], row.get("test_id", ""),
                    row.get("height_um", ""), row.get("file", ""),
                    d["element"], d["status"],
                    f"{d['fraction']:.4g}" if d["fraction"] else "",
                    f"{fr:.4g}" if fr is not None else "",
                    f"{fhi:.4g}" if fhi is not None else "",
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
                    (d["sa_share"]
                     if d.get("sa_share") is not None else ""),
                    (d["shoulder_share"]
                     if d.get("shoulder_share") is not None else ""),
                    (d["clean_anchors"]
                     if d.get("clean_anchors") is not None else ""),
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
    gas_wavelength_calibration: str = "apply",
) -> dict:
    """Notebook (nbformat-4.5 JSON dict) that inspects this directory.

    Reads ``summary.csv`` (raw composition) and ``detections.csv``
    (confounder-resolved composition) for the as-fit-vs-resolved overview,
    and re-runs the full pipeline live on one selectable spectrum for the
    standard fit inspection views.  ``stimulated_emission`` is baked into the live
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
    code_setup = f"""%matplotlib inline
import csv, glob, importlib, os
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

    code_overview = """# composition overview: as-fit (raw NNLS vertex) vs true-negative-resolved
# LEFT  panel = the raw fit composition straight from summary.csv.
# RIGHT panel = after the confounder correction (fraction_resolved from
#   detections.csv): a `confounded` element -- one whose every supporting
#   peak a genuinely-present rival could equally explain -- is credited only
#   its uncontested flux, the contested remainder reattributed to that rival
#   (so Mn, read off the shared Mg II 279.5/280.3 region, collapses into Mg;
#   an element contested only by an ABSENT rival keeps its flux instead).
ok = [r for r in SUMMARY if r['status'] == 'ok']
samples = [r['sample'] for r in ok]

comp_raw = {r['sample']: {el: (float(r[el]) if r[el] else 0.0)
                          for el in ELEMENTS} for r in ok}
comp_res = {s: {} for s in samples}
for d in DETECTIONS:
    s, v = d['sample'], d.get('fraction_resolved')
    if s in comp_res and v not in (None, ''):
        comp_res[s][d['element']] = comp_res[s].get(d['element'], 0.0) + float(v)

order = sorted(set(ELEMENTS) | {d['element'] for d in DETECTIONS
                                if d.get('fraction_resolved') not in (None, '')},
               key=element_sort_key)

def _stack(ax, comp, title):
    xpos = np.arange(len(samples))
    bottom = np.zeros(len(samples))
    for el in order:
        vals = np.array([comp.get(s, {}).get(el, 0.0) for s in samples])
        if vals.max() <= 0:
            continue
        ax.bar(xpos, vals, bottom=bottom, color=element_color(el),
               edgecolor='white', linewidth=0.3)
        bottom += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels([s[:18] for s in samples], rotation=75, fontsize=6)
    ax.set_title(title, fontsize=10)

if DETECTIONS:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(2 + 0.42 * len(samples), 5.5),
                                   sharey=True)
    _stack(axL, comp_raw, 'As fit (raw NNLS vertex — confounder-inflated)')
    _stack(axR, comp_res, 'True-negative resolved (confounded flux reattributed)')
else:
    fig, axL = plt.subplots(figsize=(2 + 0.42 * len(samples), 5.5))
    _stack(axL, comp_raw, 'As fit (raw NNLS vertex)')
    print('no detections.csv — resolved panel unavailable')
axL.set_ylabel('atom fraction of detected emitters')

seen = [el for el in order
        if any(comp_raw.get(s, {}).get(el, 0.0) or comp_res.get(s, {}).get(el, 0.0)
               for s in samples)]
handles = [Patch(facecolor=element_color(el), edgecolor='0.4',
                 label=f"{el} ({element_periodic_block(el)})") for el in seen]
fig.legend(handles=handles, title='element (periodic block)', ncol=8,
           fontsize=7, title_fontsize=8, loc='lower center',
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle('Composition by sample — as-fit vs confounder-resolved', y=1.02)
fig.tight_layout()"""

    code_overview_shift = """# corpus-mean composition shift from the confounder correction
# quantifies the left->right change above: which elements the true-negative
# resolution strips (their peaks reassigned to a present rival) and which
# gain, averaged over all samples -- the direct answer to "why so much Mn?".
if DETECTIONS:
    raw_mean = {el: float(np.mean([comp_raw[s].get(el, 0.0) for s in samples]))
                for el in order}
    res_mean = {el: float(np.mean([comp_res[s].get(el, 0.0) for s in samples]))
                for el in order}
    delta = {el: res_mean[el] - raw_mean[el] for el in order}
    shifted = sorted([el for el in order if abs(delta[el]) > 1e-4],
                     key=lambda el: delta[el])
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.4 * len(shifted))))
    yp = np.arange(len(shifted))
    ax.barh(yp, [100 * delta[el] for el in shifted],
            color=[element_color(el) for el in shifted], edgecolor='0.3')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(yp)
    ax.set_yticklabels(shifted)
    ax.set_xlabel('corpus-mean change after resolution (percentage points)')
    ax.set_title('Confounder correction: mean composition shift '
                 '(resolved - as-fit)')
    for i, el in enumerate(shifted):
        dv = 100 * delta[el]
        ax.text(dv, i, f"  {100 * raw_mean[el]:.1f}->{100 * res_mean[el]:.1f}%",
                va='center', ha='left' if dv >= 0 else 'right', fontsize=7)
    ax.margins(x=0.28)
    fig.tight_layout()
    print('largest shifts (percentage points):', ', '.join(
        f'{el} {100 * delta[el]:+.1f}' for el in
        sorted(shifted, key=lambda el: -abs(delta[el]))[:6]))
else:
    print('no detections.csv — resolved composition unavailable')"""

    code_run = f"""SPECTRUM_FILE = FILES[0]   # <-- change to inspect another spectrum
print(sample_name(SPECTRUM_FILE))
x, y = load_spectrum_csv(SPECTRUM_FILE)
a = analyze_spectrum(x, y, DB_PATH, n_calls={n_calls}, draws=16,
                     stimulated_emission={stimulated_emission!r},
                     gas_wavelength_calibration={gas_wavelength_calibration!r})
res = a['result']
print(f"T = {{res.temperature:.0f}} K   log ne = {{res.ne:.2f}}   "
      f"r^2 = {{res.r_squared:.3f}}   peaks = {{a['final']['sorted_parameter_array'].shape[0]}}")
calibration = a.get('wavelength_calibration', {{}})
for gas, evidence in calibration.get('gases', {{}}).items():
    print(gas, evidence.get('status'), 'offset nm:', evidence.get('offset_nm'),
          'uncertainty nm:', evidence.get('uncertainty_nm'))
print('Applied gas calibration regions:', calibration.get('applied_regions', []))"""

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

    code_decisions = """# refinement decisions: blends split; self-absorbed lines get a
# NON-DESTRUCTIVE tag (faithful symmetric fit kept in the table, emission
# area + tau recorded for the growth-curve amplitude recovery)
for d in sorted(a['decisions'], key=lambda d: d['center']):
    if d['action'] in ('none', 'deferred'):
        continue
    extra = ''
    if 'tau_a' in d:
        extra = (f"  tau={d['tau_a']:.2f} delta={1000*d['delta_nm']:+.0f} pm"
                 f"  emission={d['emission_area']:.3g}"
                 f" observed={d['observed_area']:.3g}")
    print(f"{d['center']:9.3f}  {d['kind']:8s} {d['verdict']:18s}"
          f" {d['action']:8s}{extra}")"""

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
# 'resolved' is the true-negative-corrected abundance (confounded elements
# credited only their uncontested flux); 'fraction' is the raw NNLS vertex.
from alibz import plot_peak_zoom

print(f"{'el':>4} {'status':>12} {'fraction':>9} {'resolved':>9} {'z':>6}"
      f" {'lines':>5} {'clear':>5} {'confounder':>10}"
      f"  {'strongest [nm]':>14} {'upper_lim':>9}")
for d in sorted(a['detections'], key=lambda d: -(d['fraction'] or 0)):
    print(f"{d['element']:>4} {d['status']:>12}"
          f" {d['fraction']:9.5f} {d.get('fraction_resolved', d['fraction']):9.5f}"
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

    md_shapes = """### Peak-shape physics QC (`alibz.profiles`)

Every fitted peak is classified per detector segment against that segment's
instrumental width floor: `instrumental` (clean, resolution-limited — the
safest quantification anchors), `broadened` (genuine plasma broadening;
Gaussian fraction separates Doppler from Stark), `shoulder` (an UNRESOLVED
overlapping line — the fitted area is contaminated), `sa-like` (core
defect: the growth-curve signature of self-absorption — the area is
saturated, NOT proportional to concentration), `narrow` (below the
instrument floor — suspect). Shoulders and self-absorption both look
"asymmetric"; their residual signatures (one-sided flank bump vs core
defect) are what tells them apart.

Per element, `sa_share`/`shoulder_share` is the flux-weighted share of its
supporting peaks that is saturated/contaminated and `clean_anchors` counts
its clean lines — a DOMINANT element with weak shape support is flagged
`dominant-weak-shape` in `summary.csv`. `corroboration` reports whether
the weak-line (pass-3) re-index was accepted or rejected by the
composition-collapse basin guard.
"""
    code_shapes = """# peak-shape physics QC for this spectrum
from alibz import analyze_peak_profiles, profile_summary
prof = analyze_peak_profiles(x, y, a['final'])
print('peak shape classes:', profile_summary(prof))
print('corroboration:', a.get('corroboration'))
rows = [d for d in a['detections'] if d.get('sa_share') is not None]
print(f"\\n{'el':>4} {'status':>12} {'resolved':>9} {'sa_share':>8}"
      f" {'shoulder':>8} {'clean':>5}")
for d in sorted(rows, key=lambda d: -(d.get('fraction_resolved') or 0))[:14]:
    print(f"{d['element']:>4} {d['status']:>12}"
          f" {d.get('fraction_resolved', 0):9.4f}"
          f" {d.get('sa_share', 0):8.2f} {d.get('shoulder_share', 0):8.2f}"
          f" {d.get('clean_anchors', ''):>5}")
# saturated or overlap-contaminated peaks, strongest first
bad = [r for r in prof if r['classification'] in ('sa-like', 'shoulder')]
print(f"\\n{len(bad)} saturated/contaminated peaks:")
for r in sorted(bad, key=lambda r: -abs(r['area']))[:10]:
    print(f"  {r['center_nm']:9.3f}  {r['classification']:9s}"
          f"  area={r['area']:9.1f}  wr={r['width_ratio']:.2f}"
          f"  gfrac={r['gaussian_fraction']:.2f}")

# shape-refit feedback: shoulder deblends + SA growth-curve area recovery
sr = a.get('shape_refit') or {}
deb = [r for r in sr.get('deblends', []) if r['action'] == 'deblended']
print(f"\\nshoulder deblends: {len(deb)} accepted")
for r in deb[:8]:
    print(f"  {r['center_nm']:9.3f} -> new component at"
          f" {r['new_center_nm']:9.3f}  area={r['area_new']:8.1f}"
          f"  snr={r['snr']:5.1f}  dBIC={r['delta_bic']:6.1f}")
sa_rec = [r for r in sr.get('sa', []) if r['action'] == 'sa-recovered']
print(f"SA area recovery ({'APPLIED' if sr.get('sa_used') else 'not applied'}):"
      f" {len(sa_rec)} lines")
for r in sa_rec[:8]:
    print(f"  {r['center_nm']:9.3f} [{r.get('species','?'):>6s}]"
          f"  observed={r['observed_area']:8.1f} ->"
          f" emission={r['emission_area']:8.1f}  (x{r['factor']:.2f},"
          f" tau={r['tau_a']:.2f})")
skipped = [r for r in sr.get('sa', []) if r['action'] == 'anchored']
if skipped:
    print(f"  ({len(skipped)} peaks left to the indexer's doublet-anchored"
          f" correction)")"""

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
        _nb_cell("markdown", "## Composition across all samples\n\nThe raw "
                 "NNLS fit (left) beside the true-negative-resolved "
                 "composition (right), then the corpus-mean shift the "
                 "confounder correction makes. The raw panel inflates "
                 "elements like Mn whose peaks sit under a present rival's "
                 "lines; the resolved panel reattributes that flux."),
        _nb_cell("code", code_overview),
        _nb_cell("code", code_overview_shift),
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
        _nb_cell("markdown", md_shapes),
        _nb_cell("code", code_shapes),
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
