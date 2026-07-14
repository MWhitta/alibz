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
DEEPEN_BARS = (3.0, 2.5, 2.0)
#: gA floor for a confident ion's database line to mark "coverage" for the
#: guarded low-bar agnostic recovery (a bump near a strong line of a
#: present ion is very likely that ion's faint line, not noise).
SUPPORT_GA_FLOOR = 1.0e6
SUPPORT_TOL_NM = 0.06


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

    db = _get_db(dbpath)
    finder = PeakyFinder.__new__(PeakyFinder)  # fit_spectrum needs no data dir
    fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False,
                              n_sigma=0)
    peaks = fit["sorted_parameter_array"]
    if peaks.size == 0:
        raise ValueError("blind fit found no peaks")

    # pooled global shift from the blind table (robust median) — enough
    # for stage 3a's coarse db evidence windows
    shift0, n_anchor = estimate_wavelength_shift(peaks, db)

    # stage 3a — DATA-ONLY refinement: blend splits and single-merges are
    # pure model evidence; the asymmetric (self-absorption) family needs
    # resonance physics of elements actually PRESENT, so those verdicts
    # are recorded but deferred until pass 1 provides an element posterior
    refined, dec_data = refine_fit(x, y, fit, db=db, shift_nm=shift0,
                                   asymmetric="defer")
    rpeaks = refined["sorted_parameter_array"]

    # per-detector-segment shifts, from the REFINED table: the three
    # segments drift independently (measured ~25-35 pm apart on MW2-112),
    # but blind centers of split/merged bright lines are displaced by up
    # to ~150 pm, so only the refined table's medians are clean enough
    # for the estimator's significance gate to separate genuine drift
    # from fit noise (segments failing the gate keep the global shift)
    shift, _ = estimate_wavelength_shift_segments(rpeaks, db)

    ne_init, ne_bounds = _halpha_ne(rpeaks)
    idx_kwargs = dict(dbpath=dbpath)
    run_kwargs = dict(sa_doublets=True, n_calls=n_calls, verbose=verbose,
                      sa_stimulated_emission=bool(stimulated_emission))
    if ne_init is not None:
        idx_kwargs["ne_init"] = ne_init
        run_kwargs["ne_bounds"] = ne_bounds

    def _db_frame(peaks: np.ndarray) -> np.ndarray:
        # indexer matches peak centers against db positions within its
        # shift_tolerance; remove each peak's SEGMENT shift first
        out = peaks.copy()
        out[:, 1] -= shift_at(shift, out[:, 1])
        return out

    bg0 = np.asarray(fit.get("background", np.zeros_like(y)), dtype=float)

    def _amp_sigma(peaks_obs: np.ndarray) -> np.ndarray:
        # per-peak area (amplitude) noise, aligned with the peak order, so
        # the indexer can gate elements on detection significance rather
        # than a fraction of the brightest peak
        return estimate_peak_uncertainties(x, y - bg0, peaks_obs)[:, 0]

    # pass 1: establish elements (a PROVISIONAL posterior — its basin can
    # be wrong; its outputs only license seeding and condition stage 3b)
    idx1 = PeakyIndexerV3(_db_frame(rpeaks), **idx_kwargs)
    idx1._amp_sigma = _amp_sigma(rpeaks)
    res1 = idx1.run(**run_kwargs)
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
        final, records = seed_minor_lines(x, y, refined, db, established,
                                          shift_nm=shift,
                                          exclude=tuple(sa_zones))
    # element-agnostic recovery: significant positive residual peaks are
    # real lines the seeder could not predict (e.g. Fe lines when the Fe
    # stage scale fails the Boltzmann trust gate) — fit them from the
    # data alone; the pass-2 indexer then identifies them.
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
    prof_pre = analyze_peak_profiles(x, y, final)
    final, deblends = deblend_shoulders(x, y, final, prof_pre,
                                        exclude=tuple(sa_zones))
    fpeaks = final["sorted_parameter_array"]

    # pass 2: identify elements + plasma state, warm-started at pass-1 state
    idx2 = PeakyIndexerV3(_db_frame(fpeaks), dbpath=dbpath,
                          temperature_init=res1.temperature,
                          ne_init=res1.ne)
    idx2._amp_sigma = _amp_sigma(fpeaks)
    res2 = idx2.run(**run_kwargs)

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
            scales, _ = match_and_scale(fpeaks, db, confirmed, shift_nm=shift)
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
                    sup.append(wl + shift_at(shift, wl))
            supported = (np.concatenate(sup) if sup
                         else np.empty(0, dtype=float))

            work = final
            n_seed_tot = n_rec_tot = 0
            rounds = []
            collapsed_at = None
            for bar in DEEPEN_BARS:
                if not confident:
                    break
                work, corr = seed_minor_lines(
                    x, y, work, db, confident, shift_nm=shift,
                    accept_snr=bar, min_expected_snr=bar,
                    robust_elements=set(confident),
                    exclude=tuple(sa_zones))
                n_seed = sum(1 for r in corr if r.get("action") == "added")
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
                idxN = PeakyIndexerV3(
                    _db_frame(work["sorted_parameter_array"]), dbpath=dbpath,
                    temperature_init=res2.temperature, ne_init=res2.ne)
                idxN._amp_sigma = _amp_sigma(work["sorted_parameter_array"])
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
    profiles = analyze_peak_profiles(x, y, final)
    result, sa_records, sa_used = recover_sa_areas(
        fidx, result, x, y, final, profiles, exclude=tuple(sa_zones),
        premeasured=tuple(sa_merges))

    # detection report + confounder (true-negative rival) analysis; when
    # SA recovery was applied, detections see the SAME corrected
    # amplitudes the re-solved result came from
    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    area_sigma = estimate_peak_uncertainties(x, y - bg, fpeaks)[:, 0]
    amp_stash = None
    if sa_used:
        amp_stash = fidx._obs_amp.copy()
        for r in sa_records:
            if r["action"] == "sa-recovered":
                fidx._obs_amp[int(r["index"])] *= float(r["factor"])
    try:
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
        n_anchor=n_anchor, ne_init=ne_init,
        result=result, established=established,
        element_uncertainty=det["element_uncertainty"],
        detections=det["detections"], support=det["support"],
        contested=det["contested"],
        resolved_fractions=det["resolved_fractions"],
        profiles=profiles, shape_quality=shape_quality,
        corroboration=corroboration,
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
    return dict(
        file=os.path.basename(path),
        sample=sample_name(path),
        status="ok",
        n_peaks=int(analysis["final"]["sorted_parameter_array"].shape[0]),
        shift_pm=round(1000.0 * float(analysis["shift"]), 1),
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
    header = ["sample", "element", "status", "fraction", "fraction_resolved",
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
                w.writerow([row["sample"], "", row.get("status", "")]
                           + [""] * (len(header) - 3))
                n += 1
                continue
            for d in row.get("detections", []):
                fr = d.get("fraction_resolved", d.get("fraction"))
                fhi = d.get("fraction_hi", d.get("fraction"))
                # fr/fhi may be a legitimate 0.0 (an element resolved away
                # by the confounder): render it as "0", not blank
                w.writerow([
                    row["sample"], d["element"], d["status"],
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
