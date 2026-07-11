"""Detection reporting and confounder (true-negative rival) analysis.

Turns a fitted whole-pattern result into a per-element DETECTION REPORT
that carries the evidence behind every number — statistical significance,
independent line support, near-limit upper bounds, and whether a rival
element could equally explain the support (a *confounder*).

Public entry points:

- :func:`analyze_detections` — the one-call analysis for a fitted
  ``PeakyIndexerV3`` + :class:`FitResult`; returns detections, support,
  contested-support, and per-element uncertainties.  This is what the
  pipeline and the inspection notebooks both call.
- :func:`classify_detections` / :func:`element_support` /
  :func:`contested_support` / :func:`merge_contests` — the stages, exposed
  for custom analysis.
- :func:`confounder_catalog` — aggregate the ``confounder`` column across a
  corpus into the recurring confusion pairs under its own (T, nₑ) range.

The confounder test answers "are the true negatives weighted?": a
supporting peak of element E is CONTESTED when some rival E', at the
LARGEST concentration E's own absent/weak lines (its true negatives)
permit, still covers most of the peak.  The archetype is Mn read at tens
of percent purely from the Mg II 279.5/280.3 nm region — genuine Mn at
that level would light its 403 nm triplet, which is absent, so Mn's own
cap collapses while Mg's (capped by a real Mg line) does not.
"""

import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from alibz.elements import element_sort_key

#: amplitude-resampling draws for the statistical uncertainty.
DEFAULT_DRAWS = 32
#: z (= fraction / 1-sigma unc) thresholds for the status ladder.
DETECT_Z = 3.0
MARGINAL_Z = 2.0
#: a peak "supports" an element when that element's summed (over ion
#: stages) contribution dominates the peak and covers at least this share
#: of the observed amplitude.
MIN_SUPPORT_FRACTION = 0.3
#: a supporting peak is contested when a rival element — at the LARGEST
#: concentration its own true negatives allow — still covers at least
#: this fraction of the peak's amplitude.
RIVAL_COVER_FRACTION = 0.5
#: rival template response at the peak must be at least this fraction of
#: the claiming element's response to count as a potential rival at all.
RIVAL_MIN_RESPONSE = 0.05
#: plasma-state grid the contest scans.  The fitted (T, n_e) cannot be
#: trusted for this: a wrong attribution drags the fit to a plasma state
#: where the rival's lines vanish and then "confirms" itself (measured:
#: Mn booking the Mg II 279.5 flux pulled T to 5.0 kK, where Mg is neutral
#: and Mg II cannot respond — the rival became invisible to its own
#: contest).  A peak is contested if a rival is viable at ANY
#: corpus-plausible state.
CONTEST_T_GRID = (6000.0, 8000.0, 10000.0, 12000.0)
CONTEST_LOGNE_GRID = (16.5, 17.5)
#: neutral plasma state at which the un-pruned contest candidate table is
#: built (see :func:`_run_contest`).
CONTEST_BUILD_STATE = (9000.0, 17.0)


# ---------------------------------------------------------------------------
# Statistical uncertainty (amplitude resampling)
# ---------------------------------------------------------------------------

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
    ``{"mean": .., "std": .., "p95": ..}`` over the draws.  The nonlinear
    plasma parameters are held fixed, so this measures how the composition
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


# ---------------------------------------------------------------------------
# Line support and the confounder contest
# ---------------------------------------------------------------------------

def element_support(
    indexer,
    result,
    shift: float = 0.0,
) -> Tuple[Dict[str, List[Tuple[float, float, float]]],
           Dict[str, List[int]], np.ndarray]:
    """Per-element supporting peaks from the fitted design.

    A peak supports an element when that element's summed contribution
    (across its ion stages) dominates the peak and covers at least
    :data:`MIN_SUPPORT_FRACTION` of the observed amplitude.  Aggregating
    across stages first fixes the stage-split undercount (Ca I 0.30 +
    Ca II 0.28 losing the per-species argmax to an Fe I 0.35); support
    peaks within one resolution element of each other are merged
    (strongest kept) so a phantom-split strong line cannot count twice.

    Returns ``(support, sup_idx, obs)``: ``support`` maps element ->
    ``[(contribution, wavelength_nm, observed_amp), ...]`` (observed
    frame, sorted strongest first); ``sup_idx`` maps element -> peak-row
    indices; ``obs`` is the observed amplitude per peak.
    """
    support: Dict[str, List[Tuple[float, float, float]]] = {}
    sup_idx: Dict[str, List[int]] = {}
    A = getattr(indexer, "_last_A", None)
    if A is None or not getattr(result, "concentrations", np.empty(0)).size:
        return support, sup_idx, np.empty(0)
    n_pk = len(indexer._obs_wl)
    A_pk = np.asarray(A)[:n_pk]
    contrib = A_pk * result.concentrations
    el_names = sorted({sp.element for sp in result.species})
    cols = {el: [j for j, sp in enumerate(result.species)
                 if sp.element == el] for el in el_names}
    E = np.stack([contrib[:, cols[el]].sum(axis=1) for el in el_names],
                 axis=1)
    from alibz.utils.wavelength import shift_at
    obs = np.asarray(indexer._obs_amp[:n_pk], dtype=float)
    dom = np.argmax(E, axis=1)
    for i in range(n_pk):
        el = el_names[dom[i]]
        con = float(E[i, dom[i]])
        if obs[i] > 0 and con >= MIN_SUPPORT_FRACTION * obs[i]:
            wl = float(indexer._obs_wl[i])
            wl_obs = wl + float(shift_at(shift, wl))
            support.setdefault(el, []).append((con, wl_obs, float(obs[i])))
            sup_idx.setdefault(el, []).append(i)
    for el, lines in support.items():
        lines.sort(reverse=True)
        merged: List[Tuple[float, float, float]] = []
        for ln in lines:
            if all(abs(ln[1] - m[1]) > 0.15 for m in merged):
                merged.append(ln)
        support[el] = merged
    return support, sup_idx, obs


def contested_support(
    A_pk: np.ndarray,
    cols: Dict[str, List[int]],
    el_names: List[str],
    obs: np.ndarray,
    area_sigma: np.ndarray,
    sup_idx: Dict[str, List[int]],
    A_nonres: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """True-negative-weighted contest of each element's support (one state).

    For each rival element the spectrum itself caps its concentration:
    ``c_max = min_j (obs_j + 3 sigma_j) / T_j`` over every peak position
    where the rival's template responds — the rival can be no more
    abundant than its FAINTEST expected line allows (its true negatives).
    A supporting peak of element E is CONTESTED when some rival at that
    cap still covers >= :data:`RIVAL_COVER_FRACTION` of the peak's
    amplitude: attribution between E and the rival is then a model choice,
    not evidence.  Archetype: Mg (capped by its observed Mg I 285.2 line)
    can still cover the Mg II 279.5/280.3 flux booked to Mn — contested;
    genuine 50% Mn would need its absent 403 nm triplet, so Mn's own cap
    collapses in samples that lack it.

    ``A_nonres``, when given, is the same design with resonance lines
    (Ei <= 0.2 eV) zeroed; the caps use it because resonance lines
    self-absorb and their observed amplitudes understate the element.

    Returns per-element ``{contested: set(peak_idx), rivals: {el: flux}}``
    for one plasma state; combine states with :func:`merge_contests`.
    """
    sig = np.where(np.isfinite(area_sigma), area_sigma, 0.1 * np.abs(obs))
    allow = np.abs(obs) + 3.0 * sig + 1e-12
    # per-element unit-concentration template response at each peak
    T = np.stack([A_pk[:, cols[el]].sum(axis=1) for el in el_names], axis=1)
    T_cap = T
    if A_nonres is not None:
        T_cap = np.stack([A_nonres[:, cols[el]].sum(axis=1)
                          for el in el_names], axis=1)
    # per-element concentration cap from its true negatives
    c_max = np.full(len(el_names), np.inf)
    for r in range(len(el_names)):
        resp = T_cap[:, r]
        if not np.any(resp > 0):
            resp = T[:, r]  # no non-resonance lines: fall back
        active = resp > 1e-3 * float(np.max(resp)) if np.any(resp > 0) \
            else np.zeros_like(resp, dtype=bool)
        if np.any(active):
            c_max[r] = float(np.min(allow[active] / resp[active]))
    out: Dict[str, dict] = {}
    for el, idxs in sup_idx.items():
        if el not in el_names:
            out[el] = dict(contested=set(), rivals={})
            continue
        e = el_names.index(el)
        contested_idx: set = set()
        rival_flux: Dict[str, float] = {}
        resp_ratio: Dict[str, list] = {}
        for i in idxs:
            for r, rel in enumerate(el_names):
                if rel == el or T[i, r] <= RIVAL_MIN_RESPONSE * max(
                        T[i, e], 1e-300):
                    continue
                if (np.isfinite(c_max[r]) and c_max[r] * T[i, r]
                        >= RIVAL_COVER_FRACTION * obs[i]):
                    contested_idx.add(i)
                    rival_flux[rel] = rival_flux.get(rel, 0.0) + float(obs[i])
                    # concentration the rival needs vs this element to make
                    # the same flux at this peak: T_E / T_rival — the factor
                    # that converts freed E-fraction into rival-fraction on
                    # reattribution
                    resp_ratio.setdefault(rel, []).append(
                        float(T[i, e] / T[i, r]))
        out[el] = dict(
            contested=contested_idx, rivals=rival_flux,
            resp_ratio={rel: float(np.median(v))
                        for rel, v in resp_ratio.items()})
    return out


def merge_contests(
    sup_idx: Dict[str, List[int]],
    obs: np.ndarray,
    per_state: List[Dict[str, dict]],
) -> Dict[str, dict]:
    """Merge per-plasma-state contest results (existential over states).

    A supporting peak is contested if ANY scanned (T, n_e) makes a rival
    viable; ``clear_lines`` counts peaks no state can contest.  The
    ``confounder`` is the rival contesting the most flux, maximised over
    states so a rival strong at one temperature is not diluted by states
    where it cannot respond.  Returns per-element ``{clear_lines,
    contested_share, confounder}``.
    """
    out: Dict[str, dict] = {}
    for el, idxs in sup_idx.items():
        contested_idx: set = set()
        rival_best: Dict[str, float] = {}
        resp_ratio: Dict[str, list] = {}
        for state in per_state:
            det = state.get(el)
            if not det:
                continue
            contested_idx |= det["contested"]
            for rel, fl in det["rivals"].items():
                rival_best[rel] = max(rival_best.get(rel, 0.0), fl)
            for rel, rr in det.get("resp_ratio", {}).items():
                resp_ratio.setdefault(rel, []).append(rr)
        total = sum(float(obs[i]) for i in idxs)
        contested_flux = sum(float(obs[i]) for i in contested_idx)
        confounder = (max(rival_best, key=rival_best.get)
                      if rival_best else None)
        out[el] = dict(
            clear_lines=sum(1 for i in idxs if i not in contested_idx),
            contested_share=round(contested_flux / total, 3)
            if total > 0 else 0.0,
            confounder=confounder,
            # median response ratio T_E/T_confounder over states — the
            # factor converting freed E-fraction into confounder-fraction
            resp_ratio=(float(np.median(resp_ratio[confounder]))
                        if confounder in resp_ratio else 1.0),
        )
    return out


def _run_contest(
    peaks_dbframe: np.ndarray,
    dbpath: str,
    obs: np.ndarray,
    area_sigma: np.ndarray,
    sup_idx: Dict[str, List[int]],
    build_state: Tuple[float, float] = CONTEST_BUILD_STATE,
) -> Dict[str, dict]:
    """Build the un-pruned contest design and scan the plasma grid.

    The contest must see the FULL candidate space, not the fit's
    survivors: the NNLS vertex hands collinear flux to one species and
    ``prune_and_refit`` then deletes the rival (measured: Mg II — 183
    candidate lines — pruned after Mn took the 279.5/280.1 region,
    leaving 50% "Mn" unfalsifiable inside the fitted design).  The table
    is therefore rebuilt at a NEUTRAL corpus-central plasma state, never
    the fitted one: ``build_candidate_matrix``'s Saha prefilter prunes
    species using T_init/ne_init, and a fit whose misattribution dragged
    T to 5 kK would prune the very rival the contest exists to consider.
    """
    from alibz.peaky_indexer_v3 import PeakyIndexerV3

    n_pk = len(obs)
    idx_c = PeakyIndexerV3(peaks_dbframe, dbpath=dbpath,
                           temperature_init=build_state[0],
                           ne_init=build_state[1])
    idx_c.build_candidate_matrix()
    lt = idx_c.line_table
    el_full = sorted({sp.element for sp in lt.species})
    cols_full = {el: [j for j, sp in enumerate(lt.species)
                      if sp.element == el] for el in el_full}
    per_state = []
    for T_c in CONTEST_T_GRID:
        for ne_c in CONTEST_LOGNE_GRID:
            lw = idx_c._line_weights(T_c, ne_c)
            A_full = np.asarray(idx_c._build_design_matrix(lw))[:n_pk]
            # non-resonance design for the caps (resonance lines
            # self-absorb and understate the element)
            lw_nr = np.where(lt.Ei > 0.2, lw, 0.0)
            A_nr = np.asarray(idx_c._build_design_matrix(lw_nr))[:n_pk]
            per_state.append(contested_support(
                A_full, cols_full, el_full, obs, area_sigma[:n_pk],
                sup_idx, A_nonres=A_nr))
    return merge_contests(sup_idx, obs, per_state)


# ---------------------------------------------------------------------------
# Classification and the one-call entry point
# ---------------------------------------------------------------------------

def classify_detections(
    result,
    stats: Dict[str, Dict[str, float]],
    support: Dict[str, List[Tuple[float, float, float]]],
    contested: Optional[Dict[str, dict]] = None,
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
    - ``confounded``   would be detected/single-line, but EVERY supporting
      peak could equally be a viable rival element's line (see
      :func:`contested_support`) — the attribution, and therefore the
      abundance, is a model choice between this element and the named
      ``confounder`` (measured archetype: Mn "detected" at 50% entirely
      from the Mg II 279.5/280.3 nm region);
    - ``marginal``     2 <= z < 3;
    - ``weak``         z < 2 (consistent with zero at ~95%);
    - ``upper-limit``  the best fit zeroed the element but its candidate
      lines are in the design: ``upper_limit`` = 95th percentile of the
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
            # true-negative demotion: a "detection" whose EVERY supporting
            # peak could equally be a rival's line (and the rival's own
            # predictions check out elsewhere) is an attribution choice,
            # not a detection — e.g. Mn "detected" at 50% purely from the
            # Mg II 279.5/280.3 region
            con = (contested or {}).get(el, {})
            if (status in ("detected", "single-line")
                    and con.get("clear_lines", 1) == 0
                    and con.get("contested_share", 0.0) >= 0.5):
                status = "confounded"
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
        con = (contested or {}).get(el, {})
        detections.append(dict(
            element=el, status=status, fraction=frac,
            # fraction_resolved / fraction_hi bracket the true-negative
            # attribution range; filled by resolve_confounded (default =
            # the point estimate for elements that are not confounded)
            fraction_resolved=frac, fraction_hi=frac,
            unc=(std if np.isfinite(std) else None),
            z=(round(min(z, 999.0), 1) if np.isfinite(z) else 999.0),
            n_lines=len(lines),
            clear_lines=con.get("clear_lines"),
            contested_share=con.get("contested_share"),
            confounder=con.get("confounder"),
            strongest_peak_nm=(round(strongest[1], 3) if strongest else None),
            strongest_obs=(round(strongest[2], 1) if strongest else None),
            upper_limit=upper,
            stage_disagreement=(round(float(d), 2) if np.isfinite(d)
                                else None),
        ))
    return detections


def resolve_confounded(
    detections: List[dict],
    contested: Dict[str, dict],
) -> Dict[str, float]:
    """Resolve confounded-element abundances by true-negative attribution.

    A ``confounded`` element is credited only for its CLEAR (uncontested)
    flux — the share no rival can cover.  Its contested share is reattributed
    to the ``confounder`` (the rival whose OWN lines are present), scaled by
    the response ratio ``T_E/T_rival`` that converts freed E-fraction into
    the rival fraction reproducing the same peak flux — but only up to what
    the rival can GLOBALLY host, with two guards:

    - the response ratio is clamped at 1 so a weak-emitter rival cannot
      manufacture composition mass (a rival needing >1x the freed fraction to
      cover the peak is evidence it is the wrong host, not licence to
      inflate it — otherwise a 4% confounded element can spawn a 40% phantom
      of a globally-absent rival);
    - the rival absorbs contested flux only up to its own evidence ceiling —
      unbounded if it is independently detected, its ``upper_limit`` if it
      survives only as an upper limit / weak, and ZERO if the fit pruned it
      entirely (its other lines are absent).  Whatever the rival cannot host
      RETURNS to the incumbent element: the fit's own assignment stands when
      no viable alternative exists.

    So a confounded element resolves toward 0 only when a genuinely-present
    rival can take its flux (the archetype: Mn -> Mg via the shared Mg II
    279.5/280.3 region); contested by an absent rival, it keeps its flux.

    Two normalised compositions are written back onto each detection:
    ``fraction_hi`` (the as-fit composition — for a confounded element the
    HIGH end of its attribution range) and ``fraction_resolved`` (the
    true-negative-resolved composition — the LOW end for a confounded
    element).  Both sum to 1, so they share a scale; note the bracket only
    reads low->high for the confounded element itself — its confounder
    GAINS flux, so its ``fraction_resolved`` exceeds its ``fraction_hi``,
    and bystanders rescale slightly with the renormalisation.  Returns the
    renormalised ``resolved_fractions`` composition.

    Order-independent under mutual/chained confounding: every element's
    clear-flux credit is applied before any reattribution, so a confounder
    that is itself confounded keeps its clear share AND gains the freed
    flux (a blind ``resolved[el] = keep`` after reattribution would wipe
    it).
    """
    contested = contested or {}
    records = {d["element"]: d for d in detections}
    vertex = {d["element"]: float(d["fraction"] or 0.0) for d in detections}

    def _norm(comp):
        tot = sum(comp.values())
        return {k: (v / tot if tot > 0 else 0.0) for k, v in comp.items()}

    # HIGH end: the as-fit composition, normalised onto the same scale as
    # the resolved composition below (so the two columns are comparable)
    hi = _norm(vertex)

    # LOW end: credit each confounded element only its CLEAR flux, then
    # collect the contested (freed) flux to reattribute AFTERWARDS.  Doing
    # all keeps first, all reattributions second, makes the result
    # independent of element order even when confounders are themselves
    # confounded (mutual A<->B or chained A->B->C).
    resolved = dict(vertex)
    reattributions = []
    for d in detections:
        el = d["element"]
        frac = vertex[el]
        if d["status"] != "confounded" or frac <= 0:
            continue
        cs = float(d.get("contested_share") or 0.0)
        keep = frac * (1.0 - cs)          # credit only clear flux
        resolved[el] = keep
        rival = d.get("confounder")
        freed = frac - keep
        if rival and freed > 0:
            # response ratio converts freed E-fraction into the rival
            # fraction that reproduces the SAME peak flux; clamp at 1 so a
            # weak-emitter rival cannot MANUFACTURE composition mass (needing
            # >1x the freed fraction to cover the peak is itself evidence the
            # rival is the wrong host, not licence to inflate it)
            rr = min(float(contested.get(el, {}).get("resp_ratio", 1.0)), 1.0)
            reattributions.append((el, rival, freed, freed * rr))
    for el, rival, freed, want in reattributions:
        # the rival can absorb contested flux only up to its own GLOBAL
        # evidence: unbounded if it is independently detected (its lines
        # carry it), its true-negative upper limit if it survives only as an
        # upper limit / weak, and zero if the fit pruned it entirely (its
        # other lines are absent, so it cannot host a phantom).  Whatever the
        # rival cannot host RETURNS to the incumbent element -- the fit's own
        # assignment stands when no viable alternative exists.
        rec = records.get(rival)
        if rec is None:
            headroom = 0.0
        elif rec["status"] in ("detected", "single-line"):
            headroom = float("inf")
        else:
            headroom = max(0.0, float(rec.get("upper_limit") or 0.0)
                           - resolved.get(rival, 0.0))
        give = want if headroom == float("inf") else min(want, headroom)
        rho = (give / want) if want > 0 else 0.0     # share the rival hosts
        if give > 0:                                 # never create a 0 key
            resolved[rival] = resolved.get(rival, 0.0) + give
        resolved[el] = resolved.get(el, 0.0) + freed * (1.0 - rho)
    resolved = _norm(resolved)

    for el, d in records.items():
        d["fraction_hi"] = hi.get(el, float(d["fraction"] or 0.0))
        d["fraction_resolved"] = resolved.get(el, float(d["fraction"] or 0.0))
    return resolved


def analyze_detections(
    indexer,
    result,
    area_sigma: np.ndarray,
    shift: float = 0.0,
    dbpath: str = "db",
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    contest: bool = True,
) -> dict:
    """Full detection + confounder analysis for one fitted spectrum.

    ``indexer`` is a :class:`PeakyIndexerV3` that has already ``run``
    (its ``peak_array`` is in the database frame and ``_last_A`` is set);
    ``result`` is its :class:`FitResult`; ``area_sigma`` are the fitted
    per-peak area 1-sigma uncertainties (aligned with the peak table);
    ``shift`` maps observed peak centers back to the database frame.

    Returns ``{detections, support, support_idx, contested,
    element_uncertainty, stats, resolved_fractions}`` (``support_idx``
    maps element -> peak-row indices, for joining shape/profile QC).  ``resolved_fractions`` is the
    composition after true-negative attribution (confounded elements
    credited only their clear flux, the rest reattributed to the
    confounder; see :func:`resolve_confounded`); each detection also gains
    ``fraction_resolved`` and ``fraction_hi``.  Set ``contest=False`` to
    skip the (few-second) confounder grid scan.
    """
    stats = element_uncertainty_stats(indexer, result, area_sigma,
                                      draws=draws, seed=seed)
    unc = {el: (stats[el]["std"] if el in stats else float("nan"))
           for el in result.element_fractions}
    support, sup_idx, obs = element_support(indexer, result, shift)
    contested: Dict[str, dict] = {}
    if contest and support:
        try:
            contested = _run_contest(np.asarray(indexer.peak_array),
                                     dbpath, obs, area_sigma, sup_idx)
        except Exception:  # noqa: BLE001 - contest is best-effort QC
            traceback.print_exc(file=sys.stderr)
            print("contested-support: full-candidate scan failed; "
                  "confounder columns will be empty", file=sys.stderr)
            contested = {}
    detections = classify_detections(result, stats, support,
                                     contested=contested)
    resolved_fractions = resolve_confounded(detections, contested)
    return dict(detections=detections, support=support,
                support_idx=sup_idx, contested=contested,
                element_uncertainty=unc, stats=stats,
                resolved_fractions=resolved_fractions)


def confounder_catalog(detections) -> "Counter":
    """Corpus confounder pairs ``(element, confounder)`` by frequency.

    ``detections`` is a flat iterable of detection dicts (pooled across a
    corpus) or an iterable of per-sample detection lists.  Returns a
    :class:`collections.Counter` — the recurring pairs are the operative
    confounders under the corpus's own (T, nₑ) range (e.g. ``("Mn",
    "Mg")`` on the JChristensen drill cores).
    """
    from collections import Counter
    flat = []
    for d in detections:
        if isinstance(d, dict):
            flat.append(d)
        else:
            flat.extend(d)
    return Counter((d["element"], d["confounder"]) for d in flat
                   if d.get("confounder"))
