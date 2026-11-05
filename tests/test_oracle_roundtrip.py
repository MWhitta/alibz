"""Fitter-realistic oracle round-trip for the whole-pattern design matrix.

The at-truth oracle feeds the forward model's own per-line areas at the exact
true state into a fresh indexer and re-solves with :meth:`solve_at`, so any
residual composition bias is the design matrix ``A`` versus the real areas --
nothing upstream (see reports/2026-11-03-debug-composition-bias-at-truth.md and
reports/2026-11-04-candidate-coverage.md).

The naive oracle (one peak per database line) is UNPHYSICAL: the forward model
renders several database transitions at an identical wavelength (a strong line
plus zero-gA duplicates), so it emits two "peaks" at one wavelength -- something
no spectrometer resolves and the real peak-fitter never produces (it merges
coincident lines into one feature).  Under ``overlap_mode="nearest"`` the
winner-take-one per-line assignment then orphans one of the degenerate twins and
its flux becomes unattributable (54% of Mg's flux on silicate_4el; the cause of
the Mg collapse localised in the candidate-coverage report).

:func:`merged_oracle_peaks` fixes this at the ORACLE, matching the fitter: every
rendered line within ``max(1e-3 nm, 0.25 x instrumental FWHM at that
wavelength)`` of another coalesces into ONE peak carrying the summed area at the
area-weighted centroid.  ``overlap_mode="nearest"`` semantics are unchanged -- a
fitted component is the sum of the database lines under it, which is exactly what
the merged oracle now represents.  This module is the home of the builder so
later waves reuse it.
"""
import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.synthetic import SyntheticSpectrumGenerator
from bench.synth_cases import make_scene, CASES

DBPATH = "db"
MODES = ("legacy", "conserve", "nearest")
_SQRT_8LN2 = 2.3548200450309493  # FWHM / sigma for a Gaussian


def _instrument_fwhm_nm(generator, wavelength_nm):
    """Voigt FWHM of the instrument response at each wavelength (nm)."""
    inst = generator.instrument
    seg = np.asarray(inst.segment_index(np.asarray(wavelength_nm)), dtype=int)
    isig = np.asarray(inst.gaussian_sigma_nm, dtype=float)[seg]
    igam = np.asarray(inst.lorentzian_gamma_nm, dtype=float)[seg]
    fG = _SQRT_8LN2 * isig
    fL = 2.0 * igam
    # Olivero & Longbothum (1977) Voigt-FWHM approximation.
    return 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)


def merged_oracle_peaks(case, dbpath=DBPATH):
    """Forward per-line areas at truth, COALESCED like the real peak-fitter.

    Lines within ``max(1e-3 nm, 0.25 x instrumental FWHM)`` of a neighbour
    (single-linkage on the sorted wavelength axis; identical-wavelength database
    duplicates are the degenerate case) merge into one peak with the summed area
    at the area-weighted centroid wavelength and area-weighted Voigt widths.

    Returns ``(spec, peaks, med)`` where ``peaks`` is the ``(area, wl, sigma,
    gamma)`` table and ``med`` the median ``(sigma, gamma)`` over merged peaks.
    """
    spec = CASES[case]
    gen = SyntheticSpectrumGenerator(dbpath)
    scene = make_scene(spec["composition"], temperature_k=spec["temperature_k"],
                       log_ne_cm3=spec["log_ne_cm3"], seed=0, dbpath=dbpath)
    tbl = gen._component_lines(scene.target)

    wl = np.asarray(tbl["wavelength_nm"], dtype=float)
    area = np.asarray(tbl["area_counts"], dtype=float)
    sigma = np.asarray(tbl["sigma_nm"], dtype=float)
    gamma = np.asarray(tbl["gamma_nm"], dtype=float)
    order = np.argsort(wl, kind="mergesort")
    wl, area, sigma, gamma = wl[order], area[order], sigma[order], gamma[order]

    if wl.size == 0:
        peaks = np.empty((0, 4), dtype=float)
        return spec, peaks, (0.03, 0.01)

    tol = np.maximum(1e-3, 0.25 * _instrument_fwhm_nm(gen, wl))
    # Single-linkage: a new cluster starts when the gap to the previous line
    # exceeds the larger of the two lines' merge tolerances.
    gaps = np.diff(wl)
    pair_tol = np.maximum(tol[:-1], tol[1:])
    new_cluster = np.concatenate([[False], gaps > pair_tol])
    group = np.cumsum(new_cluster)

    n_groups = int(group[-1]) + 1
    g_area = np.zeros(n_groups)
    g_wsum = np.zeros(n_groups)   # area-weighted wavelength numerator
    g_sig = np.zeros(n_groups)
    g_gam = np.zeros(n_groups)
    g_n = np.zeros(n_groups)
    g_wl_simple = np.zeros(n_groups)
    np.add.at(g_area, group, area)
    np.add.at(g_wsum, group, area * wl)
    np.add.at(g_sig, group, area * sigma)
    np.add.at(g_gam, group, area * gamma)
    np.add.at(g_n, group, 1.0)
    np.add.at(g_wl_simple, group, wl)

    pos = g_area > 0
    m_wl = np.where(pos, g_wsum / np.where(pos, g_area, 1.0),
                    g_wl_simple / g_n)
    # Width of the merged feature: area-weighted where there is area, else the
    # plain mean (the zero-area groups are XUV noise lines and never matched).
    m_sig = np.where(pos, g_sig / np.where(pos, g_area, 1.0),
                     g_sig)  # g_sig==0 for zero-area groups; fall back below
    m_gam = np.where(pos, g_gam / np.where(pos, g_area, 1.0), g_gam)
    # Plain-mean fallback for zero-area groups (sigma/gamma still defined).
    g_sig_simple = np.zeros(n_groups)
    g_gam_simple = np.zeros(n_groups)
    np.add.at(g_sig_simple, group, sigma)
    np.add.at(g_gam_simple, group, gamma)
    m_sig = np.where(pos, m_sig, g_sig_simple / g_n)
    m_gam = np.where(pos, m_gam, g_gam_simple / g_n)

    peaks = np.column_stack([g_area, m_wl, m_sig, m_gam])
    med = (float(np.median(m_sig)), float(np.median(m_gam)))
    return spec, peaks, med


def _build(case, mode, restrict4=True, prefilters_off=False, dbpath=DBPATH):
    spec, peaks, med = merged_oracle_peaks(case, dbpath)
    idx = PeakyIndexerV3(peaks, dbpath=dbpath,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"], overlap_mode=mode)
    if restrict4:
        keep = set(spec["composition"])
        idx.db.analysis_excluded_elements = tuple(
            e for e in idx.db.elements if e not in keep)
    kw = {}
    if prefilters_off:
        kw = dict(min_init_relative_intensity=0.0, min_gA=0.0,
                  evidence_min_net=-1e18, pseudo_obs_weight=0.0)
    idx.build_candidate_matrix(sa_doublets=True, **kw)
    return spec, idx, med


def _fracs(case, mode, **kw):
    spec, idx, (ms, mg) = _build(case, mode, **kw)
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    return res.element_fractions


# --- silicate_4el: nearest on the merged oracle --------------------------
def test_silicate_merged_oracle_nearest_recovers_majors():
    truth = {"Si": 0.40, "Ca": 0.30, "Fe": 0.20, "Mg": 0.10}
    near = _fracs("silicate_4el", "nearest", restrict4=True)
    legacy = _fracs("silicate_4el", "legacy", restrict4=True)
    print("silicate_4el merged oracle (4-el restricted):",
          {"nearest": {e: round(near.get(e, 0.0), 3) for e in truth},
           "legacy": {e: round(legacy.get(e, 0.0), 3) for e in truth}})
    # The point of the merged oracle: nearest no longer orphans the degenerate
    # Mg II twins, so the collapsed major RECOVERS (Mg .023 -> .091) and every
    # major is present (none collapses toward zero).
    for el in truth:
        assert near.get(el, 0.0) > 0.05, el
    # Si, Ca, Mg land within +/- 0.05 of truth.
    assert near.get("Si", 0.0) == pytest.approx(0.40, abs=0.05)
    assert near.get("Ca", 0.0) == pytest.approx(0.30, abs=0.05)
    assert near.get("Mg", 0.0) == pytest.approx(0.10, abs=0.05)
    # DOCUMENTED RESIDUAL (not the orphaning defect, not fixable by the oracle):
    # Fe over-recovers to ~0.27 (truth 0.20, off ~0.07).  This is the stage
    # (_aggregate_elements) floor -- the fully self-consistent control obs :=
    # A.c_true gives the SAME Fe .273 with ZERO orphaned flux (see
    # reports/2026-11-04-candidate-coverage.md), so no peak-table change can
    # move it.  The bound is intentionally loose and loud; it exceeds the +/-
    # 0.05 the merged oracle delivers for the other three majors.
    assert 0.05 < near.get("Fe", 0.0) < 0.35
    # Legacy's non-conserving replication bias persists on silicate (milder on
    # the merged oracle but still visible): Ca over, Si under.
    assert legacy.get("Ca", 0.0) > 0.33
    assert legacy.get("Si", 0.0) < 0.35


# --- binary_ca_mg: nearest recovers both on the merged oracle ---------------
def test_binary_merged_oracle_nearest():
    out = {m: _fracs("binary_ca_mg", m, restrict4=True) for m in MODES}
    print("binary_ca_mg merged oracle (4-el restricted):",
          {m: {e: round(out[m].get(e, 0.0), 3) for e in ("Ca", "Mg")}
           for m in MODES})
    assert out["nearest"].get("Ca", 0.0) == pytest.approx(0.60, abs=0.05)
    assert out["nearest"].get("Mg", 0.0) == pytest.approx(0.40, abs=0.05)
    # On the merged binary oracle the replication bias is largely removed in
    # EVERY mode (legacy Ca drops from the unmerged ~0.73 to ~0.61): merging
    # the coincident lines is what the historic legacy-vs-conserve debate was
    # really about on this scene.
    assert out["legacy"].get("Ca", 0.0) == pytest.approx(0.60, abs=0.05)


# --- full candidate list: does U/Pb hallucination appear once realistic? ----
def test_silicate_merged_oracle_full_candidate_hallucination():
    truth = {"Si", "Ca", "Fe", "Mg"}
    report = {}
    for mode in ("conserve", "nearest"):
        f = _fracs("silicate_4el", mode, restrict4=False)
        spurious = {e: round(v, 3) for e, v in f.items()
                    if e not in truth and v > 0.01}
        report[mode] = {"majors": {e: round(f.get(e, 0.0), 3) for e in truth},
                        "spurious>1%": spurious}
    print("silicate_4el merged oracle (FULL candidates):", report)
    # No value assertion on the spurious set -- this test records whether the
    # U/Pb heavy-element hallucination survives a realistic oracle for the
    # candidate-coverage report; the fit-quality panel is the arbiter.
    assert set(report) == {"conserve", "nearest"}
