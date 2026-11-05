"""Overlap-conservation modes for the whole-pattern design matrix.

Covers the Ca-over / Mg-under composition bias at the TRUE plasma state that
reports/2026-11-03-debug-composition-bias-at-truth.md localised to the
non-conserving overlap map in :class:`alibz.peaky_indexer_v3.PeakyIndexerV3`
(a single database line replicated across every observed peak it touches).

The oracle feeds the forward model's own per-line areas at the exact true
state into a fresh indexer and re-solves with :meth:`solve_at`, so any residual
bias is the design matrix ``A`` versus real areas -- nothing upstream.
"""
import numpy as np
import pytest

from scipy.special import voigt_profile as voigt

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.synthetic import SyntheticSpectrumGenerator
from bench.synth_cases import make_scene, CASES

DBPATH = "db"
MODES = ("legacy", "conserve", "nearest")


def _oracle_peaks(case):
    """Forward-model per-line (area, wavelength, sigma, gamma) at truth."""
    spec = CASES[case]
    gen = SyntheticSpectrumGenerator(DBPATH)
    scene = make_scene(spec["composition"], temperature_k=spec["temperature_k"],
                       log_ne_cm3=spec["log_ne_cm3"], seed=0, dbpath=DBPATH)
    tbl = gen._component_lines(scene.target)
    peaks = np.column_stack([tbl["area_counts"], tbl["wavelength_nm"],
                             tbl["sigma_nm"], tbl["gamma_nm"]])
    med = (float(np.median(tbl["sigma_nm"])), float(np.median(tbl["gamma_nm"])))
    return spec, peaks, med


def _build(case, mode):
    spec, peaks, med = _oracle_peaks(case)
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"], overlap_mode=mode)
    idx.build_candidate_matrix(sa_doublets=True)
    return spec, idx, med


def _oracle_fractions(case, mode):
    spec, idx, (ms, mg) = _build(case, mode)
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    return res.element_fractions


# --- (1) replication: conserve caps every line's summed overlap at 1 --------
@pytest.mark.parametrize("case", ["binary_ca_mg", "silicate_4el"])
def test_conserve_caps_line_replication(case):
    spec, idx, (ms, mg) = _build(case, "conserve")
    # the resting per-fitted-width observed map ...
    obs_sum = np.asarray(idx.peak_line_map.sum(axis=0)).ravel()
    assert obs_sum.size == 0 or obs_sum.max() <= 1.0 + 1e-9
    # ... and the shared-width map the solve actually uses.
    idx._rebuild_overlap(ms, mg)
    solve_sum = np.asarray(idx.peak_line_map.sum(axis=0)).ravel()
    assert solve_sum.size == 0 or solve_sum.max() <= 1.0 + 1e-9


# --- (2) binary oracle at truth --------------------------------------------
def test_binary_oracle_at_truth():
    # legacy reproduces the documented defect (Ca 0.73 vs 0.60 truth)
    legacy = _oracle_fractions("binary_ca_mg", "legacy")
    assert legacy["Ca"] == pytest.approx(0.730, abs=0.01)
    # conserve and nearest restore the truth to within +/- 0.02
    conserve = _oracle_fractions("binary_ca_mg", "conserve")
    nearest = _oracle_fractions("binary_ca_mg", "nearest")
    assert conserve["Ca"] == pytest.approx(0.60, abs=0.02)
    assert nearest["Ca"] == pytest.approx(0.60, abs=0.02)


# --- (3) silicate oracle at truth ------------------------------------------
def test_silicate_oracle_at_truth():
    truth = {"Si": 0.40, "Ca": 0.30, "Fe": 0.20, "Mg": 0.10}
    out = {m: _oracle_fractions("silicate_4el", m) for m in MODES}
    table = {m: {el: round(out[m].get(el, 0.0), 3) for el in truth}
             for m in MODES}
    print("silicate_4el majors by mode:", table)
    con = out["conserve"]
    # Si and Ca (the spectrally clean / semi-clean majors) land within 0.05.
    assert con.get("Si", 0.0) == pytest.approx(0.40, abs=0.05)
    assert con.get("Ca", 0.0) == pytest.approx(0.30, abs=0.05)
    # DOCUMENTED REGRESSION (not a success): conserve dilutes the dense-forest
    # majors Fe/Mg below the candidate prefilter's detectability bar, so both
    # collapse to ~0 at truth rather than toward their truth fractions.  The
    # bound is intentionally loose and loud -- see
    # reports/2026-11-03-wave2-conserving-design.md; the fit-quality panel is
    # the real arbiter.  nearest keeps all four majors present (under-recovered
    # but non-zero), which the printed table records for the panel comparison.
    assert con.get("Fe", 0.0) < 0.05
    assert con.get("Mg", 0.0) < 0.05
    assert out["nearest"].get("Fe", 0.0) > 0.05


# --- (4) clean self-consistent control: obs = A . c_true -> truth ----------
@pytest.mark.parametrize("mode", MODES)
def test_clean_control_returns_truth(mode):
    spec, idx, (ms, mg) = _build("binary_ca_mg", mode)
    idx._rebuild_overlap(ms, mg)
    lw = idx._line_weights(spec["temperature_k"], spec["log_ne_cm3"])
    A = idx._build_design_matrix(lw)
    truth = spec["composition"]
    c_true = np.array([truth.get(sp.element, 0.0)
                       for sp in idx.line_table.species], dtype=float)
    idx._obs_amp[:] = A @ c_true
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    assert res.element_fractions["Ca"] == pytest.approx(0.60, abs=0.02)
    assert res.element_fractions["Mg"] == pytest.approx(0.40, abs=0.02)


# --- (5) real-spectrum smoke under conserve (no value assertion) -----------
def test_real_spectrum_smoke_conserve():
    from alibz.pipeline import analyze_spectrum, load_spectrum_csv
    x, y = load_spectrum_csv("data/remote_samples/user_spec1.csv")
    a = analyze_spectrum(x, y, DBPATH, n_calls=8, draws=2,
                         overlap_mode="conserve")
    res = a["result"]
    frac = {k: round(v, 4) for k, v in res.element_fractions.items() if v > 0.01}
    print("user_spec1 conserve composition:", frac,
          "r2:", round(res.r_squared, 4))
    assert res.r_squared == res.r_squared  # finite / not NaN


# --- (6) legacy is bit-identical to the raw Voigt-height-ratio map ----------
def test_legacy_bit_identical_to_raw_map():
    spec, peaks, _med = _oracle_peaks("binary_ca_mg")
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"], overlap_mode="legacy")
    idx.build_candidate_matrix(sa_doublets=True)
    lt = idx.line_table
    tol = (0.2 if idx._shift_tolerance is None else float(idx._shift_tolerance))
    # Recompute the raw (non-conserving) per-fitted-width overlap map exactly
    # as the historic construction did, and require bitwise equality.
    ref = np.zeros((idx.n_peaks, lt.n_lines), dtype=float)
    for i in range(idx.n_peaks):
        sigma = max(idx.peaks[i].sigma, 1e-6)
        gamma = max(idx.peaks[i].gamma, 1e-6)
        pv0 = voigt(0, sigma, gamma)
        if pv0 < 1e-30:
            continue
        diffs = np.abs(lt.wavelengths - idx._obs_wl[i])
        within = np.where(diffs <= tol)[0]
        ov = voigt(diffs[within], sigma, gamma) / pv0
        for j, v in zip(within, ov):
            if v > 1e-6:
                ref[i, j] = float(v)
    got = np.asarray(idx.peak_line_map.todense())
    assert np.array_equal(got, ref)
    # and legacy must NOT cap dense-forest replication (so it differs from
    # conserve): Mg's summed overlap exceeds 1.
    col_sum = ref.sum(axis=0)
    assert col_sum.max() > 1.0
