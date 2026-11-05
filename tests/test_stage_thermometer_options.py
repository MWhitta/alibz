"""Wave-3 option C: bounded ion-stage temperature offset and LTE-gap flag
in the stage-consistency thermometer (both default OFF)."""
import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.synthetic import SyntheticSpectrumGenerator
from bench.synth_cases import make_scene, CASES
from tests.test_oracle_roundtrip import _build, _instrument_fwhm_nm

DBPATH = "db"

# recorded on the MAIN tree (before this change): binary_ca_mg, conserve,
# solve_at at the true state with the merged oracle
_BASE_TOTAL = 447015.68272867054
_BASE_PER = {'Mg': 447000.7670928985, 'Ca': 14.915635772049427}


def _solved(case="binary_ca_mg", **opts):
    spec, idx, (ms, mg) = _build(case, "conserve")
    for k, v in opts.items():
        setattr(idx, k, v)
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    return spec, idx, res


def test_off_path_bit_identical_to_main_tree():
    spec, idx, res = _solved()
    tot, per = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                   A_all=idx._last_A_all)
    assert tot == _BASE_TOTAL
    assert per == _BASE_PER
    # the core (original body) agrees exactly too
    t2, p2 = idx._stage_tie_cost_core(res.concentrations, idx._last_A,
                                      idx._last_A_all)
    assert (t2, p2) == (tot, per)
    assert 'stage_temperature_offset_k' not in res.convergence_info
    assert 'lte_flagged_species' not in res.convergence_info


def test_offset_on_single_T_oracle_is_small_and_not_worse():
    spec, idx, res = _solved()
    off, _ = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                 A_all=idx._last_A_all)
    idx._stage_temperature_offset_max_k = 2000.0
    on, _ = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                A_all=idx._last_A_all)
    print("single-T: off", off, "on", on, "delta", idx._last_stage_offset_k)
    assert idx._last_stage_offset_k <= 100.0
    assert on <= off + 1e-9


def _merge(tbl, gen):
    wl = np.asarray(tbl["wavelength_nm"], float)
    area = np.asarray(tbl["area_counts"], float)
    sig = np.asarray(tbl["sigma_nm"], float)
    gam = np.asarray(tbl["gamma_nm"], float)
    o = np.argsort(wl, kind="mergesort")
    wl, area, sig, gam = wl[o], area[o], sig[o], gam[o]
    tol = np.maximum(1e-3, 0.25 * _instrument_fwhm_nm(gen, wl))
    grp = np.cumsum(np.concatenate(
        [[False], np.diff(wl) > np.maximum(tol[:-1], tol[1:])]))
    n = int(grp[-1]) + 1
    a = np.zeros(n); ws = np.zeros(n); sg = np.zeros(n); gm = np.zeros(n)
    c = np.zeros(n); w0 = np.zeros(n); s0 = np.zeros(n); g0 = np.zeros(n)
    np.add.at(a, grp, area); np.add.at(ws, grp, area * wl)
    np.add.at(sg, grp, area * sig); np.add.at(gm, grp, area * gam)
    np.add.at(c, grp, 1.0); np.add.at(w0, grp, wl)
    np.add.at(s0, grp, sig); np.add.at(g0, grp, gam)
    pos = a > 0
    d = np.where(pos, a, 1.0)
    peaks = np.column_stack([
        a, np.where(pos, ws / d, w0 / c), np.where(pos, sg / d, s0 / c),
        np.where(pos, gm / d, g0 / c)])
    return peaks, (float(np.median(peaks[:, 2])), float(np.median(peaks[:, 3])))


def test_offset_recovers_two_temperature_scene():
    case = "binary_ca_mg"
    spec = CASES[case]
    T, ne, dT = spec["temperature_k"], spec["log_ne_cm3"], 1500.0
    gen = SyntheticSpectrumGenerator(DBPATH)
    cold = gen._component_lines(make_scene(
        spec["composition"], T, ne, seed=0, dbpath=DBPATH).target)
    hot = gen._component_lines(make_scene(
        spec["composition"], T + dT, ne, seed=0, dbpath=DBPATH).target)
    cm = np.asarray(cold["ion_stage"]) == 1
    hm = np.asarray(hot["ion_stage"]) >= 2
    tbl = {k: np.concatenate([np.asarray(cold[k])[cm], np.asarray(hot[k])[hm]])
           for k in ("wavelength_nm", "area_counts", "sigma_nm", "gamma_nm")}
    peaks, (ms, mg) = _merge(tbl, gen)
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH, temperature_init=T,
                         ne_init=ne, overlap_mode="conserve")
    keep = set(spec["composition"])
    idx.db.analysis_excluded_elements = tuple(
        e for e in idx.db.elements if e not in keep)
    idx.build_candidate_matrix(sa_doublets=True)
    res = idx.solve_at(T, ne, ms, mg)
    off, _ = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                 A_all=idx._last_A_all)
    idx._stage_temperature_offset_max_k = 3000.0
    on, _ = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                A_all=idx._last_A_all)
    print("two-T: off", off, "on", on, "delta", idx._last_stage_offset_k)
    assert on < off
    assert abs(idx._last_stage_offset_k - dT) <= 300.0


def _full_table_indexer():
    """Indexer with the UNFILTERED line table (gases included)."""
    from alibz.peaky_indexer_v3 import LineTable
    from tests.test_oracle_roundtrip import merged_oracle_peaks
    spec, peaks, _ = merged_oracle_peaks("binary_ca_mg")
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH, temperature_init=10000.0,
                         ne_init=17.0)
    idx.line_table = LineTable(idx.db, idx.sb, (180.0, 960.0),
                               max_ion_stage=2, min_gA=100.0)
    return idx


def test_lte_flag_argon_and_calcium():
    idx = _full_table_indexer()
    names = [(s.element, s.ion) for s in idx.line_table.species]
    de = idx._species_delta_e()
    a = names.index(("Ar", 1))
    assert 11.0 < de[a] < 12.5
    lo = idx._lte_flagged_mask(8000.0, 16.0)
    hi = idx._lte_flagged_mask(8000.0, 17.5)
    assert lo[a] and not hi[a]
    for ion in (1, 2):
        c = names.index(("Ca", ion))
        for T in (6000.0, 8000.0):  # 12 kK: Ca I legitimately flagged at log ne 15
            for ln in (15.0, 16.0, 17.0, 18.0):
                assert not idx._lte_flagged_mask(T, ln)[c], (ion, T, ln)


def test_lte_flag_skips_flagged_element_terms():
    spec, idx, res = _solved("binary_ca_mg", _lte_gap_flag=True)
    # force flag: at log ne = 12 everything is below critical
    idx._last_state = (10000.0, 12.0, idx._last_state[2])
    tot, per = idx._stage_tie_cost(res.concentrations, idx._last_A,
                                   A_all=idx._last_A_all)
    assert tot == 0.0 and per == {}
    assert any(s.startswith("Ca_") for s in idx._last_lte_flagged)
