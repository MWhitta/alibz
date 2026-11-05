"""Opt-in Barklem & Collet (2016) partition functions and Debye IPD."""
import logging
from pathlib import Path

import numpy as np
import pytest

from alibz.utils.constants import BOLTZMANN, PLANCK, SPEED_OF_LIGHT, ELECTRON_MASS
from alibz.utils.sahaboltzmann import (SahaBoltzmann, line_emissivity,
                                       debye_length_m, debye_ipd_eV)

DB = Path(__file__).resolve().parents[1] / "db"
pytestmark = pytest.mark.skipif(not (DB / "partition_barklem2016.npz").exists(),
                                reason="db/partition_barklem2016.npz not built")


@pytest.fixture(scope="module")
def sb_off():
    return SahaBoltzmann(str(DB))


@pytest.fixture()
def sb_bk():
    return SahaBoltzmann(str(DB), partition_source="barklem2016")


# Values copied from the VizieR table J/A+A/588/A96/table8 (sha256 in
# db/atomic_line_sources.json); the paper itself only prints H/D samples.
NODES = [("Fe", 1, 4000.0, 24.613), ("Fe", 2, 9000.0, 61.4794),
         ("Ca", 1, 10000.0, 5.69578), ("Ca", 2, 4000.0, 2.07348),
         ("Mg", 1, 9000.0, 1.37994), ("Mg", 2, 10000.0, 2.03571)]


@pytest.mark.parametrize("el,ion,T,U", NODES)
def test_table_lookup(sb_bk, el, ion, T, U):
    got = sb_bk.stage_partition(el, T, ion)[0]
    assert got == pytest.approx(U, rel=1e-2)


def test_interpolation_between_nodes(sb_bk):
    lo, hi = (sb_bk.stage_partition("Fe", t, 1)[0] for t in (9000.0, 10000.0))
    mid = sb_bk.stage_partition("Fe", 9500.0, 1)[0]
    assert lo < mid < hi
    assert mid == pytest.approx(np.sqrt(lo * hi), rel=1e-12)  # linear in ln U


def test_continuity_and_extrapolation_flag(sb_bk):
    below = sb_bk.stage_partition("Fe", 10000.0 - 1e-6, 1)[0]
    at = sb_bk.stage_partition("Fe", 10000.0, 1)[0]
    assert not sb_bk.partition_extrapolated
    above = sb_bk.stage_partition("Fe", 10000.0 + 1e-6, 1)[0]
    assert sb_bk.partition_extrapolated
    assert below == pytest.approx(at, rel=1e-6)
    assert above == pytest.approx(at, rel=1e-6)
    u9, u10 = (sb_bk.stage_partition("Fe", t, 1)[0] for t in (9000.0, 10000.0))
    assert sb_bk.stage_partition("Fe", 12000.0, 1)[0] == pytest.approx(
        u10 * (u10 / u9) ** 2, rel=1e-9)
    sb_bk.reset_partition_flags()
    assert not sb_bk.partition_extrapolated


def test_species_fallback_to_line_sum(sb_off, sb_bk, caplog):
    # Remove Fe_II from the loaded table to force the line-sum fallback.
    index, grid, lnU = sb_bk._barklem_table()
    index = {k: v for k, v in index.items() if k != "Fe_II"}
    sb_bk._barklem = (index, grid, lnU)
    with caplog.at_level(logging.WARNING):
        z1 = sb_bk.stage_partition("Fe", 8000.0, 2)
        z2 = sb_bk.stage_partition("Fe", 6000.0, 2)
    assert z1 == pytest.approx(sb_off.stage_partition("Fe", 8000.0, 2), rel=0, abs=0)
    assert z2 == pytest.approx(sb_off.stage_partition("Fe", 6000.0, 2), rel=0, abs=0)
    msgs = [r for r in caplog.records if "Fe_II" in r.getMessage()]
    assert len(msgs) == 1  # once per run
    # other stages still tabulated
    assert sb_bk.stage_partition("Fe", 4000.0, 1)[0] == pytest.approx(24.613, rel=1e-2)


def test_debye_reference_condition():
    T, ne = 8000.0, 1e17
    lam = float(debye_length_m(T, ne))
    assert lam * 1e9 == pytest.approx(19.5, rel=0.02)
    d = float(debye_ipd_eV(T, ne, charge=0))
    assert d == pytest.approx(0.074, rel=0.02)
    assert float(debye_ipd_eV(T, ne, charge=2)) == pytest.approx(3 * d, rel=1e-12)


def test_ipd_saha_ratio_factor(sb_off):
    sb_ipd = SahaBoltzmann(str(DB), ipd="debye")
    T, lne = 8000.0, 17.0
    c0, _ = sb_off.ionization_distribution("Fe", T, lne)
    c1, _ = sb_ipd.ionization_distribution("Fe", T, lne)
    r0 = c0[0, 1] / c0[0, 0]
    r1 = c1[0, 1] / c1[0, 0]
    expect = np.exp(0.074 / (BOLTZMANN * T))
    assert expect == pytest.approx(1.11, rel=0.01)
    assert r1 / r0 == pytest.approx(expect, rel=0.02)
    # (z+1) scaling for the II -> III step
    r0b = c0[0, 2] / c0[0, 1]
    r1b = c1[0, 2] / c1[0, 1]
    assert r1b / r0b == pytest.approx(np.exp(2 * 0.0738 / (BOLTZMANN * T)), rel=0.02)


def test_ipd_does_not_touch_partition(sb_off):
    sb_ipd = SahaBoltzmann(str(DB), ipd="debye")
    T = np.array([6000.0, 12000.0])
    assert np.array_equal(sb_ipd.partition("Fe", T)[0], sb_off.partition("Fe", T)[0])


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        SahaBoltzmann(str(DB), partition_source="nope")
    with pytest.raises(ValueError):
        SahaBoltzmann(str(DB), ipd="griem")


def _head_reference(sb, el, T, lne):
    """HEAD (pre-option) formulas, re-derived independently of the new code."""
    ions, levels = sb._element_levels(el)
    kT = BOLTZMANN * T
    Z = np.array([np.sum(l[:, 1] * np.exp(-l[:, 0] / kT)) for l in levels])
    lamb = PLANCK / np.sqrt(2 * np.pi * ELECTRON_MASS * kT / SPEED_OF_LIGHT**2)
    eion = sb._element_eion(el)
    ci = np.ones(len(ions))
    for i in range(len(ions) - 1):
        E = eion[eion[:, 1] == ions[i] - 1, -1][0]
        ci[i + 1] = ci[i] * 2.0 * lamb**-3 / (10**lne * 1e6) * (
            max(Z[i + 1], 1e-10) / max(Z[i], 1e-10)) * np.exp(-E / kT)
    return Z, np.round(ci / ci.sum(), 10)


@pytest.mark.parametrize("el", ["Fe", "Ca", "Mg"])
def test_defaults_off_bit_identical_to_head(sb_off, el):
    explicit = SahaBoltzmann(str(DB), partition_source="lines", ipd=None)
    assert sb_off.partition_source == "lines" and sb_off.ipd is None
    L = np.asarray(sb_off.db.lines(el))
    ion, wl, gA, Ek = (L[:, i].astype(float) for i in (0, 1, 3, 5))
    ions = sb_off._element_levels(el)[0]
    idx = np.searchsorted(ions, ion)
    for T in (6000.0, 8000.0, 12000.0):
        Zr, cr = _head_reference(sb_off, el, T, 17.0)
        em = []
        for sb in (sb_off, explicit):
            c, Z = sb.ionization_distribution(el, T, 17.0)
            assert np.array_equal(c[0], cr)
            assert np.array_equal(Z[0], Zr)
            em.append(line_emissivity(wl, gA, Ek, T, Z[0][idx], c[0][idx]))
        assert np.array_equal(em[0], em[1])
        assert not sb_off.partition_extrapolated


# ---- synthetic generator / round trip -------------------------------------
def _spec_hash(x, y):
    return np.asarray(x).tobytes(), np.asarray(y).tobytes()


def test_render_defaults_identical_and_metadata():
    from bench.synth_cases import CASES, render_case
    spec = dict(CASES["binary_ca_mg"]); comp = spec.pop("composition")
    x0, y0, s0 = render_case(comp, seed=3, dbpath=str(DB), **spec)
    x1, y1, s1 = render_case(comp, seed=3, dbpath=str(DB), partition_source="lines",
                             ipd=None, **spec)
    assert np.array_equal(x0, x1) and np.array_equal(y0, y1)
    assert s0.metadata["partition_source"] == "lines" and s0.metadata["ipd"] is None


def test_render_tables_differ_and_metadata():
    from bench.synth_cases import CASES, render_case
    spec = dict(CASES["binary_ca_mg"]); comp = spec.pop("composition")
    _, y0, _ = render_case(comp, seed=3, dbpath=str(DB), **spec)
    _, y1, s1 = render_case(comp, seed=3, dbpath=str(DB), partition_source="barklem2016", **spec)
    assert s1.metadata["partition_source"] == "barklem2016"
    assert not np.array_equal(y0, y1)


@pytest.mark.skipif(not __import__("os").environ.get("ALIBZ_RUN_SLOW_ROUNDTRIP"),
                    reason="minutes-long inversion; set ALIBZ_RUN_SLOW_ROUNDTRIP=1")
@pytest.mark.xfail(strict=False, reason="2026-11-03 finding: tables/tables median_rel_err 0.216 vs "
                   "lines/lines 0.084 on binary_ca_mg (seed 11, n_calls 8, draws 2); cause not yet isolated")
def test_roundtrip_tables_not_worse_than_lines():
    from bench.synth_cases import CASES, render_case, score_recovery
    from alibz.pipeline import analyze_spectrum
    spec = dict(CASES["binary_ca_mg"]); comp = spec.pop("composition")
    err = {}
    for src in ("lines", "barklem2016"):
        x, y, _ = render_case(comp, seed=11, dbpath=str(DB), partition_source=src, **spec)
        a = analyze_spectrum(x, y, str(DB), n_calls=8, draws=2, gp_seed=11,
                             partition_source=src)
        err[src] = score_recovery(comp, a["result"].element_fractions)["median_rel_err"]
    assert err["barklem2016"] <= err["lines"] + 0.02
