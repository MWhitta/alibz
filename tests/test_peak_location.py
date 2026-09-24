"""Tests for the per-segment peak-location dispatcher, ePSF loader and the
per-segment local-pitch helper (native-grid audit A0/B1)."""

import os

import numpy as np
import pytest

from alibz.utils.epsf_shapes import load_z300_shapes, segment_of
from alibz.utils.native_grid import segment_pitch_array, local_pitch_nm
from alibz.utils import peak_location as PL
from alibz.utils import peakfit

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "z300_peakloc_windows.npz")

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _native_grid():
    """A synthetic three-segment grid with Z300-like per-segment pitch."""
    xuv = np.arange(200.0, 365.0, 0.086)
    xvis = np.arange(365.0, 620.0, 0.138)
    xnir = np.arange(620.0, 950.0, 0.190)
    return np.concatenate([xuv, xvis, xnir])


def _epsf_line(shapes, seg_wl, c_true_u, u0, n, area=1000.0, bg=5.0):
    """Sample a pixel-integrated ePSF line at integer pixels u0..u0+n-1."""
    ep = shapes.epsf_at(seg_wl)
    u = np.arange(u0, u0 + n, dtype=float)
    y = area * ep(u - c_true_u) + bg
    return u, y


# ---------------------------------------------------------------------------
# per-segment pitch
# ---------------------------------------------------------------------------
def test_pitch_uniform_is_byte_identical():
    x = np.linspace(200.0, 900.0, 14001)
    pitch, inc, native = segment_pitch_array(x)
    assert not native
    assert inc == float(np.median(np.diff(x)))
    assert np.all(pitch == inc)                # exactly the legacy scalar
    assert local_pitch_nm(x) == inc
    assert local_pitch_nm(x, 5000) == inc


def test_pitch_native_is_per_segment():
    x = _native_grid()
    pitch, inc, native = segment_pitch_array(x)
    assert native
    assert local_pitch_nm(x, np.argmin(np.abs(x - 300.0))) == pytest.approx(0.086, abs=3e-3)
    assert local_pitch_nm(x, np.argmin(np.abs(x - 500.0))) == pytest.approx(0.138, abs=3e-3)
    assert local_pitch_nm(x, np.argmin(np.abs(x - 800.0))) == pytest.approx(0.190, abs=3e-3)
    # global median is dragged toward VIS (the wide middle segment)
    assert not np.isclose(pitch[np.argmin(np.abs(x - 800.0))], inc, atol=1e-3)


def test_pitch_dead_channel_excluded():
    x = _native_grid()
    xd = np.concatenate([x, np.arange(955.0, 961.0, 0.003)])  # old dead 4th channel
    pitch, inc, native = segment_pitch_array(xd)
    # NIR pitch is recovered, not dragged toward the tiny dead-channel step
    i = np.argmin(np.abs(xd - 800.0))
    assert pitch[i] == pytest.approx(0.190, abs=3e-3)


# ---------------------------------------------------------------------------
# instrument detection
# ---------------------------------------------------------------------------
def test_is_z300_native():
    assert PL.is_z300_native(_native_grid())
    assert not PL.is_z300_native(np.arange(200.0, 900.0, 1.0 / 30))     # vendor export
    assert not PL.is_z300_native(np.linspace(380.0, 420.0, 400))        # uniform synthetic
    assert not PL.is_z300_native(np.arange(5))                          # too short


# ---------------------------------------------------------------------------
# dispatcher flags + forced methods
# ---------------------------------------------------------------------------
def test_shape_flags_drive_method():
    shapes = load_z300_shapes()
    # shipped config: UV=spline, VIS=epsf, NIR=gaussian (NIR ePSF gated)
    assert shapes.method_for("UV") == "spline"
    assert shapes.method_for("VIS") == "epsf" and shapes.uses_epsf("VIS")
    assert shapes.method_for("NIR") == "gaussian" and not shapes.uses_epsf("NIR")


def test_auto_method_per_segment_on_native():
    """Each segment dispatches as configured in the shape file."""
    shapes = load_z300_shapes()
    x = _native_grid()
    y = np.full(x.size, 5.0)
    # inject an ePSF line in each of UV / VIS / NIR
    picks = {}
    for wl in (300.0, 500.0, 800.0):
        i0 = int(np.argmin(np.abs(x - wl)))
        u, yl = _epsf_line(shapes, wl, i0 + 0.3, i0 - 8, 17)
        y[i0 - 8:i0 - 8 + 17] = yl
        picks[wl] = i0
    idx = [picks[300.0], picks[500.0], picks[800.0]]
    c, s, methods = PL.locate(x, y, idx)          # auto
    assert methods == ["spline", "epsf", "gaussian"]   # UV / VIS / NIR per config
    assert all(np.isfinite(c)) and all(np.isfinite(s))


def test_forced_methods():
    shapes = load_z300_shapes()
    x = _native_grid()
    y = np.full(x.size, 5.0)
    i0 = int(np.argmin(np.abs(x - 500.0)))
    u, yl = _epsf_line(shapes, 500.0, i0 + 0.25, i0 - 8, 17)
    y[i0 - 8:i0 - 8 + 17] = yl
    for m in ("epsf", "spline", "parabola", "gaussian"):
        c, s, meth = PL.locate(x, y, [i0], method=m, instrument="z300")
        assert meth[0].startswith(m) or meth[0] == m
        assert np.isfinite(c[0])
    with pytest.raises(ValueError):
        PL.locate(x, y, [i0], method="nope")


def test_epsf_recovers_synthetic_subpixel_truth():
    """Known ePSF line per segment -> the ePSF centre lands within ~2 pm."""
    shapes = load_z300_shapes()
    x = _native_grid()
    for wl, tol_pm in (("VIS", 3.0), ("NIR", 4.0)):
        seg_wl = {"VIS": 500.0, "NIR": 800.0}[wl]
        i0 = int(np.argmin(np.abs(x - seg_wl)))
        pitch = float(segment_pitch_array(x)[0][i0])
        for phase in (-0.3, 0.0, 0.35):
            y = np.full(x.size, 5.0)
            u, yl = _epsf_line(shapes, seg_wl, i0 + phase, i0 - 9, 19)
            y[i0 - 9:i0 - 9 + 19] = yl
            c, s, meth = PL.locate(x, y, [i0], method="epsf", instrument="z300")
            c_true = float(np.interp(i0 + phase, np.arange(x.size), x))
            assert meth[0] == "epsf"
            assert abs(c[0] - c_true) * 1000 < tol_pm
            assert np.isfinite(s[0])


def test_epsf_falls_back_on_blend():
    """A close pair that the bounded ePSF cannot resolve falls back and is
    flagged in method_used, never raising."""
    shapes = load_z300_shapes()
    x = _native_grid()
    i0 = int(np.argmin(np.abs(x - 500.0)))
    y = np.full(x.size, 5.0)
    ep = shapes.epsf_at(500.0)
    u = np.arange(i0 - 9, i0 + 10, dtype=float)
    # two lines 0.6 px apart -> merged bump inside the centre bound
    y[i0 - 9:i0 + 10] = 800 * ep(u - (i0 - 0.3)) + 800 * ep(u - (i0 + 0.3)) + 5.0
    c, s, meth = PL.locate(x, y, [i0], method="epsf", instrument="z300")
    assert np.isfinite(c[0])
    assert meth[0] in ("epsf", "spline(epsf_fallback)", "gaussian(epsf_fallback)")


# ---------------------------------------------------------------------------
# non-Z300 inputs: byte-identical delegation to the existing path
# ---------------------------------------------------------------------------
def test_non_z300_locate_records_delegates_byte_identical():
    x = np.arange(200.0, 900.0, 1.0 / 30)
    y = np.full(x.size, 3.0)
    for wl in (300.0, 500.0, 800.0):
        i = int(np.argmin(np.abs(x - wl)))
        y[i - 3:i + 4] += 100 * np.exp(-0.5 * ((x[i - 3:i + 4] - x[i]) / 0.05) ** 2)
    idx = [int(np.argmin(np.abs(x - wl))) for wl in (300.0, 500.0, 800.0)]
    got = PL.locate_records(x, y, idx)
    ref = peakfit.refine_peaks(x, y, idx)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.get("ok") == r.get("ok")
        if g.get("ok"):
            assert g["center_nm"] == r["center_nm"]        # byte-identical


def test_non_z300_locate_one_record_delegates():
    x = np.arange(200.0, 900.0, 1.0 / 30)
    y = np.full(x.size, 3.0)
    i = int(np.argmin(np.abs(x - 500.0)))
    y[i - 3:i + 4] += 100 * np.exp(-0.5 * ((x[i - 3:i + 4] - x[i]) / 0.05) ** 2)
    g = PL.locate_one_record(x, y, i, sigma_px=peakfit.instrument_sigma_px(500.0))
    r = peakfit.gaussian_subpixel_center(x, y, i, sigma_px=peakfit.instrument_sigma_px(500.0))
    assert g.get("ok") == r.get("ok")
    if g.get("ok"):
        assert g["center_nm"] == r["center_nm"]


def test_locate_non_z300_uses_spline():
    x = np.arange(200.0, 900.0, 1.0 / 30)
    y = np.full(x.size, 3.0)
    i = int(np.argmin(np.abs(x - 500.0)))
    y[i - 3:i + 4] += 100 * np.exp(-0.5 * ((x[i - 3:i + 4] - x[i]) / 0.05) ** 2)
    c, s, meth = PL.locate(x, y, [i])          # auto, non-native
    assert meth[0] == "spline"


# ---------------------------------------------------------------------------
# real-data regression (trimmed corrected-run fixture)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture missing")
def test_real_data_regression_vis_fe_and_nir_ar():
    d = np.load(FIXTURE, allow_pickle=True)
    shapes = load_z300_shapes()
    expect = {"VIS": "epsf", "NIR": "gaussian"}   # shipped per-segment config
    for seg, xk, yk, dbk in (("VIS", "x_vis", "y_vis", "db_vis"),
                             ("NIR", "x_nir", "y_nir", "db_nir")):
        x = np.asarray(d[xk], float)
        y = np.asarray(d[yk], float)
        db = float(d[dbk])
        i = int(np.argmax(y))
        # a trimmed window cannot be auto-detected as native; force z300.
        c, s, meth = PL.locate(x, y, [i], instrument="z300")
        cg, sg, mg = PL.locate(x, y, [i], method="gaussian", instrument="z300")
        assert meth[0] == expect[seg], (seg, meth)
        # lands near the db line (registration residual is sub-100 pm)
        assert abs(c[0] - db) < 0.1, (seg, c[0], db)
        # configured method and the Gaussian agree on a clean isolated line
        assert abs(c[0] - cg[0]) < 0.03, (seg, c[0], cg[0])
        assert np.isfinite(s[0]) and s[0] < 0.05
