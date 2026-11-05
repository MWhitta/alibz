"""Regression tests for the native-grid fit_peaks window gate / width filter.

On the native Z300 grid (per-segment pitch ~0.089/0.129/0.179 nm) a resolution-
limited line spans only 1-2 samples, so the span between neighbouring minima is
3-4 samples.  The pre-fix ``fit_peaks`` seeded a peak only inside
``if len(x_window) > 5`` with no else-branch, silently discarding 85/96/76 % of
UV/VIS/NIR native detections; the survivors were then thinned again by the
per-channel ``< key_inc/2`` width gate after the equal Gaussian/Lorentzian seed
split put both width channels below half a pixel.  The strong Fe I VIS lines of
the fresh Fe standard were lost this way, Fe I dropped out of the indexer, and
the solve railed to a hot Fe II-only basin
(reports/2026-11-04-fe-hot-basin-root-cause.md, reports/2026-11-05-native-gate-fix.md).

The fix: narrow native windows seed the detected peak directly (floored at the
instrument LSF), and the native width filter tests the COMBINED Voigt width
against the instrument FWHM instead of each channel against half a pixel.  Both
new paths are gated on ``segment_pitch_array(...).is_native``; on a uniform /
vendor-export grid the behaviour is byte-identical to the pre-fix code.
"""
import json
import os

import numpy as np
import pytest

from alibz import PeakyFinder
from alibz.utils.native_grid import segment_pitch_array

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "z300_fe_native_trimmed.json")

#: Fe I lines whose 3-4 sample windows the fix must recover.  404.58 / 438.35 /
#: 440.48 are the ones killed by BOTH the gate and the per-channel width filter;
#: 430.79 is killed by the gate alone.
_FE1_RECOVERED_NM = (404.58, 430.79, 438.35, 440.48)


@pytest.fixture(scope="module")
def native_fe():
    with open(FIXTURE) as fh:
        d = json.load(fh)
    return np.asarray(d["x"], float), np.asarray(d["y"], float)


def _centers(fit):
    spa = np.asarray(fit["sorted_parameter_array"])
    return spa[:, 1] if spa.size else np.array([])


def test_fixture_is_native(native_fe):
    x, _ = native_fe
    _pitch, _inc, is_native = segment_pitch_array(x)
    assert is_native, "fixture must span two native segments with distinct pitch"


def test_narrow_window_regime(native_fe):
    """The target lines really do have 3-4 sample windows (the failing regime)."""
    x, y = native_fe
    f = PeakyFinder.__new__(PeakyFinder)
    bg = f.find_background(x, y, segment_edges=f.DEFAULT_SEGMENT_EDGES)
    y_bg = y - bg
    seg_idx = np.searchsorted(x, np.sort(np.asarray(f.DEFAULT_SEGMENT_EDGES, float)))
    peak_idx, *_ = f.fourier_peaks(y_bg, n_sigma=0, segment_indices=seg_idx)
    seen_narrow = False
    for t in _FE1_RECOVERED_NM:
        near = peak_idx[np.abs(x[peak_idx] - t) < 0.3]
        assert near.size, f"{t} nm must be detected by filter_peaks"
        for k in near:
            _fm, _fw, _hw, nr, nl = f.peak_parameter_guess(y_bg, int(k))
            if (nr - nl) <= 5:
                seen_narrow = True
    assert seen_narrow, "expected <=5 sample windows for the Fe I lines"


def test_native_lines_reach_table(native_fe):
    """After the fix the gate+filter Fe I victims survive to the blind table."""
    x, y = native_fe
    f = PeakyFinder.__new__(PeakyFinder)
    fit = f.fit_spectrum(x, y, subtract_background=True, plot=False, n_sigma=0)
    cen = _centers(fit)
    assert cen.size, "native fit produced an empty table"
    for t in _FE1_RECOVERED_NM:
        assert np.any(np.abs(cen - t) < 0.3), (
            f"Fe I line {t} nm missing from the native blind-fit table")


def test_uniform_grid_not_rescued(native_fe):
    """Export/uniform regime: the narrow-window rescue must NOT fire.

    Placing the same intensities on a single-pitch (non-native) grid flips
    ``is_native`` to False; the narrow windows are then discarded exactly as the
    pre-fix code did, so the lines do not appear.  This pins the fix to the
    native regime and guards the byte-identical export path.
    """
    x, y = native_fe
    x_uniform = x[0] + 0.129 * np.arange(x.size)
    _pitch, _inc, is_native = segment_pitch_array(x_uniform)
    assert not is_native
    f = PeakyFinder.__new__(PeakyFinder)
    fit = f.fit_spectrum(x_uniform, y, subtract_background=True, plot=False, n_sigma=0)
    cen = _centers(fit)
    # map each target nm to its position on the uniform grid
    missing = 0
    for t in _FE1_RECOVERED_NM:
        ti = int(np.argmin(np.abs(x - t)))
        if not (cen.size and np.any(np.abs(cen - x_uniform[ti]) < 0.3)):
            missing += 1
    assert missing == len(_FE1_RECOVERED_NM), (
        "narrow-window peaks should stay discarded on a uniform grid")
