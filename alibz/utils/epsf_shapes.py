"""Packaged instrument ePSF shapes (SciAps Z300) with a cached loader.

The empirical line-spread function measured in ``cal/`` is shipped inside the
package as ``alibz/data/epsf-z300-20260924.json`` (see that file's
``packaging`` block for provenance).  This module loads it once, exposes the
per-segment shapes as :class:`~alibz.utils.epsf.EPSF` objects, interpolates the
sub-band shapes in wavelength (the recommended "ePSF lambda" estimator), and
carries the per-segment ``location_method`` (UV spline / VIS epsf / NIR gaussian)
so the choice of peak-location method is data-driven, never hard-coded.

Instrument scope: these shapes are for Z300 *native-grid* data only.  Callers
decide whether a spectrum is native (see :func:`alibz.utils.native_grid` /
:func:`alibz.utils.peak_location`); this module only serves the shapes.

Deterministic; numpy + scipy only.  Never raises on a missing/other-instrument
request (returns ``None``), so callers fall back gracefully.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import numpy as np

from alibz.utils.epsf import EPSF

#: Packaged shape file, relative to the package root.
DEFAULT_SHAPE_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "data",
                                  "epsf-z300-20260924.json")

SEGMENT_EDGES = (365.0, 620.0)
SEGMENT_NAMES = ("UV", "VIS", "NIR")


def segment_of(wavelength) -> str:
    """UV / VIS / NIR for a scalar wavelength [nm]."""
    return SEGMENT_NAMES[int(np.digitize(float(wavelength), SEGMENT_EDGES))]


class Z300Shapes:
    """Per-segment ePSF shapes and location flags for the Z300 native grid."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.instrument = doc.get("instrument", "")
        self._full = {}          # segment -> EPSF (segment average)
        self._band_wl = {}       # segment -> np.ndarray of sub-band centres
        self._band_coef = {}     # segment -> list[np.ndarray] of coefficients
        self._h = {}
        self._R = {}
        self._unstable = {}      # segment -> set of sub-band indices to skip
        self.location = {}       # segment -> {location_method, accuracy_pm, ...}
        self.pitch_nm = {}       # segment -> reference median pitch [nm]
        for sn in SEGMENT_NAMES:
            seg = doc["segments"][sn]
            ep = seg["epsf"]
            self._h[sn] = float(ep["h"])
            self._R[sn] = float(ep["R"])
            self._full[sn] = EPSF(np.asarray(ep["coef"], float), ep["h"], ep["R"],
                                  meta={"segment": sn, "kind": "segment_average"})
            bands = seg.get("sub_bands", [])
            self._band_wl[sn] = np.array([b["wl_center_nm"] for b in bands], float)
            self._band_coef[sn] = [np.asarray(b["epsf"]["coef"], float) for b in bands]
            self._unstable[sn] = set(seg.get("unstable_sub_bands", []))
            self.location[sn] = dict(seg.get("location", {}))
            self.pitch_nm[sn] = float(seg["reference"].get("median_pitch_nm", np.nan))

    # -- location policy --------------------------------------------------
    def method_for(self, segment: str) -> str:
        """Configured peak-location method for a segment, from the shape file's
        ``location.location_method`` ('epsf', 'gaussian', 'spline', 'parabola').

        The shipped config is UV='spline', VIS='epsf', NIR='gaussian' (NIR ePSF
        is gated for a sub-band-seam location bias; see the file's gating_note).
        A future UV/NIR ePSF is enabled by editing that one field.
        """
        loc = self.location.get(segment, {})
        return str(loc.get("location_method")
                   or loc.get("method")  # legacy key
                   or "spline")

    def uses_epsf(self, segment: str) -> bool:
        return self.method_for(segment) == "epsf"

    # -- shape access -----------------------------------------------------
    def segment_epsf(self, segment: str) -> EPSF:
        """Segment-average shape."""
        return self._full[segment]

    def epsf_at(self, wavelength: float) -> EPSF:
        """Sub-band shapes linearly interpolated (in coefficients) at ``wavelength``.

        This is the "ePSF lambda" estimator of the validation report.  Sub-bands
        flagged unstable (NIR 622-645 nm) are dropped from the interpolation, so
        a request inside/below them clamps to the nearest stable band.  Falls
        back to the segment average when no stable sub-band exists.
        """
        sn = segment_of(wavelength)
        wl = self._band_wl[sn]
        coef = self._band_coef[sn]
        keep = [j for j in range(wl.size) if j not in self._unstable[sn]]
        if not keep:
            return self._full[sn]
        wk = wl[keep]
        ck = [coef[j] for j in keep]
        wlv = float(wavelength)
        j = int(np.searchsorted(wk, wlv))
        if j <= 0:
            c = ck[0]
        elif j >= wk.size:
            c = ck[-1]
        else:
            t = (wlv - wk[j - 1]) / (wk[j] - wk[j - 1])
            c = (1.0 - t) * ck[j - 1] + t * ck[j]
        return EPSF(c, self._h[sn], self._R[sn],
                    meta={"segment": sn, "kind": "sub_band_lambda", "wl": wlv})


@lru_cache(maxsize=4)
def load_z300_shapes(path: Optional[str] = None) -> Z300Shapes:
    """Load and cache the packaged Z300 ePSF shapes.

    ``path`` defaults to the packaged file.  Cached by path; call
    ``load_z300_shapes.cache_clear()`` to force a reload.
    """
    p = path or DEFAULT_SHAPE_FILE
    with open(p) as fh:
        doc = json.load(fh)
    return Z300Shapes(doc)


__all__ = ["Z300Shapes", "load_z300_shapes", "segment_of",
           "SEGMENT_EDGES", "SEGMENT_NAMES", "DEFAULT_SHAPE_FILE"]
