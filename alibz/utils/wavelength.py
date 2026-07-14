"""Vacuum-air wavelength conversion.

The pickled NIST line lists store Ritz VACUUM wavelengths, while LIBS
spectrometers are calibrated to standard-air wavelengths (the ASD
convention: air above 200 nm, vacuum below).  Left unconverted, the
mismatch is the Edlen dispersion difference — 0.11 nm at 400 nm rising to
0.21 nm at 770 nm — which exceeds the indexer's matching tolerance
everywhere and makes observed peaks silently match wrong lines.
"""

import numpy as np


def vacuum_to_air(wavelength_vac_nm):
    """Convert Ritz vacuum wavelengths [nm] to standard air.

    Uses the Edlen (1966) dispersion of standard air as adopted by the
    NIST Atomic Spectra Database:

        n - 1 = 1e-8 * (8342.13 + 2406030/(130 - s^2) + 15997/(38.9 - s^2))

    with ``s = 1/lambda_vac`` in inverse micrometres.  Wavelengths below
    200 nm are returned unchanged: the ASD (and instrument-calibration)
    convention quotes vacuum wavelengths in the VUV, where the formula is
    not valid and air is opaque anyway.
    """
    wl = np.asarray(wavelength_vac_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_sq = (1.0e3 / wl) ** 2
        n = 1.0 + 1.0e-8 * (
            8342.13
            + 2406030.0 / (130.0 - sigma_sq)
            + 15997.0 / (38.9 - sigma_sq)
        )
        converted = wl / n
    return np.where(wl >= 200.0, converted, wl)


def air_to_vacuum(wavelength_air_nm, iterations=6):
    """Convert standard-air wavelengths [nm] to vacuum wavelengths.

    This is the numerical inverse of :func:`vacuum_to_air`.  The Edlen
    refractive index depends on the *vacuum* wavelength, so a short fixed-point
    iteration is used.  Six iterations converges far below the precision of
    the bundled atomic line lists.

    Values below the standard-air image of the 200 nm vacuum boundary are
    returned unchanged, matching the NIST ASD and Kurucz convention used by
    the import pipeline.
    """
    air = np.asarray(wavelength_air_nm, dtype=float)
    vac = air.copy()
    # Vacuum 200.0 nm is reported as ~199.935 nm in standard air.  Testing
    # against a literal 200 nm air value would make the boundary non-invertible.
    conversion_boundary_air = float(vacuum_to_air(200.0))
    for _ in range(int(iterations)):
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_sq = (1.0e3 / vac) ** 2
            n = 1.0 + 1.0e-8 * (
                8342.13
                + 2406030.0 / (130.0 - sigma_sq)
                + 15997.0 / (38.9 - sigma_sq)
            )
        vac = np.where(air >= conversion_boundary_air, air * n, air)
    return vac


#: fewest unambiguous anchor matches for a trustworthy median shift.
MIN_SHIFT_MATCHES = 5


def _anchor_catalog(db, min_gA=1e6):
    """Locally-dominant bright-line catalog for shift estimation.

    The raw database is too dense for nearest-neighbour matching (a
    shifted peak's nearest line is usually a DIFFERENT line near the
    shifted position, collapsing the median to zero).  Keep only lines
    that DOMINATE their +-0.3 nm neighbourhood in Boltzmann-weighted
    strength — the classic calibration-line property.
    """
    kT = 0.9  # eV; only used to rank line strengths locally
    wl_all, s_all = [], []
    for el in db.elements:
        if el in db.no_lines:
            continue
        if el in getattr(db, "strength_uncertain_elements", ()):
            continue
        arr = db.lines(el)
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ek = arr[:, 5].astype(float)
        keep = (ion <= 2) & (gA >= min_gA) & (wl > 180.0) & (wl < 1000.0)
        wl_all.append(wl[keep])
        s_all.append(gA[keep] * np.exp(-Ek[keep] / kT))
    wl_all = np.concatenate(wl_all)
    s_all = np.concatenate(s_all)
    order = np.argsort(wl_all)
    wl_all, s_all = wl_all[order], s_all[order]

    # Merge components closer than an instrument linewidth (unresolved
    # multiplets such as the Li I 670.776/670.791 doublet act as ONE
    # observable feature and must not veto each other in the dominance
    # test): strength-weighted centroids of clusters with <60 pm gaps.
    merged_wl, merged_s = [], []
    i = 0
    while i < wl_all.size:
        j = i + 1
        while j < wl_all.size and wl_all[j] - wl_all[j - 1] < 0.06:
            j += 1
        s_c = s_all[i:j]
        merged_wl.append(float(np.sum(wl_all[i:j] * s_c) / np.sum(s_c)))
        merged_s.append(float(np.sum(s_c)))
        i = j
    wl_all = np.asarray(merged_wl)
    s_all = np.asarray(merged_s)

    anchors = []
    for i in range(wl_all.size):
        lo = np.searchsorted(wl_all, wl_all[i] - 0.3)
        hi = np.searchsorted(wl_all, wl_all[i] + 0.3)
        neighbours = np.concatenate([s_all[lo:i], s_all[i + 1 : hi]])
        if neighbours.size == 0 or s_all[i] >= 10.0 * np.max(neighbours):
            anchors.append(wl_all[i])
    return np.asarray(anchors)


def _match_deltas(peaks, anchor_wl, tolerance_nm, n_peaks, ambiguity_ratio):
    """(fitted - database) offsets of unambiguously matched bright peaks.

    Returns ``(deltas, matched_wl)`` — matched_wl are the peak centers,
    so callers can group the offsets by detector segment.
    """
    deltas, where = [], []
    strongest = np.argsort(peaks[:, 0])[::-1][: int(n_peaks)]
    for mu in peaks[strongest, 1]:
        j = np.searchsorted(anchor_wl, mu)
        cand = anchor_wl[max(j - 2, 0) : j + 2]
        if cand.size == 0:
            continue
        d = np.abs(cand - mu)
        k = int(np.argmin(d))
        if d[k] > tolerance_nm:
            continue
        d_sorted = np.sort(d)
        if d_sorted.size > 1 and d_sorted[1] < ambiguity_ratio * max(d_sorted[0], 1e-6):
            continue
        deltas.append(mu - cand[k])
        where.append(mu)
    return np.asarray(deltas), np.asarray(where)


def estimate_wavelength_shift(
    peak_array,
    db,
    tolerance_nm=0.15,
    n_peaks=40,
    min_gA=1e6,
    ambiguity_ratio=3.0,
):
    """Systematic instrument wavelength shift from bright matched lines.

    Spectrometers calibrated against a standard (e.g. a steel shot) drift
    with instrument temperature, producing a systematic line shift that
    must be removed before database matching (measured ~-28 pm on
    MW2-112 data).  For the ``n_peaks`` strongest peaks, the nearest
    database line within ``tolerance_nm`` is taken as the match when it
    is UNAMBIGUOUS (the second-nearest line is at least
    ``ambiguity_ratio`` times farther); the shift is the median of the
    matched (fitted - database) differences, robust to the occasional
    misassignment.

    Returns ``(shift_nm, n_matches)``; subtract ``shift_nm`` from the
    peak centers (or add it to the database positions).  Returns
    ``(0.0, 0)`` when fewer than ``MIN_SHIFT_MATCHES`` unambiguous
    matches exist.
    """
    peaks = np.atleast_2d(np.asarray(peak_array, dtype=float))
    if peaks.size == 0:
        return 0.0, 0
    anchor_wl = _anchor_catalog(db, min_gA=min_gA)
    if anchor_wl.size == 0:
        return 0.0, 0
    deltas, _ = _match_deltas(peaks, anchor_wl, tolerance_nm, n_peaks,
                              ambiguity_ratio)
    if deltas.size < MIN_SHIFT_MATCHES:
        return 0.0, int(deltas.size)
    return float(np.median(deltas)), int(deltas.size)


class SegmentShift:
    """Per-detector-segment instrument wavelength shift.

    The three detector segments are independently calibrated and drift
    independently (measured on MW2-112: segments differ by ~33 pm while
    the pooled global shift splits the difference, leaving 15-25 pm of
    systematic matching error in two of the three segments).  Convention
    matches :func:`estimate_wavelength_shift`: observed = db + shift.

    ``at(wl)`` evaluates the shift at wavelength(s) ``wl`` — either frame
    is acceptable, the shifts (tens of pm) are negligible against the
    segment widths.  ``float()`` returns the pooled global shift, so
    summary consumers keep working.
    """

    __slots__ = ("edges", "shifts", "global_shift", "n_matches", "applied")

    def __init__(self, edges, shifts, global_shift, n_matches, applied=None):
        self.edges = tuple(float(e) for e in edges)
        self.shifts = np.asarray(shifts, dtype=float)
        self.global_shift = float(global_shift)
        self.n_matches = tuple(int(n) for n in n_matches)
        self.applied = (tuple(bool(a) for a in applied)
                        if applied is not None
                        else (False,) * len(self.shifts))

    def at(self, wl):
        """Shift [nm] at wavelength(s) ``wl`` (scalar in, scalar out)."""
        arr = np.asarray(wl, dtype=float)
        out = self.shifts[np.digitize(arr, self.edges)]
        return float(out) if arr.ndim == 0 else out

    def __float__(self):
        return self.global_shift

    def __repr__(self):
        pm = ", ".join(f"{1000 * s:+.1f}" + ("" if a else "(=global)")
                       for s, a in zip(self.shifts, self.applied))
        return (f"SegmentShift([{pm}] pm, global "
                f"{1000 * self.global_shift:+.1f} pm, n={self.n_matches})")


def shift_at(shift, wl):
    """Evaluate a scalar or :class:`SegmentShift` at wavelength(s) ``wl``.

    Every consumer of ``shift_nm`` should convert through this helper so
    plain floats (legacy, tests) and per-segment shifts both work.
    """
    if hasattr(shift, "at"):
        return shift.at(wl)
    return shift


def estimate_wavelength_shift_segments(
    peak_array,
    db,
    segment_edges=(365.0, 620.0),
    tolerance_nm=0.15,
    n_peaks=60,
    min_gA=1e6,
    ambiguity_ratio=3.0,
):
    """Per-detector-segment wavelength shifts from bright matched lines.

    Same anchor matching as :func:`estimate_wavelength_shift`, but the
    matched offsets are grouped by detector segment.  A segment gets its
    OWN median only when that median is trustworthy: at least
    ``MIN_SHIFT_MATCHES`` unambiguous matches AND a deviation from the
    pooled global median that exceeds twice the median's standard error
    (1.2533 x 1.4826 x MAD / sqrt(n)).  Everything else falls back to the
    global (which itself falls back to 0.0 below the match floor, exactly
    like the scalar estimator).  The significance gate matters: measured
    on MW2-112 BLIND-fit tables, a segment's median can sit 35 pm from
    the global purely because the blind centers of split/merged bright
    lines are displaced (per-segment MAD ~60 pm) — call this on the
    REFINED table, whose bright-line centers are physical, for the gate
    to pass on genuine calibration drift.

    ``n_peaks`` is higher than the scalar default so each segment keeps
    a usable share of the matches.

    Returns ``(SegmentShift, n_matches_total)``.
    """
    edges = tuple(float(e) for e in np.sort(np.atleast_1d(
        np.asarray(segment_edges, dtype=float))))
    peaks = np.atleast_2d(np.asarray(peak_array, dtype=float))
    n_seg = len(edges) + 1
    if peaks.size == 0:
        return SegmentShift(edges, [0.0] * n_seg, 0.0, [0] * n_seg), 0
    anchor_wl = _anchor_catalog(db, min_gA=min_gA)
    deltas, where = (np.empty(0), np.empty(0)) if anchor_wl.size == 0 else \
        _match_deltas(peaks, anchor_wl, tolerance_nm, n_peaks,
                      ambiguity_ratio)
    if deltas.size < MIN_SHIFT_MATCHES:
        return (SegmentShift(edges, [0.0] * n_seg, 0.0, [0] * n_seg),
                int(deltas.size))
    global_shift = float(np.median(deltas))
    seg = np.digitize(where, edges)
    shifts, counts, applied = [], [], []
    for s in range(n_seg):
        ds = deltas[seg == s]
        counts.append(int(ds.size))
        use_own = False
        if ds.size >= MIN_SHIFT_MATCHES:
            med = float(np.median(ds))
            mad = float(np.median(np.abs(ds - med)))
            se = 1.2533 * 1.4826 * mad / np.sqrt(ds.size)
            use_own = abs(med - global_shift) > 2.0 * max(se, 1e-6)
        shifts.append(med if use_own else global_shift)
        applied.append(use_own)
    return (SegmentShift(edges, shifts, global_shift, counts, applied),
            int(deltas.size))
