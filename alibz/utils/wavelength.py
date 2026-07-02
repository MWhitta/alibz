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
    ``(0.0, 0)`` when fewer than 5 unambiguous matches exist.
    """
    peaks = np.atleast_2d(np.asarray(peak_array, dtype=float))
    if peaks.size == 0:
        return 0.0, 0

    # Anchor-line catalog: the raw database is too dense for
    # nearest-neighbour matching (a shifted peak's nearest line is usually
    # a DIFFERENT line near the shifted position, collapsing the median
    # to zero).  Keep only lines that DOMINATE their +-0.3 nm
    # neighbourhood in Boltzmann-weighted strength — the classic
    # calibration-line property.
    kT = 0.9  # eV; only used to rank line strengths locally
    wl_all, s_all = [], []
    for el in db.elements:
        if el in db.no_lines:
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
    anchor_wl = np.asarray(anchors)
    if anchor_wl.size == 0:
        return 0.0, 0

    strongest = np.argsort(peaks[:, 0])[::-1][: int(n_peaks)]
    deltas = []
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

    if len(deltas) < 5:
        return 0.0, len(deltas)
    return float(np.median(deltas)), len(deltas)
