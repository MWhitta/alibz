"""Per-segment, per-peak profile analysis: physics encoded in peak shapes.

Production form of the ``peaky_data`` notebook's key thread (per-detector
segment, per-peak shape-component quantification).  For every fitted peak
this module answers, from the DATA in the peak's own window:

1. **How broad is it relative to the instrument?**  The per-segment
   instrumental width floor is measured from the spectrum's own narrow
   peaks (each detector segment has its own line-spread function), so
   excess width is attributable to plasma physics rather than optics.
2. **Which broadening mechanism?**  The Voigt decomposition of the excess
   (Gaussian vs Lorentzian) separates Doppler/instrumental from
   Stark/pressure broadening per peak.
3. **Is the shape a single line at all?**  The residual of the peak's own
   fitted component against its neighbour-subtracted window is classified
   by signature:

   - ``shoulder``  — a localised one-sided positive bump on a flank: an
     UNRESOLVED OVERLAPPING line, not asymmetry of this line.  These are
     deblend candidates; their fitted areas are contaminated.
   - ``sa-like``   — a core defect (flat-top or dip at centre, the model
     overshooting the data at the maximum) with the wings intact: the
     growth-curve signature of SELF-ABSORPTION.  The fitted area is a
     lower bound on emission and NOT proportional to concentration.
   - ``broadened`` — symmetric excess width with a clean residual:
     genuine plasma broadening (see 2).
   - ``narrow``    — narrower than the segment's instrument floor:
     physically impossible for a real emission line; usually a noise
     spike or fit artefact.
   - ``instrumental`` — consistent with the segment reference: a clean,
     optically-thin, resolution-limited line.  The safest quantification
     anchors.
   - ``irregular`` — significant residual structure matching none of the
     above.

   Shoulders and self-absorption both make a peak "asymmetric", but their
   residual signatures differ (one-sided localised bump vs core defect),
   which is what lets a shoulder be split while true SA is modelled
   rather than split (splitting SA manufactures phantom lines at
   wavelengths that then match the wrong elements).

Optionally, when a corpus-trained :class:`alibz.PeakyPCA` is supplied,
each window is also projected onto its principal components (the same
normalisation as PCA training) and the scores are mapped through the
perturbation sensitivities (``decompose_all_components``) into per-peak
Doppler / Stark / self-absorption attributions — the notebook's
"PC-based peak characterisation for downstream indexing".

The classifications are physics CONSTRAINTS for downstream consumers:
``shoulder`` peaks are split candidates, ``sa-like`` areas are saturated
(quantification from them is a model choice, cf. the Ca II 393.3
resonance line carrying 99% Ca in a failed fit), and the per-segment
width floor bounds what a legitimate new component may look like.
"""

from typing import Dict, List

import numpy as np

from alibz.utils.voigt import multi_voigt as _multi_voigt
from alibz.utils.voigt import voigt as _voigt
from alibz.utils.voigt import voigt_width as _voigt_width

#: peaks with FWHM <= NARROW_FACTOR x segment median narrow-width form the
#: instrument-floor sample for their segment.
NARROW_QUANTILE = 0.2
#: FWHM below this multiple of the segment floor is flagged ``narrow``.
NARROW_FLOOR_RATIO = 0.7
#: FWHM above this multiple of the segment floor counts as ``broadened``.
BROAD_FLOOR_RATIO = 1.8
#: residual significance (in local-noise sigma) for shoulder / core tests.
SHAPE_SIGMA = 3.0
#: flank band (in units of the peak's HWHM) searched for shoulder bumps.
SHOULDER_BAND = (0.8, 3.0)
#: half-window in units of the peak's fitted FWHM.
WINDOW_FWHM_MULT = 3.0
#: absolute half-window clamp in nm.
WINDOW_NM_RANGE = (0.15, 1.5)


def _segment_index(mu: float, segment_edges) -> int:
    edges = np.sort(np.asarray(segment_edges, dtype=float)) \
        if segment_edges is not None and len(segment_edges) else np.array([])
    return int(np.searchsorted(edges, float(mu)))


def segment_width_floor(peaks: np.ndarray, segment_edges=None) -> Dict[int, float]:
    """Per-detector-segment instrumental FWHM floor from the narrow peaks.

    The lower quantile of fitted Voigt widths in each segment estimates the
    instrument line-spread width there (the narrowest real lines are
    resolution-limited).  Segments with fewer than 5 peaks fall back to the
    global floor.
    """
    peaks = np.atleast_2d(np.asarray(peaks, dtype=float))
    if peaks.size == 0:
        return {}
    fwhm = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                        np.maximum(peaks[:, 3], 1e-6))
    segs = np.array([_segment_index(mu, segment_edges) for mu in peaks[:, 1]])
    glob = float(np.quantile(fwhm, NARROW_QUANTILE))
    floors = {}
    for s in np.unique(segs):
        w = fwhm[segs == s]
        floors[int(s)] = (float(np.quantile(w, NARROW_QUANTILE))
                          if w.size >= 5 else glob)
    floors[-1] = glob  # fallback key
    return floors


def analyze_peak_profiles(x, y, fit_dict, segment_edges=None,
                          pca=None, pca_decomposition=None,
                          pca_half_window_nm=None) -> List[dict]:
    """Per-peak, per-segment shape physics for a fitted spectrum.

    Parameters
    ----------
    x, y : arrays — the measured spectrum (raw; the fit's stored background
        is subtracted internally).
    fit_dict : a PeakyFinder-style fit dict (``sorted_parameter_array``,
        optional ``background``).
    segment_edges : detector-segment boundaries in nm (defaults to
        ``PeakyFinder.DEFAULT_SEGMENT_EDGES``).
    pca : optional trained :class:`alibz.PeakyPCA` — adds per-peak PC
        scores (``pc_scores``) using the training normalisation.
    pca_decomposition : optional output of ``pca.decompose_all_components``
        — adds per-peak mechanism attributions (``pc_doppler``,
        ``pc_stark``, ``pc_sa``) from the PC perturbation sensitivities.
    pca_half_window_nm : the half-window (nm) the PCA basis was TRAINED
        with; scores are only comparable at that width, so projection is
        skipped when it is neither passed nor found on the ``pca`` object
        (``half_window_nm`` attribute).

    Returns one record per fitted peak::

        {index, center_nm, area, segment, fwhm_nm, floor_nm, width_ratio,
         gaussian_fraction, asym, core_defect_sigma, shoulder_sigma,
         shoulder_side, shoulder_offset_nm, classification, ...}
    """
    from alibz.peaky_finder import PeakyFinder

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"],
                                     dtype=float))[:, :4]
    if peaks.size == 0 or x.size < 16:
        return []
    if segment_edges is None:
        segment_edges = PeakyFinder.DEFAULT_SEGMENT_EDGES
    segment_edges = tuple(np.atleast_1d(
        np.asarray(segment_edges, dtype=float)).ravel())

    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    seg_idx = (np.searchsorted(x, np.sort(np.asarray(segment_edges)))
               if len(segment_edges) else None)
    noise = PeakyFinder._noise_scale_local(y_bgsub, segment_indices=seg_idx)

    floors = segment_width_floor(peaks, segment_edges)
    fwhm_all = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                            np.maximum(peaks[:, 3], 1e-6))
    model_all = _multi_voigt(x, np.ravel(peaks[:, :4]))

    records = []
    for j in range(peaks.shape[0]):
        area, mu, sig, gam = (float(peaks[j, 0]), float(peaks[j, 1]),
                              float(max(peaks[j, 2], 1e-6)),
                              float(max(peaks[j, 3], 1e-6)))
        fwhm = float(fwhm_all[j])
        seg = _segment_index(mu, segment_edges)
        floor = floors.get(seg, floors.get(-1, fwhm))
        wr = fwhm / max(floor, 1e-9)
        # Gaussian share of the width: Doppler/instrumental (sigma) vs
        # Stark/pressure (gamma).  fG/(fG+fL) on FWHM contributions.
        fg = 2.3548 * sig
        fl = 2.0 * gam
        gfrac = fg / max(fg + fl, 1e-12)

        half_nm = float(np.clip(WINDOW_FWHM_MULT * fwhm, *WINDOW_NM_RANGE))
        # isolate this peak: subtract every OTHER fitted component
        got = None
        m = (x >= mu - half_nm) & (x <= mu + half_nm)
        if int(np.sum(m)) >= 9:
            xw = x[m]
            own = area * _voigt(xw - mu, sig, gam)
            # neighbour-subtracted window: data minus every OTHER component
            y_iso = y_bgsub[m] - (model_all[m] - own)
            got = (xw, y_iso, own, noise[m])
        rec = dict(index=j, center_nm=mu, area=area, segment=int(seg),
                   fwhm_nm=fwhm, floor_nm=float(floor),
                   width_ratio=float(wr), gaussian_fraction=float(gfrac),
                   asym=None, core_defect_sigma=None, shoulder_sigma=None,
                   shoulder_side=None, shoulder_offset_nm=None,
                   classification="unresolved-window")
        if got is None:
            records.append(rec)
            continue
        xw, y_iso, own, nz = got
        resid = y_iso - own
        hwhm = 0.5 * fwhm

        # --- shape metrics ------------------------------------------------
        # core defect: model minus data around the fitted CENTRE (positive =
        # the model overshoots a flattened/dipped core -> self-absorption).
        # Measured NARROW (0.3 x HWHM): a converged least-squares fit of a
        # flat-topped line balances residuals across the core, so the median
        # over the full +-HWHM band flips negative for moderate saturation,
        # while the centre overshoot survives at every measured clip level
        # (verified 50-85% clip refits).
        pitch = float(np.median(np.diff(xw))) if xw.size > 1 else 0.01
        core = np.abs(xw - mu) <= max(0.3 * hwhm, 2.0 * pitch)
        core_defect = (float(np.median(-resid[core] / np.maximum(nz[core], 1e-12)))
                       if np.any(core) else 0.0)
        # flank asymmetry of the DATA about the fitted centre
        lm = (xw < mu) & (np.abs(xw - mu) <= 2.5 * hwhm)
        rm = (xw > mu) & (np.abs(xw - mu) <= 2.5 * hwhm)
        a_l = float(np.trapezoid(y_iso[lm], xw[lm])) if np.sum(lm) > 2 else 0.0
        a_r = float(np.trapezoid(y_iso[rm], xw[rm])) if np.sum(rm) > 2 else 0.0
        asym = (a_r - a_l) / max(abs(a_r) + abs(a_l), 1e-12)
        # flank bumps: max positive residual in the shoulder band, PER side.
        # A shoulder is ONE-SIDED by definition (an overlapping line sits on
        # one flank); symmetric double lobes are instead the wing signature
        # a symmetric fit leaves on a flat-topped/self-absorbed core.
        bump = {}
        for side, mask in (("blue", (xw < mu)), ("red", (xw > mu))):
            band = mask & (np.abs(xw - mu) >= SHOULDER_BAND[0] * hwhm) \
                        & (np.abs(xw - mu) <= SHOULDER_BAND[1] * hwhm)
            if np.sum(band) < 3:
                continue
            rb = resid[band] / np.maximum(nz[band], 1e-12)
            k = int(np.argmax(rb))
            bump[side] = (float(rb[k]), float(xw[band][k] - mu))
        if bump:
            sh_side = max(bump, key=lambda s: bump[s][0])
            sh_sig, sh_off = bump[sh_side]
        else:
            sh_sig, sh_side, sh_off = 0.0, None, None
        other_sig = min((v[0] for v in bump.values()), default=0.0) \
            if len(bump) == 2 else 0.0
        one_sided = sh_sig >= SHAPE_SIGMA and other_sig < 0.5 * sh_sig
        double_lobed = (len(bump) == 2 and sh_sig >= SHAPE_SIGMA
                        and other_sig >= 0.5 * sh_sig)

        # --- classification ladder (most specific first) -------------------
        if wr < NARROW_FLOOR_RATIO:
            cls = "narrow"
        elif one_sided and sh_sig >= abs(core_defect):
            cls = "shoulder"
        elif core_defect >= SHAPE_SIGMA or double_lobed:
            # explicit core overshoot, OR symmetric high-significance lobes
            # on BOTH flanks: at production sampling (~4 px per FWHM) a
            # converged fit of a flat-topped line leaves near-equal two-
            # sided lobes while the centre-defect SIGN is pixel-unstable
            # (measured: -4 to +450 sigma across 50-85% clip refits), so the
            # lobe symmetry itself is the robust saturation signature -- and
            # a genuine blend contributing equally to both flanks is the far
            # rarer case, with the same consequence (area untrustworthy)
            cls = "sa-like"
        elif (np.max(np.abs(resid) / np.maximum(nz, 1e-12)) >= 2 * SHAPE_SIGMA):
            cls = "irregular"
        elif wr >= BROAD_FLOOR_RATIO:
            cls = "broadened"
        else:
            cls = "instrumental"
        rec.update(asym=float(asym), core_defect_sigma=float(core_defect),
                   shoulder_sigma=float(sh_sig), shoulder_side=sh_side,
                   shoulder_offset_nm=sh_off, classification=cls)

        # --- optional PC projection (corpus-trained basis) ------------------
        # scores are comparable only when the window matches the TRAINING
        # half-window, so a separate extraction at that width is used (the
        # shape-metric window above is per-peak adaptive); peaks whose
        # training-width window is unavailable are skipped
        if pca is not None and getattr(pca, "components", None) is not None:
            train_half = pca_half_window_nm
            if train_half is None:
                train_half = getattr(pca, "half_window_nm", None)
            scores = None
            if train_half is not None:
                mt = (x >= mu - float(train_half)) & (x <= mu + float(train_half))
                if int(np.sum(mt)) >= 5:
                    y_iso_t = y_bgsub[mt] - (model_all[mt]
                                             - area * _voigt(x[mt] - mu,
                                                             sig, gam))
                    scores = _project_onto_pca(x[mt], y_iso_t, pca)
            if scores is not None:
                rec["pc_scores"] = [float(s) for s in scores]
                if pca_decomposition:
                    dop = st = sa = 0.0
                    for i, s in enumerate(scores):
                        d = (pca_decomposition[i]
                             if i < len(pca_decomposition) else None)
                        if d is None:
                            continue
                        dop += abs(s) * float(d.get("gaussian_fraction", 0.0))
                        st += abs(s) * float(d.get("lorentzian_fraction", 0.0))
                        sa += abs(s) * float(d.get("asymmetry_fraction", 0.0))
                    tot = max(dop + st + sa, 1e-12)
                    rec.update(pc_doppler=float(dop / tot),
                               pc_stark=float(st / tot),
                               pc_sa=float(sa / tot))
        records.append(rec)
    return records


def _project_onto_pca(xw, yw, pca):
    """Project one peak window onto a trained PeakyPCA basis.

    Reproduces the training normalisation: linear endpoint baseline,
    min-zeroed, range-normalised, resampled to the training grid.
    """
    comp = np.asarray(pca.components, dtype=float)
    mean = np.asarray(pca.mean_peak, dtype=float)
    n_pts = mean.shape[0]
    yw = np.asarray(yw, dtype=float).copy()
    if yw.size < 5:
        return None
    baseline = np.linspace(yw[0], yw[-1], yw.size)
    yw = yw - baseline
    rng = float(np.max(yw) - np.min(yw))
    if rng <= 0:
        return None
    yw = (yw - np.min(yw)) / rng
    grid = np.linspace(0.0, 1.0, yw.size)
    fixed = np.linspace(0.0, 1.0, n_pts)
    win = np.interp(fixed, grid, yw)
    return (win - mean) @ comp.T


def profile_summary(records: List[dict]) -> Dict[str, int]:
    """Classification counts, e.g. for logging/notebook display."""
    out: Dict[str, int] = {}
    for r in records:
        out[r["classification"]] = out.get(r["classification"], 0) + 1
    return out


def element_shape_quality(support_idx: Dict[str, List[int]],
                          records: List[dict]) -> Dict[str, dict]:
    """Per-element shape QC of its supporting peaks.

    ``support_idx`` maps element -> indices into the fitted peak table (the
    peaks whose dominant assignment is that element).  Returns, per element,
    the flux-weighted share of its support that is ``sa-like`` (area
    saturated -- quantification from those peaks is a lower-bound/model
    choice) and ``shoulder`` (area contaminated by an unresolved overlap),
    plus the count of clean anchors (``instrumental``/``broadened``).

    This is the shape analogue of the true-negative ``clear_lines`` count:
    an element whose dominance rests on saturated or contaminated peaks is
    flagged, not trusted.
    """
    by_idx = {r["index"]: r for r in records}
    out = {}
    for el, idxs in (support_idx or {}).items():
        tot = sat = cont = 0.0
        clean = 0
        for j in idxs:
            r = by_idx.get(int(j))
            if r is None:
                continue
            a = abs(float(r.get("area") or 0.0))
            tot += a
            if r["classification"] == "sa-like":
                sat += a
            elif r["classification"] == "shoulder":
                cont += a
            elif r["classification"] in ("instrumental", "broadened"):
                clean += 1
        if tot <= 0:
            continue
        out[el] = dict(sa_share=sat / tot, shoulder_share=cont / tot,
                       clean_anchors=clean)
    return out
