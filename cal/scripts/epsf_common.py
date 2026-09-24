"""Shared data handling for the empirical-LSF (ePSF) study (2026-09-23).

Loads native Z300 run means per detector segment in PIXEL coordinates,
estimates a per-pixel noise model, reproduces the coordinator's
identification-free peak selection (cal/scripts/peak_phase_uniformity.py),
and provides the comparison estimators (spline-root, parabola, free-width
pixel-integrated Gaussian) exactly as that script defines them.

Pixel coordinate u: index of the sample within its segment in ascending
wavelength order (the native file is sorted ascending; the detector's pixel 0
is the red end, so u = N_seg - 1 - pixel up to a constant; a sub-pixel phase
is therefore phase(u) = -phase(pixel)).  The 4th (dead) channel >= 955 nm is
excluded.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import erf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FE_ACQ = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
          "39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/acq")
FE_LEDGER = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
             "39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/ledger.json")
V_ACQ = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
         "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/vrun2/acq")
V_LEDGER = os.path.join(REPO, "cal", "data", "v-pure-run2-ledger.json")

CORR_ROOT = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
             "1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/corrected")

#: dataset presets.  'native-20260923': the uncorrected native files (vendor
#: pixToNm evaluated at the stored-sample index; ~18.3 px axis offset, found by
#: pattern matching).  'corrected-20260924': the same runs re-derived by
#: pantheum with PIXEL_OFFSET = -18 (reports/2026-09-23-native-axis-18px-offset.md),
#: plus V run1; 5,848 samples, 4th channel dropped.
DATASETS = {
    "native-20260923": dict(kind="legacy", tag="20260923", register="pattern"),
    "corrected-20260924": dict(kind="ledger", root=CORR_ROOT, ledger=os.path.join(CORR_ROOT, "ledger.json"),
                               tag="20260924", register="zero"),
}

SEGS = {"UV": (180.0, 365.0), "VIS": (365.0, 620.0), "NIR": (620.0, 955.0)}
# coordinator's analysis windows (peak_phase_uniformity.py)
COORD_SEG = {"UV": (190, 362), "VIS": (368, 617), "NIR": (623, 945)}


class Segment:
    """One detector segment of one run mean, in pixel coordinates."""

    def __init__(self, name, wl, y, shots):
        self.name = name
        self.wl = wl
        self.y = y
        self.n = wl.size
        self.u = np.arange(self.n, dtype=float)
        # smooth pixel->nm map (cubic, like the vendor pixToNm); CSV rounding 5e-5 nm
        self.poly = np.polyfit(self.u - self.n / 2, wl, 3)
        self.nshot = shots.shape[0]
        self.sigma = noise_model(y, shots)

    def u_to_nm(self, u):
        return np.polyval(self.poly, np.asarray(u, float) - self.n / 2)

    def pitch(self, u):
        return np.polyval(np.polyder(self.poly), np.asarray(u, float) - self.n / 2)

    def nm_to_u(self, wl):
        wl = np.asarray(wl, float)
        u = np.interp(wl, self.wl, self.u)
        for _ in range(3):
            u = u - (self.u_to_nm(u) - wl) / self.pitch(u)
        return u


def noise_model(y, shots):
    """Per-pixel 1-sigma for fitting the shot MEAN.

    sigma^2 = s0^2 + a * S + (b * S)^2, S = signal above a 41-px running median.
    s0: robust first-difference scatter of the mean.  On these data s0 (~15
    counts) is ~4x the shot-to-shot noise of the mean in line-free pixels
    (~3.5): it is dominated by the detector's static pixel-to-pixel pattern,
    which does not average down but acts as noise for a line fit.  a, b: the
    shot-to-shot variance of the mean above the line-free level, binned in S
    and fitted non-negatively (Poisson-like + multiplicative jitter), after
    rescaling each shot to the mean segment total (common-mode plasma
    intensity changes no line shape).
    """
    from scipy.optimize import nnls
    n = shots.shape[0]
    d1 = np.diff(y)
    s0 = 1.4826 * np.median(np.abs(d1 - np.median(d1))) / np.sqrt(2)
    base = running_median(y, 41)
    S = np.clip(y - base, 0, None)
    a = b2 = 0.0
    if n >= 3:
        tot = np.clip(shots, 0, None).sum(1)
        sc = tot.mean() / np.where(tot > 0, tot, 1.0)
        # remove each shot's common intensity scale (it changes no line shape)
        var_m = (shots * sc[:, None]).var(0, ddof=1) / n
        v0 = np.median(var_m[S < 3 * s0])
        edges = s0 * np.array([5, 10, 20, 40, 80, 160, 320, 640, 1e9])
        X, Yv = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = (S >= lo) & (S < hi)
            if q.sum() >= 5:
                sm = np.median(S[q]); X.append([sm, sm ** 2]); Yv.append(max(np.median(var_m[q]) - v0, 0))
        if len(X) >= 2:
            X = np.array(X); Yv = np.array(Yv)
            w = 1 / np.maximum(Yv, 1.0)          # relative-error weighting
            coef, _ = nnls(X * w[:, None], Yv * w)
            a, b2 = coef
    return np.sqrt(s0 ** 2 + a * S + b2 * S ** 2)


def running_median(y, w):
    from scipy.ndimage import median_filter
    return median_filter(y, size=w, mode="nearest")


class Run:
    def __init__(self, run_id, family, meta, segs, path):
        self.id = run_id
        self.family = family          # 'Fe' or 'V'
        self.meta = meta
        self.segs = segs              # name -> Segment
        self.path = path

    @property
    def delay(self):
        return self.meta.get("delay")

    @property
    def period(self):
        return self.meta.get("period")


def load_run(path, family, meta=None):
    fs = sorted(glob.glob(os.path.join(path, "shots", "shot-*.csv")))
    fs = [f for f in fs if not f.endswith("shot--1.csv")]
    if not fs:
        return None
    A = [np.loadtxt(f, delimiter=",", skiprows=1) for f in fs]
    x = A[0][:, 0]
    Y = np.array([a[:, 1] for a in A])
    segs = {}
    for name, (lo, hi) in SEGS.items():
        m = (x >= lo) & (x < hi)
        segs[name] = Segment(name, x[m], Y[:, m].mean(0), Y[:, m])
    meta = dict(meta or {})
    try:
        man = json.load(open(os.path.join(path, "manifest.json")))
        meta.setdefault("params", man.get("params"))
        meta.setdefault("created_at", man.get("created_at"))
    except Exception:
        pass
    return Run(os.path.basename(path), family, meta, segs, path)


def load_all(cache=None, verbose=True, dataset="native-20260923"):
    """Load a dataset preset (see DATASETS)."""
    if cache and os.path.exists(cache):
        import pickle
        return pickle.load(open(cache, "rb"))
    if DATASETS[dataset]["kind"] == "ledger":
        runs = _load_ledger(DATASETS[dataset])
        if verbose:
            print(f"loaded {len(runs)} runs ({sum(r.family == 'Fe' for r in runs)} Fe, "
                  f"{sum(r.family == 'V' for r in runs)} V) from {DATASETS[dataset]['root']}")
        if cache:
            import pickle
            pickle.dump(runs, open(cache, "wb"))
        return runs
    fe_led = {r["run"]: r for r in json.load(open(FE_LEDGER))}
    v_led = {r["run"]: r for r in json.load(open(V_LEDGER))}
    runs = []
    for d in sorted(glob.glob(os.path.join(FE_ACQ, "run-*"))):
        rid = os.path.basename(d)
        meta = dict(fe_led.get(rid, {}))
        if not meta:
            try:
                p = json.load(open(os.path.join(d, "manifest.json")))["params"]
                meta = dict(delay=p.get("intergrationDelay"), period=p.get("intergrationPeriod"))
            except Exception:
                pass
        meta["in_ledger"] = rid in fe_led
        r = load_run(d, "Fe", meta)
        if r:
            runs.append(r)
    for d in sorted(glob.glob(os.path.join(V_ACQ, "*"))):
        rid = os.path.basename(d)
        r = load_run(d, "V", dict(v_led.get(rid, {})))
        if r:
            runs.append(r)
    if verbose:
        print(f"loaded {len(runs)} runs "
              f"({sum(r.family == 'Fe' for r in runs)} Fe, {sum(r.family == 'V' for r in runs)} V)")
    if cache:
        import pickle
        pickle.dump(runs, open(cache, "wb"))
    return runs


def _load_ledger(cfg):
    """Corrected tree: acq/<run>/shots/shot-*.csv + ledger.json.  Family from
    the ledger 'element'; 6 early runs without it are Fe (they are in the
    2026-09-22 Fe ledger)."""
    fe_old = {r["run"] for r in json.load(open(FE_LEDGER))} if os.path.exists(FE_LEDGER) else set()
    runs = []
    for meta in sorted(json.load(open(cfg["ledger"])), key=lambda r: r["run"]):
        el = meta.get("element") or ("Fe" if meta["run"] in fe_old else None)
        if el not in ("Fe", "V"):
            continue
        m = dict(meta)
        m["family_source"] = "ledger" if meta.get("element") else "fe-ledger-20260922"
        r = load_run(os.path.join(cfg["root"], "acq", meta["run"]), el, m)
        if r:
            try:
                man = json.load(open(os.path.join(cfg["root"], "acq", meta["run"], "manifest.json")))
                r.meta["axis_check"] = man.get("axis_check")
            except Exception:
                pass
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Coordinator estimators, verbatim semantics (nm coordinates on the run mean)
# ---------------------------------------------------------------------------
def _edges(x):
    m = 0.5 * (x[1:] + x[:-1])
    return np.r_[x[0] - (m[0] - x[0]), m, x[-1] + (x[-1] - m[-1])]


def pixint_nm(x, y, i, half=3, sigma=None):
    s = slice(i - half, i + half + 1); xs, ys = x[s], y[s]; e = _edges(x)[i - half:i + half + 2]; w = np.diff(e)
    pitch = np.median(w)

    def model(p):
        sg = p[2] if sigma is None else sigma
        c = 0.5 * (1 + erf((e - p[1]) / (sg * np.sqrt(2)))); return p[0] * np.diff(c) / w + p[-1]
    p0 = [max(ys.max(), 1) * pitch * 1.5, x[i]] + ([0.6 * pitch] if sigma is None else []) + [0.0]
    lo = [0, x[i] - pitch] + ([0.25 * pitch] if sigma is None else []) + [-np.inf]
    hi = [np.inf, x[i] + pitch] + ([2.5 * pitch] if sigma is None else []) + [np.inf]
    r = least_squares(lambda p: model(p) - ys, p0, bounds=(lo, hi))
    return r.x[1], (r.x[2] if sigma is None else sigma), r.x[0]


def spline_root(x, y, i, half=4):
    s = slice(i - half, i + half + 1); cs = CubicSpline(x[s], y[s], bc_type='not-a-knot')
    xx = np.linspace(x[i - 1], x[i + 1], 4001); return xx[np.argmax(cs(xx))]


def vendor_spline(x, y, i, half=8, grid_nm=1.0 / 30):
    """Emulated vendor export: not-a-knot cubic spline through the native
    samples (+-half around i).  Returns (root, parabola):
      root     = maximum of the continuous spline (vendor-spline root);
      parabola = 3-point parabola on the spline resampled to the vendor
                 1/30 nm grid (what a user of the exported CSV would compute)."""
    s = slice(max(i - half, 0), i + half + 1)
    cs = CubicSpline(x[s], y[s], bc_type="not-a-knot")
    xx = np.linspace(x[i - 1], x[i + 1], 8001)
    root = float(xx[np.argmax(cs(xx))])
    g = np.arange(np.ceil(x[i - 2] / grid_nm) * grid_nm, x[i + 2], grid_nm)
    gy = cs(g)
    j = int(np.argmax(gy)); j = min(max(j, 1), g.size - 2)
    a, b, _ = np.polyfit(g[j - 1:j + 2] - g[j], gy[j - 1:j + 2], 2)
    par = float(g[j] - b / (2 * a)) if a < 0 else float(g[j])
    return root, par


def parab(x, y, i):
    a, b, c = np.polyfit(x[i - 1:i + 2] - x[i], y[i - 1:i + 2], 2); return x[i] - b / (2 * a)


def bgsub(x, y, i):
    idx = np.r_[i - 9:i - 4, i + 5:i + 10]; a, b = np.polyfit(x[idx] - x[i], y[idx], 1)
    return y - (a * (x - x[i]) + b)


def coordinator_peaks(run):
    """Reproduce peak_phase_uniformity.py's selection on this run's mean.

    Works on the concatenated <955 nm array exactly as the coordinator did,
    then maps each accepted peak to (segment, u index).
    """
    warnings.filterwarnings('ignore')
    xs = np.concatenate([run.segs[s].wl for s in ("UV", "VIS", "NIR")])
    ys = np.concatenate([run.segs[s].y for s in ("UV", "VIS", "NIR")])
    m = xs < 955; x = xs[m]; y = ys[m]
    offs = {"UV": 0, "VIS": run.segs["UV"].n, "NIR": run.segs["UV"].n + run.segs["VIS"].n}
    out = []
    for s, (lo, hi) in COORD_SEG.items():
        for i in np.where((x > lo) & (x < hi))[0]:
            if i < 12 or i > len(x) - 12 or y[i] != y[i - 4:i + 5].max():
                continue
            yb = bgsub(x, y, i); q = np.r_[yb[i - 9:i - 4], yb[i + 5:i + 10]]; nz = q.std() + 1
            h = yb[i]
            if h / nz < 30:
                continue
            if max(yb[i - 4:i - 1].max(), yb[i + 2:i + 5].max()) > 0.6 * h:
                continue
            pitch = abs(x[i + 1] - x[i])
            try:
                mu8, sg, a8 = pixint_nm(x, yb, i)
            except Exception:
                continue
            if sg > 2.0 * pitch:
                continue
            k = i - offs[s]
            out.append(dict(run=run.id, family=run.family, seg=s, k=int(k), wl_pk=float(x[i]),
                            snr=float(h / nz), height=float(h),
                            p8=float(mu8), sp=float(spline_root(x, yb, i)), pa=float(parab(x, yb, i)),
                            vr=vendor_spline(x, yb, i)[0], vp=vendor_spline(x, yb, i)[1],
                            sg=float(sg / pitch), pitch=float(pitch), a8=float(a8)))
    return out


def estimators_at(seg, k):
    """spline-root / parabola / free-width pixint centres (nm) at local max k
    of a segment (same definitions as the coordinator, segment-local arrays)."""
    x, y = seg.wl, seg.y
    if k < 10 or k > seg.n - 11:
        return None
    yb = bgsub(x, y, k)
    try:
        mu8, sg, a8 = pixint_nm(x, yb, k)
    except Exception:
        return None
    vr, vp = vendor_spline(x, yb, k)
    return dict(p8=float(mu8), sp=float(spline_root(x, yb, k)), pa=float(parab(x, yb, k)), vr=vr, vp=vp,
                sg=float(sg / abs(x[k + 1] - x[k])), a8=float(a8))


# ---------------------------------------------------------------------------
# Line database, registration and identification
# ---------------------------------------------------------------------------
KB_EV = 8.617333e-5
_DB = None


def get_db():
    global _DB
    if _DB is None:
        from alibz.utils.database import Database
        _DB = Database(os.path.join(REPO, "db"))
    return _DB


def db_table(elements, T=9000.0, ion_scale=1.0):
    """Rows for ion stages I/II of ``elements``: dict of arrays.

    strength = gA exp(-Ek/kT) / lambda (thin-plasma relative emissivity);
    ion II rows multiplied by ``ion_scale``.  Wavelengths in AIR (Database).
    """
    db = get_db()
    cols = dict(el=[], ion=[], wl=[], gA=[], Ei=[], Ek=[], s=[])
    for el in elements:
        a = db.lines(el)
        if a.size == 0:
            continue
        ion = a[:, 0].astype(float); wl = a[:, 1].astype(float)
        gA = a[:, 3].astype(float); Ei = a[:, 4].astype(float); Ek = a[:, 5].astype(float)
        m = (ion <= 2) & (wl > 185) & (wl < 950) & (gA > 0)
        s = gA * np.exp(-Ek / (KB_EV * T)) / wl * np.where(ion == 2, ion_scale, 1.0)
        for k, v in (("ion", ion), ("wl", wl), ("gA", gA), ("Ei", Ei), ("Ek", Ek), ("s", s)):
            cols[k].append(v[m])
        cols["el"].append(np.array([el] * int(m.sum())))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.argsort(out["wl"])
    return {k: v[o] for k, v in out.items()}


def pattern_offset(seg, tab, n_top=40, rng=(-40.0, 40.0), step=0.05):
    """Pixel offset du (observed u = u(db) + du) maximising the match of the
    ``n_top`` strongest predicted lines with the observed spectrum.
    Returns (du, z_best, z_second) with z a robust score significance."""
    y = np.clip(seg.y - running_median(seg.y, 41), 0, None) / seg.sigma
    y = np.sqrt(y)
    m = (tab["wl"] > seg.wl[0]) & (tab["wl"] < seg.wl[-1])
    w = tab["wl"][m]; ss = tab["s"][m]
    top = np.argsort(ss)[::-1][:n_top]
    u0 = seg.nm_to_u(w[top]); wt = np.sqrt(ss[top] / ss[top].max())
    offs = np.arange(rng[0], rng[1] + step / 2, step)
    sc = np.array([np.sum(wt * np.interp(u0 + o, seg.u, y)) for o in offs])
    med = np.median(sc); mad = 1.4826 * np.median(np.abs(sc - med)) + 1e-12
    z = (sc - med) / mad
    j = int(np.argmax(z))
    far = np.abs(offs - offs[j]) > 1.5
    return float(offs[j]), float(z[j]), float(z[far].max())


def find_peaks_u(seg, snr_min=8.0):
    """Local maxima (+-2 px) with height above a running-median baseline >=
    snr_min * sigma.  Returns list of (k, snr)."""
    y = seg.y - running_median(seg.y, 41)
    out = []
    for k in range(6, seg.n - 6):
        if y[k] == y[k - 2:k + 3].max() and y[k] > y[k - 1] and y[k] >= snr_min * seg.sigma[k]:
            out.append((k, float(y[k] / seg.sigma[k])))
    return out


def identify(seg, tab, du0, peaks, tol=0.45, dom=5.0, iso_px=2.0, rel_min=0.02,
             refine=True, weights=None):
    """Identify observed peaks with db lines after pattern registration.

    A peak (sub-pixel centre uc) is CONFIDENTLY identified with db line L if
      * |uc - (u_db(L) + du(u))| <= tol px, du = registration (constant, then
        refined to linear in u from the confident set with 3-MAD clipping);
      * L's predicted strength dominates every other db line within
        +-iso_px px (u-space) by >= dom (db-isolated: no blend partner);
      * L's predicted strength is >= rel_min of the strongest predicted line in
        the segment (it is expected to be visible);
      * no other observed peak lies within +-iso_px px.
    Returns (list of dict, du_model(poly coeffs in u)).
    """
    m = (tab["wl"] > seg.wl[0] - 1) & (tab["wl"] < seg.wl[-1] + 1)
    T = {k: v[m] for k, v in tab.items()}
    if weights:
        # strengths are only comparable WITHIN an element (different T_exc,
        # abundance); normalise per element in this segment, then weight
        inseg = (T["wl"] > seg.wl[0]) & (T["wl"] < seg.wl[-1])
        s_eff = np.zeros_like(T["s"])
        for el in np.unique(T["el"]):
            e = T["el"] == el
            mx = T["s"][e & inseg].max() if np.any(e & inseg) else T["s"][e].max()
            s_eff[e] = weights.get(el, 0.0) * T["s"][e] / mx
        T["s"] = s_eff
    udb = seg.nm_to_u(T["wl"])
    smax = T["s"][(T["wl"] > seg.wl[0]) & (T["wl"] < seg.wl[-1])].max()
    pk_u = np.array([p[0] for p in peaks], float)
    poly = np.array([0.0, du0])
    for it in range(3 if refine else 1):
        res = []
        for p_ in peaks:
            k, snr = p_[0], p_[1]
            uc = float(p_[2]) if len(p_) > 2 else float(k)
            pred = udb + np.polyval(poly, udb)
            d = uc - pred
            cand = np.where(np.abs(d) <= tol)[0]
            if cand.size == 0:
                continue
            j = cand[np.argmax(T["s"][cand])]
            near = np.abs(pred - pred[j]) <= iso_px
            near[j] = False
            if near.any() and T["s"][j] < dom * T["s"][near].max():
                continue
            if T["s"][j] < rel_min * smax:
                continue
            others = np.abs(pk_u - k) <= iso_px
            if others.sum() > 1:
                continue
            res.append(dict(k=int(k), uc=float(uc), snr=snr, el=str(T["el"][j]), ion=int(T["ion"][j]),
                            wl_db=float(T["wl"][j]), gA=float(T["gA"][j]), Ei=float(T["Ei"][j]),
                            Ek=float(T["Ek"][j]), s_rel=float(T["s"][j] / smax), u_db=float(udb[j]),
                            resid=float(d[j] - 0.0)))
        if not refine or len(res) < 6:
            break
        x = np.array([r["u_db"] for r in res]); yv = np.array([r["uc"] - r["u_db"] for r in res])
        keep = np.ones(x.size, bool)
        for _ in range(3):
            c = np.polyfit(x[keep], yv[keep], 1)
            rr = yv - np.polyval(c, x)
            s = 1.4826 * np.median(np.abs(rr[keep] - np.median(rr[keep])))
            keep = np.abs(rr) <= max(3 * s, 0.1)
        poly = c
    for r in res:
        r["du_model"] = float(np.polyval(poly, r["u_db"]))
        r["resid"] = r["uc"] - r["u_db"] - r["du_model"]
    return res, poly
