"""Local pitch/phase search for native knots in segments without d4 brackets.

For each overlapping window: grid-search (pitch, phase) minimising the
cubic-spline lstsq residual, get an explicit knot list; link knots across
overlapping windows into a global index; polyfit degree 3; global refine.
"""
import sys, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

PX = 1.0 / 30.0


def kernel(xk, xe):
    M = CubicSpline(xk, np.eye(xk.size), axis=0)(xe)
    M[np.abs(M) < 1e-12] = 0.0
    return M


def relres_dense(xk, xe, ye):
    M = kernel(xk, xe)
    sol, *_ = np.linalg.lstsq(M, ye, rcond=None)
    r = ye - M @ sol
    return np.linalg.norm(r) / np.linalg.norm(ye)


def relres_sparse(xk, xe, ye):
    M = csr_matrix(kernel(xk, xe))
    sol = lsqr(M, ye, atol=1e-14, btol=1e-14, iter_lim=20000)[0]
    return np.linalg.norm(ye - M @ sol) / np.linalg.norm(ye)


def window_search(xe, ye, p_lo, p_hi, dp=0.02, dphi=0.1):
    """Return best (phase_nm, pitch_nm, relres) for one window; knots at
    xe[0] + phase + k*pitch (nm), extended one knot beyond both ends."""
    best = (None, None, np.inf)
    x0 = xe[0]
    span = xe[-1] - xe[0]
    for p_px in np.arange(p_lo, p_hi + 1e-9, dp):
        p = p_px * PX
        for phi_px in np.arange(0.0, p_px, dphi):
            phi = phi_px * PX
            k = np.arange(-2, int(np.ceil((span - phi) / p)) + 3)
            xk = x0 + phi + k * p
            rr = relres_dense(xk, xe, ye)
            if rr < best[2]:
                best = (phi, p, rr)
    # local polish in continuous (phi, p)
    phi0, p0, r0 = best

    def obj(q):
        phi, p = q
        if p <= 0 or p < p_lo * PX or p > p_hi * PX:
            return 1e9
        k = np.arange(-2, int(np.ceil((span - phi) / p)) + 3)
        return relres_dense(x0 + phi + k * p, xe, ye)

    res = minimize(obj, [phi0, p0], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-9, maxiter=300))
    phi, p = res.x
    k = np.arange(-2, int(np.ceil((span - phi) / p)) + 3)
    xk = x0 + phi + k * p
    inside = (xk >= xe[0]) & (xk <= xe[-1])
    return xk[inside], p, res.fun


OUT = ''
def run_segment(x, y, lo, hi, p_lo, p_hi, win_nm=6.0, step_nm=3.0, margin_nm=1.0,
                verbose=True):
    m = (x >= lo + margin_nm) & (x < hi - margin_nm)
    xs, ys = x[m], y[m]
    # trim leading/trailing constant runs (zero-padded detector edges)
    a = 0
    while a < ys.size - 1 and ys[a + 1] == ys[0]:
        a += 1
    b = ys.size - 1
    while b > 0 and ys[b - 1] == ys[-1]:
        b -= 1
    if a > 3 or b < ys.size - 4:
        xs, ys = xs[a + 1:b], ys[a + 1:b]
        print(f"  trimmed dead edges -> {xs[0]:.2f}-{xs[-1]:.2f} nm", flush=True)
    t0 = time.time()
    knots_all = []   # list of arrays per window
    pitches = []
    starts = np.arange(xs[0], xs[-1] - win_nm, step_nm)
    for i, s in enumerate(starts):
        w = (xs >= s) & (xs < s + win_nm)
        xe, ye = xs[w], ys[w]
        if np.all(ye == ye[0]):
            knots_all.append(np.array([])); pitches.append(np.nan); continue
        xk, p, rr = window_search(xe, ye, p_lo, p_hi)
        knots_all.append(xk); pitches.append(p / PX)
        if verbose and i % 10 == 0:
            print(f"  win {i}/{len(starts)} {s:.1f} nm: pitch {p/PX:.3f} px relres {rr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    # Link windows: every window's knot set is correct to ~0.05 px (verified
    # against the d4-bracket knots), so the union of all windows IS the knot
    # list once duplicates (same knot seen by overlapping windows) are merged
    # with a tolerance of half the local pitch; indices then follow from the
    # gaps divided by the local pitch.
    centers = np.array([s + 0.5 * win_nm for s in starts])
    pit = np.array(pitches, float)
    good = np.isfinite(pit) & (pit > p_lo + 0.02) & (pit < p_hi - 0.02)
    # pitch continuity: robust quadratic fit of pitch vs window centre;
    # windows whose pitch deviates > 3 % are unreliable (strong lines,
    # dead pixels) and are dropped from the knot chain
    for _ in range(4):
        cp = np.polyfit(centers[good], pit[good], 2)
        dev = np.abs(pit - np.polyval(cp, centers)) / np.polyval(cp, centers)
        good = np.isfinite(pit) & (dev < 0.03)
    print(f"  windows kept for chaining: {good.sum()}/{good.size}", flush=True)
    knots_all = [k if g else np.array([]) for k, g in zip(knots_all, good)]
    def local_pitch_px(pos):
        return float(np.interp(pos, centers[good], pit[good]))
    allk = np.sort(np.concatenate([k for k in knots_all if k.size]))
    merged = [allk[0]]
    for v in allk[1:]:
        if (v - merged[-1]) / PX < 0.5 * local_pitch_px(v):
            merged[-1] = 0.5 * (merged[-1] + v)
        else:
            merged.append(v)
    merged = np.array(merged)
    gaps = np.diff(merged) / PX
    idx = [0.0]
    for g, pos in zip(gaps, merged[1:]):
        idx.append(idx[-1] + max(1, int(round(g / local_pitch_px(pos)))))
    idx = np.array(idx)
    keep = np.ones(idx.size, dtype=bool)
    for _ in range(6):
        c = np.polyfit(idx[keep], merged[keep], 3)
        r = (merged - np.polyval(c, idx)) / PX
        newkeep = np.abs(r) < max(3.0 * 1.4826 * np.median(np.abs(r[keep])), 0.5)
        if np.array_equal(newkeep, keep):
            break
        keep = newkeep
    print(f"  robust polyfit kept {keep.sum()}/{idx.size} knots; rejected rms {r[~keep].std() if (~keep).sum()>1 else 0:.2f} px", flush=True)
    r = r[keep]
    np.save(OUT + "_chain.npy", np.array([merged, idx, keep], dtype=object), allow_pickle=True)
    print(f"  linked knots {merged.size}, index span {idx[-1]:.0f}, polyfit rms {r.std():.3f} px max {np.abs(r).max():.2f} px, "
          f"pitch {np.polyval(np.polyder(c), idx[0])/PX:.3f}->{np.polyval(np.polyder(c), idx[-1])/PX:.3f} px "
          f"({np.polyval(np.polyder(c), idx[0]):.4f}->{np.polyval(np.polyder(c), idx[-1]):.4f} nm)", flush=True)
    np.save(OUT + f"_windows.npy", np.array([knots_all, pitches], dtype=object), allow_pickle=True)
    # global refinement of the 4 coefficients on the full segment residual
    n_ext = np.arange(-6, idx[-1] + 7)

    inner = (xs >= xs[0] + 2.0) & (xs <= xs[-1] - 2.0)
    def obj(cc):
        xk = np.polyval(cc, n_ext)
        if np.any(np.diff(xk) <= 0):
            return 1e9
        return relres_sparse(xk, xs[inner], ys[inner])

    r0 = obj(c)
    scale = np.abs(c) + 1e-12
    res = minimize(lambda q: obj(q * scale), c / scale, method="Nelder-Mead",
                   options=dict(xatol=1e-9, fatol=1e-12, maxiter=400))
    c2 = res.x * scale
    print(f"  global relres: polyfit {r0:.3e} -> refined {res.fun:.3e} ({time.time()-t0:.0f}s)")
    return dict(coeffs=c2, relres=res.fun, relres0=r0, n_knots=merged.size, knots=merged, idx=idx)


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "REE_01"
    segs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["NIR", "VIS", "UV"]
    d = np.loadtxt(f"/Users/mwhittaker/Projects/github/alibz/data/remote_samples/{f}.csv", delimiter=",", skiprows=1)
    x, y = d[:, 0], d[:, 1]
    cfg = dict(NIR=(620, 961, 4.6, 6.2), VIS=(365, 620, 3.2, 4.8), UV=(190, 365, 2.1, 3.2))
    out = {}
    for s in segs:
        lo, hi, plo, phi = cfg[s]
        print(f"== {f} {s} {lo}-{hi}", flush=True)
        globals()["OUT"] = f"/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/c9201306-a054-4205-98ee-19371a81d6ea/scratchpad/lps_{f}_{s}"
        out[s] = run_segment(x, y, lo, hi, plo, phi)
        np.save(f"/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/c9201306-a054-4205-98ee-19371a81d6ea/scratchpad/lps_{f}_{s}.npy",
                np.array([out[s]["coeffs"], [out[s]["relres"], out[s]["relres0"], out[s]["n_knots"], 0]], dtype=object), allow_pickle=True)
