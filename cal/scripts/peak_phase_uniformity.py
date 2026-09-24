"""Identification-free test of phase-dependent peak-location bias on REAL native spectra.
For many isolated strong peaks, compute centres with several estimators and plot pairwise
differences vs sub-pixel phase phi (fractional position of the fitted centre between pixel
centres). A phase-dependent bias appears as a periodic curve in phi; noise does not."""
import glob, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import erf
def edges(x):
    m = 0.5 * (x[1:] + x[:-1]); return np.r_[x[0] - (m[0] - x[0]), m, x[-1] + (x[-1] - m[-1])]
def pixint(x, y, i, half=3, sigma=None):
    s = slice(i - half, i + half + 1); xs, ys = x[s], y[s]; e = edges(x)[i - half:i + half + 2]; w = np.diff(e)
    pitch = np.median(w)
    def model(p):
        sg = p[2] if sigma is None else sigma
        c = 0.5 * (1 + erf((e - p[1]) / (sg * np.sqrt(2)))); return p[0] * np.diff(c) / w + p[-1]
    p0 = [max(ys.max(), 1) * pitch * 1.5, x[i]] + ([0.6 * pitch] if sigma is None else []) + [0.0]
    lo = [0, x[i] - pitch] + ([0.25 * pitch] if sigma is None else []) + [-np.inf]
    hi = [np.inf, x[i] + pitch] + ([2.5 * pitch] if sigma is None else []) + [np.inf]
    r = least_squares(lambda p: model(p) - ys, p0, bounds=(lo, hi)); return r.x[1], (r.x[2] if sigma is None else sigma)
def spline_root(x, y, i, half=4):
    s = slice(i - half, i + half + 1); cs = CubicSpline(x[s], y[s], bc_type='not-a-knot')
    xx = np.linspace(x[i - 1], x[i + 1], 4001); return xx[np.argmax(cs(xx))]
def parab(x, y, i):
    a, b, c = np.polyfit(x[i-1:i+2] - x[i], y[i-1:i+2], 2); return x[i] - b / (2 * a)
def bgsub(x, y, i):
    idx = np.r_[i-9:i-4, i+5:i+10]; a, b = np.polyfit(x[idx] - x[i], y[idx], 1); return y - (a * (x - x[i]) + b)
SEG = {'UV': (190, 362), 'VIS': (368, 617), 'NIR': (623, 945)}
dirs = sorted(glob.glob('/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/acq/run-*')) \
     + sorted(glob.glob('/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/vrun2/acq/*'))
rows = {s: [] for s in SEG}
for d in dirs:
    fs = sorted(glob.glob(d + '/shots/*.csv'))
    if not fs: continue
    A = [np.loadtxt(f, delimiter=',', skiprows=1) for f in fs]
    m = A[0][:, 0] < 955; x = A[0][m, 0]; y = np.mean([a[m, 1] for a in A], 0)
    for s, (lo, hi) in SEG.items():
        for i in np.where((x > lo) & (x < hi))[0]:
            if i < 12 or i > len(x) - 12 or y[i] != y[i-4:i+5].max(): continue
            yb = bgsub(x, y, i); q = np.r_[yb[i-9:i-4], yb[i+5:i+10]]; nz = q.std() + 1
            h = yb[i]
            if h / nz < 30: continue
            if max(yb[i-4:i-1].max(), yb[i+2:i+5].max()) > 0.6 * h: continue      # isolated, not flat-topped blend
            pitch = abs(x[i+1] - x[i])
            try:
                mu8, sg = pixint(x, yb, i)
            except Exception: continue
            if sg > 2.0 * pitch: continue
            r = dict(p8=mu8, sp=spline_root(x, yb, i), pa=parab(x, yb, i), pitch=pitch, sg=sg / pitch,
                     phi=((mu8 - x[i]) / pitch))
            rows[s].append(r)
for s in SEG:
    R = rows[s]; print(f"\n{s}: {len(R)} peaks; fitted sigma median {np.median([r['sg'] for r in R]):.2f} px")
    phi = np.array([r['phi'] for r in R]); pitch = np.array([r['pitch'] for r in R])
    for name, k in (('spline_root - pixint', 'sp'), ('parabola - pixint', 'pa')):
        dd = np.array([(r[k] - r['p8']) / r['pitch'] for r in R])     # in pixels
        bins = np.linspace(-0.5, 0.5, 11); med = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (phi >= a) & (phi < b); med.append(np.median(dd[sel]) if sel.sum() >= 5 else np.nan)
        med = np.array(med)
        print(f"  {name:22s} diff vs phase (px, bins -0.5..0.5): " + ' '.join(f"{v:+.3f}" for v in med)
              + f" | amplitude {np.nanmax(med)-np.nanmin(med):.3f} px = {1000*(np.nanmax(med)-np.nanmin(med))*np.median(pitch):.0f} pm p-p")
