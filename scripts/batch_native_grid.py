"""Batch native-grid recovery over a directory of vendor CSVs (parallel).

usage: python3 batch_native_grid.py OUT.csv DIR [DIR ...] [--workers N] [--limit N]
Writes one row per (file, segment) with the calibrated-mode diagnostics.
Imports alibz.utils.native_grid when alibz is installed; otherwise a copy of
native_grid.py next to this script (so it runs on a bare compute node with
only numpy/scipy).  BLAS threads are pinned to 1 per worker BEFORE numpy is
imported: 56 workers x default BLAS threads oversubscribe a 112-core box 20x.
"""
import csv, glob, os, sys, time, traceback
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # must precede the numpy import: 64 procs x BLAS threads oversubscribe
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
try:
    from alibz.utils import native_grid as ng
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import native_grid as ng

FIELDS = ["campaign", "file", "segment", "ok", "error", "exact", "relres", "relres_full", "delta_px", "stretch_ppm",
          "c0", "c1", "c2", "c3", "n_ref", "windows_used", "windows_total", "on_model_lo", "on_model_hi",
          "n_native", "pitch_start", "pitch_end", "max_window_relres", "seconds"]


def load(path):
    try:
        v = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
        return v[:, 0], v[:, 1]
    except Exception:
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split(",")
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except Exception:
                    continue
        a = np.array(rows)
        return a[:, 0], a[:, 1]


ROOT = ""


def one(path):
    t0 = time.time()
    out = []
    camp = os.path.relpath(os.path.dirname(path), ROOT) if ROOT else os.path.basename(os.path.dirname(path))
    try:
        x, y = load(path)
        xn, yn, info = ng.recover_native_grid(x, y, fallback=False)
        for name, seg in info["segments"].items():
            c = list(seg.get("correction_coef") or [np.nan] * 4)
            out.append(dict(campaign=camp, file=os.path.basename(path), segment=name, ok=1, error="",
                            exact=int(seg["exact"]), relres=seg["relres"], relres_full=seg["relres_full"],
                            delta_px=seg["delta_px"], stretch_ppm=seg["stretch_ppm"],
                            c0=c[0], c1=c[1], c2=c[2], c3=c[3], n_ref=seg.get("correction_n_ref"),
                            windows_used=seg["windows_used"], windows_total=seg["windows_total"],
                            on_model_lo=seg["on_model_range"][0], on_model_hi=seg["on_model_range"][1],
                            n_native=seg["n_native"], pitch_start=seg["pitch_start"], pitch_end=seg["pitch_end"],
                            max_window_relres=float(np.max(seg["window_relres"])),
                            seconds=time.time() - t0))
    except Exception as exc:
        out.append(dict(campaign=camp, file=os.path.basename(path), segment="", ok=0,
                        error=f"{type(exc).__name__}: {exc}"[:300], seconds=time.time() - t0))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]

    workers = 8; limit = None
    if "--workers" in args:
        i = args.index("--workers"); workers = int(args[i + 1]); del args[i:i + 2]
    if "--limit" in args:
        i = args.index("--limit"); limit = int(args[i + 1]); del args[i:i + 2]
    per_dir = None
    if "--per-dir" in args:
        i = args.index("--per-dir"); per_dir = int(args[i + 1]); del args[i:i + 2]
    recursive = "--recursive" in args
    if recursive:
        args.remove("--recursive")
    if "--root" in args:
        i = args.index("--root"); ROOT = args[i + 1]; del args[i:i + 2]
    out_csv, dirs = args[0], args[1:]
    if recursive:
        files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    else:
        files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.csv")))
    files = [f for f in files if not os.path.basename(f).lower().startswith(("detections", "summary"))]
    if per_dir:
        # deterministic per-directory sample: evenly spaced in sorted order
        bydir = {}
        for f in files:
            bydir.setdefault(os.path.dirname(f), []).append(f)
        picked = []
        for d, fs in sorted(bydir.items()):
            if len(fs) <= per_dir:
                picked.extend(fs)
            else:
                idx = np.linspace(0, len(fs) - 1, per_dir).round().astype(int)
                picked.extend(fs[i] for i in sorted(set(idx)))
        files = picked
    if limit:
        files = files[:limit]
    print(f"{len(files)} files, {workers} workers", flush=True)
    t0 = time.time()
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, f): f for f in files}
            for k, fut in enumerate(as_completed(futs), 1):
                for row in fut.result():
                    w.writerow({f: row.get(f, "") for f in FIELDS})
                fh.flush()
                if k % 50 == 0 or k == len(files):
                    print(f"  {k}/{len(files)} done ({time.time()-t0:.0f}s)", flush=True)
    print("wrote", out_csv, flush=True)
