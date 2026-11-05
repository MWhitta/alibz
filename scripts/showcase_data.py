#!/usr/bin/env python3
"""Named, sha256-verified access to the manuscript showcase spectra.

    python scripts/showcase_data.py --list | --check | --stage | --build

load(key, root=None) -> (x, y, entry).  Drive files live under DRIVE_PREFIX;
ALIBZ_SHOWCASE_ROOT (or root=) replaces that prefix on another machine.  A
staged copy data/showcase/<key>.csv, when present, is preferred and verified
against the manifest's staged_sha256.  --build regenerates the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
MANIFEST = REPO / "manuscript" / "data" / "showcase.json"
STAGE_DIR = REPO / "data" / "showcase"
DRIVE_PREFIX = "/Users/mwhittaker/Library/CloudStorage/GoogleDrive-mwhittaker@lbl.gov/My Drive"
FE_NPZ = "data/triage-real/fe_aesar_9998_means_20260922.npz"
JC = DRIVE_PREFIX + "/CMI/Data/LIBS/JChristensen/2026-07-01/"
MW2 = DRIVE_PREFIX + "/Postdocs/Xuan Cao/Data/LIBS/MW2-112/raw/"

# Static specification used by --build (everything else is computed).
SPEC = [
    dict(key="mw2_1000", loader="vendor_csv", grid="vendor 1/30 nm",
         role="stage-by-stage walkthrough; refinement zooms (K I 766.5 phantom split, Li I 670.8 SA tag)",
         source=MW2 + "1000-Test #1000 10 38, Mar 25 2024.03.25.10.38.csv"),
    dict(key="jc_mdd017_24_15_2nd_1", loader="vendor_csv", grid="vendor 1/30 nm",
         role="confounder contest: Mn on two peaks inside the Mg II doublet -> confounded(Mg) -> resolved",
         source=JC + "MDD017-24-15 2nd #1 Kaleb_20260617_110533_AM_AverageSpectrum.csv"),
    dict(key="jc_alaskire", loader="vendor_csv", grid="vendor 1/30 nm",
         role="K-dominant dominant-weak-shape case: self-absorbed K I 766/770 doublet; segment-response sensitivity",
         source=JC + "Alaskire MDD27 box 26_20260614_083522_AM_AverageSpectrum.csv"),
    dict(key="jc_mdd017_29_2c_1", loader="vendor_csv", grid="vendor 1/30 nm",
         role="Mn/Mg confounder contest (current defaults): Mn 0.102 as-fit -> confounded(Mg) -> 0.000 resolved (largest fraction change of 6 confounded rows in the 38-spectrum survey)",
         source=JC + "MDD017-29-2c #1 Kaleb_20260617_102644_AM_AverageSpectrum.csv"),
    dict(key="user_spec1", loader="vendor_csv", grid="vendor 1/30 nm",
         role="full-spectrum overview (data, background, model, residual/sigma, three segments)",
         source="data/remote_samples/user_spec1.csv"),
    dict(key="fe_fresh_d5p25", loader="fe_npz", grid="Z300 native",
         role="known composition: Fe I / Fe II Saha-Boltzmann stage tie; composition bar",
         source=FE_NPZ, run_id="run-07c25af6dd9b4fa6bc801e18a40661a4", run_index=17,
         pixel_offset=18, delay=5, period=25, site="raster fresh"),
    dict(key="fe_reused_fail", loader="fe_npz", grid="Z300 native",
         role="counterexample: Fe fraction 0.000 on the same defaults (failed native-grid fit)",
         source=FE_NPZ, run_id="run-674f4feb6d2149c4bac365e395a4d327", run_index=9,
         pixel_offset=18, delay=5, period=25, site="reused spot"),
]


class ShowcaseError(RuntimeError):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _manifest() -> dict:
    return {e["key"]: e for e in json.loads(MANIFEST.read_text())["entries"]}


def _resolve(src: str, root) -> Path:
    root = root or os.environ.get("ALIBZ_SHOWCASE_ROOT")
    if src.startswith(DRIVE_PREFIX):
        if root:
            src = str(root).rstrip("/") + src[len(DRIVE_PREFIX):]
        return Path(src)
    return REPO / src


def _read_fe(entry):
    from scripts.fit_quality_panel import load_fe
    return load_fe(entry["run_id"], pixel_offset=entry["pixel_offset"])


def _read_csv(path):
    from alibz.pipeline import load_spectrum_csv
    x, y = load_spectrum_csv(str(path))
    return np.asarray(x, float), np.asarray(y, float)


def load(key, root=None):
    entries = _manifest()
    if key not in entries:
        raise ShowcaseError(f"unknown showcase key {key!r}; known: {sorted(entries)}")
    e = entries[key]
    staged = STAGE_DIR / f"{key}.csv"
    if staged.is_file():
        want, path = e.get("staged_sha256"), staged
    else:
        want, path = e["sha256"], _resolve(e["source"], root)
    if not path.is_file():
        raise ShowcaseError(f"[{key}] file not found: {path}")
    try:
        got = sha256(path)
    except OSError as exc:
        raise ShowcaseError(f"[{key}] cannot read {path}: {exc}") from exc
    if want is None or got != want:
        raise ShowcaseError(f"[{key}] sha256 mismatch for {path}: got {got}, manifest {want}")
    if staged.is_file() or e["loader"] == "vendor_csv":
        x, y = _read_csv(path)
    else:
        x, y = _read_fe(e)
    return x, y, e


def stats(x, y, grid):
    med = float(np.median(y))
    return dict(n_points=int(x.size), wl_min_nm=float(x.min()), wl_max_nm=float(x.max()),
                max_counts=float(y.max()), negative_fraction=float(np.mean(y < 0)),
                median_abs_dev_noise=float(1.4826 * np.median(np.abs(y - med))), grid=grid)


def build():
    entries = []
    old = _manifest() if MANIFEST.is_file() else {}
    for s in SPEC:
        e = dict(s)
        p = _resolve(e["source"], None)
        e["sha256"], e["size_bytes"] = sha256(p), p.stat().st_size
        x, y = _read_fe(e) if e["loader"] == "fe_npz" else _read_csv(p)
        e["stats"] = stats(x, y, e["grid"])
        if "staged_sha256" in old.get(e["key"], {}):
            e["staged_sha256"] = old[e["key"]]["staged_sha256"]
        entries.append(e)
    MANIFEST.write_text(json.dumps({"drive_prefix": DRIVE_PREFIX, "entries": entries}, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)} ({len(entries)} entries)")


def stage():
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(MANIFEST.read_text())
    for e in data["entries"]:
        dst = STAGE_DIR / f"{e['key']}.csv"
        if e["loader"] == "vendor_csv" and e["source"].startswith(DRIVE_PREFIX):
            src = _resolve(e["source"], None)
            if sha256(src) != e["sha256"]:
                raise ShowcaseError(f"[{e['key']}] source sha256 mismatch: {src}")
            shutil.copyfile(src, dst)
        elif e["loader"] == "fe_npz":
            x, y = _read_fe(e)
            with open(dst, "w") as fh:
                fh.write(f"# run_id={e['run_id']} pixel_offset={e['pixel_offset']}\n")
                for a, b in zip(x, y):
                    fh.write(f"{float(a)!r},{float(b)!r}\n")
        else:
            continue
        e["staged_sha256"] = sha256(dst)
        print(f"staged {e['key']} -> {dst.relative_to(REPO)}")
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def check():
    bad = 0
    for k in _manifest():
        try:
            x, y, e = load(k)
            if x.size != e["stats"]["n_points"]:
                raise ShowcaseError(f"[{k}] n_points {x.size} != manifest {e['stats']['n_points']}")
            print(f"OK   {k}  n={x.size}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"FAIL {k}  {exc}")
    return 1 if bad else 0


def list_():
    print(f"{'key':24s} {'grid':15s} {'n_points':>8s}  source | role")
    for e in _manifest().values():
        print(f"{e['key']:24s} {e['stats']['grid']:15s} {e['stats']['n_points']:8d}  "
              f"{Path(e['source']).name[:50]} | {e['role'][:60]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("list", "check", "stage", "build"):
        g.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        list_()
    elif a.check:
        return check()
    elif a.stage:
        stage()
    else:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
