"""Corpus-level PCA for background and detector artifact removal.

Runs PCA on full standardised spectra (not peak windows) to identify
common-mode features: detector junction artifacts, baseline shapes,
and instrument response.  These can be subtracted before peak fitting.

With ``--segment``, runs a separate PCA on each detector segment
(UV, VIS, NIR) using the junction positions from
``corrections/detector.json``.  This captures segment-specific
variance that the full-spectrum PCA may dilute.

Usage:
    background-pca /path/to/data --out data/bg_pca.pkl
    background-pca /path/to/data --segment --out data/bg_pca_segments.pkl
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

from alibz.peaky_corpus import PeakyCorpus
from alibz.gpu import gpu_available


# ---------------------------------------------------------------------------
# Detector config
# ---------------------------------------------------------------------------

_CORRECTIONS_DIR = Path(__file__).resolve().parent.parent / "corrections"
_DETECTOR_CONFIG = _CORRECTIONS_DIR / "detector.json"


def _load_detector_config(config_path=None):
    """Load junction positions and segment labels from detector config."""
    cfg_path = Path(config_path) if config_path else _DETECTOR_CONFIG
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core PCA routine (shared by full-spectrum and per-segment paths)
# ---------------------------------------------------------------------------

def _run_incremental_pca(spectra, n_components, batch_size, label=""):
    """Run IncrementalPCA on a spectra matrix, return results dict."""
    n_spectra, n_channels = spectra.shape
    prefix = f"  [{label}] " if label else "  "

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    t0 = time.time()
    for start in range(0, n_spectra, batch_size):
        end = min(start + batch_size, n_spectra)
        batch = np.array(spectra[start:end], dtype=np.float64)
        pca.partial_fit(batch)
        if (start // batch_size) % 5 == 0:
            print(f"{prefix}fitted {end}/{n_spectra}")

    print(f"{prefix}PCA fit done in {time.time() - t0:.1f}s")

    # Compute scores
    scores = np.zeros((n_spectra, n_components), dtype=np.float64)
    for start in range(0, n_spectra, batch_size):
        end = min(start + batch_size, n_spectra)
        batch = np.array(spectra[start:end], dtype=np.float64)
        scores[start:end] = pca.transform(batch)

    return {
        "components": pca.components_,
        "mean": pca.mean_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "singular_values": pca.singular_values_,
        "scores": scores,
        "n_spectra": n_spectra,
        "n_channels": n_channels,
    }


# ---------------------------------------------------------------------------
# Full-spectrum PCA
# ---------------------------------------------------------------------------

def _run_full(corpus, args):
    """Run PCA on the full wavelength range."""
    wl = corpus.common_wavelength
    n_spectra = corpus.spectra.shape[0]

    print(f"\n=== Running IncrementalPCA ({args.n_components} components, "
          f"batch_size={args.batch_size}) ===")
    print(f"  {n_spectra} spectra x {len(wl)} channels")

    result = _run_incremental_pca(
        corpus.spectra, args.n_components, args.batch_size)
    result["wavelength"] = wl
    result["csv_files"] = corpus.csv_files

    print(f"  Explained variance: {result['explained_variance_ratio']}")
    return result


# ---------------------------------------------------------------------------
# Per-segment PCA
# ---------------------------------------------------------------------------

def _run_segmented(corpus, args):
    """Run a separate PCA on each detector segment."""
    cfg = _load_detector_config()
    junctions = sorted(cfg["junctions_nm"])
    seg_cfgs = sorted(cfg.get("segments", []), key=lambda s: s["index"])

    wl = corpus.common_wavelength
    n_spectra = corpus.spectra.shape[0]

    # Build segment boundaries
    boundaries = [0]
    for jw in junctions:
        boundaries.append(int(np.argmin(np.abs(wl - jw))))
    boundaries.append(len(wl))

    results = {
        "mode": "segmented",
        "junctions_nm": junctions,
        "wavelength": wl,
        "csv_files": corpus.csv_files,
        "n_spectra": n_spectra,
        "segments": {},
    }

    for i, seg_cfg in enumerate(seg_cfgs):
        label = seg_cfg.get("label", str(i))
        idx_lo, idx_hi = boundaries[i], boundaries[i + 1]
        seg_wl = wl[idx_lo:idx_hi]
        seg_spectra = corpus.spectra[:, idx_lo:idx_hi]
        n_ch = idx_hi - idx_lo

        n_comp = min(args.n_components, n_spectra, n_ch)

        print(f"\n=== Segment {label} ({seg_wl[0]:.1f}-{seg_wl[-1]:.1f} nm, "
              f"{n_ch} channels, {n_comp} components) ===")

        seg_result = _run_incremental_pca(
            seg_spectra, n_comp, args.batch_size, label=label)
        seg_result["wavelength"] = seg_wl
        seg_result["label"] = label
        seg_result["idx_lo"] = idx_lo
        seg_result["idx_hi"] = idx_hi

        print(f"  [{label}] Explained variance: {seg_result['explained_variance_ratio']}")
        print(f"  [{label}] Cumulative: {np.cumsum(seg_result['explained_variance_ratio'])}")

        results["segments"][label] = seg_result

    return results


# ---------------------------------------------------------------------------
# Junction analysis (shared)
# ---------------------------------------------------------------------------

def _print_junction_analysis(result):
    """Print which components are concentrated at junction zones."""
    cfg = _load_detector_config()
    zone_hw = cfg.get("junction_zone_half_width_nm", 5.0)

    if result.get("mode") == "segmented":
        # For segmented mode, junctions are at segment edges — check
        # each segment's first/last few PCs for boundary concentration
        for label, seg in result["segments"].items():
            wl = seg["wavelength"]
            evr = seg["explained_variance_ratio"]
            print(f"\n  [{label}] {wl[0]:.1f}-{wl[-1]:.1f} nm: "
                  f"{len(evr)} PCs, "
                  f"cumulative = {evr.sum():.1%}")
    else:
        wl = result["wavelength"]
        components = result["components"]
        evr = result["explained_variance_ratio"]

        for jw in cfg["junctions_nm"]:
            zone = (wl >= jw - zone_hw) & (wl <= jw + zone_hw)
            zone_frac = zone.sum() / len(wl)
            print(f"\n  Junction {jw:.0f} nm "
                  f"(zone {jw-zone_hw:.0f}-{jw+zone_hw:.0f}, "
                  f"{zone_frac:.1%} of channels):")
            for i in range(min(10, len(evr))):
                pc = components[i]
                total = np.sum(pc ** 2)
                jfrac = np.sum(pc[zone] ** 2) / total if total > 0 else 0
                tag = " ARTIFACT" if jfrac > 2 * zone_frac else ""
                print(f"    PC{i+1}: {evr[i]:.4%} variance, "
                      f"{jfrac:.1%} in zone{tag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dirs', nargs='+',
                        help='Root directories of LIBS CSV spectra')
    parser.add_argument('--pattern', default='**/*AverageSpectrum.csv')
    parser.add_argument('--wl-min', type=float, default=190.0)
    parser.add_argument('--wl-max', type=float, default=910.0)
    parser.add_argument('--wl-step', type=float, default=0.01)
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of PCA components to retain (per segment '
                             'when --segment is used)')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--segment', action='store_true', default=False,
                        help='Run a separate PCA on each detector segment '
                             '(UV/VIS/NIR) instead of the full spectrum')
    parser.add_argument('--out', default='data/bg_pca.pkl')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--no-gpu', action='store_true', default=False)
    args = parser.parse_args()

    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = None

    t0 = time.time()

    # Load and standardise
    print(f"=== Loading corpus from {len(args.data_dirs)} directories ===")
    corpus = PeakyCorpus(
        args.data_dirs if len(args.data_dirs) > 1 else args.data_dirs[0],
        wl_min=args.wl_min, wl_max=args.wl_max, wl_step=args.wl_step,
        memmap=True, pattern=args.pattern, use_gpu=use_gpu,
    )
    corpus.load_corpus()

    print(f"\n=== Standardizing to {corpus.n_channels} channels ===")
    corpus.standardize_all()

    # Run PCA
    if args.segment:
        result = _run_segmented(corpus, args)
    else:
        result = _run_full(corpus, args)

    # Junction analysis
    print("\n=== Junction analysis ===")
    _print_junction_analysis(result)

    # Save
    import os
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'wb') as f:
        pickle.dump(result, f)

    elapsed = time.time() - t0
    print(f"\nSaved to {args.out}")
    mode = "segmented" if args.segment else "full-spectrum"
    print(f"Mode: {mode}, total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
