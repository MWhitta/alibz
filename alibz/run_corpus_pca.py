#!/usr/bin/env python
"""Run the peak-shape PCA pipeline on a LIBS corpus.

Usage (on moissanite):
    python run_corpus_pca.py /path/to/libs/data [--pattern '**/*AverageSpectrum.csv'] [--out results.pkl]
    python run_corpus_pca.py /dir1 /dir2 /dir3 --gpu --out results.pkl

Produces a .pkl file containing width statistics, PCA components,
scores, decompositions, and peak metadata.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

from alibz.peaky_corpus import PeakyCorpus
from alibz.peaky_pca import PeakyPCA
from alibz.gpu import gpu_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dirs', nargs='+',
                        help='Root directories of LIBS CSV spectra '
                             '(multiple directories are combined)')
    parser.add_argument('--pattern', default='**/*AverageSpectrum.csv',
                        help='Glob pattern to filter CSV files')
    parser.add_argument('--wl-min', type=float, default=190.0)
    parser.add_argument('--wl-max', type=float, default=910.0)
    parser.add_argument('--wl-step', type=float, default=0.01)
    parser.add_argument('--n-sigma', type=float, default=0,
                        help='Peak detection threshold')
    parser.add_argument('--n-components', type=int, default=5,
                        help='Number of PCA components')
    parser.add_argument('--window-multiplier', type=float, default=3.0,
                        help='Half-window width in multiples of median FWHM')
    parser.add_argument('--half-window-nm', type=float, default=None,
                        help='Fixed half-window width in nm (overrides --window-multiplier)')
    parser.add_argument('--out', default='corpus_pca_results.pkl',
                        help='Output pickle file')
    parser.add_argument('--no-memmap', action='store_true',
                        help='Disable memory mapping (load everything into RAM)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel CPU workers for peak fitting '
                             '(default: 1 = sequential)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-spectrum fit timeout in seconds '
                             '(default: 120, 0 to disable)')
    parser.add_argument('--fit-checkpoint', default=None,
                        help='Path to save/load fit results checkpoint. '
                             'If the file exists and matches the corpus, '
                             'fitting is skipped.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Force GPU acceleration (auto-detected if omitted)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Disable GPU even if available')
    args = parser.parse_args()

    # Resolve GPU preference
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
        if not gpu_available:
            print("WARNING: --gpu requested but no CUDA device found. "
                  "Falling back to CPU.")
            use_gpu = False
    else:
        use_gpu = None  # auto-detect

    t_start = time.time()
    dirs_str = ', '.join(args.data_dirs)
    print(f"=== Loading corpus from {len(args.data_dirs)} director{'y' if len(args.data_dirs) == 1 else 'ies'}: {dirs_str} ===")
    if gpu_available:
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"  GPU: {n_gpus} CUDA device(s) detected")

    corpus = PeakyCorpus(
        args.data_dirs if len(args.data_dirs) > 1 else args.data_dirs[0],
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        wl_step=args.wl_step,
        memmap=not args.no_memmap,
        pattern=args.pattern,
        use_gpu=use_gpu,
    )
    corpus.load_corpus()

    print(f"\n=== Standardizing to {corpus.n_channels} channels "
          f"({args.wl_min}-{args.wl_max} nm, step {args.wl_step} nm) ===")
    corpus.standardize_all()

    # --- Fit peaks (with optional checkpoint) ---
    loaded_checkpoint = False
    if args.fit_checkpoint:
        print(f"\n=== Checking fit checkpoint: {args.fit_checkpoint} ===")
        loaded_checkpoint = corpus.load_fit_results(args.fit_checkpoint)

    if not loaded_checkpoint:
        w_label = f"{args.workers} workers" if args.workers > 1 else "sequential"
        print(f"\n=== Fitting peaks (n_sigma={args.n_sigma}, {w_label}) ===")
        corpus.fit_all_spectra(n_sigma=args.n_sigma, workers=args.workers,
                               timeout=args.timeout or None)

        if args.fit_checkpoint:
            print(f"\n=== Saving fit checkpoint ===")
            corpus.save_fit_results(args.fit_checkpoint)

    print(f"\n=== Width statistics ===")
    stats = corpus.peak_width_statistics()
    print(f"  Min  FWHM: {stats['min']:.4f} nm")
    print(f"  Max  FWHM: {stats['max']:.4f} nm")
    print(f"  Mean FWHM: {stats['mean']:.4f} nm")
    print(f"  Median FWHM: {stats['median']:.4f} nm")
    print(f"  Std  FWHM: {stats['std']:.4f} nm")

    print(f"\n=== Multimodality check ===")
    corpus.detect_width_modes()
    print(f"  Smallest mode mean: {stats['smallest_mode_mean']:.4f} nm")

    hw_label = (f"{args.half_window_nm} nm" if args.half_window_nm
                else f"{args.window_multiplier}x median FWHM")
    print(f"\n=== PCA ({args.n_components} components, half-window={hw_label}) ===")
    pca_analyzer = PeakyPCA(
        corpus,
        window_multiplier=args.window_multiplier,
        n_components=args.n_components,
        half_window_nm=args.half_window_nm,
        use_gpu=use_gpu,
        workers=args.workers,
    )
    pca_analyzer.extract_peak_windows()
    print(f"  Half-window: {pca_analyzer.half_window:.4f} nm")
    print(f"  Extracted {pca_analyzer.windows.shape[0]} peak windows")

    pca_analyzer.fit_pca()
    print(f"  Explained variance: {pca_analyzer.explained_variance_ratio}")

    print(f"\n=== Characterising mean peak ===")
    mf = pca_analyzer.characterize_mean_peak()
    print(f"  Offset: {pca_analyzer.mean_offset:.4f} (subtracted so min=0)")
    print(f"  Fit region: [{mf['fit_left']}, {mf['fit_right']}] of {pca_analyzer.n_window_points} pts")
    print(f"  sigma={mf['sigma']:.4f}  gamma={mf['gamma']:.4f}  "
          f"tau={mf['tau']:.4f}  FWHM={mf['fwhm']:.4f}")
    print(f"  Gaussian fraction: {mf['gaussian_fraction']:.1%}")
    print(f"  Residual norm: {mf['residual_norm']:.4f}")

    print(f"\n=== Decomposing components (perturbation analysis) ===")
    pca_analyzer.decompose_all_components()
    for i, d in enumerate(pca_analyzer.decompositions):
        print(f"  PC{i+1}: d_sigma={d['d_sigma']:+.4f}  "
              f"d_gamma={d['d_gamma']:+.4f}  "
              f"d_tau={d['d_tau']:+.4f}  "
              f"-- {d['physical_interpretation']}")

    print(f"\n=== Classifying peaks ===")
    labels = pca_analyzer.classify_peaks()
    from collections import Counter
    counts = Counter(labels)
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

    # Save results — store score summary stats instead of the full
    # 13M x n_comp array to keep the pickle small and fast to load.
    all_scores = pca_analyzer.scores
    score_stats = {
        'mean': np.mean(all_scores, axis=0),
        'std': np.std(all_scores, axis=0),
        'min': np.min(all_scores, axis=0),
        'max': np.max(all_scores, axis=0),
        'percentiles': {
            q: np.percentile(all_scores, q, axis=0)
            for q in [1, 5, 25, 50, 75, 95, 99]
        },
        'n_samples': all_scores.shape[0],
    }

    results = {
        'csv_files': corpus.csv_files,
        'width_stats': {k: v for k, v in stats.items()},
        'components': pca_analyzer.components,
        'explained_variance_ratio': pca_analyzer.explained_variance_ratio,
        'mean_peak': pca_analyzer.mean_peak,
        'mean_peak_zeroed': pca_analyzer.mean_peak_zeroed,
        'mean_offset': pca_analyzer.mean_offset,
        'mean_fit': pca_analyzer.mean_fit,
        'score_stats': score_stats,
        'decompositions': pca_analyzer.decompositions,
        'peak_metadata': pca_analyzer.peak_metadata,
        'peak_classifications': labels,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)

    elapsed = time.time() - t_start
    m, s = divmod(elapsed, 60)
    print(f"\nResults saved to {args.out}")
    print(f"Total time: {int(m)}m {s:.1f}s"
          f" (GPU: {'yes' if corpus.use_gpu else 'no'})")


if __name__ == '__main__':
    main()
