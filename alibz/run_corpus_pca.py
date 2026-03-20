#!/usr/bin/env python
"""Run the peak-shape PCA pipeline on a LIBS corpus.

Usage (on moissanite):
    python run_corpus_pca.py /path/to/libs/data [--pattern '**/*AverageSpectrum.csv'] [--out results.npz]

Produces an .npz file containing width statistics, PCA components,
scores, decompositions, and peak metadata.
"""

import argparse
import pickle
import sys

import numpy as np

from alibz.peaky_corpus import PeakyCorpus
from alibz.peaky_pca import PeakyPCA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', help='Root directory of LIBS CSV spectra')
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
    args = parser.parse_args()

    print(f"=== Loading corpus from {args.data_dir} ===")
    corpus = PeakyCorpus(
        args.data_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        wl_step=args.wl_step,
        memmap=not args.no_memmap,
        pattern=args.pattern,
    )
    corpus.load_corpus()

    print(f"\n=== Standardizing to {corpus.n_channels} channels "
          f"({args.wl_min}-{args.wl_max} nm, step {args.wl_step} nm) ===")
    corpus.standardize_all()

    print(f"\n=== Fitting peaks (n_sigma={args.n_sigma}) ===")
    corpus.fit_all_spectra(n_sigma=args.n_sigma)

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

    # Save results
    results = {
        'csv_files': corpus.csv_files,
        'width_stats': {k: v for k, v in stats.items()},
        'components': pca_analyzer.components,
        'explained_variance_ratio': pca_analyzer.explained_variance_ratio,
        'mean_peak': pca_analyzer.mean_peak,
        'mean_peak_zeroed': pca_analyzer.mean_peak_zeroed,
        'mean_offset': pca_analyzer.mean_offset,
        'mean_fit': pca_analyzer.mean_fit,
        'scores': pca_analyzer.scores,
        'decompositions': pca_analyzer.decompositions,
        'peak_metadata': pca_analyzer.peak_metadata,
        'peak_classifications': labels,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
