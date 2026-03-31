"""Corpus-level PCA for background and detector artifact removal.

Runs PCA on full standardised spectra (not peak windows) to identify
common-mode features: detector junction artifacts, baseline shapes,
and instrument response.  These can be subtracted before peak fitting.

Usage:
    python -m alibz.background_pca /path/to/data --out bg_pca.pkl
"""

import argparse
import pickle
import time
import numpy as np
from sklearn.decomposition import IncrementalPCA

from alibz.peaky_corpus import PeakyCorpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dirs', nargs='+',
                        help='Root directories of LIBS CSV spectra')
    parser.add_argument('--pattern', default='**/*AverageSpectrum.csv')
    parser.add_argument('--wl-min', type=float, default=190.0)
    parser.add_argument('--wl-max', type=float, default=910.0)
    parser.add_argument('--wl-step', type=float, default=0.01)
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of PCA components to retain')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--out', default='bg_pca.pkl')
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

    n_spectra = corpus.spectra.shape[0]
    n_channels = corpus.spectra.shape[1]
    print(f"  {n_spectra} spectra × {n_channels} channels")

    # IncrementalPCA on full spectra
    print(f"\n=== Running IncrementalPCA ({args.n_components} components, "
          f"batch_size={args.batch_size}) ===")

    pca = IncrementalPCA(n_components=args.n_components,
                         batch_size=args.batch_size)

    t_pca = time.time()
    for start in range(0, n_spectra, args.batch_size):
        end = min(start + args.batch_size, n_spectra)
        batch = np.array(corpus.spectra[start:end], dtype=np.float64)
        pca.partial_fit(batch)
        if (start // args.batch_size) % 5 == 0:
            print(f"  fitted {end}/{n_spectra}")

    print(f"  PCA fit done in {time.time() - t_pca:.1f}s")

    # Transform: get scores for all spectra
    print(f"\n=== Computing scores ===")
    scores = np.zeros((n_spectra, args.n_components), dtype=np.float64)
    for start in range(0, n_spectra, args.batch_size):
        end = min(start + args.batch_size, n_spectra)
        batch = np.array(corpus.spectra[start:end], dtype=np.float64)
        scores[start:end] = pca.transform(batch)

    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    # Identify artifact-heavy components by checking their loading
    # in the 600-650 nm junction zone
    wl = corpus.common_wavelength
    junction_mask = (wl >= 600) & (wl <= 650)
    junction_frac = junction_mask.sum() / len(wl)

    print(f"\n=== Junction zone analysis (600-650 nm = {junction_frac:.1%} of channels) ===")
    for i in range(min(10, args.n_components)):
        pc = pca.components_[i]
        # Fraction of component's total variance in the junction zone
        total_var = np.sum(pc ** 2)
        junction_var = np.sum(pc[junction_mask] ** 2)
        junction_frac_pc = junction_var / total_var if total_var > 0 else 0
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4%} variance, "
              f"{junction_frac_pc:.1%} in junction zone "
              f"({'ARTIFACT?' if junction_frac_pc > 2 * junction_frac else 'ok'})")

    # Save results
    results = {
        'components': pca.components_,           # (n_comp, n_channels)
        'mean': pca.mean_,                       # (n_channels,)
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'scores': scores,                        # (n_spectra, n_comp)
        'wavelength': wl,
        'csv_files': corpus.csv_files,
        'n_spectra': n_spectra,
        'n_channels': n_channels,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nSaved to {args.out}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
