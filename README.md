# alibz

LIBS (Laser-Induced Breakdown Spectroscopy) data analysis toolkit for spectral peak finding, Voigt profile fitting, elemental indexing, and PCA-based peak-shape decomposition.

## Installation

```bash
pip install -e .
```

This installs `alibz` as an editable package with all dependencies (numpy, scipy, scikit-learn, matplotlib, pulp).

## Modules

| Module | Description |
|---|---|
| `alibz.peaky_finder` | Peak detection, FFT background removal, multi-Voigt fitting |
| `alibz.peaky_indexer` | Peak-to-element matching and MILP spectrum/reference scaling |
| `alibz.peaky_fitter` | Extended fitting with temperature estimation |
| `alibz.peaky_maker` | Forward spectral synthesis via Saha-Boltzmann |
| `alibz.peaky_corpus` | Batch loading, standardization, width statistics |
| `alibz.peaky_pca` | PCA peak-shape decomposition and broadening classification |
| `alibz.utils` | Shared Voigt utilities, physical constants, database loader |

## Quick Start

```python
from alibz import PeakyFinder, PeakyIndexer

# Load data and fit spectra
finder = PeakyFinder("path/to/raw/data")
results = finder.fit_spectrum_data(sample_index, n_sigma=1)

# Match peaks to elemental database entries
indexer = PeakyIndexer(finder)
matches = indexer.peak_match(
    results["sorted_parameter_array"],
    element_list=["Li", "Na", "Ca", "K", "Rb", "Cs", "Ba"],
)
```

## Notebooks

- `peaky_demo_v1.ipynb` — Peak finding, fitting, and elemental indexing workflow
- `peaky_pca_demo.ipynb` — PCA peak-shape analysis and broadening mechanism classification

## CLI

```bash
run-corpus-pca /path/to/libs/data --out results.pkl
```

Runs the full corpus PCA pipeline (load, standardize, fit, width stats, PCA, decomposition).

## Tests

```bash
python -m unittest discover -s tests
```

## License

[Unlicense](LICENSE) (public domain)
