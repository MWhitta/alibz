# alibz

LIBS (Laser-Induced Breakdown Spectroscopy) data analysis toolkit for spectral peak finding, Voigt profile fitting, elemental indexing, and PCA-based peak-shape decomposition.

## Installation

```bash
pip install -e .
```

This installs `alibz` as an editable package with all dependencies (numpy, scipy, scikit-learn, scikit-optimize, matplotlib, pulp).

For GPU acceleration (optional):

```bash
pip install -e ".[gpu]"
```

## Modules

| Module | Description |
|---|---|
| `alibz.peaky_finder` | Peak detection, FFT background removal, multi-Voigt fitting |
| `alibz.peaky_indexer_v3` | Whole-pattern spectral indexer (Bayesian optimisation + NNLS); the only supported indexer |
| `alibz.peaky_maker` | Forward spectral synthesis via Saha-Boltzmann |
| `alibz.peaky_corpus` | Batch loading, standardisation, parallel fitting, width statistics |
| `alibz.peaky_pca` | PCA peak-shape decomposition and broadening classification |
| `alibz.background_pca` | Corpus-level PCA for detector artifact and baseline removal |
| `alibz.utils` | Physical constants, Voigt utilities, NIST database interface, Saha-Boltzmann solver |

Legacy `alibz.peaky_indexer` and `alibz.peaky_indexer_v2` are deprecated
compatibility shims that route to v3. `PeakyFitter` has been removed.

## Quick Start

```python
from alibz import PeakyFinder, PeakyIndexer

# Load data and fit peaks
finder = PeakyFinder("path/to/raw/data")
fit_dict = finder.fit_spectrum_data(0, n_sigma=1, plot=False)

# Whole-pattern indexing
indexer = PeakyIndexer(fit_dict["sorted_parameter_array"])
result = indexer.run(shift_tolerance=0.1, max_ion_stage=2, n_calls=20, verbose=False)

print(result.temperature, result.ne)       # plasma T and log10(ne)
print(result.concentrations)               # per-species concentrations
```

The indexer returns a `FitResult` dataclass with temperature, electron density,
per-species concentrations, residuals, R-squared, and peak assignments.

## CLI

### Peak-shape PCA pipeline

```bash
run-corpus-pca /path/to/libs/data \
    --pattern '**/*AverageSpectrum.csv' \
    --n-components 5 \
    --workers 4 \
    --timeout 120 \
    --fit-checkpoint fits.pkl \
    --out results.pkl
```

### Background / artifact PCA

```bash
background-pca /path/to/libs/data \
    --n-components 20 \
    --batch-size 200 \
    --out bg_pca.pkl
```

Both commands accept `--gpu` / `--no-gpu` flags and support multiple input directories.

## Notebooks

- `peaky_demo_v1.ipynb` -- Peak finding, fitting, and elemental indexing workflow
- `peaky_pca_demo.ipynb` -- PCA peak-shape analysis and broadening mechanism classification

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

[Unlicense](LICENSE) (public domain)
