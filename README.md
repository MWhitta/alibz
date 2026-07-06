# alibz

LIBS (Laser-Induced Breakdown Spectroscopy) data analysis toolkit for spectral peak finding, Voigt profile fitting, elemental indexing, production directory analysis, and PCA-based peak-shape decomposition.

## Installation

`alibz` requires Python 3.9 or newer.

### From GitHub

Install the current `main` branch from any working directory:

```bash
python -m pip install "alibz @ git+https://github.com/MWhitta/alibz.git@main"
```

This installs the Python package, command-line tools, scientific dependencies, and the bundled atomic database. The default database path (`"db"`) works even when Python or the CLI is launched outside the source checkout.

Optional extras:

```bash
python -m pip install "alibz[notebook] @ git+https://github.com/MWhitta/alibz.git@main"
python -m pip install "alibz[gpu] @ git+https://github.com/MWhitta/alibz.git@main"
```

Use the `notebook` extra when you want `alibz-analyze` to execute its generated inspection notebook. Use the `gpu` extra only on machines with a compatible CUDA 12 runtime.

### Development Checkout

```bash
git clone https://github.com/MWhitta/alibz.git
cd alibz
python -m pip install -e ".[dev,notebook]"
```

The database lookup order is:

1. an explicit `dbpath` argument or CLI `--db` path
2. `$ALIBZ_DB`
3. `./db`
4. the source checkout `db`
5. the installed bundled database

To force a specific database:

```bash
export ALIBZ_DB=/path/to/alibz/db
```

## Use

### Directory Analysis

For routine use, run the production pipeline on a directory of two-column wavelength/intensity CSV files:

```bash
alibz-analyze /path/to/libs/data \
    --pattern '*AverageSpectrum.csv' \
    --workers 4 \
    --n-calls 40
```

This writes:

- `summary.csv`: one row per spectrum with fitted plasma parameters, detected-element fractions, per-element statistical uncertainties, and diagnostic flags
- `detections.csv`: long-format per-sample/per-element detection status, line-support evidence, z-scores, and upper limits for non-detections or borderline elements
- `fit_inspection.ipynb`: an inspection notebook for one live spectrum, including fit plots, refinement decisions, seeded minor lines, and composition charts

If notebook dependencies are not installed, either install the `notebook` extra or pass `--no-execute`.

Self-absorption optical depths use the current default convention unless you pass `--stimulated-emission` to include the induced-emission factor.

### Python API

Lower-level peak fitting and indexing:

```python
from alibz import PeakyFinder, PeakyIndexer
from alibz.utils.stark import halpha_ne_bounds

finder = PeakyFinder("/path/to/libs/data")
fit_dict = finder.fit_spectrum_data(0, n_sigma=1, plot=False)
peaks = fit_dict["sorted_parameter_array"]

run_kwargs = {
    "shift_tolerance": 0.1,
    "max_ion_stage": 2,
    "n_calls": 20,
    "sa_doublets": True,
    "verbose": False,
}

indexer_kwargs = {}
ne_bounds = halpha_ne_bounds(peaks)
if ne_bounds is not None:
    indexer_kwargs["ne_init"] = 0.5 * (ne_bounds[0] + ne_bounds[1])
    run_kwargs["ne_bounds"] = ne_bounds

indexer = PeakyIndexer(peaks, **indexer_kwargs)
result = indexer.run(**run_kwargs)

print(result.temperature, result.ne)       # T [K], log10(ne / cm^-3)
print(result.element_fractions)            # detected-element fractions
print(result.stage_disagreement)           # LTE / stage-consistency diagnostic
```

`PeakyIndexer` returns `FitResult`. `element_fractions` are normalized over active detected emitters in the fitted model. They are useful quantitative estimates for research workflows, not certified material compositions; systematic error from LTE assumptions, atomic-data quality, matrix effects, and unresolved self-absorption is not included in the statistical uncertainty.

## Modules

| Module | Description |
|---|---|
| `alibz.pipeline` / `alibz-analyze` | End-to-end directory analysis, summary CSV output, inspection notebook generation |
| `alibz.peaky_finder` | Peak detection, arPLS background removal, multi-Voigt fitting |
| `alibz.refinement` | Blend vs self-absorption refinement for ambiguous peak features |
| `alibz.minor_lines` | Prior-driven fitting of minor lines from established elements |
| `alibz.peaky_indexer_v3` | Experimental whole-pattern spectral indexer (Bayesian optimisation + NNLS); the only supported indexer |
| `alibz.peaky_maker` | Forward spectral synthesis via Saha-Boltzmann |
| `alibz.peaky_corpus` | Batch loading, standardisation, parallel fitting, width statistics |
| `alibz.peaky_pca` | PCA peak-shape decomposition and broadening classification |
| `alibz.background_pca` | Corpus-level PCA for detector artifact and baseline removal |
| `alibz.detector` | Three-segment detector model: junction detection, artifact removal, segmented background subtraction |
| `alibz.utils` | Physical constants, Voigt utilities, NIST database interface, Saha-Boltzmann solver |

`alibz.peaky_indexer_v3` is the only indexer module. `PeakyFitter` has been removed from the public API.

## Additional CLI Tools

Peak-shape PCA pipeline:

```bash
run-corpus-pca /path/to/libs/data \
    --pattern '**/*AverageSpectrum.csv' \
    --n-components 5 \
    --workers 4 \
    --timeout 120 \
    --fit-checkpoint data/fits.pkl \
    --out data/results.pkl
```

Background / artifact PCA:

```bash
background-pca /path/to/libs/data \
    --n-components 20 \
    --batch-size 200 \
    --out data/bg_pca.pkl
```

Both commands accept `--gpu` / `--no-gpu` flags and support multiple input directories.

## Notebooks

- `notebooks/demo_notebook.ipynb`: end-to-end production demo — one-call `analyze_spectrum`, the confounder-resolved detection report, fit-inspection plots, and the directory/`alibz-analyze` workflow
- `notebooks/fit_inspection.ipynb`: fit inspection for one live spectrum (the notebook `alibz-analyze` generates), with fit plots, refinement decisions, seeded minor lines, and the raw-vs-resolved composition charts
- `notebooks/peaky_data.ipynb`: corpus-level PCA, peak width statistics, peak-shape decomposition, and PC-based peak characterisation (a visual record of the whole-corpus development target)

## Tests

```bash
python -m pip install -e ".[dev,notebook]"
python -m pytest tests/
```

## License

[Unlicense](LICENSE) (public domain)
