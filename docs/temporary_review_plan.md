# Temporary Review Plan

Date: 2026-03-21

## Purpose

This document captures the main repository findings from the current review,
the concrete fixes required to stabilize the codebase, and the next
development items that should be tackled only after those fixes are in place.

The goal is to restore consistency between the implemented pipeline and the
project's stated physics-first LIBS roadmap:

1. Forward model the plasma emission.
2. Fit experimental peaks robustly.
3. Index peaks and solve for composition.
4. Extend to corpus-scale and full-spectrum inference.

## Summary Of Current State

The repository has a strong peak-fitting and PCA analysis path, but the
forward-model and inverse-composition paths are less reliable than the docs
imply. The immediate priority is not adding new features. It is repairing the
broken or inconsistent physics and API layers so the current architecture is
internally coherent.

## Phase 1: Required Fixes

### 1. Repair `PeakyFitter` or explicitly retire it

Status:
- High priority
- Structural/API mismatch

Problem:
- `PeakyFitter` depends on `PeakyFinder` attributes and methods that no longer
  exist or are no longer attached in the expected way.
- This makes the temperature-estimation and extended fitting path unreliable.

Observed issues:
- Calls missing finder methods such as `calculate_peaks` and `rank`.
- Reads missing attached objects such as `finder.indexer`, `finder.db`,
  `finder.maker`, and `finder.boltzmann_constant`.
- Assumes an older finder/indexer integration model that no longer matches the
  current code.

Relevant files:
- [alibz/peaky_fitter.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_fitter.py)
- [alibz/peaky_finder.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_finder.py)
- [alibz/peaky_indexer.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_indexer.py)
- [alibz/peaky_maker.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_maker.py)

Fix:
- Choose one of two paths and do it decisively:
  - Rebuild `PeakyFitter` against the current finder/indexer APIs.
  - Or mark it experimental/deprecated and remove it from the advertised core
    workflow until it is rebuilt.
- If rebuilt, inject required dependencies explicitly rather than assuming they
  are hanging off `PeakyFinder`.
- Add tests for the fitted-temperature workflow before advertising it as
  working.

Definition of done:
- No stale method calls remain.
- The fitter can run end-to-end on a small synthetic example.
- Its public API is documented and covered by tests.

### 2. Fix the single-ion-stage bug in `SahaBoltzmann.partition`

Status:
- High priority
- Forward-model correctness bug

Problem:
- `partition()` computes a value in the `len(ions) == 1` branch but does not
  return it.
- `ionization_distribution()` then unpacks the result as if a tuple was
  returned.
- This can break the forward model for hydrogen or any species represented with
  only one ion stage in the database.

Relevant files:
- [alibz/utils/sahaboltzmann.py](/Users/mwhittaker/Projects/github/alibz/alibz/utils/sahaboltzmann.py)

Fix:
- Make `partition()` return a consistent shape and type for all branches.
- Standardize its contract so every caller receives the same tuple structure.
- Add unit tests for:
  - single-ion species
  - multi-ion species
  - elements with missing line data

Definition of done:
- `partition()` and `ionization_distribution()` agree on return types.
- Forward synthesis runs for hydrogen without special-case failure.

### 3. Repair `PeakyMaker.batch_maker` and parameter/unit consistency

Status:
- High priority
- Synthetic data path is broken

Problem:
- `batch_maker()` references nonexistent attributes like `self.elements` and
  `self.elem_abund`.
- It calls `peak_maker()` with `temp=...` even though the function expects
  `temperature=...`.
- Electron density handling is inconsistent between log-scale and absolute
  units.

Relevant files:
- [alibz/peaky_maker.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_maker.py)
- [alibz/utils/database.py](/Users/mwhittaker/Projects/github/alibz/alibz/utils/database.py)

Fix:
- Replace `self.elements` with `self.db.elements`.
- Replace `self.elem_abund` with a correctly prepared array derived from the
  database abundance table.
- Make `peak_maker()` and `batch_maker()` use the same parameter names.
- Define one canonical convention for `ne`:
  - either `log10(n_e [cm^-3])`
  - or absolute `n_e [cm^-3]`
- Apply that convention consistently across docstrings, defaults, code, and
  downstream callers.

Definition of done:
- `batch_maker()` generates synthetic spectra without attribute or keyword
  errors.
- `ne` semantics are unambiguous across the forward model.

### 4. Correct the meaning of `spectrum_match()` outputs

Status:
- High priority
- Inverse-model semantics bug

Problem:
- `spectrum_match()` returns `c` as if it were a vector of composition
  fractions.
- The current optimization does not enforce sum-to-unity or a shared plasma
  state.
- Reference intensities are normalized independently, so the resulting `c`
  values are scale factors, not defensible abundances.

Relevant files:
- [alibz/peaky_indexer.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_indexer.py)

Fix:
- In the short term, rename outputs and documentation so the solver is not
  overstated.
- Treat current `c` values as reference-scale parameters, not composition.
- In the medium term, replace the formulation with a physically coupled inverse
  model that shares `T`, `n_e`, broadening, and composition across all peaks.

Definition of done:
- The API no longer mislabels scale factors as abundances.
- The docs clearly describe what the current solver can and cannot infer.

### 5. Fix lazy loading in `fit_spectrum_data()`

Status:
- Medium priority
- Usability bug

Problem:
- `fit_spectrum_data()` calls `self.data.load_data()` and stores the return
  value, but `load_data()` mutates internal state and returns `None`.
- This makes lazy loading fail unless the caller manually preloads the data.

Relevant files:
- [alibz/peaky_finder.py](/Users/mwhittaker/Projects/github/alibz/alibz/peaky_finder.py)
- [alibz/utils/dataloader.py](/Users/mwhittaker/Projects/github/alibz/alibz/utils/dataloader.py)
- [README.md](/Users/mwhittaker/Projects/github/alibz/README.md)

Fix:
- Update `fit_spectrum_data()` to call `load_data()` and then read from
  `self.data.data`.
- Add a regression test for the lazy-load path.
- Align the README examples with the supported usage.

Definition of done:
- `fit_spectrum_data()` works with and without an explicit prior
  `load_data()` call.

### 6. Add baseline test and environment coverage for the physics path

Status:
- Medium priority
- Quality gap

Problem:
- Current tests cover only a narrow part of the repo.
- The forward model, batch synthesis, indexer, and fitter paths are largely
  untested.
- In the current environment the suite also cannot run because scientific
  dependencies are missing.

Relevant files:
- [tests/test_peaky_finder_fast.py](/Users/mwhittaker/Projects/github/alibz/tests/test_peaky_finder_fast.py)
- [tests/test_peaky_pca.py](/Users/mwhittaker/Projects/github/alibz/tests/test_peaky_pca.py)
- [pyproject.toml](/Users/mwhittaker/Projects/github/alibz/pyproject.toml)

Fix:
- Add tests for:
  - Saha-Boltzmann partition and ionization distribution
  - synthetic spectrum generation
  - indexer output semantics
  - lazy-load data flow
  - at least one end-to-end synthetic spectrum round trip
- Add a documented dev setup that actually installs the scientific stack used
  by the test suite.

Definition of done:
- The test suite runs in a clean environment.
- The main physics and inverse-model paths have regression coverage.

## Phase 2: Cleanup After Required Fixes

These items should happen immediately after the repair work above.

### 7. Align docs with actual maturity

Fix:
- Update README and development docs to distinguish:
  - production-ready peak fitting and corpus PCA
  - experimental forward model
  - incomplete composition inference

### 8. Define explicit API boundaries

Fix:
- Clarify which classes are public and stable.
- Separate internal helpers from supported workflows.
- Remove assumptions that one class automatically owns all others.

### 9. Standardize physical units everywhere

Fix:
- Define canonical units for:
  - temperature
  - electron density
  - widths
  - line intensity
- Encode them consistently in docstrings and variable names.

## Phase 3: TODO Items After Fixes

These are the next developments to pursue once the current code is stable.

### TODO 1. Build a shared whole-spectrum inverse model

Replace the current peak-by-peak matching plus weakly constrained MILP with a
single inference loop that fits:
- composition
- plasma temperature
- electron density
- wavelength shift
- instrument width
- optional baseline parameters

This is the most important architectural step for turning the project into a
real calibration-free LIBS solver.

### TODO 2. Make broadening explicitly physical

Use the peak/PCA outputs to constrain:
- Gaussian width from instrumental response plus Doppler broadening
- Lorentzian width from Stark broadening
- asymmetry from self-absorption or radiative-transfer effects

The current PCA classification should become a prior or constraint in the
refitting stage, not just a descriptive label.

### TODO 3. Add continuum and optical-depth modeling

Extend the forward model to include:
- continuum background
- self-absorption / escape-factor treatment
- better handling of strong resonance lines

This will improve both synthesis realism and inverse-model identifiability.

### TODO 4. Move to hierarchical multi-shot inference

Model multiple LIBS shots from the same sample jointly:
- shared composition
- shot-specific nuisance parameters
- uncertainty on sample heterogeneity

This is a better match to how LIBS data behaves experimentally.

### TODO 5. Create a synthetic-to-real validation harness

Use the repaired forward model to generate synthetic spectra with controlled:
- composition
- temperature
- electron density
- broadening
- noise
- baseline artifacts

Then measure whether the analysis stack can recover known parameters before
applying it to real standards.

### TODO 6. Add standards-based validation

Once the inverse model is credible, validate it on certified reference
materials and quantify:
- bias
- variance
- limits of detection
- failure modes by matrix type

## Recommended Execution Order

1. Fix `SahaBoltzmann.partition`.
2. Fix `fit_spectrum_data()` lazy loading.
3. Repair `PeakyMaker.batch_maker` and unit consistency.
4. Decide whether `PeakyFitter` is rebuilt now or temporarily retired.
5. Correct `spectrum_match()` semantics and documentation.
6. Add tests and runnable dev environment support.
7. Only then start the larger inverse-model and physics developments.

## Exit Criteria For This Temporary Plan

This temporary plan can be retired once:
- the broken APIs are repaired or intentionally removed,
- the forward model has basic regression coverage,
- the inverse-model outputs are no longer mislabeled,
- and a replacement roadmap exists for whole-spectrum composition inference.
