# alibz Development Guide

## Project Goal

Quantify elemental compositions from LIBS (Laser-Induced Breakdown
Spectroscopy) data with maximum precision and accuracy — **without
calibration standards**.  The approach is fundamentally physics-first:
forward-model the plasma emission, fit the observed spectra, and solve the
inverse problem for composition using first-principles constraints rather
than empirical calibration curves.

---

## Architecture Overview

The repository is organised around five sequential components.  Each builds
on the outputs of the previous one.

```
 [1] Forward Model ──> [2] Peak Fitting ──> [3] Indexing & Whole-Pattern Fit
                                                        │
                                                        v
                            [5] Production  <──  [4] Transformer Model
```

---

## Component 1 — Forward Model (LTE Plasma Emission)

### Purpose

Simulate LIBS line intensities from first principles so that observed
spectra can be compared to physically-grounded predictions.

### What it does

| Module | Role |
|--------|------|
| `utils/sahaboltzmann.py` | Saha ionization equilibrium + Boltzmann thermal population.  Computes partition functions, ionization-state fractions, and line intensities for any element at a given temperature *T* and electron density *n_e*. |
| `peaky_maker.py` | Synthetic spectrum generator.  Combines Saha-Boltzmann intensities with Voigt broadening to produce full spectra for arbitrary multi-element mixtures. |
| `utils/database.py` | NIST atomic line database (92 elements, wavelengths, oscillator strengths *gA*, energy levels, ionization energies). |
| `utils/constants.py` | Physical constants (Boltzmann, Planck, *c*, *m_e*) in CGS/eV. |

### Physics included

- Saha ionization equilibrium (multi-stage)
- Boltzmann thermal level populations
- Partition functions from spectral line data
- Voigt broadening (Doppler + Stark convolution)
- Ground-state resonant absorption (GSRA) correction
- Time-gated spontaneous emission decay
- 92 elements (H–U) with natural crustal abundances

### Current status: ~95% complete

The forward model is the most mature component.  It reliably generates
synthetic LIBS spectra that match the gross features of experimental data.

### What's missing / next steps

- **Pressure broadening** (van der Waals) — significant for high-pressure
  or atmospheric plasmas.
- **Continuum background** (bremsstrahlung + recombination) — currently
  ignored; real spectra have a slowly-varying continuum that affects
  baseline fitting downstream.
- **Multi-temperature plasma** — current model assumes a single *T_e*.
  A two-temperature model (ions vs electrons) would improve accuracy for
  early-gate spectra.
- **Self-absorption within the forward model** — currently only available
  as a post-processing correction; integrating it into the forward model
  would make the inverse problem better-posed.

---

## Component 2 — Peak Fitting (Current Development Focus)

### Purpose

Extract precise peak parameters (wavelength, amplitude, Gaussian width,
Lorentzian width, asymmetry) from experimental LIBS spectra.  These
parameters are the primary input to indexing and composition solving.

### Pipeline stages

```
raw spectrum
    │
    ├─ (a) Fast peak finding + Voigt profile fitting
    │      ↓
    ├─ (b) PCA refinement of peak shapes
    │      ↓
    └─ (c) Robust profile fitting (instrumental + physical broadening)
```

### What exists

| Module | Stage | Status |
|--------|-------|--------|
| `peaky_finder.py` | (a) Fast peak detection via local-maxima search; FFT-based background subtraction; multi-Voigt least-squares fitting with adaptive bounds; fast-mode parameter estimation without optimiser. | **Working** |
| `peaky_fitter.py` | (a,c) Legacy ion-by-ion fitting path. Removed from the supported API pending a ground-up rebuild against current finder/indexer interfaces. | **Retired** |
| `peaky_corpus.py` | (a,b) Batch fitting across a corpus; common-grid standardisation (GPU-accelerated); FWHM width statistics; GMM mode detection. | **Working** |
| `peaky_pca.py` | (b) PCA decomposition of normalised peak windows; perturbation analysis mapping each PC to a physical broadening mechanism (Doppler, Stark, self-absorption); peak classification. | **Working** |
| `gpu.py` | All | GPU acceleration (CuPy): batch interpolation, SVD-based PCA, pseudo-Voigt evaluation, window extraction. | **Working** |
| `utils/voigt.py` | (a) | Thompson FWHM approximation; vectorised multi-Voigt; GPU dispatch. | **Working** |

### Current status: ~80% complete

Stages (a) and (b) are functional and tested.  Stage (c) — using the PCA
broadening decomposition to feed back into a physics-informed re-fit — is
the **critical gap** that must be closed before moving to Component 3.

### What needs to happen next

1. **Close the PCA → re-fit loop.**  The PCA decomposition (Component 2b)
   currently classifies peaks by dominant broadening mechanism but does not
   feed those classifications back into the fitter.  The next step is to
   use the per-peak broadening classification to:
   - Constrain the Gaussian width to instrumental + Doppler contributions
     (from the PCA Gaussian component).
   - Constrain the Lorentzian width to Stark broadening (from the PCA
     Lorentzian component).
   - Model self-absorption via the asymmetry parameter (PCA asymmetry
     component).
   This produces physically-meaningful Voigt parameters rather than
   purely empirical curve fits.

2. **Corpus-wide parameter consistency.**  Peaks from the same element/ion
   should share broadening parameters (same *T*, same *n_e*).  Implement
   a hierarchical or global constraint that ties Stark widths across peaks
   of the same species.

3. **Confidence intervals on fitted parameters.**  Currently only point
   estimates are returned.  Add bootstrap or profile-likelihood uncertainty
   quantification for each peak's amplitude, position, sigma, gamma, and
   tau.

4. **Performance.**  The current run on 4,909 spectra takes ~2 min per
   spectrum (CPU-bound `least_squares`).  Investigate:
   - Warm-starting from fast-mode estimates
   - Reducing the number of least-squares iterations
   - Parallel fitting across spectra (multiprocessing)
   - GPU-accelerated Levenberg-Marquardt (e.g. `gpufit`)

5. **Edge cases.**  Improve handling of overlapping multiplets, saturated
   peaks, and low-SNR spectra where the current fitter sometimes fails
   silently.

---

## Component 3 — Peak Indexing & Whole-Pattern Fitting

### Purpose

Assign each fitted peak to a specific {element, ion, transition} using its
wavelength, lineshape, and forward-model predictions.  Then combine all
peak assignments into a self-consistent whole-pattern fit that yields
elemental concentrations, plasma temperature, and electron density.

### What exists

| Module | Role | Status |
|--------|------|--------|
| `peaky_indexer_v3.py` | Experimental whole-pattern spectral fitter combining candidate generation, Saha-Boltzmann line weighting, and NNLS/Bayesian optimisation. This is now the only supported indexer. | **Experimental** |

### Current status: experimental research prototype

The legacy `peaky_indexer.py` and `peaky_indexer_v2.py` paths have been
retired as active development targets and remain only as deprecated
compatibility shims. The v3 whole-pattern solver is the single indexing
path going forward, but it is not yet reliable enough for quantitative use.

### What needs to happen next

1. **Complete the MILP composition solver.**  The solver should minimise
   the residual between observed and forward-modelled line intensities
   subject to:
   - Sum-to-unity constraint on element fractions
   - Consistent temperature and *n_e* across all peaks
   - Self-absorption corrections for strong lines

2. **Whole-pattern fitting.**  Move beyond peak-by-peak matching to a
   full-spectrum forward fit:
   - Generate a synthetic spectrum from the current composition estimate
   - Compare to the observed spectrum (not just peak positions)
   - Iterate to refine composition, *T*, *n_e*, and broadening parameters

3. **Validation against certified reference materials.**  Compare
   quantified compositions to known standards (NIST SRMs, geological
   reference powders) to assess accuracy and identify systematic biases.

4. **Multi-spectrum aggregation.**  Combine results across multiple shots
   on the same sample to reduce statistical uncertainty and detect spatial
   heterogeneity.

---

## Component 4 — Transformer Model

### Purpose

Train a deep-learning model on synthetic spectra (generated by Component 1
with empirical peak parameters from Component 2 and whole-pattern
attributes from Component 3) to perform fast, accurate composition
inference.  The training data should include realistic artifacts: noise,
baseline curvature, detector nonlinearity, self-absorption, shot-to-shot
variation, etc.

### Current status: 0% — not yet started

No neural network code, no PyTorch/TensorFlow dependencies, no training
infrastructure.

### What needs to happen (future)

1. **Training data generation.**  Use `PeakyMaker` to generate large
   synthetic datasets with:
   - Randomised multi-element compositions
   - Realistic temperature and *n_e* distributions
   - Empirical broadening parameters drawn from PCA statistics
   - Synthetic noise, baseline, and self-absorption artifacts

2. **Architecture.**  A transformer encoder that ingests a standardised
   spectrum (or a set of extracted peak features) and predicts:
   - Element concentrations (regression head)
   - Plasma parameters *T*, *n_e* (auxiliary regression head)
   - Confidence intervals (e.g. via MC dropout or evidential regression)

3. **Transfer learning.**  Pre-train on synthetic data, fine-tune on
   labelled experimental data (certified reference materials).

4. **Evaluation.**  Benchmark against the physics-based pipeline
   (Components 1–3) on the same test set.

---

## Component 5 — Production Model

### Purpose

A deployment-ready system that ingests raw LIBS data (spectra, spatial
coordinates, timestamps) and returns comprehensive chemometrics:
- Single-point elemental composition with uncertainties
- LIBS maps (2D spatial composition images)
- Depth profiles (composition vs ablation depth)
- Timeseries (temporal evolution of plasma or sample)

### Current status: ~30% — research prototype only

The `run_corpus_pca.py` CLI can batch-process a corpus of spectra through
the peak-fitting and PCA pipeline, but there is no inference pathway for
new spectra and no spatial/temporal analysis.

### What needs to happen (future)

1. **Inference API.**  A function or service that takes a single spectrum
   and returns composition + uncertainties using either the physics
   pipeline (Components 1–3) or the transformer (Component 4).

2. **Spatial mapping.**  Associate spectra with (x, y) coordinates from
   the LIBS instrument's raster scan.  Produce 2D elemental maps with
   interpolation and uncertainty overlays.

3. **Depth profiling.**  Track composition as a function of shot number
   (ablation depth) at a single location.

4. **Timeseries.**  Monitor plasma parameters and composition across a
   sequence of measurements (e.g. for process control).

5. **Output formats.**  HDF5 or NetCDF for structured scientific data;
   JSON/CSV for interoperability.

6. **Testing & validation.**  Expand test coverage from the current 2 test
   files to comprehensive unit + integration tests across all components.

---

## Summary

| # | Component | Status | Completeness | Blocking? |
|---|-----------|--------|-------------|-----------|
| 1 | Forward Model | Mature | ~95% | No |
| 2 | **Peak Fitting** | **Active development** | **~80%** | **Yes — blocks Component 3** |
| 3 | Indexing & Whole-Pattern | Partial | ~50% | Blocked by Component 2 |
| 4 | Transformer Model | Not started | 0% | Blocked by Components 2+3 |
| 5 | Production Model | Prototype | ~30% | Blocked by Components 3+4 |

**Current priority**: Close the PCA-to-re-fit loop in Component 2 (robust
profile fitting with physically-informed broadening constraints), then
complete the MILP composition solver in Component 3.
