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

The legacy pre-v3 indexer paths have been removed. The v3 whole-pattern
solver is the single indexing path going forward, but it is not yet
reliable enough for quantitative use.

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

The `run-corpus-pca` CLI can batch-process a corpus of spectra through
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

## Future work — instrument line-spread function (forward model)

**Status: planned, not started. Interim mitigation (robust loss) is in place.**

### The problem

The spectrometer exports every spectrum on an exactly-uniform 0.03333 nm
grid (verified: `np.diff(x)` std ≈ 1e-14 straight from the instrument CSV;
this is the instrument's own CCD→wavelength export, not a resampling we
apply). Two consequences show up on the brightest, sharpest lines and
contaminate the second-iteration self-absorption fit
([refinement.py](../alibz/refinement.py) model **A**):

1. **Export ringing notches.** A bright line leaves a single-pixel
   undershoot on its steep flank — e.g. Sr II 407.77 dips to −50.9 raw
   counts one pixel past its edge (≈19σ below the local continuum),
   followed by a small positive side-lobe. This is a Gibbs/apodization
   signature of the instrument's internal interpolation onto the uniform
   grid, not photon noise. It is rare (≈0.1% of samples) and localised to a
   few of the brightest lines.
2. **Line-shape model inadequacy.** Even away from the notch, a Voigt (or
   attenuated Voigt) leaves a *systematic, oscillatory* ~5–7σ residual
   across the whole profile of the brightest lines. The true instrument
   line-spread function (LSF) is not a Voigt, so the self-absorption
   parameters (τ, δ) partly absorb LSF error rather than plasma physics.
   Measured on Sr II 407.77: model-A residual RMS 5σ (blue) / 7σ (red), and
   τ carries a ~20% systematic from the notch alone.

Crucially, **upsampling the data does not help** — the instrument already
exports its finest grid, so spline-interpolating to a finer axis adds no
information (verified: refitting model A on a ×5 spline-upsampled window
returns identical τ, δ). The fix has to be on the *model* side.

### Interim mitigation (implemented)

`refinement._fit` uses a robust `soft_l1` loss with `f_scale =
SA_ROBUST_F_SCALE = 8` sigma
([refinement.py](../alibz/refinement.py#L84)). This *keeps* every sample
(no rejection) but rolls the penalty from quadratic to linear past ~8σ, so
a single-pixel export notch cannot lever τ/δ while genuine few-σ line-shape
structure is untouched. On clean synthetic data (no notches) it reduces to
ordinary least squares, so the refinement fixtures are unaffected. Measured
effect on MW2-112 spectrum 0: profile RMS 36.8→34.9, verdict counts stable,
Sr II 407.77 τ 0.54→0.49 (toward the notch-masked 0.52), notch-free lines
(Li 670.75, K 766.5) unchanged. This addresses the *notch* but not the
*LSF-shape* inadequacy.

### The plan (a forward-modeled LSF)

1. **Characterise the LSF empirically.** Stack isolated, bright,
   *unsaturated* lines spanning the range; normalise and align them to
   extract the mean instrument profile and its wavelength dependence (width
   grows toward the NIR). Capture the ringing side-lobe structure, which
   encodes the export interpolation kernel.
2. **Model the export kernel.** The uniform-grid ringing is consistent with
   a fixed interpolation/apodization kernel. Represent it (e.g. a
   windowed-sinc or cubic-convolution kernel), **forward-convolve** the
   physical model with it, and **pixel-integrate** onto the 0.03333 nm grid
   — i.e. fit `LSF ⊛ (attenuated Voigt)` sampled as the detector samples
   it, rather than a bare point-sampled Voigt.
3. **Refit self-absorption against the true LSF.** With the LSF folded into
   the forward model, τ and δ measure plasma self-absorption rather than
   compensating for instrument shape. Feeds the `emission_area`
   reconstruction the indexer consumes.
4. **Validate on fit accuracy.** Success criteria: the ~5–7σ systematic
   residual on bright isolated lines collapses toward the noise band; τ/δ
   stabilise under the notch-masking / robust-loss perturbations that
   currently move them; and doublet-ratio checks (e.g. K I 766.5/769.9)
   move toward the optically-thin prediction after correction.

A **Component-2** enhancement that also sharpens Component-3
quantification; deferred behind the PCA-to-re-fit loop and the MILP solver.
The robust loss is sufficient until then.

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

---

## Appendix — CF-LIBS prototype (dev1) salvaged ideas

An experimental self-contained CF-LIBS engine lived under `dev/dev1/cflibs/`
— a full-spectrum variable-projection (VarPro) inverter with its own
Saha/Boltzmann, forward model, self-absorption, and validation scaffolding.
A module-by-module comparison against the production engine
(`peaky_indexer_v3.py`, `utils/sahaboltzmann.py`, `utils/absorption.py`,
`utils/voigt.py`, `utils/stark.py`, `refinement.py`) found its **core
redundant and superseded**: the VarPro architecture (outer nonlinear
plasma search around an inner non-negative linear concentration solve with
a damped self-absorption fixed point), the Saha/Boltzmann chain, the
partition sum, the `(1−e^−τ)/τ` escape factor, the wofz Voigt and its FWHM
approximation, and the robust noise estimator all already exist in main —
and main is *more* complete (Aitken-accelerated SA with a divergence guard,
doublet-anchored per-element τ via `invert_doublet_tau`, the resolved-shape
cold-absorber `sa_voigt`, an explicit Stark-width nₑ channel, and the
`stage_disagreement` diagnostic). The prototype was therefore removed. What
follows is the salvaged novelty — ideas genuinely absent from main —
recorded here so nothing of value was lost with the code.

**None of these has been adopted: each is a physics/solver change to a
tuned pipeline and must be A/B-tested for fit accuracy (against the `bench/`
harness and the synthetic self-absorption round-trips) before adoption.**
Priority order below is by expected accuracy impact / risk.

### Ready to A/B test (concrete, accuracy-relevant)

1. **Stimulated-emission factor on the optical depth** *(highest value, ~1
   line)*. The prototype multiplies the line-center optical depth by
   `(1 − exp(−hc/λk_BT))` (`forward.py:110`). Main's absorbing strength
   `_kappa_raw` ([peaky_indexer_v3.py:1127](../alibz/peaky_indexer_v3.py#L1127))
   omits it. It is **wavelength-dependent** (~0.91 at Na 589, ~0.83 at Rb
   794, ~0.82 at Cs 852 for T≈10⁴ K), so it is **not** absorbable into
   main's single global `sa_tau_scale`, and it shifts *relative*
   escape-factor compression by ~9–18% across exactly the alkali resonance
   lines (K 766/770, Na 589, Rb 780/795, Cs 852) that are this codebase's
   known quantification pain point (see the sim-to-real notes: alkali
   doublet ratios sit ~35% below optically thin). Integration point:
   `_kappa_raw`. Accuracy check: does it move the K/Na doublet ratios
   toward the optically-thin prediction?

2. **Debye ionization-potential-depression (continuum lowering) in the Saha
   balance.** The prototype lowers χ by `Δχ = (z+1)e²/4πε₀λ_D` before the
   Saha step (`physics.py`). Main's `ionization_distribution`
   ([sahaboltzmann.py:177](../alibz/utils/sahaboltzmann.py#L177)) has no IPD
   term, so at LIBS densities (nₑ≈10¹⁷ cm⁻³, several-tenths-eV lowering) the
   effective ionization energy is unlowered and the ion balance is biased.
   Physically load-bearing; changes every ion-stage fraction feeding
   quantification.

3. **Robust Tukey-biweight IRLS in the concentration solve.** The prototype
   wraps the inner linear solve in IRLS reweighting (`inversion.py`) so
   unmodeled lines / DB-absent blends / detector artifacts cannot bias T,
   nₑ, or composition. Main's whole-pattern NNLS
   ([peaky_indexer_v3.py:1250](../alibz/peaky_indexer_v3.py#L1250)) uses
   plain squared error. This is the *indexer-stage* analogue of the
   `soft_l1` robust loss just added to `refinement._fit` — same philosophy,
   different stage, and a natural next step.

4. **Classical Boltzmann-plot T seed + independent cross-check.** The
   prototype runs a per-species robust `ln(Iλ/gA)` vs `Ek` regression
   (`classical.py`) for a data-driven T, used both to seed the outer search
   and as an independent physical check. Main starts the outer search from a
   **fixed** `T_init` (default 10 kK,
   [peaky_indexer_v3.py:377](../alibz/peaky_indexer_v3.py#L377)) with no
   data-derived initialization and no independent T diagnostic. A cheap
   Boltzmann-plot from the isolated strong peaks the finder already extracts
   could steer the multimodal outer search off spurious minima and
   cross-check the full-pattern T/nₑ. (Complements the existing
   stage-consistency thermometer.)

5. **Non-negative LASSO trace-element suppression.** `sparse_refine`
   (`inversion.py`) applies a principled sparsity operator once at the
   best-fit θ to zero spurious trace elements whose weak/blended lines only
   soaked up noise — a cleaner alternative/complement to main's heuristic
   false-positive stack (evidence penalties, pseudo-observations,
   `prune_and_refit`).

6. **Physics-based thermal Doppler width.** `doppler_sigma_nm(T, mass)`
   (`lineshape.py:18`) ties the Gaussian width to fitted T and per-element
   atomic mass (heavier → narrower), added in quadrature with an instrument
   σ. Main uses a fixed width in the forward model and fits σ as a free
   scalar; grounding it in physics removes/constrains a free parameter.
   (Related to the LSF work above.)

### Future / tooling (document, lower priority)

- **Bootstrap uncertainty** over θ and concentrations (Gaussian-resample +
  local re-fit) — main returns point estimates only.
- **External critically-evaluated (T,U) partition tables** (cubic-spline
  interpolation), decoupling U from the incomplete line-derived level set
  main is forced to sum; plus **Griem truncation** of the partition sum
  under continuum lowering.
- **McWhirter LTE-validity diagnostic** `nₑ ≥ 1.6e12·√T·ΔE³` — report
  whether the LTE assumption holds for a given fit; none exists in main.
- **NIST ASD "Lines Form" text parser** (`atomic.parse_nist_asd_lines`,
  header-name matching, tolerant of NIST quoting) — main ingests only
  pre-pickled line arrays, so there is no text-ingestion / provenance path.
- **In-solver joint Legendre continuum** — only relevant if a future path
  fits raw spectrum pixels rather than the extracted peak table; moot on
  main's current peak-table input.

### Removed as redundant (already in main)

Voigt profile + FWHM (`utils/voigt.py`), Saha/Boltzmann + partition sum
(`utils/sahaboltzmann.py`), `(1−e^−τ)/τ` self-absorption fixed point
(`utils/absorption.py` + `peaky_indexer_v3.py`, main more complete), the
robust noise estimator (duplicated verbatim in `peaky_finder.py:131`),
Stark-width∝nₑ (`utils/stark.py`, main more sophisticated), bounded-LSQ /
column normalization, and the synthetic-observation / procedural-database
validation scaffolding (superseded by the committed `bench/` harness).
