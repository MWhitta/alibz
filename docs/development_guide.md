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
| `peaky_corpus.py` | (a,b) Batch fitting across a corpus; common-grid standardisation (GPU-accelerated); FWHM width statistics; GMM mode detection. | **Working** |
| `peaky_pca.py` | (b) PCA decomposition of normalised peak windows; perturbation analysis mapping each PC to a physical broadening mechanism (Doppler, Stark, self-absorption); peak classification. | **Working** |
| `profiles.py` | (c) Per-detector-segment, per-peak shape physics on a FITTED spectrum: segment instrumental width floors from the spectrum's own narrow peaks; Doppler-vs-Stark decomposition of excess width; residual-signature classification separating shoulders (unresolved overlaps) from true self-absorption asymmetry; optional PC-score projection with a trained `PeakyPCA`; per-element support shape QC (`sa_share`, `shoulder_share`, `clean_anchors` in `detections.csv`). | **Working** |
| `gpu.py` | All | GPU acceleration (CuPy): batch interpolation, SVD-based PCA, pseudo-Voigt evaluation, window extraction. | **Working** |
| `utils/voigt.py` | (a) | Thompson FWHM approximation; vectorised multi-Voigt; GPU dispatch. | **Working** |
| `background_pca.py` | (b) Corpus background/segment PCA. Superseded by `PeakyFinder.find_background` + `DetectorModel` + `profiles.py`. | **Deprecated** |

### Current status: ~85% complete

Stages (a) and (b) are functional and tested.  Stage (c) — shape physics
feeding back into the analysis — is now closed end to end.
`alibz.profiles` classifies every fitted peak per segment, QCs each
element's supporting flux (reported in `detections.csv` and flagged in
`summary.csv` as `dominant-weak-shape`), and FEEDS THE FIT:

- **`deblend_shoulders`** (pre-identification): a `shoulder`-flagged peak
  (one-sided flank bump = unresolved overlap) is refit as two components
  plus pedestal, the main confined near its parameters and the new one
  pinned at the bump, gated by matched-filter SNR (≥4) and window BIC
  (≥6); refinement's asymmetric-merge zones are excluded.  The pass-2/3
  indexers then see the decontaminated areas.
- **`recover_sa_areas`** (post-identification): an `sa-like` peak of a
  species NOT doublet-anchored is refit with the SA model (`sa_voigt`,
  whose area parameter is the unattenuated emission area) against a
  symmetric control; acceptance needs a 10-BIC win, tau below the
  ceiling, and amplification ≤ 5x.  Accepted emission areas correct the
  observed amplitudes and the composition is RE-SOLVED linearly at the
  fitted plasma state (the `element_uncertainty_stats` pattern — no new
  Bayesian pass, no basin risk), with the collapse guard applied to the
  corrected composition as final safety.  Doublet-anchored species (K I,
  Na D, ...) are skipped — their depths already act on the RESPONSE side
  of the design, and correcting the data side too would double-count.
  The same re-solve also propagates the refinement merges' PRE-MEASURED
  emission/observed ratios (`premeasured=`): a merged row carries the
  observed area, its zone is excluded from the growth-curve refit, and —
  audited 2026-07-09 — for species the doublet channel does not anchor
  (Li I is *never* anchorable: its 670.776/.791 doublet is unresolved)
  the asymmetric fit's measured correction (Li I ×1.4, Mg I ×1.5,
  Al I ×1.6, Fe I ×2–7 per MW2-112 spectrum) was previously computed and
  then dropped by every channel.

Measured on JChristensen (38 spectra): deblends fired on 26, SA recovery
on 24; corpus means moved Si +3.1 pp / K −3.5 pp with no collapse.  On
MW2-112: deblends split the Mg II 279.75/280.0 and Hg 253.6 regions;
recovered lines (Al I 396.1 ×1.35, Mg I 383.8 ×1.60) carry physically
sensible optical depths (τ ≈ 0.5).

An honest measured caveat from JChristensen (2026-07-06): the
single-element-dominated failures there (Hg 1.000 from one 194 nm line,
Fe 0.991, Ca 0.990) were NOT peak-shape failures — the driver peaks
classify `instrumental` (clean).  They were **inverse-solve basin
failures**: the corroboration (pass-3) re-index, after seeding dozens of
weak low-excitation lines, drifted into a low-T basin (~5 200 K) where one
element's tiny Saha-Boltzmann response explains everything — with
r-squared even IMPROVING (0.87), because a line-rich element fits
anything.  The fix is the **composition-collapse basin guard** in
`analyze_spectrum` (`COLLAPSE_TOP_FRACTION`/`COLLAPSE_JUMP`): the
corroborated re-index is rejected when its top fraction newly collapses
(measured A/B: Pam Hg 1.000 -> K/Li/Si; MDD006 Fe 0.991 -> Si/K/Li), and
`summary.csv` records `corroboration-rejected(collapse)`.  Shape QC and
the basin guard are complementary constraints, not substitutes.

**Staged refinement (2026-07-09).**  `refine_fit` is now split into a
DATA-ONLY pass and a PHYSICS pass (`asymmetric='defer'|'only'`):
stage 3a applies blend splits and single-merges (pure model evidence,
BIC) before any identification and defers the whole asymmetric family;
after pass 1, stage 3b re-adjudicates the deferred features with the
resonance gates conditioned on the *retained candidate species* of the
whole-pattern solve (broader than the established list, so a wrong
pass-1 basin cannot veto a real resonance merge — but far tighter than
the whole periodic table, which is what the old single-pass had to gate
against).  The merge zones are now computed BEFORE seeding, and
`seed_minor_lines` gained the same `exclude=` gate as recovery and
deblending — seeds landing inside a merge zone measurably eroded the
merged rows' observed areas by 21–93% (K, Ca, Si archetypes).

Two follow-up defects found by eye on the MW2-112 248.3–249.5 nm window
(dense Fe forest): (1) when the DATABASE supports a blend (two distinct
lines matching both fitted centers with a consistent separation), the
near-degenerate-statistics escape that let `classify_feature` call a
pair "asymmetric" anyway is now barred — Fe I 248.814/249.064 had been
merged into a fictitious τ≈2.7 line whose exclusion zone then blocked
recovery of the real residuals; (2) exclude-zone rows are now FROZEN in
neighbouring joint refits of both the seeder and residual recovery (they
render through `model_others`) — a merged row's observed-area proxy
previously walked +30 pm and lost 40% of its area to the refit next
door.  Remaining known gap in that window: features below recovery's 4σ
prominence bar (e.g. the small 248.55 nm bump) stay unmodeled.

**Per-segment wavelength shift (2026-07-09).**
`estimate_wavelength_shift_segments` estimates one shift per detector
segment from the refined table, but applies a segment's own median only
when it deviates from the pooled global by more than twice the median's
standard error.  Measured honestly: on MW2-112 the per-segment medians
differ by 10–35 pm but with 26–71 pm MADs (even post-refinement), so the
gate refuses and behavior is identical to the global shift — the
machinery is armed for sessions where a segment genuinely drifts.  An
UNGATED per-segment shift measurably corrupted matching (a +20 pm
segment-1 pseudo-shift conjured 25% Ti on MW2-112 #1000).

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

### Current status: ~45% — research prototype with a CLI

The `alibz-analyze` CLI is the current production-facing path for new
spectra. It runs the peak-fitting, refinement, minor-line seeding,
whole-pattern indexing, H-alpha electron-density initialization when
available, doublet self-absorption anchoring, and amplitude-resampling
uncertainty workflow over a directory of CSV spectra. It writes
`summary.csv`, `detections.csv`, and an inspection notebook into the input
directory.

`run-corpus-pca` remains available for corpus-scale peak-shape and PCA
work, not for composition inference. Spatial mapping, depth profiling,
and standards-validated reporting are still future work.

### What needs to happen (future)

1. **Harden the inference API.**  Stabilize `alibz.pipeline.analyze_spectrum`
   and `alibz-analyze` around a versioned output schema, explicit database
   provenance, and systematic-error diagnostics.

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

## Spectral confounders: the true-negative contest (implemented 2026-07-05)

### The archetype

On the JChristensen drill cores the whole-pattern fit reported Mn at
40–53% atom fraction in several samples — resting on exactly two peaks at
279.49/279.76 nm, inside the **Mg II 279.553/280.270** resonance doublet
region (Mn I/II have lines at 279.48/279.83/280.11). Mg itself read
`single-line` from Mg I 285.2. The chain that lets this happen:

1. the NNLS design has nearly collinear Mn and Mg columns in 279–281 nm;
   the solve lands on a vertex and hands ALL the flux to one of them;
2. `prune_and_refit` then deletes the zeroed rival (Mg II, 183 candidate
   lines) from the design — the alternative hypothesis becomes
   *unrepresentable*;
3. the misattribution drags the fitted temperature (measured: T → 5.0 kK,
   where Mg is neutral), so even a post-hoc check at the fitted plasma
   state finds Mg II unable to respond — the error certifies itself.

### The general method

For each element E with fitted support, and each rival element E', ask:
**could E' cover E's supporting peaks without violating its own true
negatives?** Concretely (`pipeline._contested_support`):

- cap the rival's concentration by its own absent/weak lines:
  ``c_max(E') = min_j (obs_j + 3σ_j) / T_j(E')`` over every fitted peak
  position where E''s template responds — computed from E''s
  **non-resonance lines only** (Eᵢ > 0.2 eV), because resonance lines
  self-absorb and understate the element (measured: the classic
  self-absorbed Mg I 285.2 line otherwise caps Mg at 12% of the flux it
  can physically supply);
- a supporting peak of E is **contested** if some rival at its cap still
  covers ≥ 50% of the peak amplitude;
- the scan runs over a **corpus-plausible plasma grid**
  (T ∈ {6, 8, 10, 12} kK × log nₑ ∈ {16.5, 17.5}), existentially — never
  only at the fitted state (see failure 3 above) — and against the
  **pre-prune candidate table** built at a neutral 9 kK / 17.0 state
  (see failures 1–2).

`detections.csv` reports per element: `clear_lines` (supporting peaks no
rival can cover at any scanned state), `contested_share` (fraction of
supporting flux on contested peaks), and `confounder` (the rival
contesting the most flux). An element whose *every* supporting peak is
contested is demoted from `detected`/`single-line` to **`confounded`**
and flagged in `summary.csv` (`Mn:confounded(Mg)`); its abundance is an
attribution choice, not a measurement. Validated: both hot-Mn samples
demote (contested = 1.0, confounder = Mg) while Li/Al/Na detections with
genuinely independent lines are untouched.

The corpus-level confounder catalog is the aggregation of the
`confounder` column across a run — pairs that recur (Mn⇄Mg here) are the
corpus's operative confounders under its actual (T, nₑ) range.

### Resolving the quantification (`resolve_confounded`)

The guard also *resolves* the abundance, not just flags it
([`detections.resolve_confounded`](../alibz/detections.py)). The
principle is the same true-negative accounting: a `confounded` element is
credited only its **clear (uncontested) flux** — the share no rival can
cover — and the contested remainder is reattributed to the `confounder`
(the rival whose *own* lines are present), scaled by the response ratio
`T_E/T_rival` that converts freed E-fraction into rival-fraction; the
whole composition is then renormalised. This is the "group penalty" in
the honest sense — each member of a confounded set is penalised by its
own true negatives, so the arbitrary NNLS vertex is replaced by the
physically-consistent split.

Reported per element: `fraction` (raw NNLS vertex), plus two *normalised*
compositions — `fraction_hi` (the as-fit split, the HIGH end for a
confounded element) and `fraction_resolved` (the true-negative-resolved
split, the defensible value) — all in `detections.csv`; `analyze_spectrum`
returns the full `resolved_fractions` composition (sums to 1). Both
`fraction_hi` and `fraction_resolved` are renormalised onto the same scale
(each sums to 1) so they are directly comparable; the low→high bracket
reads that way only for the confounded element itself — its confounder
*gains* flux (so `fraction_resolved` > `fraction_hi` there) and bystanders
rescale slightly with the renormalisation. Validated on the JChristensen
samples: **Mn 53% → 0%** (fully contested, no clear line), with the Mg II
279.5 flux reattributed to **Mg 8% → 31%** (which carries the independent
Mg I 285.2 line); genuine detections (Li, Al, Ca…) with clear lines are
untouched apart from renormalisation.

The reattribution only ever moves flux to a rival that can **globally
host** it, under two guards that stop a small confound from spawning a
large phantom:

- **No mass creation** — the response ratio `T_E/T_rival` is clamped at 1.
  A weak-emitter rival needing >1× the freed fraction to cover the peak is
  evidence it is the *wrong* host, not licence to inflate it. (Measured:
  Be↔Fe carried `resp_ratio ≈ 12`, so an uncapped reattribution turned a
  3.8% Be confound into a **42% Fe** phantom.)
- **Bounded by the rival's own evidence** — a rival absorbs contested flux
  only up to its evidence ceiling: unbounded if it is independently
  detected, its `upper_limit` if it survives only as an upper limit / weak,
  and **zero if the fit pruned it entirely** (its other lines are absent —
  42% Fe would light up hundreds of Fe lines the spectrum does not show).
  Whatever the rival cannot host **returns to the incumbent element**: the
  fit's own assignment stands when no viable alternative exists. So Mn → 0
  (Mg is present and takes the 279.5 flux) but **Be keeps its 3.8%** when
  its only rival, Fe, is globally absent.

The reattribution is also **order-independent under mutual or chained
confounding**: every confounded element's clear-flux credit is applied
before *any* reattribution, so a confounder that is itself confounded (A↔B
mutually, or A→B→C chained) keeps its clear share and — since a
mutually-confounded rival is not independently detected — neither can steal
the other, so both simply retain their vertex split (the ambiguity is
flagged, not silently resolved by processing order).

An honest early guard against a too-aggressive metric: a first attempt
using the ratio of each element's true-negative cap to the concentration
its own peaks demand **failed** — self-absorption on Li's resonance line
made it look over-predicted and gutted genuine Li while leaving Mn intact.
The `clear_lines`/`contested_share` accounting above (which never touches
an element's own line ratios, only whether a *rival* can cover the peak)
is the version that behaves.

### What this does NOT yet do

`fraction_resolved` reattributes contested flux to the *single* strongest
rival at a first-order response ratio; a full joint re-solve of a
many-way confounded group (three-plus collinear elements) or a proper
posterior over the split is future work. `summary.csv` still carries the
raw vertex `fraction` as the primary column (with the resolved value in
`detections.csv`) so the two stay comparable; flipping the primary
composition to resolved is a deliberate downstream choice.

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
| 5 | Production Model | Prototype CLI | ~45% | Blocked by validation and Component 3 hardening |

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

1. **Stimulated-emission factor on the optical depth** — **TESTED
   2026-07-04: implemented, A/B-refused as default; flag retained.**
   The factor `(1 − exp(−hc/λk_BT))` is implemented as
   `stimulated_emission_factor` and applied (behind
   `sa_stimulated_emission` / `--stimulated-emission`, default OFF) in
   `_kappa_raw`, the frozen-reference normalisation, and
   `_doublet_line_scale`. It is wavelength-dependent (~0.91 at Na 589,
   ~0.89 at K 770, ~0.999 in the UV at 8.3 kK), so it is not absorbable
   into a single global `sa_tau_scale`.

   **A/B result (38 JChristensen spectra, deterministic optimizer,
   pre-registered metrics):** fit quality and internal consistency were
   *neutral* — median Δr² = 0.0000 and median Δstage_disagreement =
   0.0000 — but alkali/Si compositions swung >20% on **14/38** samples
   (K/Li/Na typically ±30–60%, one sample Li +1080% with Δr² −0.081).
   A physics-free control (perturbing only the optimizer trajectory,
   `n_calls` 40→41, stim off) produced median 0% shifts with only 3/38
   samples >20% — so the swings are attributable to the factor, not
   optimizer noise. Interpretation: with only the doublet-anchored SA
   channel active, the factor re-tilts a **near-degenerate alkali-SA /
   composition direction** without adding constraining information; the
   result is instability, not accuracy. **Refused as default on fit
   accuracy.** Revisit when (a) the global `sa_fit` channel is in
   production use (the cross-λ effect is then first-order rather than a
   perturbation), and (b) the emission-side weights and the doublet
   `thin_ratio` carry a consistent treatment. The A/B also showed the
   fixed-θ statistical uncertainty understates total uncertainty on
   samples with an active alkali-SA degeneracy (flagged by
   `stage_spread` in `summary.csv`).

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
