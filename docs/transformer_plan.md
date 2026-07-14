# Irregular-spectrum transformer plan

## Objective

Build an accuracy-first transformer that maps one spectrum containing one or
more observed `(wavelength, intensity)` regions to stage-resolved plasma
abundances for the 92 fixed H--U positions. The model must accept nonuniform
samples, disjoint spectral regions, and different numerical resolutions
without depending on a fixed channel index. It is calibrated only for the
current instrument and its 190--910 nm response; accepting an arbitrary input
grid does not imply extrapolation outside that physical band or below the
instrument's resolving power. The first model processes one spectrum at a
time; repeat-spectrum aggregation is a later hierarchical layer. Training and
first-version inference use individual-shot spectra, not shot averages.

Synthetic spectra are the supervised source of truth. Their atomic emission
physics is deterministic for a versioned scene seed. Corpus spectra provide
quality-gated priors for nuisance parameters and the conditional PCA peak
profile distribution, not composition labels.

Natural-composition scene shapes and element co-occurrence come from the
versioned Gard et al. whole-rock prior described in `whole_rock_prior.md`.
This empirical prior augments rather than replaces the balanced element-stage
and abundance-decade schedule. Its bulk-to-plasma transfer is explicitly
provisional until standards are available.

Training weights the nine classified anhydrous lithologic strata and the two
separately fitted carbonate/volatile-rich strata equally. It does not inherit
the corpus's igneous, volcanic, or geographic sampling frequencies. LOI-only
records cannot supervise H/C/S/O abundance because the lost material is not
chemically speciated.

## Locked scientific contract

- The output schema always has 92 positions in atomic-number order. Pm, Po,
  At, Rn, and Pa remain present and explicitly marked `unsupported`; they are
  never silently removed or reported as zero.
- Tc, Fr, Ra, and Ac retain technically supported output positions but are
  explicitly `training_excluded` in this development round. They are absent
  from both scheduled targets and synthetic background mixtures; exclusion is
  not evidence of zero abundance in an input spectrum.
- Se, Th, and U lines are included as described in `atomic_data.md`. Se II is
  detection-only for now. Stage III is used only where quantitative lines in
  the instrument band actually support it.
- The target is emitting-plasma abundance by element and ion stage I--III,
  `x[e,s]`. Elemental nuclei abundance is `x[e] = sum_s x[e,s]`, with
  `sum_e x[e] = 1` after ambient gas is excluded.
- Stage abundances are independent model inputs. No Saha-equilibrium loss,
  synthetic-data rule, architectural coupling, or post-processing projection
  may impose an ionization distribution.
- Excited levels within each populated stage follow a Boltzmann distribution
  at an independent `T[e,s]`.
- The simulator has no explicit multiphase decomposition in this development
  round.
- Self-absorption uses a provisional effective column density. Gate delay and
  width are fixed by the current acquisition software. A bespoke path-length
  model is deferred.
- Electron density is `n_e[e]`: one effective value per element, shared across
  that element's stages I--III and coupled to other elements by explicit
  cross-element constraints. There is no single global electron density.
- Effective column density is `N_eff[e]`: one value per element, shared across
  its stages and coupled by explicit cross-element constraints. It is not a
  per-line free parameter.
- The ambient gas is a continuous Ar/air mixture. Gas metadata is optional;
  when absent, the model infers Ar evidence from the spectrum. It must not
  claim to infer physical canister pressure.
- The air endpoint uses a fixed dry-air composition in this round. Humidity is
  excluded from the gas model.
- Accuracy has priority over inference latency. Deployment optimization comes
  only after the scientific acceptance gates pass.
- `T[e,s]` is returned with a calibrated 95% confidence interval when line
  ratios make it identifiable. A prior-dominated temperature is retained
  internally but is not presented as a measurement.
- Synthetic target abundances include exact zero and the continuous range
  `1e-8` through 1. Coverage is balanced by abundance decade rather than left
  to incidental random draws.
- Input intensities are exported per-channel counts, not spectral density.
  They are modeled as integrals through the channel's effective instrument and
  export response, not as point evaluations or ideal rectangular-bin
  integrals.
- A partial spectrum with insufficient total evidence returns
  `composition_complete=false`; observed elements are not renormalized.
- Air-confounded target N or O is `detected-unquantifiable`, not assigned a
  spuriously precise target abundance.
- Positivity is enforced on latent photon emission and expected
  pre-subtraction detector counts. Signed exported observations are retained
  because dark/offset subtraction and vendor remapping can produce negative
  values.

## Output and evidence contract

The internal posterior covers all supported stage abundances so that mass is
conserved. The user-facing record for every element-stage contains:

- support flag and reason;
- evidence status: exactly `unobservable`, `detected-unquantifiable`, or
  `quantified`;
- presence probability and, for non-detections, an upper limit when one can be
  calibrated;
- abundance median and confidence interval when quantified;
- `T[e,s]` median and confidence interval only when independently
  identifiable;
- per-element `n_e[e]` and `N_eff[e]` posteriors as internal nuisance values,
  with exposure controlled separately from the `T[e,s]` output policy;
- observed wavelength support, contributing transitions, and warning codes;
- an uncertainty decomposition: measurement/aleatoric, model/ensemble, atomic
  data, and missing-coverage contributions.

Element totals are computed from joint posterior samples rather than by
adding marginal confidence bounds. The response also includes the unassigned
abundance-mass interval and a `composition_complete` flag. A partial region is
allowed to produce useful estimates, but missing elements are never treated as
zero and visible elements are never automatically renormalized to 100%.

The evidence status is based on coverage, signal, line attribution, and local
identifiability, not just a neural-network probability threshold. In
particular, one detected line can support presence but normally cannot
separate abundance from `T[e,s]` and self-absorption.

## Coordinate-aware input representation

### Meaning of an exported channel count

For a nonnegative continuous photon spectrum `S(lambda)`, physical collection
and signed export processing are modeled separately:

```text
c[j] ~ photon/readout model(integral S(lambda) R[j](lambda) d(lambda))
y[i] = sum_j W[i,j] * (gain[j] * c[j] - dark_estimate[j])
       + offset[i] + export_error[i]
```

Here `R[j]` is the nonnegative optical and physical detector-pixel response;
the expected photon contribution to `c[j]` is nonnegative. `W[i,j]` is the
instrument software's remapping/apodization onto the exported wavelength grid
and can introduce correlated ringing. Subtracting an estimated dark/offset
from a noisy measurement can also make `y[i]` negative. Thus,
"channel-integrated" does not mean simply evaluating `S` at the reported
wavelength, nor does it assume an ideal top-hat integral between adjacent
wavelength midpoints.

MW2-112 and JChristensen exports each contain 23,431 rows from 180--961 nm on
an exactly uniform 0.033333 nm grid, of which 21,600 rows span the active
190--910 nm domain. The uniform grid is the vendor export grid and need not be
identical to the physical detector-pixel grid. Explicit channel edges describe
external binnings when available; native current-instrument data use calibrated
physical `R[j]` and export `W[i,j]` responses.

The exported count can be floating-point and background-shifted; it is not
required to be a nonnegative integer. For example, representative active-band
files have the following raw characteristics:

| Corpus example | Median | 99th percentile | Maximum | Negative samples |
|---|---:|---:|---:|---:|
| MW2-112 #1000 | 72.1 | 881.6 | 8,070.5 | 0.028% |
| JChristensen MDD006 B0x | 29.2 | 820.2 | 6,512.2 | 26.2% |

These are examples, not fixed targets. Their large difference in negative
baseline fraction demonstrates why segment-, session-, and acquisition-mode
nuisance distributions are required.

Negative exported values do not relax the physical positivity constraint.
They are modeled only after nonnegative photon generation and detector
collection. They must not be clipped before inference: clipping creates a
point mass at zero, biases local baselines upward, changes weak-line
likelihoods, and erases information about the noise/offset distribution. The
neural input uses a noise-scaled signed transform such as
`asinh(y / sigma_local)` while preserving raw `y` as an auxiliary feature.

The negative-value source is not uniquely identifiable from a two-column CSV.
Morphology provides provisional classes: isolated negative pixels or short
oscillations adjacent to a bright line are export-kernel ringing, while long
low-signal negative runs indicate dark/offset subtraction. In a 100-file
sample of MW2-112 individual shots, the median negative fraction was 0.236%,
the 95th percentile was 2.71%, and the maximum was 7.62%, proving that negative
values are not solely an averaging artifact. Dark/blank spectra or vendor
processing documentation are needed to identify the electronic terms more
specifically.

### Spectral cells

Sort and validate the input samples, split it at genuine gaps, and infer cell
edges from adjacent wavelengths. Every input token represents a physical
spectral cell and contains:

- cell center wavelength and left/right width in nm, accepting optional
  explicit channel edges;
- intensity and a robust transform of intensity;
- estimated local noise and signal-to-noise ratio;
- saturation, clipping, missing-data, and detector-junction flags;
- current detector segment identity (190--365, 365--620, or 620--910 nm);
- optional acquisition and gas fields, each with an explicit missingness bit.

No learned channel-number embedding is permitted. Wavelength uses continuous
Fourier features spanning empirically selected sub-line-width through
full-band scales. Attention receives signed wavelength separation, cell
overlap, and log cell-width as relative features.

Input intensity is the exported detector count for the spectral cell.
Explicit left/right channel edges are accepted when available; otherwise
nominal edges are inferred from wavelength centers. Synthetic spectra are
passed through the calibrated effective instrument/export response before
being emitted on those cells. The model receives cell width explicitly and
may derive a nominal count-density feature for comparison across grids, but
the original count remains available. Attention uses cell width and an
effective-resolution indicator so a wider channel is not interpreted as
proportionally stronger plasma emission and subdividing a native channel does
not create extra evidence.

### Grid and region augmentation

Each continuous synthetic scene is rendered into several views:

- the native current-instrument grid (nominally 0.03333 nm);
- regular coarser and finer grids within the empirically valid range;
- locally irregular and jittered grids;
- single contiguous crops and multiple disjoint regions;
- missing blocks, masked cells, and regions crossing detector junctions.

All views share a scene identifier and physical target. Training and test
splits occur at the scene level so alternative samplings cannot leak across
splits. Losses never force a cropped view to be confident about a stage for
which the crop contains no evidence.

The native-grid view is always generated through the measured current-
instrument export model. Coarser grids are count-conserving rebinnings of that
measurement process. Finer numerical views preserve the native optical
resolution and correlated information content; they must not imply that
interpolation created new resolving power.

## Model architecture

The proposed model is a sparse, hierarchical transformer rather than a dense
transformer over a fixed raster:

```text
(lambda, intensity, cell width, quality) tokens
                  |
       wavelength-local attention
                  |------------------------|
                  v                        v
       catalog transition queries    global spectral latents
       (quantitative + observed)      (continuum, artifacts, gas)
                  |                        |
                  +------ blend graph -----+
                              |
                    element-stage queries
                              |
                       element queries
                              |
         abundance / evidence / T / nuisance posteriors
```

### 1. Local point encoder

A small shared MLP embeds cell features. Transformer blocks then exchange
information only inside wavelength-radius neighborhoods measured in nm, with
several physical radii covering line cores, wings, blends, and local
background. This is linear-sparse in the number of samples and does not change
its physical receptive field when sample density changes.

### 2. Atomic transition queries

Each catalog transition is a query carrying wavelength, stage, lower and
upper energy, transition strength when known, source/uncertainty class, and
quantitative-readiness. It cross-attends only to nearby covered cells.

Quantitative transitions participate in abundance and temperature inference.
Observed-only NIST records may support a presence explanation but never
receive an invented oscillator strength or a quantitative intensity target.
Uncovered transitions receive an explicit coverage mask, not a zero-intensity
observation.

A sparse blend graph links transitions whose instrument-convolved profiles
overlap. Graph-masked attention lets competing elemental explanations be
resolved jointly rather than peak by peak.

### 3. Global spectral latents

A modest Perceiver-style latent bank cross-attends to chunks of all spectral
cells. It represents continuum shape, detector response, wavelength drift,
PCA profile state, saturation, unmatched peaks, and Ar/air evidence that is
not local to one catalog line. Chunked cross-attention supports long or
high-resolution inputs without quadratic full-spectrum attention.

### 4. Stage and element hierarchy

There are 276 fixed element-stage queries (92 elements by stages I--III).
Each gathers its transition evidence and global context. Element queries then
combine their three stages without a Saha constraint. Unsupported or
atomic-data-inadequate queries are masked with explicit reasons.

The abundance posterior is a zero-aware logistic-normal distribution: a
presence gate handles true absence, while posterior samples of active logits
are normalized jointly to enforce elemental-nuclei mass conservation.
Separate heads predict evidence/identifiability, `log(T[e,s])`, per-element
`log(N_eff[e])`, per-element `log(n_e[e])`, wavelength offsets, intensity
gain, and Ar evidence. Hierarchical cross-element constraint layers couple
the density and column predictions without collapsing them to global values
or imposing Saha equilibrium. These nuisance outputs feed a differentiable
spectral-rendering consistency loss but are not all necessarily part of the
production response.

### 5. Uncertainty

Use heteroscedastic posterior heads plus an independently trained ensemble.
Calibrate final 95% abundance and temperature intervals on untouched
synthetic scenes, stratified by element, stage, abundance decade, gas mixture,
coverage, and signal quality. Conformal calibration can correct nominal
coverage but cannot turn a structurally unidentifiable quantity into a
measurement.

`T[e,s]` is exposed only when all of the following pass calibrated thresholds:

- the stage is detected and has quantitative line data in the observed
  regions;
- the observed transitions span enough upper-level energy to constrain a
  Boltzmann slope;
- blends, saturation, and self-absorption do not make the local forward-model
  Jacobian rank deficient;
- ensemble and posterior intervals satisfy held-out coverage tests.

## Synthetic generator

### Scene variables

Generate a versioned scene from a recorded random seed and manifest:

1. Draw a sparse-to-dense elemental composition using element-balanced,
   abundance-decade-balanced sampling across exact zero and `1e-8` through 1.
   Include pure, binary, ternary, realistic multielement, and deliberately
   adversarial blended mixtures. Draw realistic multielement cases from the
   rock-stratified whole-rock copula, using balanced strata for training and
   retaining corpus-weighted sampling as a documented evaluation condition.
2. Draw independent stage I--III fractions for each present element. Do not
   derive them from Saha equilibrium.
3. Draw `T[e,s]` for every populated, quantitatively supported stage from
   corpus-anchored bounds, then apply Boltzmann excitation within that stage.
4. Draw per-element electron densities and other broadening variables from a
   cross-element distribution anchored by QC-approved corpus evidence that
   does not depend on Saha-derived fitted values.
5. Draw source-aware perturbations of oscillator strengths and energy data;
   keep both the nominal truth and perturbed realization in the manifest.
6. Draw cross-element-constrained per-element effective column densities and
   apply self-absorption, fixed gate response, and the current instrument's
   wavelength- and segment-dependent optical response.
7. Draw a continuous mixture between pure Ar and fixed dry air, excluding
   humidity, and render its spectral contribution separately from the target
   abundance denominator. Label target N/O as `detected-unquantifiable`
   whenever air makes their source attribution non-identifiable.
8. Draw empirical peak-profile, baseline, response, wavelength-drift,
   saturation, photon/read noise, and dark-estimation variables; integrate
   nonnegative emission through the physical channel response; then apply the
   signed vendor-export kernel or a count-conserving derived-grid
   transformation.

All random draws are pseudorandom but replayable. Given the simulator version,
catalog hashes, PCA artifact hash, and scene seed, spectrum generation is
deterministic.

### Comprehensive coverage

Do not rely on an unconstrained Dirichlet draw to cover the periodic table.
Build a coverage scheduler and fail dataset generation unless every supported
element-stage has the required number of scenes in every applicable abundance
decade, temperature band, gas band, detector segment, and blend-difficulty
bucket. Pairwise schedules must oversample known line-confusion pairs. Stage
III cells with no quantitative lines are recorded as structurally
unobservable, not simulated with fabricated data.

The five unsupported element positions are included in every target and
metric table with a fixed support mask. Tc, Fr, Ra, and Ac are separately
retained as technically supported schema positions with
`training_enabled=false`. Dataset generation must reject them as focus
elements and as incidental mixture backgrounds. Consequently, current-round
coverage gates apply to 83 training-enabled positions, while all 92 positions
remain present in model outputs and metric tables.

## PCA profile model from `peaky_data`

The existing corpus analysis covers 4,909 spectra and about 13.5 million peak
windows; ten PCs explain about 99.2% of normalized-window variance. Those
summary numbers do not by themselves define a generative distribution.

Before synthesis:

1. Refit or reconstruct the PCA with full per-window scores and immutable
   spectrum/peak identifiers.
2. Exclude failed fits, severe saturation, unresolved multi-peak windows,
   detector-edge artifacts, and other windows that do not represent a single
   generative profile. Preserve excluded classes separately for artifact
   augmentation.
3. Split at the 365 and 620 nm detector junctions and characterize conditional
   score distributions versus wavelength, fitted width, amplitude/SNR,
   distance to a junction, and other supported instrument variables.
4. Fit a held-out-validated conditional density for PCA scores and width
   parameters. Sample it with a recorded seed; reject shapes that violate
   positivity, area, or observed profile bounds.
5. Keep atomic mass, Stark, Doppler, and self-absorption effects in the
   physical renderer. The empirical PCA residual must not learn an element ID
   or otherwise leak composition into line shape.
6. Validate generated profiles against held-out corpus distributions by
   segment, wavelength, width, asymmetry, and reconstruction error.

This refit is a release gate. The currently saved PCA summary is not
sufficient for conditional profile generation if its full score/metadata
mapping cannot be recovered reliably.

## Experimental-characteristic matching gate

The legacy `PeakyMaker` nonzero/replay tests are necessary but do not establish
that its spectra resemble MW2-112 or JChristensen. Before synthetic training,
the new renderer must match held-out real-corpus distributions, separately for
the UV, visible, and NIR detector segments, including:

- wavelength pitch, active range, detector junctions, missing/zero channels,
  and cross-segment gain;
- continuum level and curvature, negative-baseline fraction, individual-shot
  photon/read noise, noise autocorrelation, and dynamic range;
- isolated-line FWHM, area/height, asymmetry, PCA score distribution, and
  wavelength dependence;
- strong-line ringing/notches, side lobes, clipping/saturation, and effective
  export-kernel correlations;
- peak density, blend multiplicity, self-absorption depth, and amplitude/area
  distributions conditional on signal-to-noise;
- distribution of residuals after the same peak-fitting pipeline is run on
  synthetic and experimental spectra.

Validation uses held-out spectrum groups, per-feature distribution distances,
and a real-versus-synthetic discriminator. A discriminator that easily
separates the two corpora is evidence of a missing nuisance mechanism, not a
training objective to fool by adding unconstrained noise. The generator must
also preserve its known physics labels and deterministic replay while passing
these empirical checks.

## Training objectives

Use a weighted multi-task objective, with weights checked for gradient
conflict rather than selected only by scale:

- stage-presence focal/Brier loss;
- zero-aware compositional likelihood and Aitchison/log-ratio error;
- abundance error per decade, with explicit major/minor/trace strata;
- masked heteroscedastic `log(T[e,s])` likelihood;
- evidence-status and identifiability classification;
- Ar/air evidence and nuisance-parameter likelihoods;
- differentiable forward-rendering likelihood on the actually observed cells;
- profile/line-attribution auxiliary losses;
- cross-grid consistency for outputs supported in both views, plus an
  uncertainty-monotonicity loss when spectral evidence is removed.

No loss compares an independent-stage prediction to a Saha distribution.
Clean, full-range synthetic scenes form the first curriculum stage; line-data
uncertainty, empirical artifacts, difficult blends, random grids, and partial
regions are added progressively. Masked-spectrum self-supervision on unlabeled
corpus spectra may initialize the point encoder, but supervised abundance and
temperature truth remains synthetic until standards exist.

## Data splits and validation

Split by underlying physical scene before rendering alternate grids. Group
real corpus spectra by physical sample/acquisition group before any
self-supervised train/validation split. Maintain dedicated challenge sets for:

- unseen element combinations and abundance ratios;
- each supported element and stage, including rare heavy elements;
- strong blends and near-coincident lines;
- self-absorption and saturation extremes;
- Ar-rich, air-rich, and intermediate mixtures;
- wavelength drift and detector-junction artifacts;
- withheld PCA profile clusters;
- coarser, finer, irregular, cropped, and disjoint sampling patterns;
- atomic-data perturbations not used for a training scene.

Report errors and interval coverage for every element-stage cell, never only
a pooled average. Synthetic acceptance targets should include:

- calibrated 95% interval coverage in each adequately populated stratum;
- low false-quantification rate for absent or unidentifiable stages;
- abundance error by decade and Aitchison distance for complete compositions;
- `T[e,s]` error and interval coverage only on identifiable cases;
- stable posterior medians for equivalent samplings of the same continuous
  spectrum, with wider uncertainty as useful regions are removed;
- exact support masks and mass conservation in posterior samples;
- no measurable dependence on channel ordinal index or resampling density.

Numeric error thresholds are set after a deterministic baseline establishes
the irreducible error from noise, line-data uncertainty, and missing coverage.
Passing synthetic tests establishes simulator inversion accuracy, not
real-world composition accuracy. Certified standards remain required for
external calibration and accuracy claims.

## Implementation phases and release gates

1. **Scientific schema and observability audit**
   - Freeze versioned 92-by-3 target, support, status, and reason-code schemas.
   - Audit quantitative and observed-only I--III coverage for all supported
     elements in 190--910 nm.
   - Produce per-stage observability and line-confusion matrices.

2. **Corpus refit and nuisance priors**
   - Refit quality-gated PCA profiles with recoverable scores.
   - Estimate non-Saha corpus distributions for widths, shifts, backgrounds,
     noise, saturation, gain, and defensible temperature/broadening priors.
   - Freeze versioned artifacts with train/validation provenance.

3. **Explicit-stage deterministic simulator**
   - Refactor synthesis so stage fractions and `T[e,s]` are direct inputs.
   - Integrate source-aware line uncertainty, effective self-absorption,
     Ar/air nuisance emission, empirical profiles, and pixel integration.
   - Add replay, conservation, no-Saha, and line-ratio unit tests.

4. **Irregular-grid data layer and baselines**
   - Implement spectral-cell validation, gap detection, cell-width inference,
     coverage masks, and multi-grid rendering.
   - Establish deterministic nonlinear-fit and small point-network baselines.
     The transformer must beat them by element-stage stratum.

5. **Hierarchical transformer**
   - Implement local wavelength attention, transition queries, blend graph,
     global latents, stage/element decoders, and posterior heads.
   - Verify coordinate, masking, permutation, subdivision, and mass invariants
     before large training runs.

6. **Beryl training and ablation study**
   - Use reproducible manifests, distributed mixed-precision training,
     checkpoints, and full per-stratum evaluation.
   - Ablate atomic queries, global latents, PCA conditioning, renderer loss,
     uncertainty components, and grid augmentation to verify each claim.

7. **Calibration and production gate**
   - Train the accuracy-first ensemble, calibrate intervals, and freeze model,
     simulator, catalog, and preprocessing hashes together.
   - Shadow-test unlabeled current-instrument spectra for residuals and OOD
     behavior. Do not claim bulk-fraction accuracy until standards are added.

## Resolved interface decisions

- Intensities are exported per-channel counts formed through an effective
  instrument/export response. Explicit channel edges are accepted; nominal
  boundaries are inferred from wavelength centers otherwise.
- Synthetic target abundance covers exact zero and `1e-8` through 1.
- Reported abundance and `T[e,s]` intervals are equal-tailed 95% intervals;
  non-detections use one-sided 95% upper limits.
- Insufficient partial spectra return `composition_complete=false` and are not
  renormalized over visible elements.
- Air-confounded target N/O is `detected-unquantifiable`.
- The first model processes one spectrum at a time. Repeat aggregation is
  deferred to a later hierarchical model.
- Training and first-version inference use individual-shot spectra.
  `AverageSpectrum` files may validate mean profile/response behavior but do
  not define the individual-shot noise distribution.
- The evidence states are `unobservable`, `detected-unquantifiable`, and
  `quantified`. The five unsupported element positions additionally carry an
  orthogonal `unsupported` support flag; unsupported is not a fourth evidence
  state.
- Tc, Fr, Ra, and Ac carry an orthogonal `training_excluded` policy flag. The
  model must not present an untrained quantitative estimate for those
  positions as calibrated.
- Electron density is per element and cross-element constrained; it is not
  global and is shared across stages of that element.
- Effective column density is per element and cross-element constrained; it
  is shared across stages of that element.
- The Ar/air mixture uses fixed dry-air composition without humidity.
- Latent emission and expected pre-subtraction detector counts are
  nonnegative. Signed exported counts are retained and modeled after dark/
  offset subtraction and vendor remapping; they are not clipped to zero.

## Constraint details still to resolve

Before freezing simulator priors and posterior heads, define the mathematical
form and strength of the cross-element constraints for `n_e[e]` and
`N_eff[e]`, decide whether either nuisance is user-facing, and decide how the
ambient-gas elements participate in those constraints.
