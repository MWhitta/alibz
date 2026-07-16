# MW2-112 LIBS profile: full analysis plan

**Status:** historical pre-run plan retained to show the decisions and gates
specified before production. The run is complete; see
[provenance.md](provenance.md) for the executed method and verified outcome.  
**Prepared:** 2026-07-15  
**Input:** `/Users/mwhittaker/Library/CloudStorage/GoogleDrive-mwhittaker@lbl.gov/My Drive/Li/Data/LIBS/MW2-112/raw`

## 1. Scientific objective

Use the MW2-112 LIBS line profile as a spatially resolved chemical anchor for
correlative X-ray and neutron diffraction interpretation of a finely layered
lithium mudrock. The analysis will:

1. estimate per-position elemental composition and detection evidence;
2. preserve sharp stratigraphic changes at the native 100 micrometre spacing;
3. distinguish heavy/transition-metal chemistry important to X-ray contrast
   from light-element chemistry important to neutron contrast;
4. identify chemical associations consistent with lithium clay, quartz,
   calcite, authigenic K-feldspar, and accessory/trace phases; and
5. align the chemical profile to the X-ray/neutron coordinate system without
   treating LIBS alone as a phase-refinement measurement.

The present `alibz` result is an **atom fraction of detected plasma emitters**.
It is not yet a certified bulk-rock weight fraction: element-dependent
ablation, transport, detector response, atomic-data error, and plasma-model
systematics are not fully calibrated. Until standards close that boundary, the
primary defensible products are relative spatial profiles, detection status,
composition brackets, and systematic-sensitivity envelopes.

## 2. Input audit

The directory was inspected read-only on 2026-07-15.

- 929 CSV spectra, 772 MB total.
- Test IDs are unique and consecutive from 989 through 1917.
- Every inspected file has columns `wavelength,intensity` and 23,431 rows on
  the same nominal 180--961 nm, 0.033333 nm vendor grid. The physically active
  working domain is 190--910 nm (21,600 channels).
- If ID 989 is position zero and every ID is one 100 micrometre step, the
  nominal center-to-center profile length is 92.8 mm.
- IDs 989--1915 were acquired from approximately 10:35 to 12:17. IDs 1916 and
  1917 were acquired at 15:13 and 15:15, after a 176-minute interruption.
- The local `chemical-tomography` repository contains a second 929-file raw
  mirror and 927 vendor composition-summary CSVs. The summaries omit IDs 1916
  and 1917. Treat those vendor percentages as an uncalibrated comparison, not
  ground truth or a training target.
- ID 1632 is an all-zero spectrum and must be retained in the inventory as a
  failed/missing observation, never interpreted as zero elemental abundance.
- A four-minute acquisition gap occurs between IDs 1874 and 1875.
- Negative exported channels are real and spatially variable. In the active
  band the audited corpus median negative fraction is about 0.2%, with a tail
  above 3% concentrated in several late-profile spectra. Negative values must
  not be clipped before background/noise modeling.
- There are no sidecar coordinate, standard, dark, or acquisition metadata
  files in the raw directory.

The confirmed spatial mapping is

```text
height_above_profile_bottom_um = (test_id - 989) * 100
```

ID 989 is the bottom of the sample and increasing ID follows the upward
sedimentation/gravity direction. IDs 1916--1917 remain profile positions despite
their acquisition delay. Absolute registration to diffraction coordinates
still requires common endpoints or fiducials.

## 3. Quantities to report

### 3.1 Element-level outputs

For every spatial point and element, retain:

- raw NNLS atom fraction (`fraction`);
- true-negative-resolved fraction (`fraction_resolved`);
- the raw/resolved attribution bracket;
- detection class, statistical uncertainty, upper limit, z score, number of
  supporting and uncontested lines, and named confounder;
- ion-stage disagreement and peak-shape QC;
- whether support is concentrated in self-absorbed, blended, saturated, or
  detector-sensitive lines; and
- systematic envelopes from detector-response and model sensitivities.

Do not silently replace non-detections with zero. Spatial plots should show
detected values, upper limits, and missing/failed spectra distinctly.

### 3.2 Scattering-oriented summaries

Element grouping alone is not a scattering calculation. Report both the
underlying elemental profiles and explicitly labeled derived proxies:

1. **Transition-metal pool:** at minimum Sc--Zn, with 4d/5d transition metals
   retained individually if detected. Fe, Mn, Ti, V, Cr, Co, Ni, Cu, and Zn
   must remain visible rather than only appearing in a pooled curve.
2. **X-ray contrast proxy:** a provisional composition-weighted proxy using
   tabulated atomic form factors over the actual diffraction Q range. A simple
   `sum(x_i Z_i^2)` screen may be supplied for visualization, but it is not a
   substitute for phase structure factors.
3. **Light-element/neutron-sensitive pool:** H, Li, B, C, N, O, and F reported
   individually where spectroscopically defensible.
4. **Neutron coherent contrast proxy:** composition-weighted bound coherent
   scattering lengths, calculated only after isotope assumptions and the
   desired normalization are fixed.
5. **Neutron nuisance flags:** incoherent scattering and absorption reported
   separately. H can dominate incoherent background; Li is strongly
   isotope-dependent and may be more important through absorption than coherent
   scattering. These should not be collapsed into one unsigned "neutron
   strength" score.

Use natural isotopic abundances for every scattering and absorption calculation.
Report all three neutron mechanisms separately: bound coherent contrast,
incoherent scattering, and wavelength-dependent absorption. The principal
scientific product is relative spatial abundance of the elements producing the
largest X-ray or neutron contrast, not absolute bulk concentration.

H and Li require special caution. H-alpha is also used as an electron-density
diagnostic and may include surface/ambient contributions. Li I 670.8 nm is a
strong, self-absorption-prone line in the NIR detector segment. Without
composition standards, their absolute bulk concentrations should be labeled
semi-quantitative even when their spatial trends are strong.

### 3.3 Mineral-association anchors

Construct chemical association indices, not standalone LIBS phase fractions:

- lithium clay: Li--Mg--Si with Al/Fe substitution and possible Na/F support;
- dioctahedral aluminous clay: Al--Si with Li/Mg/Fe covariation;
- quartz: Si enrichment without corresponding Al, Li, Mg, K, or Ca support;
- calcite: Ca with C support when the C detection is clean and not ambient;
- authigenic K-feldspar: coupled K--Al--Si enrichment; and
- accessory phases: spatially coherent Fe/transition-metal or trace-element
  peaks with multiple uncontested lines.

These indices become priors/covariates for the diffraction interpretation.
Final phase proportions must come from the joint diffraction model, not from
forcing elemental profiles into ideal mineral stoichiometries.

## 4. Pre-run readiness gates

The full 929-spectrum run does not start until all gates pass.

### G1. Freeze spatial and acquisition metadata (partly satisfied)

- The ID mapping, 100 micrometre pitch, bottom origin, upward direction, and
  membership of the full consecutive series are confirmed.
- Each CSV is a single-pulse shot. Preserve this explicitly because there is no
  within-position averaging or replicate-derived measurement precision.
- Laser spot diameter is approximately 50 micrometres. Gate delay/width,
  atmosphere/purge, pulse energy, and translation details are currently
  unavailable and must be recorded as unknown rather than inferred.
- No standards, blanks/darks, replicate shots, bulk assays, or XRF/ICP
  measurements are currently available. This limits interpretation to relative
  composition/contrast profiles and systematic-sensitivity brackets.

### G2. Close detector-response bookkeeping

The current uncommitted pipeline estimates the VIS-to-NIR response at 620 nm
and falls back to 3.9 when the continuum is weak. A stratified 93-spectrum audit
found 74 measurable ratios spanning 1.57--4.98 and 19 fallback cases. Before
production:

- scale the final detection/resampling `area_sigma` by the same segment
  response as the corrected amplitudes;
- record response ratio, uncertainty, and source (`measured`, `fallback`, or
  `invalid`) in every per-spectrum row;
- add tests for fallback loading and end-to-end amplitude/uncertainty scaling;
- treat the fallback as a distribution, not an exact value; and
- pre-register response sensitivity runs (at least 3.0, 3.9, and 5.2 for
  fallback spectra, or an empirically improved session-specific distribution).

The 365 nm discontinuity remains additive and belongs to background handling,
not the multiplicative response correction.

### G3. Make the run immutable, resumable, and separate from raw data

The present CLI writes outputs into its input directory and only returns rows
after workers finish. Add a run wrapper that:

- accepts separate input and output directories;
- never writes to the Google Drive `raw` directory;
- writes one atomic JSON result/checkpoint per completed spectrum;
- resumes by validating input path/size/hash and configuration;
- assembles CSV/Parquet-style tables only from checkpoints;
- records git commit plus dirty patch hash, database/source hashes, Python and
  package versions, command/configuration, and input manifest; and
- keeps logs, timeouts, and failures without losing completed work.

Suggested run root:

```text
runs/mw2_112_profile/<UTC timestamp>_<git-or-patch-id>/
```

### G4. Fix analysis configuration before tuning

- Use the fitted pass-1/pass-2 temperature for minor-line prediction instead
  of the fixed 0.76 eV corpus temperature.
- Carry corrected peak covariance/noise into the composition and detection
  stages as far as the current solver permits.
- Do not claim electron density when H-alpha is absent or instrument-width
  subtraction makes it uninformative.
- Decide primary behavior for stimulated emission before the run. Recommended:
  off for the primary result, on only in a systematic subset because the
  existing A/B test exposed an alkali/self-absorption degeneracy.
- Pin the candidate database and element policy. Detection-only or
  source-uncertain elements must be labeled accordingly.

### G5. Define pilot acceptance criteria

The pilot must establish runtime and reject silent scientific failures. At
minimum, require:

- the all-zero ID 1632 becomes an explicit QC failure;
- no unexpected hard errors or timeouts;
- valid wavelength-shift anchors, with failures flagged rather than silently
  treated as zero shift;
- detector response and source recorded for every successful spectrum;
- no new one-element composition collapse;
- stable major-element rankings under `n_calls=40` versus a higher-call subset;
- standardized residuals and profile classifications visually acceptable in
  each detector segment; and
- Li, H, Fe, Si, Al, Mg, K, Ca, Na, Ti, and Mn line evidence manually reviewed
  across representative lithologies before scaling to the corpus.

## 5. Execution sequence

### Phase A. Inventory and QC table

Create `input_inventory.csv` with test ID, confirmed height, timestamp,
file size/hash, wavelength-grid checks, active-channel count, intensity
quantiles, negative fraction, missing/zero status, acquisition-gap flags, and
eventual detector-response metadata. Preserve all 929 rows.

### Phase B. Deterministic pilot

Run approximately 20 positions spanning the profile, including IDs 989, 1000,
representatives near 1100/1200/1300/1400/1500/1600/1700/1800/1900, the zero
file 1632, the 1874/1875 acquisition boundary, and 1915--1917. Add points chosen
from low/high intensity and low/high negative-fraction strata.

Use production settings (`n_calls=40`, `draws=32`, standard timeout) for the
timing pilot. Repeat a smaller subset at higher optimizer effort and detector
response alternatives. Inspect the generated fit notebook/plots before the
full run.

### Phase C. Full per-spectrum inference

After pilot approval, run all confirmed profile spectra with atomic
checkpoints. Use the local workstation for the pilot and execute the production
run on **Beryl** after its environment, storage, and CPU resources are
inventoried. Pilot wall time is authoritative for resource planning. Stage an
immutable copy or manifest-verified mirror of the raw inputs on Beryl rather
than analyzing a live cloud-mounted directory. Code, database, configuration,
and input hashes must match the run manifest, and checkpoints/results must be
mirrored back incrementally.

Primary per-point output uses the true-negative-resolved fractions for spatial
interpretation while retaining raw fractions and attribution brackets.

### Phase D. Systematic sensitivity ensemble

On a stratified subset and every scientifically important boundary, rerun:

- detector fallback response at its calibrated central and IQR values;
- stimulated emission off/on;
- optimizer effort (40 versus at least 80 calls);
- minor-line seeding/deepening on/off or tightened thresholds; and
- global versus accepted segment wavelength shifts when both are identifiable.

Use these differences as a model-systematic envelope. Do not combine them with
fixed-plasma amplitude resampling and call the result a purely statistical
standard deviation.

### Phase E. Spatial analysis

- Map IDs to confirmed coordinates; retain the 50 micrometre beam diameter and
  100 micrometre sampling pitch as separate metadata.
- Plot raw points before any smoothing.
- Analyze compositions in log-ratio space with explicit censored/non-detected
  handling rather than ordinary correlations on zero-filled fractions.
- Detect layer boundaries with robust multivariate change-point or fused
  segmentation, retaining both unsmoothed and segmented profiles. Do not impose
  a smoothing length larger than the beam footprint without diffraction
  evidence.
- Calculate transition-metal, light-element, and scattering proxies with
  propagated composition/systematic uncertainty.
- Test mineral-association indices and element covariance within, not only
  across, inferred layers.

### Phase F. Correlative diffraction registration

The X-ray and neutron data products are being compiled in the
`chemical-tomography` repository. When available, register LIBS coordinates to
their coordinates using endpoints or fiducials. Preserve the different beam
footprints by forward-averaging the
latent layer model into each instrument's sampling kernel rather than simply
interpolating all datasets to the finest grid.

Compare LIBS chemistry to diffraction-derived phase abundance using held-out
checks: fit/interpret selected layers, then predict chemical signatures in
other layers. Report disagreements as possible solid-solution variation,
amorphous material, trace phases, registration error, or LIBS matrix effects.
The vendor LIBS summaries already stored there may be plotted as a secondary
comparison, with their missing terminal positions and unknown calibration made
explicit.

## 6. Deliverables

```text
run_manifest.json
input_inventory.csv
config.json
checkpoints/<test_id>.json
summary_raw.csv
detections.csv
composition_long.csv
composition_resolved.csv
point_qc.csv
scattering_proxies.csv
layer_boundaries.csv
mineral_association_indices.csv
figures/profile_elements.*
figures/profile_scattering_groups.*
figures/profile_qc.*
figures/correlation_xray_neutron_libs.*
fit_inspection.ipynb
analysis_report.md
```

The report must distinguish measurement noise, model sensitivity, spatial
heterogeneity, upper limits, and uncalibrated bulk-to-plasma uncertainty.

## 7. Current readiness verdict

### Available now

- all 929 raw files are readable and compatible with the loader;
- wavelength/air conversion, segmented background/noise, refinement, line
  evidence, confounder analysis, self-absorption channels, and spatially useful
  detection outputs exist;
- the atomic database and whole-rock-prior artifacts resolve locally;
- required core Python dependencies are installed;
- 67 relevant detector/pipeline/finder tests pass on the current worktree;
- approximately 233 GB local disk is free; and
- the workstation exposes 10 logical CPUs.

### Not yet ready for the defensible full run

- the current detector-response edits are uncommitted and lack complete
  end-to-end uncertainty/provenance handling;
- no separate-output, atomic-checkpoint/resume production runner exists;
- minor-line ratios still use a fixed corpus temperature;
- fixed-plasma statistical errors omit important detector, plasma, atomic, and
  self-absorption systematics;
- no standards or replicates exist, so outputs must remain relative and
  single-shot robustness must be assessed spatially and through line-level
  consistency rather than empirical replicate precision;
- Beryl is not reachable from the current execution environment, so its alibz
  checkout, Python environment, CPU/storage capacity, and data staging paths
  remain unverified;
- the `chemical-tomography` diffraction products and registration information
  are not yet available; and
- the repository's independent benchmark path is stale, while certified
  reference-material validation remains outstanding.

**Decision:** proceed next with the small technical hardening tasks and the
stratified pilot. Do not launch or publish the full 929-spectrum relative-
composition run until those gates pass. Diffraction registration can follow as
soon as the `chemical-tomography` products are available.

## 8. Confirmed investigator decisions

1. IDs are consecutive 100 micrometre positions from the sample bottom upward
   along the sedimentation/gravity direction.
2. Every CSV is one pulse; no further acquisition metadata, standards, blanks,
   darks, replicates, or assays are available at present.
3. Neutron reporting will include coherent, incoherent, and absorption
   mechanisms using natural isotopic abundances.
4. Correlative diffraction data are being assembled in the
   `chemical-tomography` repository; registration details remain pending.
5. Relative abundance profiles for the strongest X-ray- and neutron-contrast
   elements are sufficient; absolute bulk weight percentages are out of scope.
6. The pilot may run locally; the production analysis should run remotely on
   Beryl after remote-readiness checks and manifest-verified data staging.

## 9. Execution record (2026-07-15)

The plan was executed through the production gate.

- Detector amplitude and final uncertainty frames are now consistent.
- Response ratio, uncertainty, and measured/shared/fallback provenance are
  recorded per shot. No additional detector metadata are assumed forthcoming.
- Minor-line predictions use each shot's fitted plasma temperature and remain
  within the detector segment that calibrated the line scale.
- A separate-output, manifest-hashed, atomic-checkpoint/resume runner exists.
- A data-internal session prepass pools neighboring-shot detector response,
  wavelength shifts, and peak widths without crossing the 4-minute or
  176-minute acquisition breaks; trustworthy per-shot estimates take precedence.
- Natural-isotope X-ray form factors and neutron coherent, incoherent, and
  absorption coefficients are tabulated separately.

The authoritative 20-shot quantitative pilot completed all 19 nonzero spectra;
the all-zero ID 1632 was correctly retained as an analysis error. Shared shift
calibration reduced ID 1400 from a 900-second timeout to 276 seconds. However,
only 3 positions passed quantitative physical QC, 4 warned, and 12 nonzero
positions failed because of low/negative physical-pattern R-squared, plasma
parameters at bounds, or weak-shape/composition collapse. Doubling optimizer
effort and varying the 620 nm fallback response produced large composition
changes without stable fit improvement. The full 929-shot closed-sum CF-LIBS
run therefore remains **rejected by the pre-registered production gate** and
was not sent to Beryl.

A more defensible relative-profile branch was completed locally for all 929
spectra. It uses 255 same-stage atomic multiplets, shared segment profile
characteristics, matched-filter areas, shot-yield normalization, and a
vendor-independent requirement of at least two spatially coherent lines per
ion stage. Its within-element spatial ranks validate against the independent
vendor summaries for Si (Spearman rho 0.934), Ti (0.883), Fe (0.705), Li
(0.657), and Mg (0.598). Al is internally coherent but externally discordant
(rho 0.030). H, B, C, F, several transition-metal traces, and other elements
without coherent multi-line support are retained as exploratory line records
but excluded from the primary profiles.

Primary completed run:

```text
runs/mw2_112_profile/relative_profiles_primary_20260715/
```

These scores are comparable along the profile for one element, not between
different elements. They are appropriate chemical anchors/covariates for the
forthcoming `chemical-tomography` registration, not bulk concentrations or
phase fractions.

## 10. Peak-window PCA execution record (2026-07-16)

The independent corpus-prior method in
[peak_window_pca.md](peak_window_pca.md) ran across all 929 positions. It uses
the training-consistent 0.155706 nm window and PC1--PC5, explicitly clusters
non-independent windows, audits all expected cross-element competitors, and
requires two uncontested coherent lines per ion stage.

The detector-shift prior was tested rather than assumed. Variable correction
worsened common-centroid dispersion in UV and visible data, so those segments
use constant median offsets; it improved NIR dispersion from 4.98 to 1.52 pm
and was retained there. The final run supports Al, Ba, Ca, Fe, Li, Mg, Na, Si,
Sr, and Ti. Preliminary Ag, Cu, Ga, Ge, and V assignments were rejected as
blends or single-line evidence. A subsequent broad-window resonance analysis
restored K after demonstrating strong optical depth; its relative trend is
reported separately because the narrow corpus PCA is not valid for those
0.4--0.47 nm-wide lines.

Promoted local run:

```text
runs/mw2_112_profile/peak_window_pca_shift_validated_20260716/
```

All 21 inventoried products verify against `product_checksums.json`, and the
original Drive raw/vendor inputs pass the upstream provenance verifier.
