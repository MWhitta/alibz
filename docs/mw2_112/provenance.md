# MW2-112 LIBS relative profiles: provenance and regeneration

For workflow commands and document navigation, start at the
[MW2-112 index](README.md). Open work is maintained in [TODO.md](TODO.md).

## Scope and scientific claim

This document describes the lineage of the MW2-112 LIBS profile product in
`relative_profiles_primary_20260715`. The primary values are **within-element
relative spatial scores**. A value may be compared between positions for the
same element; it is not an absolute concentration, a closed-sum composition,
or a scale on which Fe can be compared numerically with Li.

The quantitative CF-LIBS pilot failed its pre-registered production gate. It
was therefore not promoted to the primary product. This is a deliberate
scientific boundary, not missing processing.

## Data lineage

```text
929 raw single-pulse spectra
  -> immutable filename/size/SHA-256 inventory and 100 µm coordinates
  -> session-wide detector response, wavelength-shift, and width priors
  -> frozen same-stage multi-line feature manifest
  -> per-shot, per-line matched areas and uncertainties
  -> within-line scaling and spatial-coherence filtering
  -> within-element relative profiles and QC
  -> separate X-ray / neutron contrast annotations
  -> layer boundaries, mineral-association covariates, figures, report
```

The canonical regeneration route starts from the raw spectra but reuses the
frozen feature and calibration tables. This reproduces the stated analysis,
rather than silently re-estimating priors with a future database or algorithm.

## Raw observations and coordinates

- Sample: MW2-112, finely stratified lithium mudrock.
- Files: 929 vendor CSV spectra, test IDs 989 through 1917 inclusive.
- Observation unit: one laser pulse per CSV; there are no within-position
  replicates.
- Nominal wavelength grid: 23,431 rows from approximately 180 to 961 nm at
  1/30 nm pitch. The active analysis range is 190--910 nm.
- Scan geometry: consecutive positions from the bottom of the sample upward
  along the sedimentation/gravity direction.
- Position mapping:

  `height_above_bottom_um = (test_id - 989) * 100`

- Step: 100 µm center-to-center. Approximate beam diameter: 50 µm.
- Nominal span: 92.8 mm between the first and last centers.
- ID 1632 is an all-zero spectrum. It is an explicit missing/failed
  observation, never zero elemental abundance.
- A four-minute acquisition gap occurs between IDs 1874 and 1875. IDs 1916 and
  1917 follow a roughly 176-minute interruption but remain spatially
  consecutive profile positions.
- Gate timing, pulse energy, atmosphere/purge, and detector model/settings are
  unavailable and recorded as unknown. No later detector metadata are
  inferred.

`input_inventory.csv` is authoritative for every filename, byte size, SHA-256,
timestamp parsed from the name, coordinate, wavelength-grid check, intensity
quantiles, negative-channel fraction, and zero-spectrum flag. Absolute source
paths in that historical table are informational. Regeneration resolves files
by the inventoried basename so the verified mirror at
`data/LIBS/MW2-112/raw` can be used on another machine.

## Frozen instrumental priors

`session_calibration.csv` contains the per-position calibration state used in
measurement. These are shared **instrument characteristics**, not shared
chemical abundance:

1. A continuum-derived multiplicative response across the 620 nm detector
   junction. A locally measured value is preferred. Where unavailable, the
   neighboring session profile is used. The earlier corpus fallback has median
   3.9 and IQR 3.0--5.2; its source and uncertainty are explicit.
2. Separate UV, VIS, and NIR wavelength-shift priors inferred from atomic
   anchors in neighboring shots. A missing local anchor does not silently
   become zero shift.
3. Shared robust peak FWHM estimates by detector segment. This reduces
   unstable single-shot width fitting while retaining segment dependence.

The 365 nm discontinuity is handled as an additive background feature, not a
multiplicative gain. Negative exported channels are retained through
background/noise modeling rather than clipped.

The file's SHA-256 and a canonical JSON calibration hash are recorded in the
provenance and run manifests. Canonical regeneration consumes this exact table
instead of recomputing it.

## Atomic-line features and physical priors

`line_feature_manifest.json` freezes 255 candidate features for the declared
element policy. Each record includes element, ion stage, component
wavelengths, component weights, thermal reference strength, rank, and named
cross-element competitors.

Selection was constrained as follows:

- ion stages I and II are treated separately;
- components within 0.06 nm form one same-stage multiplet;
- the reference excitation temperature is 9000 K;
- at most five high-ranked features per element/stage are retained after
  removing detector-junction neighborhoods;
- potential competitors within 0.12 nm are explicitly marked rather than
  assumed absent; and
- an ion stage enters the primary profile only if at least two lines have a
  median cross-line spatial Spearman correlation of at least 0.15.

For each feature, a Gaussian multiplet template with the shared segment width
is fitted jointly with a local pedestal. The non-negative matched area and its
robust residual-noise uncertainty are retained in `line_profiles.csv`.
Response correction and wavelength shift are applied before area estimation.

The atomic database used to create the frozen manifest contained 252 files.
Its historical hash in `run_manifest.json` includes an absolute path and is
therefore not portable. `provenance/atomic_database_inventory.csv` instead
records every relative path, size, and SHA-256 plus a path-independent
aggregate. The selected numerical line data needed for canonical regeneration
are already embedded in the feature manifest, so the 462 MB database is not a
runtime dependency of that route.

## Construction of relative element profiles

For each shot and line:

1. subtract the data-derived background;
2. divide the NIR residual by the frozen 620 nm response;
3. shift the multiplet centers by the frozen segment shift;
4. estimate matched area and area uncertainty at the shared segment FWHM; and
5. divide the area by the positive, response-corrected shot line power.

Each line profile is then normalized by its own corpus 90th percentile. Line
profiles are admitted only through the multi-line stage-coherence rule above.
A line is detected at SNR >= 3. Clean lines are preferred to contested lines;
at least two clean detections are required for the strongest `detected` class.
Ion-stage scores and disagreements remain in the output.

This hierarchy separates four concepts that the earlier pipeline sometimes
conflated: expected line existence, measured line evidence, ion-stage support,
and relative abundance trend. It also makes unsupported elements explicit.
H, B, C, F, and several transition-metal traces do not have defensible
multi-line support in this run.

## Derived corpus peak-window PCA product

An independent relative-profile method now projects every reliable known-line
window into the fixed corpus peak-shape basis. Its full method, result audit,
and regeneration instructions are in [peak_window_pca.md](peak_window_pca.md).
The promoted local result is
`runs/mw2_112_profile/peak_window_pca_shift_validated_20260716/`.

That run references this bundle's frozen calibration and raw-input provenance,
but reselects windows from the hashed atomic database. It records 718 priors,
466 summed-spectrum-screened windows, 52 accepted independent windows, five
active PCs, segment-specific shift-validation decisions, all overlap
competitors, exact software state, and checksums for every product. The
original Drive raw and vendor directories passed `mw2-112 provenance verify`
before promotion.

The PCA output does not supersede the frozen matched-filter bundle. Agreement
between the independent methods is validation evidence; disagreement, notably
for Al versus the vendor summary, remains a flagged systematic limitation.

## Contrast and geological interpretation products

`relative_contrast_profiles.csv` attaches physical contrast factors without
turning element-specific scores into a cross-element composition:

- X-ray form factor at Q = 4 Å⁻¹ and its squared, profile-weighted annotation;
- natural-abundance neutron coherent scattering;
- natural-abundance neutron incoherent scattering; and
- natural-abundance absorption at the stated reference wavelength.

Coherent, incoherent, and absorption terms stay separate. Signed coherent
length is retained where applicable. The `cross_element_ranking_valid` field
is false because the elemental scores have independent normalizations.

`mineral_association_indices.csv` contains standardized chemical covariates
for lithium clay, aluminous clay, K-feldspar, quartz contrast, and Fe-Ti
accessories. They are not phase fractions. C lacks coherent support, so no
calcite index is reported. These covariates are intended to anchor joint X-ray
and neutron diffraction interpretation, with phase proportions determined by
the diffraction model.

Layer boundaries are generated by a multivariate PELT ensemble on robustly
standardized log profiles. The fixed penalty multipliers are 0.5, 1.0, and 2.0
times a BIC-scale penalty. A reported boundary must appear under at least two
penalties; full stability means all three. The zero shot is interpolated only
for segmentation, not for elemental output.

## Independent validation data

The 927 vendor summary CSVs are used only after fitting for spatial-rank
comparison. They omit IDs 1916 and 1917 and are not a ground truth,
calibration standard, or training target. Their individual hashes are in
`provenance/vendor_input_inventory.csv`.

Reported Spearman correlations are Si 0.934, Ti 0.883, Fe 0.705, Li 0.657,
Mg 0.598, and Al 0.030. The Al profile is internally coherent but externally
discordant and must remain flagged as systematic/model limited.

## Software and environment provenance

The production run began from `alibz` commit
`dd0c5019171aa9e82a88762c925ae302537aefb9` on branch `main`, using Python
3.13.5 on macOS 15.7.3 arm64. Primary package versions include NumPy 2.4.4,
SciPy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, periodictable 2.1.0,
PuLP 3.3.0, and ruptures 1.1.10. The complete captured environment is
`provenance/environment-py3.13-macos-arm64.lock.txt`.

Important limitation: the production Git worktree was dirty, and several
analysis modules and scripts were untracked. The run manifest preserved the
commit, dirty flag, untracked filenames, and hashes, but not the complete dirty
patch. A hash proves identity only when the bytes still exist; it cannot by
itself restore them. The original path-dependent source hashes are therefore
retained as historical evidence but are not represented as a complete source
archive.

This was remediated after the run by shipping:

- a `git archive` of the recorded base commit;
- `provenance/source_overlay/`, containing the complete runtime files needed
  for the current frozen-prior regeneration path;
- the exact environment lock;
- raw, vendor, database, source, and artifact file inventories; and
- an executable regeneration that checks the scientific tables against the
  preserved reference bytes.

The source capture is deliberately labeled as post-run remediation. It is not
misrepresented as a byte-for-byte reconstruction of the original dirty tree.

## Canonical regeneration procedure

The canonical route reproduces the published computation from the immutable
raw spectra, frozen feature manifest, and frozen session calibration:

```bash
mkdir mw2_112_reproduction
tar -xzf BUNDLE/provenance/alibz_base_dd0c5019171aa9e82a88762c925ae302537aefb9.tar.gz \
  -C mw2_112_reproduction
rsync -a BUNDLE/provenance/source_overlay/ mw2_112_reproduction/
cd mw2_112_reproduction
python3.13 -m venv .venv
./.venv/bin/python -m pip install \
  -r BUNDLE/provenance/environment-py3.13-macos-arm64.lock.txt
./.venv/bin/python -m scripts.mw2_112 provenance verify BUNDLE \
  --input-dir RAW_DIR --vendor-dir VENDOR_DIR
./.venv/bin/python -m scripts.mw2_112 regenerate \
  BUNDLE RAW_DIR NEW_OUTPUT --vendor-dir VENDOR_DIR --workers 6
```

Here `BUNDLE` is this result directory, `RAW_DIR` is the verified 929-file
mirror, and `VENDOR_DIR` is the verified summary directory. The script refuses
to use a nonempty output directory, verifies all raw bytes first, measures all
255 frozen features, rebuilds postprocessing products, and compares eight
deterministic scientific tables byte for byte with the reference bundle.

The number of workers controls throughput only. Python, NumPy/SciPy, BLAS, and
plotting versions should be kept fixed for strict byte comparison. Figures and
the prose report contain regenerated metadata and are verified scientifically
through their source tables rather than required to be byte-identical.

## Alternative audit and recovery routes

- `line_profiles.csv` is the lossless per-line intermediate. Running
  `python -m scripts.mw2_112 reassemble BUNDLE_COPY` and then the report
  command regenerates all downstream tables without reading raw spectra.
- The 929 JSON files under `checkpoints/` independently preserve each shot's
  measured line records and QC. If `line_profiles.csv` is absent, the
  reassembly script reconstructs it from these checkpoints.
- A from-first-principles audit may recompute line selection and session
  calibration using the exact inventoried database and the commands in
  `reproduction_recipe.json`. That tests the prior-generation logic; it is not
  the canonical frozen-prior reproduction and need not be byte-identical.

## Integrity policy

`provenance/artifact_checksums.csv` covers every shipped report, table, figure,
checkpoint, source-capture file, environment file, and provenance record. Run
the verifier before analysis and after transfer. New interpretations should be
written outside this dated bundle. If the bundle itself must change, create a
new dated version, regenerate the checksum inventory, and retain this one.

The verified raw mirror is not duplicated inside the result directory. Its
929 SHA-256 values remain in `input_inventory.csv`. Absolute file locations,
hostnames, and timestamps are provenance metadata, not logical dependencies.

## Known irreducible limitations

- No certified standards, blanks/darks, replicate shots, or bulk assays are
  available; absolute concentration and empirical precision are unsupported.
- Single-pulse heterogeneity includes sampling and ablation variability.
- Detector/acquisition settings are incomplete, so unmodeled systematic error
  may remain despite internal response/shift/width correction.
- Natural isotopic abundance is assumed; sample-specific Li or H isotope
  composition was not measured.
- H has no defensible multi-line profile, so the largest potential neutron
  incoherent contribution cannot be anchored by this LIBS run.
- O may retain ambient/plasma sensitivity; Al fails the independent trend
  comparison; trace lines remain susceptible to blends and database error.
- LIBS chemical covariates do not establish crystallographic phase identity or
  phase fraction without the correlative diffraction model.
