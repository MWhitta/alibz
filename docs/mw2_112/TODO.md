# MW2-112 TODO

Items are ordered by the risk they pose to durable reuse or scientific
interpretation. Completed implementation details belong in the provenance
record, not as permanently checked TODO boxes here.

## P0 — durable handoff

- [ ] Track and commit the currently untracked MW2-112 source, tests, and
  documentation as one reviewable change. Do not mix unrelated
  notebook/log worktree changes into that commit.
- [ ] Transfer the dated 179 MB result bundle to
  `chemical-tomography/analysis/libs_profile/`, run the checksum verifier
  after transfer, and retain the immutable source copy until two verified
  copies exist.
- [ ] Put `chemical-tomography` under version control or document its external
  archival/versioning system. It currently has no Git metadata, so a local
  copy alone is not durable provenance.
- [ ] Store the 462 MB atomic-database snapshot in a content-addressed archive
  keyed by the portable hash in `atomic_database_inventory.csv`. The frozen
  feature manifest is sufficient for canonical regeneration, but the complete
  database is needed to audit feature selection from first principles.

## P1 — correlative diffraction integration

- [ ] Establish the affine coordinate transform from LIBS height to the X-ray
  and neutron tomography coordinates using common endpoints or fiducials;
  record orientation, offset, scale, uncertainty, and any excluded interval.
- [ ] Ingest the relative profiles, QC flags, and stable/ensemble layer
  boundaries into the chemical-tomography data model without converting them
  to phase fractions.
- [ ] Define the joint interpretation model: diffraction supplies phase
  identity/proportion, while LIBS element profiles enter as chemical
  covariates or priors with explicit uncertainty and unsupported-element masks.
- [ ] Decide the diffraction Q values and neutron wavelength(s) used in the
  final experiment, then recompute the scattering annotations at those values;
  retain coherent, incoherent, and absorption mechanisms separately.

## P1 — scientific validation

- [ ] Run a pre-registered peak-window PCA sensitivity grid over the material
  competitor ratio, detection coverage, and common-observation gate. Report
  element-support stability explicitly. Keep the K broad-resonance exception
  separate from this grid because its lines exceed the corpus window width.
- [ ] Replace the single global peak-shape basis with detector-segment-specific
  corpus bases while preserving training scores and normalization metadata;
  compare physical-coordinate stability before changing primary profiles.
- [ ] Manually inspect representative high/low-profile shots for every
  two-window element (Li, Ba, Sr) and all lines with large abundance-to-shape
  correlations. Confirm that endpoint baselines and broad wings do not drive
  the spatial trend.
- [ ] Calibrate the abundance-weighted 0.25 material-competitor ratio using
  synthetic mixtures or standards. Until then it is an explicit identification
  prior, not a quantitative cross-element response model.
- [ ] Acquire or identify matrix-matched standards, replicate shots, blanks,
  and an independent bulk assay if absolute or cross-element concentrations
  become necessary. Until then, preserve the within-element claim boundary.
- [ ] Investigate the externally discordant Al profile using manually reviewed
  lines, detector-segment sensitivity, database alternatives, and mineralogical
  context. Do not use Al as an unqualified anchor meanwhile.
- [ ] Treat H as unavailable from this run. If neutron-incoherent contrast is
  essential, obtain an independent H-sensitive measurement rather than
  inferring H from unsupported LIBS evidence.
- [ ] Quantify sensitivity to the SNR threshold, line-coherence threshold,
  number of features per stage, shared-width window, response prior, rolling
  median width, and PELT penalties. Report boundary/profile stability rather
  than selecting settings against desired mineralogy.
- [ ] Review Li I 670.8 nm and O lines for self-absorption, ambient/plasma
  contributions, and detector-response sensitivity across representative
  lithologies.

## P1 — reproducibility engineering

- [ ] Add a Linux/Beryl environment lock or container image. The preserved
  production lock is macOS arm64 and is exact for audit, not portable binary
  deployment.
- [ ] Promote path-independent source/database/input hashes into run-manifest
  schema version 2; retain the historical path-dependent hashes for lineage.
- [ ] Add schema validation for every CSV/JSON product and a small committed
  synthetic fixture that exercises inventory → measurement → reassembly →
  report in CI.
- [ ] Make the provenance freeze fail when the worktree contains uncaptured
  runtime changes, unless an explicit source overlay is created first.
- [ ] Add a semantic/tolerance comparator alongside byte comparison for future
  platform migrations where BLAS or plotting metadata may differ.

## P2 — repository hygiene

- [ ] Audit ignored top-level `data/`, `jobs/`, database binaries, caches, and
  downloaded papers. Assign each to generated, external-source, archived, or
  versioned categories; do not delete scientific assets merely because they
  are ignored.
- [ ] Decide whether sample-specific orchestration remains in `scripts/` or
  moves to a maintained `alibz.workflows` package once a second sample uses the
  same relative-profile design.
- [ ] Replace hard-coded reporting element lists and association formulas with
  a versioned declarative configuration after the current result is archived.
- [ ] Add a concise data catalog in `chemical-tomography` linking raw LIBS,
  vendor summaries, frozen result bundle, X-ray volumes, neutron volumes, and
  coordinate transforms by checksum and sample identifier.
