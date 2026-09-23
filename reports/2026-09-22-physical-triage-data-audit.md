# Physical-triage real-data audit (2026-09-22)

## Outcome

The local workspace has one composition-labelled real LIBS material suitable
for a physical-element-triage recall test: an Aesar nominal 99.98% Fe target.
It has 26 live Z300 runs and 252 shots. The acquisition ledger, raw native CSVs,
and contemporaneous analysis remain available under the acquisition scratch
tree. No independently labelled non-Fe matrix was found. Therefore an Fe recall
test is supported, but the broad-validation gate across diverse compositions
must fail honestly.

The reusable inventory is
`provenance/physical-triage-real-data-20260922.json`. It pins nine averaged
spectra selected before benchmarking: five fresh-site representative runs,
three held-out verification runs, and one reused-spot nuisance run. Each entry
contains its path, SHA-256, native wavelength span, acquisition parameters,
truth strength, evidence, split, and limitations. All 252 shot CSVs remain in
the source acquisition tree but are intentionally not treated as 252
independent samples.

## Source and label strength

- Source root:
  `/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/acq`
- Ledger: sibling `ledger.json`, SHA-256
  `31062f933f84c32da9ac2df19aa929ee6c37fe2df23df98de841ecc5cd779537`.
- The sorted ledger has 26 runs, consistently labels the sample
  `Acquire panel (Fe Aesar 99.98%)`, and records run ids, timestamps, settings,
  locations, requested shot counts, and recovery details.
- `reports/2026-09-22-fe-calibration-plasma-analysis.md` independently
  characterizes those same 26 runs and 252 stored shots. Its pinned SHA-256 is
  `c2fe4a90e90c246fe673647f2ec07c53e6aeccfe9f6fb9024a1691e3505a1c0c`.
- The label is **well-supported nominal identity**, not an independently
  measured assay in this repo. No vendor certificate or trace-element table is
  archived. Thus Fe is an expected positive; there are no certified negatives.
  Ar, H, O, N and trace candidates are context only and must not be scored as
  target composition truth.
- Existing alibz/indexer detections and compositions were excluded as truth.

## Loading and wavelength details

Each selected input is `<run>/average.csv`, header
`wavelength,intensity`, loaded as:

```python
a = numpy.loadtxt(path, delimiter=",", skiprows=1)
wavelength_nm, intensity_counts = a[:, 0], a[:, 1]
```

Every average has 7,914 data rows. Later grids span 186.0521–961.0000 nm;
the oldest selected run begins at 186.0551 nm. The real continuous coverage is
186.0521–947.9357 nm, followed by a 12.2471 nm gap and an anomalously dense
2,066-sample terminal interval from 960.1828–961.0000 nm. Peak benchmarks
should ordinarily exclude that terminal interval. Use each run's own native
axis without resampling.

The prior Fe analysis reports `observed = database_air + shift`, approximately
−119 pm in UV and −154 pm in VIS/NIR. Those values are useful audit evidence,
but using them in the benchmark would leak Fe identity. Blind benchmarking must
estimate its wavelength correction from generic anchors or data-only evidence.

## Prespecified split

- Representative: ledger indices 17–21, five fresh raster locations spanning
  delay/period settings 5/25, 5/10, 10/10, 5/50, and 10/50.
- Held out: indices 23–25, the verification location at 20/50/1000,
  20/50/100, and 20/10/100. These share one spot and are robustness replicates,
  not independent materials.
- Nuisance: index 0 from the heavily reused location. Use it for surface/depth
  or stability stress only.

This split avoids tuning on the verification conditions, but no split can
manufacture matrix diversity from one target.

## Other local real spectra

`data/remote_samples/{REE_01,REE_44,argon_noAr,scan9x9,user_spec1}.csv` are
real full-range exports useful for runtime, wavelength, null-translation, and
qualitative diagnostics. They have no independent reference composition in
the workspace. The repo's own
`reports/2026-09-21-stage-consistency-thermometer.md` explicitly records the
missing ground truth for user_spec1, scan9x9, and REE_44. Filenames and past
indexer outputs are insufficient evidence, so none can contribute true-positive
recall or false-positive counts.

`data/ree_*.pkl` are derived peak/PCA artifacts and are neither independent
truth nor preferred loading inputs. `tests/fixtures/z300_real_spectrum_trimmed.json`
is a useful ingestion fixture without supported composition truth. The older
47-shot Fe purity checkout is documented in
`reports/2026-09-21-fe-purity-checkout.md`, but its spectra are recorded as a
remote beryl scratch product and are not locally present in the audited paths.

The measured-data K resonance diagnostic is pinned as exploratory evidence.
Recomputing the K I 766.49/769.90 **integrated-area** ratio from
`provenance/physical-triage-k-20260922/k_wide_line_profiles.csv`
for pairs with both areas positive and both SNR at least 5 gives 928 pairs,
5/50/95 percentiles 1.351270188/1.440606688/1.527383171, and 926/928 outside
±20% of the optically thin ratio 2.006045. This is a strong empirical
counterexample to rejecting K solely for a non-thin doublet ratio. It is
processed measured-data evidence, not an independent composition label and
not proof that every fitted pair is K. Peak-height ratios in
`k_doublet_optical_depth.csv` must not be substituted for the integrated-area
test.
The archived CSV is byte-identical to the original measured-data product at
`runs/mw2_112_profile/k_resonance_diagnostic_20260716/k_wide_line_profiles.csv`
(SHA-256 `35dc5ca86a9346775790aa25f11e34a8a0dd225c01f2438038d50f06063de280`).

## Gates and limitations

- Fe expected-positive retention can be measured on nine prespecified run
  averages (or all 26 as technical replication).
- Runtime and candidate-reduction effects can be measured on all real spectra.
- Unlabelled elements are unknown, not false positives. The nominal purity does
  not provide certified element-by-element absence thresholds.
- The diverse independently labelled matrix gate is **FAIL**: one Fe target,
  zero supported non-Fe target matrices.
- The source path is transient `/private/tmp`. The 26 run means are preserved
  in `provenance/physical-triage-real-data-20260922.npz`, a deterministic
  compressed NPZ with SHA-256
  `c8d04ab5b650a5f911b56dc8e0586b8aaff207bbe6c573ec5857bbe319a766b8`.
  It contains native wavelength and intensity arrays shaped `(26, 7914)`, run
  ids, and JSON metadata pinning source-average hashes and acquisition fields.
  Full mean regeneration or shot-level replay still requires the 252 raw shot
  CSVs in the source acquisition tree.

## Verification before edits

Required system-Python command:

`PYTHONPATH=src python3 -m pytest tests/ -q`

Result: collection failed with 30 errors because that interpreter lacks SciPy;
zero tests ran.

Required project-environment command:

`PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`

Result: 455 passed, 4 skipped, 71 subtests passed, one warning, in 286.54 s.
No provider switch occurred.

## Verification after deliverables

- `PYTHONPATH=src python3 -m pytest tests/ -q`: collection failed with 34
  missing-SciPy errors after collaborating work added triage/gas tests; zero
  tests ran.
- Pymatgen-enabled required invocation: 498 passed, 4 skipped, 71 subtests
  passed, one warning, in 274.53 s.
- `tests/test_triage_benchmark.py`: 4 passed; the final full run exercised the
  archived K path and offline 26-mean replay.
- Both normal and `--replay-archive` dry runs reported one dataset, nine
  prespecified samples, 26 expanded run means, and the honest failed broad
  validation gate.
