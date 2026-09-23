# Physical-triage real-data benchmark harness (2026-09-22)

## Outcome

Implemented a deterministic real-data benchmark harness, a conservative
composition-truth manifest, two self-contained provenance archives, and focused
tests. The harness is ready for the main session's 26-mean quick run and
selected production/full-indexer comparisons.

The validation gate intentionally fails unless all supported positive elements
survive the actual post-filter candidate set, candidates decrease beyond the
existing filters, median candidate-build time improves, and at least three
independently labelled material matrices are present. The current corpus has
one Fe material, so broad validation cannot pass regardless of speed.

## Changes

- `scripts/benchmark_physical_triage.py:69` validates the manifest schema and
  raw input hashes. Replay mode verifies the archive hash without requiring the
  transient raw CSV paths.
- `scripts/benchmark_physical_triage.py:96` expands the prespecified nine-run
  split to all 26 ledger means as technical replicates; unselected runs are
  nuisance/stability observations, not independent materials.
- `scripts/benchmark_physical_triage.py:142` loads and validates the offline
  NPZ replay archive, including aligned shapes, metadata lengths, and run-id
  ordering.
- `scripts/benchmark_physical_triage.py:170` crops both quick and production
  extraction to audited continuous coverage. It excludes the anomalously dense
  960.18–961.00 nm terminal export interval for an explicit data-quality reason.
- `scripts/benchmark_physical_triage.py:178` implements blind quick extraction:
  segment-local median background, robust local MAD noise, 4-sigma peaks,
  local integrated areas, fitted-width proxies, and integrated-area noise.
- `scripts/benchmark_physical_triage.py:225` supplies the optional production
  path through `PeakyFinder.fit_spectrum`. Its `sqrt(area)` uncertainty is
  labelled a proxy and no missing-line claim is enabled from it.
- `scripts/benchmark_physical_triage.py:234` estimates generic segmented
  wavelength shifts without Fe truth. Coverage and local-noise coordinates are
  transformed to the database frame with the peaks.
- `scripts/benchmark_physical_triage.py:254` compares the existing candidate
  build with opt-in prune mode on the same corrected peaks. Confirmed Ar/O gas
  detections run first and are passed as protected elements to direct triage,
  prune candidate builds, null controls, and optional full indexers.
- `scripts/benchmark_physical_triage.py:286` records positive-element evidence,
  matched groups, score/rank, deferral/removal status, and actual off/prune
  candidate retention. Null translations of +1.37, −1.91, and +2.73 nm record
  the same diagnostics rather than treating `keep_elements` as proof.
- `scripts/benchmark_physical_triage.py:307` independently reproduces the
  measured-data K integrated-area counterexample: 928 >=5-sigma pairs,
  5/50/95% ratios 1.351270188/1.440606688/1.527383171, and 926/928 outside
  ±20% of the thin ratio 2.006045.
- `scripts/benchmark_physical_triage.py:333` writes a deterministic compressed
  NPZ with fixed ZIP metadata so identical inputs have an identical archive
  hash.
- `scripts/benchmark_physical_triage.py:365` runs per-sample triage, gas,
  candidate, null, and optional 10-call off/prune full-indexer comparisons and
  writes detailed JSON. Its gate combines recall, reduction, runtime, and
  material diversity.
- `scripts/benchmark_physical_triage.py:537` exposes `--dry-run`, `--manifest`,
  `--out`, `--limit`, `--indexer-cases` (default 0), `--peak-method
  quick|production`, `--archive-inputs`, and `--replay-archive`.
- `tests/test_triage_benchmark.py:17` tests manifest hash rejection;
  `tests/test_triage_benchmark.py:43` tests coverage-respecting quick peaks;
  `tests/test_triage_benchmark.py:56` pins the K integrated-area result; and
  `tests/test_triage_benchmark.py:67` verifies offline replay of all 26 native
  7,914-row means.
- `provenance/physical-triage-real-data-20260922.json:1` records truth policy,
  nine prespecified split entries, loading/grid details, all source/evidence
  hashes, limitations, and the honest broad-validation failure.
- `provenance/physical-triage-real-data-20260922.npz` preserves all 26 run means
  and metadata. Shape is `(26, 7914)` for both wavelength and intensity;
  SHA-256 is
  `c8d04ab5b650a5f911b56dc8e0586b8aaff207bbe6c573ec5857bbe319a766b8`.
- `provenance/physical-triage-k-20260922/k_wide_line_profiles.csv` is a
  byte-identical copy of the otherwise ignored measured-data result, SHA-256
  `35dc5ca86a9346775790aa25f11e34a8a0dd225c01f2438038d50f06063de280`.
- `reports/2026-09-22-physical-triage-data-audit.md:1` documents the full data
  audit, supported truth, loading, splits, exploratory exclusions, and gates.

No existing runtime source was edited by this task. Other working-tree runtime
changes belong to the collaborating triage/gas agents and were preserved.

## Commands and evidence

Dry-run validation:

`PYTHONPATH=src:<venv-site> python3 scripts/benchmark_physical_triage.py --dry-run`

Result: status `ok`, one dataset, nine prespecified samples, 26 expanded run
means, broad-validation status `fail`.

Offline dry-run:

`... benchmark_physical_triage.py --dry-run --replay-archive provenance/physical-triage-real-data-20260922.npz`

Result: status `ok`, archive hash/order/shape valid, 26 replayable means.

One-sample end-to-end smoke, both raw and offline replay: both exited 0 and
wrote valid detailed JSON. It extracted 166 quick peaks. Direct triage retained
Fe with five matched features across three ion stages, but the pre-existing
candidate filters removed Fe in both off and prune modes. Thus prune added no
Fe loss, while actual final candidate recall was still zero. The generic blind
shift was +25.6 pm rather than the Fe-audit diagnostic shift; no Fe-derived
calibration was injected. This is a benchmark finding, not a harness success.

Focused command:

`PYTHONPATH=src:<venv-site> python3 -m pytest tests/test_triage_benchmark.py -q`

Result before the final path-only K provenance update: 4 passed in 1.52 s. The
subsequent full suite exercised the final archived path and passed.

## Required test counts

Before changes:

- `PYTHONPATH=src python3 -m pytest tests/ -q`: collection failed with 30
  missing-SciPy errors; zero tests ran.
- Pymatgen-enabled invocation: 455 passed, 4 skipped, 71 subtests passed,
  one warning, 286.54 s.

After changes:

- `PYTHONPATH=src python3 -m pytest tests/ -q`: collection failed with 34
  missing-SciPy errors after the suite gained triage/gas tests; zero tests ran.
- `PYTHONPATH=src:$(.venv/bin/python -c
  "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`:
  498 passed, 4 skipped, 71 subtests passed, one warning, 274.53 s.

## What remains unverified

- No independently labelled non-Fe material exists locally. Diverse-matrix
  scientific validation remains blocked and the broad gate must stay failed.
- There is no vendor certificate or trace assay, so unlabelled elements cannot
  be scored as false positives or certified negatives.
- The full 26-sample final benchmark and production/full-indexer comparison are
  owned and rerun by the main session against the final shared triage/gas code.
- The required system interpreter lacks SciPy; only the project-environment
  invocation can execute the suite.
- No provider switch occurred. No hardware or remote host was contacted.
