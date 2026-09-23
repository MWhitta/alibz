# Legacy Fe candidate-gate diagnosis

Date: 2026-09-22

## Finding

The known Fe positive in production benchmark case 3 is retained by the
initial-strength filter and then removed by the legacy multi-line evidence
filter.  At the benchmark's blind shift, both Fe I and Fe II fail the 0.20
coverage threshold and exceed the maximum of six missing strong lines.  Fe I
also has only one supported strong line where two are required.  Fe II reaches
the two-line support minimum, but its 7.20% coverage and 18 missing strong
lines still remove it.

The independently reported truth-assisted shift of -0.154 nm improves Fe I
coverage from 4.40% to 32.10% and supported strong lines from one to three.
Fe I still fails because 17 of its 20 strong lines are treated as missing,
well above the allowed six.  Thus blind wavelength calibration worsens the
evidence, but calibration error alone does not explain the Fe loss; the
legacy missing-line gate removes Fe even under the truth-assisted diagnostic.
This is a read-only diagnosis.  No legacy rule was changed.

## Reproduced input

- Benchmark report:
  `reports/physical-triage-production-benchmark-20260922.json`, SHA-256
  `c7de2480c057f2eb38ed187f8d6a814dd5ffa2b46c45a1e9b0373f644f02ea08`.
- Replay archive:
  `provenance/physical-triage-real-data-20260922.npz`, independently verified
  SHA-256
  `c8d04ab5b650a5f911b56dc8e0586b8aaff207bbe6c573ec5857bbe319a766b8`.
- Case 3 run: `run-eae9d54f22834928ae2f6af8d4a23779`, source-record
  SHA-256 recorded in the pinned archive metadata:
  `b0d1e1b5d89a7d7bf24aa230387c78e75c7822c720fc31f18100707ffd2f43f0`.
- Audited valid range: 186.0521--947.9357 nm; 5,846 native samples after
  cropping.  The known dense 960.1828--961.0 nm tail and the preceding gap
  were excluded as specified by the manifest.
- `production_peaks` extracted 40 peaks.  Their observed-frame contiguous
  little-endian float64 array has SHA-256
  `60cf493a6d6306a560758c7241e46ec87311cc5e34f16e67ba94faa9d73fa861`.
- Blind shift: -0.021742941321406306 nm from five anchors.  The diagnostic
  comparison used the separately reported truth-assisted shift -0.154 nm and
  is not benchmark evidence.

## Instrumentation method

The existing helpers `production_peaks` and `blind_shift` were called without
modification.  Instance wrappers recorded inputs and outputs around
`PeakyIndexerV3._prefilter_species_by_initial_strength` and
`PeakyIndexerV3._prefilter_species_by_line_evidence`; each wrapper then called
the original method.  Metrics were computed through the indexer's existing
`_line_weights`, `_build_design_matrix_from_map`, `_species_line_evidence`,
and `_species_evidence_keep_mask` methods.  No source or runtime rule was
edited.

Relevant implementation locations are
`scripts/benchmark_physical_triage.py:225` (`production_peaks`),
`scripts/benchmark_physical_triage.py:234` (`blind_shift`),
`alibz/peaky_indexer_v3.py:910` (initial-strength filter), and
`alibz/peaky_indexer_v3.py:1084` (line-evidence filter).

## Exact gate measurements at T = 10,000 K, log10(ne) = 17

Initial-strength thresholds were relative signal >= 0.001 of the global
maximum and, for a species touching at most one peak, relative signal >= 0.05.
Only 10,000 K was in the configured prefilter ladder for this run.

| shift | stage | signal | global-relative | peaks touched | passes initial gate |
|---|---:|---:|---:|---:|---|
| blind -0.02174294 nm | Fe I | 4,308.4687 | 0.0126227 | 20 | yes |
| blind -0.02174294 nm | Fe II | 14,941.2474 | 0.0437741 | 30 | yes |
| truth-assisted -0.154 nm | Fe I | 17,063.9736 | 0.0433951 | 22 | yes |
| truth-assisted -0.154 nm | Fe II | 12,501.0536 | 0.0317912 | 31 | yes |

Both stages remained after the initial-strength filter under both shifts.

The line-evidence thresholds were: top 20 lines; strong line relative
threshold 0.10; presence threshold 0.25; minimum coverage 0.20; minimum two
supported strong lines; maximum six missing strong lines.  The filter did not
require net evidence at this stage.

| shift | stage | matched / total mass | coverage | supported strong | missing strong | result |
|---|---:|---:|---:|---:|---:|---|
| blind | Fe I | 0.260112 / 5.917895 | 0.043954 | 1 / 20 | 19 | remove: coverage, support, missing |
| blind | Fe II | 0.508556 / 7.059672 | 0.072037 | 2 / 20 | 18 | remove: coverage, missing |
| truth-assisted | Fe I | 1.899885 / 5.917895 | 0.321041 | 3 / 20 | 17 | remove: missing |
| truth-assisted | Fe II | 0.155188 / 7.059672 | 0.021982 | 1 / 20 | 19 | remove: coverage, support, missing |

After the line-evidence filter no Fe stage remained under either shift.  The
blind run then retained only Ru, reproducing the benchmark case-3 candidate
result.  The truth-assisted comparison retained Ag and Ni, confirming that it
is diagnostic only and not a substitute benchmark result.

## Practical limitation established

The legacy line-evidence filter treats unmatched strong database lines as
negative evidence without local absent-line noise, response, saturation, or
self-absorption safeguards.  In this labelled Fe spectrum, that assumption is
the decisive false-negative mechanism.  The new physical triage independently
kept Fe in the same report with five strong matched features across two stages;
the legacy downstream filter subsequently removed it.
