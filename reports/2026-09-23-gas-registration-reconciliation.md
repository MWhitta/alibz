# Reconciling the two argon wavelength estimators — 2026-09-23

Two sessions implemented the same request ("use Ar I and O I lines as separate
wavelength calibrations") in parallel:

- `alibz/gas_calibration.py` + `alibz/wavelength_calibration.py` (closed
  session): regional Ar I / O I offsets from a native-maxima peak table,
  consensus over ≥ 3 (Ar) / ≥ 2 (O) independent groups, applied only between
  bracketing anchors (`gas_wavelength_calibration`, default `apply`).
- `alibz/wavelength_registration.py` (this session, commit 88cd8fb):
  line-shape-aware two-pass matching with a Gaussian instrumental-profile
  sub-pixel centre, per-segment offset/slope (`wavelength_registration`,
  default `ambient`: NIR only, ≥ 3 lines).

Both are committed together here (the closed session's work was uncommitted and
interleaved with mine in `alibz/pipeline.py`). This report records what each
does on the same real data, why they disagreed, and the rule that now binds
them. Data: the closed session's pinned archive of 26 Fe run means
(`provenance/physical-triage-real-data-20260922.npz`, native 7,914-point grid,
99.98 % Fe, 10-shot batches, argon purge).

## 1. Outcome on the 26 real run means

| estimator | gate | calibrated / applied | tentative | no evidence |
|---|---|---|---|---|
| gas engine, Ar, default (`min_snr` 6, native maxima) | ≥ 3 consistent groups | **0** | 2 | 24 |
| gas engine, Ar, `min_snr` 3 | ≥ 3 consistent groups | **1** (run-2e03f73d: −0.156 ± 0.039 nm, 3 inliers) | 24 | 1 |
| gas engine, O (either gate) | ≥ 2 groups | 0 | 0 | 26 (O I absent under the purge) |
| ambient registration, NIR | ≥ 3 lines to apply | **0 applied**; 1 line in 24 runs ("weak", recorded), 0 lines in 2 | — | — |

The gas engine's per-group reason is `no_observed_peak_in_search_window` for
almost every Ar group: its native-maxima peak discovery (three-cell parabola,
sideband noise, SNR ≥ 6) finds no candidate at the argon positions in a 10-shot
Fe mean. The ambient registration measures the same features at SNR 6–17 but
accepts only one of them per run. Neither applies a correction to this data;
both record what they saw. **So the "disagreement" was never about the
offset: one engine saw nothing, the other one line.**

The gas engine's semi-synthetic recovery (`reports/gas-calibration-real-20260923.json`,
`semi_synthetic_recovery`) returns the injected ±0.070 / ∓0.060 nm offsets with
errors < 1e-13 nm. That validates the bookkeeping (frames, regional
application, Ar/O separation); it is not evidence about real spectra, because
the injected Gaussians sit on a flat background at exactly the anchor
positions.

## 2. Why one line: the argon anchors in a pure-iron target

Per-anchor shift (observed − database) over the 26 run means, measurements
with SNR ≥ 5, composition-blind matching pass:

| Ar I anchor (nm) | n | median shift (nm) | MAD (nm) | median SNR | best non-argon explanation within the window |
|---|---|---|---|---|---|
| 696.543 | 25 | **−0.188** | 0.011 | 17.0 | none (Fe I 696.302 and N II 696.681 are 0.24 / 0.14 nm away) |
| 772.376 (+772.421) | 5 | **−0.123** | 0.010 | 6.4 | pair centroid; consistent with −0.16 if 772.421 dominates |
| 801.479 | 6 | **−0.165** | 0.033 | 7.6 | Fe II 801.692 / O II 801.316 at 0.19 nm (gated as ambiguous) |
| 706.722 | 6 | −0.657 | 0.020 | 6.1 | no strong database line; not argon at this registration |
| 738.398 | 16 | −0.379 | 0.017 | 9.1 | blended (88 %); Ar II 738.042 / N I 737.851 nearby |
| 763.511 | 5 | −0.373 | 0.009 | 6.8 | Fe II 763.193 (0.14 nm from the feature after −0.19) |
| 794.818 | 11 | −0.342 | 0.002 | 11.7 | Fe II 794.506 / 794.523 (0.15 nm) |
| 811.531 | 4 | −0.824 | 0.006 | 7.7 | Fe II 810.978 (0.08 nm) |
| 826.452 | 2 | +0.294 | 0.004 | 5.8 | Fe II 826.620 (0.32 nm) |
| 842.465 | 1 | −0.782 | — | 5.5 | Fe II 842.056 (0.18 nm) |
| 750.387, 912.297 | 0 | — | — | < 5 | too weak |

Three anchors (696.5, 772.4, 801.5 nm) agree on a registration of about
**−0.17 ± 0.03 nm**, each repeatable to 10–30 pm across runs. Every other
"argon" feature is equally repeatable but sits 0.2–0.6 nm from that value: in
a 99.98 % Fe target the NIR is not empty, and Fe II lines within 0.1–0.2 nm of
the Ar I 763/794/811/842 nm anchors (plus three unlisted repeatable features)
are what both estimators find there. The 696.5 nm line is the only anchor with
SNR > 10 in a 10-shot mean, which is why the ambient registration reports one
line, and why in 5 runs its robust mode locked onto the 794 nm iron feature
(−0.34 nm) instead.

This corrects the "−0.18 nm, trusted" wording of
`reports/2026-09-23-wavelength-registration.md` §"Coordinator verification":
the value stands (median −0.196 nm, MAD 10 pm over 46 of 53 runs once the
composition gate is on; it was 48 before, the gate removed exactly the two
outliers at +0.48 and −0.33 nm), but it is a
**single-line** (696.5 nm) measurement, not a multi-line registration, and its
±50 pm drift bound applies to that line only.

## 3. What was changed

- `ambient_registration(..., composition=...)`: the sample composition now
  gates the ambient anchors (a `composition` config key existed but the config
  validator rejected it, so the path was dead). The composition only removes
  anchors; it never supplies them. On this data it flags 801.5 nm as ambiguous
  (O II 801.316 / Fe II 801.692) and changes nothing else, because the strong
  competitor heuristic (Boltzmann strength at 9 kK relative to the band
  maximum) does not predict the NIR Fe II lines that actually appear. Lowering
  the isolation fraction to 0.03–0.003 was tried and rejected: it removes the
  genuine 696.5 nm anchor (Fe I 696.302 enters the window) and keeps 794 nm.
- Pipeline order and precedence (unchanged, now documented): ambient
  registration may replace the NIR segment shift (≥ 3 lines, quality ok/weak);
  the gas engine's regional calibration then overrides inside its bracketed
  support when it reaches `calibrated`. Both are argon-based, so where both
  fire they must agree; `analysis['wavelength_registration']['gas_cross_check']`
  records their difference with a 20 pm single-line floor, and the summary QC
  gains `ar-registration-disagreement` (> 3 σ). New summary columns:
  `wavelength_registration_mode`, `wavelength_registration_applied`,
  `ambient_nir_shift_pm`, `ambient_nir_n_lines`, `ar_registration_n_sigma`.
- CLI: `--wavelength-registration {off,report,ambient,apply_element}` and
  `--subpixel {gaussian,parabolic}` beside the closed session's
  `--gas-wavelength-calibration` and `--physical-triage`.
- Two of the closed session's tests were stale against its own code and were
  updated to the documented behaviour: `test_missing_argon_group_abstains`
  (bracket evidence is kept as `tentative` segments, never an offset) and
  `test_controlled_injection_uses_independent_gas_origins_on_native_grid`
  (the injection returns truth records and the benchmark re-extracts peaks
  blindly).
- Defaults retained: `gas_wavelength_calibration="apply"` and
  `wavelength_registration="ambient"`. Both apply only when their gates pass,
  and on 10-shot Fe means neither does, so production output is unchanged
  except for the new records and columns.

## 4. What it would take to register the NIR from argon

A dedicated registration acquisition: accumulate 50–100 shots (or a longer
gate) on a target without a NIR Fe II forest (Al, Cu, or a glass) under the
purge, so that Ar I 696.5, 706.7, 738.4, 750.4, 763.5, 772.4, 794.8, 801.5 nm
reach SNR > 10 unblended. Then the ambient registration applies automatically
(≥ 3 lines), the gas engine can calibrate the bracketed 696–801 nm region, and
the cross-check becomes meaningful. O I cannot serve under the argon purge
(absent in all 26 + 53 runs). UV/VIS still need a line lamp or a certified
multi-element reference (previous report).

## Verification

- Focused suites after the edits: `tests/test_wavelength_registration.py`
  (19, incl. the composition-gate and cross-check tests),
  `tests/test_peakfit.py`, `tests/test_gas_calibration_integration.py`: 44
  passed. Closed-session suites `tests/test_gas_calibration*.py`,
  `test_gas_detection.py`, `test_triage*.py`: 106 collected, 2 stale tests
  fixed, all pass.
- Full suite after the edits: 563 passed, 4 skipped, 71 subtests (7 min 45 s).
- Study re-run with the composition gate (`scripts/registration_study.py`,
  53 runs): ambient NIR 46/53 non-null (48 before; the two removed are the
  +0.48 / −0.33 nm outliers), median −0.196 nm, MAD 9.5 pm; drift slopes and
  the Gaussian-vs-parabolic comparison (47.8 vs 51.9 pm median residual,
  n = 1786) unchanged.
- Measurement scripts for the tables above were run inline against the pinned
  archive (`provenance/physical-triage-real-data-20260922.npz`, sha256
  c8d04ab5…); the per-anchor table is reproducible with
  `wavelength_registration.ambient_registration(x, y, db)['lines']` over the
  26 run means.

No provider switch, deployment or hardware action.
