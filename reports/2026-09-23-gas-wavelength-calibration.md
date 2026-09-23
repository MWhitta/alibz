# Independent Ar I and O I wavelength calibration

Status: implemented and committed 2026-09-23 (reconciled by the coordinating
session after the implementing session closed); real-data outcome in §Outcome.

User requested known Ar I/O I wavelength references separate from the internal
Fe calibration and explicitly selected application within supported regions,
with Ar and O reported separately.

Implementation owners: independent gas estimator/tests, real-data validation,
calibration-boundary audit delegated; main owns application logic, pipeline,
CLI, cross-component tests and final verification.

Design: calibrate untouched incoming x/y with original observed-frame peaks,
without Fe/mixed-element priors. Absolute observed-minus-database offsets replace
baseline corrections only where independent same-segment gas anchors bracket a
region. Keep the incoming instrument calibration, mixed-element baseline, and
both gas evidence records. Ar/O disagreements beyond combined uncertainty keep
the baseline in their overlap. Multiplet ambiguity must enter uncertainty or
cause abstention. Unknown/weak references never force a calibration.

Modes: apply (default per user), report, off. Detailed per-spectrum evidence is
exported to wavelength_calibration.json, with separate Ar/O summary fields.
Native calibration coefficients, frozen session tables and raw spectra are
unchanged. Verification and real-data performance are pending.

No provider switch, deployment or hardware action.

## Outcome (2026-09-23, added at commit time)

Real data (`reports/gas-calibration-real-20260923.json`, 26 Fe run means,
pinned archive): Ar `calibrated` 0, `tentative` 2, `no_evidence` 24; O
`no_evidence` 26 (absent under the argon purge). Nothing was applied to any
real spectrum. The dominant per-group reason is
`no_observed_peak_in_search_window`: the native-maxima peak discovery finds no
candidate at SNR ≥ 6 in a 10-shot Fe mean. With `min_snr` 3 one run calibrates
(−0.156 ± 0.039 nm, 3 inliers), consistent with the line-shape-aware ambient
registration on the same runs (Ar I 696.5 nm: −0.188 nm, MAD 11 pm). The
semi-synthetic recovery (< 1e-13 nm error) validates frames and regional
application only.

Several Ar I anchor groups (763, 794, 811, 842 nm) coincide within 0.1–0.2 nm
with Fe II lines that are present in a pure-iron target; the competitor check
must see the sample composition to exclude them (see the reconciliation report).
The `apply` default is retained: it only acts when ≥ 3 independent groups agree,
which on this data never happens. Two tests that were stale against the final
code were updated (`test_missing_argon_group_abstains`,
`test_controlled_injection_uses_independent_gas_origins_on_native_grid`).
Binding rule with `wavelength_registration`, cross-check record and the
recommended argon-rich registration acquisition:
`reports/2026-09-23-gas-registration-reconciliation.md`.
