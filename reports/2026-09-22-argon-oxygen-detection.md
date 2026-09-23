# Early Ar I and O I background-gas detector

Date: 2026-09-22

## Delivered API

`alibz.gas_detection` provides `detect_argon`, `detect_oxygen`, and
`detect_background_gases(x, y, db, *, shift_nm=0.0, peak_array=None,
config=None)`. The combined call returns JSON-safe `Ar` and `O` evidence
records with `detected`, `tentative`, `no_evidence`, or `out_of_range` status.

The detector runs on the native spectrum. It obtains exact stage-I wavelengths,
upper energies, and positive `gA` values from `Database.lines()`. Rounded anchor
ranges only select the documented Ar I and O I groups. Multiplets, including O I
777 nm and the close Ar I 772 nm pair, count as one independent feature.

## Conservative gates

- Local core and sideband windows expand with local native pitch, retain at
  least three samples, and reject physical gaps.
- A signed baseline-subtracted area must align with an expected component and
  exceed five calibrated local area sigmas. The point-noise scale is the median
  of left MAD, right MAD, and a Gaussian-calibrated difference estimate.
- Saturated or flat-topped support is amplitude-unreliable and not clean.
- Ar I requires three clean independent groups; O I requires two. O I 777 alone
  remains tentative.
- A blend requires a locally strong transition selected within the same ion
  and 100 nm band plus at least three distinct matched features outside the gas
  window. The target-window peak cannot corroborate its own competitor. This
  flags the plausible Fe II 645.638 nm/O I 645 nm ambiguity without allowing
  strong UV Fe lines to suppress the NIR competitor catalog.
- Without a peak table, interference cannot be checked and positive evidence
  remains tentative.

No fixed line ratio is required. The statuses do not infer gas concentration,
O2, purge origin, ambient origin, or sample origin. The detector covers neutral
Ar I/O I anchors only; their absence does not establish elemental absence when
other ion stages or timing regimes dominate.

## Verification

`tests/test_gas_detection.py` has 11 passing tests. They cover blank spectra,
Ar and O group thresholds, O I 777 grouping, missing peak tables, range and gap
guards, wavelength shifts, explicit and externally supported interference,
invalid arrays, a 0.18 nm native grid, and 40 seeded native-grid Gaussian null
spectra. The pooled noise estimate recovers the injected sigma within 15%, and
none of the null spectra returns `detected`.

A truth-assisted check of native mean
`run-e825d9f567204c5cbc6fc33c9c16bbf7`, using the recorded -0.154 nm shift and
a prominence-10 peak table, returns Ar I `tentative` (one supported 696 nm
group, blend-flagged) and O I `no_evidence`. A blind production-peak replay of
the first three Fe-session means returns `no_evidence` for Ar and O in all
three. These are calibration and nuisance checks on reused Fe-session data,
not independent gas-mixture validation. Threshold performance still needs
held-out positive and negative gas standards across instruments and gate times.
