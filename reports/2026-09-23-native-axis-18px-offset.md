# ROOT CAUSE: native Z300 wavelength axis is offset by ~18 detector pixels — 2026-09-23

## Finding (coordinator-verified)
Pantheum builds every native/API spectrum with `pixels_to_wavelength`
(pantheum-I pantheum/alibz/z300_calibration.py:174-180). It evaluates the per-segment pixToNm
cubic at the raw 0-based stored-sample index `p`. The polynomial is defined against the
physical detector column, and the stored 2,066-sample array starts 18 columns later. This was
already solved and documented for the FlatBuffers path: scripts/z300_fb_decode.py:29-37, 72, 156
(`PIXEL_OFFSET = -18.0`, solved against the vendor export: r 0.05 → 0.91–0.98, "shifts lines by
up to ~3.5 nm, which silently mis-assigns elements"). The API path never applied it (since
pantheum-I 6de6524). Result: every line in native/API data is labelled ~18 px too BLUE
(UV ≈ −1.6 nm, VIS ≈ −2.3 nm, NIR ≈ −3.3 nm).

## Evidence (V_pure_run2 median of 200 native shots)
- Cross-correlation of the observed high-passed spectrum with a predicted V I/II + Ar I stick
  spectrum vs pixel offset: best UV −18.50 px (r 0.58 vs 0.13 at 0, next-best >3 px away 0.21),
  VIS −18.25 (0.68 vs 0.13 / 0.21), NIR −18.50 (0.25 vs 0.07 / 0.14).
- Ar I NIR: the strongest peaks at 808.30/807.05/798.14/797.24/791.51 nm are Ar I
  811.53/810.37/801.48/800.62/794.82 shifted −3.24…−3.37 nm.
- V I 437.92/438.47/439.00 at −18 px: 2239/1833/1358 counts, the correct order; at 0 px
  491/1142/453. V I 411.18: 2681 at −18 px vs 260 at 0.
- Independent discovery by the ePSF build (judgment agent): unique offset −18.0…−18.65 px in all
  86 runs and all 3 segments (cal/reports/2026-09-23-empirical-lsf.md).
The residual beyond −18.0 (−0.25…−0.5 px) is the genuine calibration/registration shift.

## What this invalidates (anything that assigned line identities on API-path data)
- Pantheum window-quality scores (reference windows land ~2 nm from the lines) and the
  delay/period "best" choices that rest on them.
- The "−154 pm native shift", the grid-family shifts in docs/z300_device_notes.md, the Ar I NIR
  anchor conclusions (memory ar-nir-registration-anchors), the wavelength-registration study and
  engines (alibz/wavelength_registration.py, gas_detection/gas_calibration, commit 88cd8fb).
- cal/ studies: Fe plasma T and line lists; line-matching/self-absorption (V 417.94/415.97 = 0.75
  "self-absorption" → 2.10 [1.99, 2.18] ≈ 2.05 thin once correctly identified); V timing Q1
  ion/neutral identities and Q3. The pixel-level findings are NOT affected: period = exposure,
  the period-scaled fixed pattern, flat line signal vs period, dead channel, per-segment pitch,
  and the undersampling/estimator results.
- The old 23,250-point Pantheum files: same map, same error. Vendor Profile Builder exports and
  FlatBuffers-decoded data carry the offset correctly.

## Fix (not applied; production change + user-run deploy)
`wl = _polyval(coeffs, p + PIXEL_OFFSET)` with PIXEL_OFFSET = −18 (one shared constant with
z300_fb_decode). Then re-derive every stored native dataset (raw/*.json keeps pixels +
calibration, so no refetch is needed), re-score the sessions, and re-run the cal analyses.
Add a regression test: synthetic Ar I pattern → correct wavelengths; plus a live guard that
cross-correlates Ar I NIR and rejects |offset| > 2 px.
