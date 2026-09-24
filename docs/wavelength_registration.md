# Wavelength registration

`alibz/wavelength_registration.py` re-registers the wavelength axis of an
individual spectrum against lines whose air wavelengths are known exactly, so
observed peaks match the right database lines. It complements — and, per
segment, can replace — the legacy mixed-element anchor shift in
`alibz/utils/wavelength.py`.

## Why

The dominant database-mismatch term on native Z300 data was a **~18 px axis
offset** (every spectrum ~18 px too blue: UV −1.6 / VIS −2.3 / NIR −3.3 nm),
fixed on 2026-09-24 (`PIXEL_OFFSET −18`; `reports/2026-09-23-native-axis-18px-offset.md`).
After the fix the residual is only **≈ −0.25 px (tens of pm)** — per-calibration
and run-to-run variable, not a fixed offset — so registration is now a fine
correction on top of a correct axis, not a rescue of a broken one. The Z300's
pixel→nm calibration is done on a cold instrument (≈25 °C) and samples collected
as it warms; any thermal drift is bounded to ~1 pm/h (negligible). This module
measures and, per segment, applies the small remaining shift so observed peaks
match the right lines. See `reports/2026-09-24-wavelength-registration.md`.

## Convention

A *shift* is `observed − database` in nm (same convention as
`estimate_wavelength_shift`). To move an observed axis into the database frame,
subtract the shift: that is `apply_registration(x, registration)`.

## The three estimators

- **`ambient_registration(x, y, db, composition=None)`** — from Ar I / O I
  (and optionally N I, Hα) present via the argon purge and ambient air. These
  live almost entirely in the NIR segment. Output: per-segment shift (and slope
  if ≥4 lines), a per-line table, the species detected, and a quality flag.
  `composition` (or `config["composition"]`) lists the sample elements whose
  strong lines gate the anchors: it only *removes* anchors a sample line could
  be mistaken for, never supplies any. On the **corrected axis (2026-09-24)** the
  full Ar I NIR anchor set (696.5, 706.7, 738.4, 750.4, 763.5, 772.4, 794.8,
  801.5, 811.5, 826.5, 842.5, 912.3 nm) registers as genuine argon within
  ~±30 pm on both Fe and V 10-shot means (10–12 clean lines/run), so the gate
  rarely fires and the NIR segment passes its ≥3-line gate on ~100 % of runs.
  (On the offset-era 18-px-blue data the 763/794/811/842 nm anchors coincided
  with Fe II features 0.2–0.6 nm away and only 696.5/772.4/801.5 nm read as
  argon — that reading is withdrawn.)
- **`element_registration(x, y, db, composition)`** — DETERMINISTIC golden-line
  estimator. `golden_lines(db, composition, segment)` selects lines unambiguous
  *by construction* — a composition I/II line in the segment's top ~150 by
  predicted strength (`gA·exp(−E_k/9 kK)`, ion II ×0.5) with NO composition +
  Ar/N/O/H line within ±0.50 nm above 3 % of its strength (Ar I is the NIR
  golden set). Each is measured with `gaussian_subpixel_center` (nearest SNR ≥ 6
  peak within ±0.45 nm, blend rejected) and a robust (Huber) per-segment
  **offset** is fit; < 3 golden lines ⇒ the segment is `unregistered`.
- **`vote_mode_diagnostic(...)`** — the earlier statistical vote/window estimator,
  kept **diagnostic only**: it measures each element's alias pattern, not the
  instrument (a vote-score offset+slope gives Fe VIS +1.86 vs V VIS −1.49 pm/nm
  on the identical axis), so it must not drive the registration.
- **`combined_registration(ambient, element)`** — deployment rule: the golden
  offset where a segment registers; Ar as the NIR anchor; pooled fallback
  otherwise; a flag when the element and Ar NIR shifts differ by >2σ.

### The per-segment offset is a shared instrument quantity

On the corrected axis (2026-09-24) the per-segment **offset** transfers cleanly
between metals: Fe→V VIS transfer residual is **40 pm** (was 597 pm on the old
axis), VIS run-to-run SD is ≤ 12 pm (all segments now ≤ 30 pm), and the
within-segment VIS slope is ~0 and identical for Fe and V (was Fe −5.8 vs
V +1.5 pm/nm — an axis artefact). Only the offset (the Huber intercept) is
applied; the fitted slope stays a diagnostic because the **NIR** per-line
residual is still large for a pure metal (Fe NIR ~170 pm) — the strongest NIR
lines are optically thick / displaced, which the axis fix does not change.
UV registers only where ≥ 3 golden lines exist (Fe 3 / V 6 candidates).

## Sub-pixel centring

Line centres come from `alibz/utils/peakfit.py`
(`gaussian_subpixel_center`): a weighted single-Gaussian fit at the instrument
width (σ ≈ 0.6 px UV/VIS, 0.7 px NIR = FWHM/2.355) plus a local linear
baseline, returning the centre with a covariance 1σ, the fitted width, χ²/ν, an
SNR and a blend flag (with a two-Gaussian fallback for the nearest pair). It is
unbiased across the pixel phase and more accurate than parabolic interpolation
(SNR 20: ≈0.035 px vs ≈0.08 px). Selectable via the `subpixel` config
(`"gaussian"` default, `"parabolic"` legacy).

## The temperature correction — `thermal_drift_model(records)`

Given per-run registrations plus provenance (minutes since calibration,
temperature or None, ambient NIR shift), it fits per segment `Δλ = a + b·minutes`
(and vs temperature / vs the ambient NIR shift as a thermal proxy), reporting
the slope, its σ and the 2σ detection limit. **Fit within one calibration
epoch**: runs from different calibrations have different fixed offsets, so
pooling across epochs lets high-leverage old-calibration runs mask a real
within-epoch drift. When the data cannot resolve a drift, `predict_shift`
returns 0.0 and `resolved` is False, with the bound reported.

## Vendor calibration change — `vendor_calibration_shift(base, current, knots)`

Converts the handheld's stored pixel→nm cubics (`calibration.json` base vs
`currentwlcalibration.json` current) into a per-segment Δλ(λ), so the size of a
recalibration can be compared directly with the observed offsets.

## Pipeline integration

`analyze_spectrum` (and `AnalysisConfig`) gain `wavelength_registration`
(`off`/`report`/`ambient`/`apply_element`, **default `ambient`**) and `subpixel`
(`gaussian`/`parabolic`, default `gaussian`). The pipeline always computes and
records the ambient, golden-line element, combined and vote-mode-diagnostic
registrations under `analysis['wavelength_registration']` (element under
`['element']`, vote-mode under `['diagnostic']`). What it APPLIES depends on the
mode:

- **`ambient`** (default) — apply only the composition-independent ambient
  (Ar/O/N/H) registration to the **NIR** segment; keep the legacy anchor shift
  for UV/VIS. On the corrected axis this now applies a real ~−0.25 px NIR
  correction on ~100 % of runs (it was effectively a no-op on the offset-era
  one-line data). Element registration is recorded but not applied.
- **`apply_element`** — additionally replace a segment's shift with the element
  (golden) offset where that segment passes quality. On the corrected axis
  (2026-09-24) **element VIS/UV registration is now validated** (VIS run-to-run
  SD ≤ 12 pm, Fe→V VIS transfer 40 pm, flat shared slope; the offset-era
  "optically-thick displaced anchors" objection was an axis artefact). The
  pipeline default is kept at `ambient` rather than promoted to `apply_element`
  only because changing the inversion's wavelength input pipeline-wide should be
  gated on a dedicated inversion-accuracy benchmark, not on registration
  self-consistency; the residual is small (~14 pm), so deferring costs little.
- **`report`** records everything, applies nothing; **`off`** computes nothing.

**Relation to the gas-calibration engine.** `alibz.gas_calibration` (regional
Ar I / O I offsets from a native-maxima peak table, `gas_wavelength_calibration`,
default `apply`) runs in the same pipeline. Order: the ambient registration may
replace the NIR segment shift first; the gas engine's regional calibration then
overrides *inside its bracketed anchor support* when it reaches `calibrated`
(≥ 3 consistent Ar groups). Both are argon-based, so where both fire they must
agree: `['gas_cross_check']` records the difference (20 pm single-line floor)
and the summary QC adds `ar-registration-disagreement` beyond 3 σ. On the
corrected axis both argon gates now pass on ~100 % of 10-shot means (gas engine:
20/20 Fe, 87/87 V calibrated; ambient: 10–12 clean NIR lines/run), with the two
argon estimates agreeing to a few tens of pm (gas −88 vs ambient −50 pm on V —
watch the cross-check). CLI: `--wavelength-registration`, `--subpixel`,
`--gas-wavelength-calibration`.

`['applied_segments']` logs which segments were changed. `PeakyIndexerV3` accepts
the combined registration as a prior and reports its own anchor shift beside it
with a disagreement flag (`wavelength_registration_report()`); matching is
unchanged when none is supplied.

See `reports/2026-09-24-wavelength-registration.md` for the current
(corrected-axis) measurement study and the verdict on the warm-up-drift
hypothesis (2026-09-23 is superseded); regenerate it with
`scripts/registration_study.py`.
