# Wavelength registration

`alibz/wavelength_registration.py` re-registers the wavelength axis of an
individual spectrum against lines whose air wavelengths are known exactly, so
observed peaks match the right database lines. It complements — and, per
segment, can replace — the legacy mixed-element anchor shift in
`alibz/utils/wavelength.py`.

## Why

The Z300's pixel→nm calibration is done on a cold instrument (≈25 °C, right
after power-up); samples are collected minutes later as the spectrometer warms
toward 35–40 °C. Thermal expansion of the grating/detector moves lines by a
*wavelength-dependent* amount (a scale change, not a constant offset). The
legacy estimator captures only one median offset per detector segment. Left
uncorrected, the mismatch reaches ±0.3 nm and silently mis-assigns lines.

## Convention

A *shift* is `observed − database` in nm (same convention as
`estimate_wavelength_shift`). To move an observed axis into the database frame,
subtract the shift: that is `apply_registration(x, registration)`.

## The three estimators

- **`ambient_registration(x, y, db)`** — composition-independent, from Ar I / O I
  (and optionally N I, Hα) present via the argon purge and ambient air. These
  live almost entirely in the NIR segment. Output: per-segment shift (and slope
  if ≥4 lines), a per-line table, the species detected, and a quality flag.
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

### Offset is the instrument; the within-segment slope is database error

Only the per-segment **offset** is a shared instrument quantity: Fe and V VIS
offsets agree within 17 pm and V VIS is reproducible to 24 pm run-to-run. The
within-segment slope does **not** transfer between elements (Fe VIS −5.8 vs
V VIS +1.5 pm/nm; a Fe-fit Δλ(λ) applied to V leaves 597 pm), so it is dominated
by per-element database wavelength errors and is recorded as a diagnostic, never
applied. Golden-line registration therefore deploys an offset per segment; a
true wavelength-dependent instrument function would need a lamp or a
multi-element standard, not single-element metal forests (per-line db-error
floor ~150–240 pm).

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
  for UV/VIS. Element registration is recorded but not applied.
- **`apply_element`** — additionally replace a segment's shift with the element
  (golden) offset where that segment passes quality. **Element UV/VIS
  registration is diagnostic until validated against a line lamp or a certified
  reference: the strongest lines of a pure-metal plasma are optically thick and
  displaced, so they are unsuitable as wavelength anchors** (Fe I 438.35 has no
  maximum at its position; the V I 437.92/438.47/439.00 triplet collapses to a
  single ~438.31 peak; the strongest-60 match fraction to each metal's own
  top-300 list is only 50–62 %).
- **`report`** records everything, applies nothing; **`off`** computes nothing.

`['applied_segments']` logs which segments were changed. `PeakyIndexerV3` accepts
the combined registration as a prior and reports its own anchor shift beside it
with a disagreement flag (`wavelength_registration_report()`); matching is
unchanged when none is supplied.

See `reports/2026-09-23-wavelength-registration.md` for the measurement study
and the verdict on the warm-up-drift hypothesis; regenerate it with
`scripts/registration_study.py`.
