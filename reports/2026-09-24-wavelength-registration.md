# Wavelength registration on the CORRECTED (−18 px) axis — 2026-09-24

Specialist (Opus 4.8). Provider: Claude subscription (no switch). Step 3c of the
Z300 axis-offset fix. Supersedes `reports/2026-09-23-wavelength-registration.md`
(and the offset-era gas reports), every number of which was measured on the
native axis that was ~18 px too blue (`reports/2026-09-23-native-axis-18px-offset.md`).

## What changed and why re-run

Before 2026-09-24 every native/API spectrum was labelled ~18 px too blue
(UV ≈ −1.6, VIS ≈ −2.3, NIR ≈ −3.3 nm). The fix (`PIXEL_OFFSET −18`) was
deployed and every run re-derived in place on 2026-09-24; each manifest now
carries `axis_check` (Ar I NIR cross-correlation) ≈ −0.25 px. The registration
and gas engines, the anchor sets and the study were all built on the wrong
axis, so their conclusions ("only 696.5/772.4/801.5 nm are argon", "−0.18 nm
NIR", "warm-up drift unsupported", the within-segment slope, the Fe→V
non-transfer) had to be re-measured.

## Data

`scratchpad/corrected/`: 113 runs; **107 analysed** (20 Fe, 87 V; 6 carry no
declared composition and are skipped). 10-shot means (8–10 shots), 5,848-sample
native grid 187.8–950.8 nm, three segments (UV ≤365 / VIS 365–620 / NIR ≥620),
NIR pitch ≈ 0.179 nm/px. Element from the ledger; calibration epoch, `axis_check`
and calibration cubics from each run's `manifest.json`. Regenerate with
`scripts/registration_study.py` (now single-ledger + per-run manifest);
JSON `reports/registration-study-20260924.json`, figures
`reports/figures/registration-20260924/`.

**Three calibration epochs** (from the manifests): `2026-09-22T09:21:50`
(15 Fe, "pre-recal"), `2026-09-22T13:39:21` (5 Fe + 87 V = 92, "post-recal"),
`2026-09-21T14:55:07` (1 run, no composition, skipped). Warm-up minutes =
`created_at`(UTC) − (`wl_calibration_time`(analyzer-local) + 3 h); the analyzer
clock is mis-set ~+3 h, so the absolute zero is uncertain but the drift slope is
offset-invariant. `analyzer_temperature_c` is still null (not logged).

## Method

Unchanged engines (`alibz/wavelength_registration.py`), shift = observed −
database: `ambient_registration` (Ar I/O I/N I/Hα anchor offsets, composition
gate removes anchors only), `element_registration` (deterministic golden-line
Huber offset per segment; slope diagnostic only), `combined_registration` (Ar
anchors NIR), `thermal_drift_model` (fit within one epoch), `vendor_calibration_shift`.
Gas: `detect_background_gases` + `calibrate_background_gases` on the raw axis
with a native-maxima peak table.

## Results

### Does a residual ~−0.25 px remain? Is it constant?

**Yes, ~−0.25 px remains, and it is NOT a single constant** — it is
per-calibration and varies run-to-run. Two independent measurements agree:

| measure | median | spread |
|---|---|---|
| manifest `axis_check.offset_px` (Ar NIR x-corr) | **−0.25 px** | −0.50 … +0.25 (n=107) |
| study ambient Ar I NIR shift, converted to px | **−0.26 px** | MAD 0.07 px |

Broken out (ambient Ar I NIR, pm; 1 px ≈ 179 pm NIR):

| group | ambient NIR (pm) | ≈ px | manifest axis_check (px) |
|---|---|---|---|
| Fe / 09:21 (pre-recal) | −28 (MAD 13) | −0.16 | median 0.00 |
| Fe / 13:39 (post-recal) | +4 (MAD 5) | +0.02 | (within 13:39) |
| V / 13:39 | −50 (MAD 12) | −0.28 | median −0.25 |

So the residual clusters at ≈ −0.25 px for the dominant 13:39/V set, is ≈ 0 for
the 09:21 epoch, and swings −0.5 … +0.25 px run-to-run. It is a **per-run /
per-calibration residual centred near −0.25 px**, not a fixed offset — i.e. the
last ~quarter-pixel of registration is worth applying but is not a further
constant to bake into the axis.

### Residual element shift per segment (median pm)

| group | UV | VIS | NIR |
|---|---|---|---|
| Fe / 09:21 | — (n<3) | −29 | −53 |
| Fe / 13:39 | −13 | −12 | −26 |
| V / 13:39 | — (n<3) | −49 | −82 |

All are small (tens of pm). UV registers only when ≥3 golden lines exist (Fe 3 /
V 6 candidates), so most 10-shot means leave UV `unregistered`.

### Wavelength dependence (within-segment slope, golden lines)

| segment | Fe (pm/nm) | V (pm/nm) | old axis |
|---|---|---|---|
| VIS | +0.1 (resid 14) | +0.0 (resid 19) | Fe −5.8, V +1.5 |
| NIR | +1.1 (resid 169) | −0.1 (resid 64) | Fe −0.4, V +1.0 |
| UV | −0.8 (resid 15) | — | Fe +3.2 |

**The VIS slope is now ~0 and identical for both metals** — the offset-era
opposite-sign slopes (Fe −5.8 vs V +1.5) were an axis artefact, not per-element
database error. NIR still carries a large per-line residual for Fe (169 pm):
the strongest pure-metal NIR lines are optically thick / displaced, which the
axis fix does not change.

### Pre vs post the 13:39 recalibration (Fe)

| segment | pre 09:21 elem | post 13:39 elem | Δ | ambient NIR pre→post |
|---|---|---|---|---|
| VIS | −29 | −12 | +17 | — |
| NIR | −53 | −26 | +26 | −28 → +4 (+32) |

The 13:39 recalibration moved the residual toward zero by ~17–32 pm (a fraction
of a pixel), consistent across the element and ambient estimators.

### Drift vs minutes since calibration — thermal-drift verdict

Fit within one (element, epoch) group:

| group | segment | slope (pm/h) | 2σ limit | resolved | n |
|---|---|---|---|---|---|
| **V / 13:39** (396–1728 min) | VIS | −1.0 | 0.2 | yes | 87 |
| | NIR | −0.7 | 0.3 | yes | 87 |
| Fe / 09:21 (250–437 min) | VIS | −6.4 | 2.2 | yes | 15 |
| | NIR | −8.9 | 30.4 | no | 15 |

**The warm-up-temperature hypothesis is again not supported as a significant,
let alone dominant, cause.** In the best-sampled group (V/13:39, 87 runs over
22 h) the drift is statistically resolved but physically tiny — ~1 pm/h, i.e.
~30 pm over the whole 22 h span. Fe/09:21 shows −6 to −9 pm/h. All are small
against the fixed per-segment offsets and the vendor recalibration step. The
corrected axis tightens the bound relative to 2026-09-23 (the run-to-run scatter
that hid the slope is gone). Analyzer temperature is still not logged, so the
direct temperature regression cannot be run.

### Run-to-run reproducibility (element offset SD; target ≤ 30 pm)

| group | VIS SD | NIR SD |
|---|---|---|
| Fe / 09:21 | 8 | 26 |
| Fe / 13:39 | 3 | 13 |
| V / 13:39 | 12 | 16 |

**Every segment now meets the ≤ 30 pm target** (offset-era: only V VIS at 24 pm;
the rest 60–83 pm).

### Fe→V transfer: the instrument function is now shared

Fit Δλ(λ) from Fe golden lines, apply to V golden lines; report the V residual:

| segment | V residual median (pm) | old axis |
|---|---|---|
| UV | 11 | 533 |
| VIS | **40** | 597 |
| NIR | 69 | 256 |

The offset-era "the slope does not transfer between elements; it is per-element
database error" conclusion is **withdrawn**: on the corrected axis Fe→V VIS
transfers to 40 pm (near the run-to-run floor), so the per-segment correction is
a genuine shared instrument quantity, not a database artefact.

### Post-registration residual

| element | median |res| (pm) | rms (pm) | old axis median |
|---|---|---|---|---|
| Fe | 14 | 122 (NIR opacity outliers) | 185 |
| V | 14 | 45 | 141 |

The per-line residual collapses from ~185 pm to ~14 pm: almost all of it was the
axis error. The Fe rms tail is the optically thick NIR lines.

### Gas detection / calibration (corrected axis)

| element | Ar detected | Ar calibrated | Ar offset (pm) | O detected | O calibrated |
|---|---|---|---|---|---|
| Fe (20) | 20/20 | 20/20 | −24 (MAD 36) | 6/20 | 0 |
| V (87) | 87/87 | 87/87 | −88 (MAD 0) | 72/87 | 0 |

- **Ar I now detects and calibrates on every run** with the full NIR anchor set
  (offset-era gas engine: 0/26 calibrated, one usable line). Ar is expected
  (argon purge), so these are true positives, not false. The gas-engine V Ar
  offset (−88 pm) sits ~38 pm from the ambient estimate (−50 pm); both are small
  and the competitor gate is active, but the two argon engines are not identical
  and the cross-check flag (`gas_cross_check`) should be watched.
- **O I is now detected** (V 72/87, Fe 6/20) where the offset-era study found
  none — the corrected axis lands the O I 777/844 nm windows on real features.
  O never reaches the `calibrated` gate (it needs two resolved 777-triplet
  components at a common offset), so it is recorded as evidence only. No
  independent O truth is available, so "false positive" cannot be scored; the
  competitor gate did not flag them and the lines are physically plausible
  (ambient air / oxide), so they are reported as tentative O evidence, not a
  validated detection.
- No spurious Ar/O firing was attributable to the Fe/V forests (the competitor
  interference gate removes supported blends).

The vendor base→current cubic change (unchanged, same cubic pair): UV +70,
VIS −94, NIR −223 pm — this is the 13:39 recalibration's own scale move, larger
than any warm-up drift. Sub-pixel comparison (Gaussian −12 % rms vs parabolic)
is axis-independent and carries over from 2026-09-23 (`--no-subpixel` here).

## Engine audit — offset-era artefacts found and fixed

The engine *logic* is composition- and database-driven and proved robust to the
axis correction; the offset-era assumptions lived in **docstrings/comments and
the docs**, not in tolerances, seeds or gates (the unit tests are synthetic and
axis-independent — 71 pass unchanged):

- `alibz/wavelength_registration.py:82` (`composition` default doc) and `:556`
  (`ambient_registration` docstring): removed the "only 696.5/772.4/801.5 nm are
  argon; 763/794/811/842 nm are Fe II" claim — on the corrected axis all Ar I
  NIR anchors register as genuine argon within ~±30 pm (measured: 696.5, 706.7,
  738.4, 750.4, 763.5, 772.4, 794.8, 801.5, 811.5, 826.5, 842.5, 912.3 nm all
  clean). The composition gate is retained as a conservative safeguard (it now
  rarely fires).
- `alibz/wavelength_registration.py:916` (`element_registration` offset comment):
  updated — the within-segment VIS slope now transfers (Fe→V 40 pm, flat shared
  slope); the offset-only deployment stays because the NIR slope/residual is
  still dominated by optically thick strong lines.
- `scripts/registration_study.py`: rewritten data plumbing for the corrected
  single-ledger + per-run-manifest layout (calibration epoch, `axis_check`,
  warm-up from the manifest); added per-group drift, pre/post-recal, residual-px
  and gas-detection/calibration summaries.
- No offset-era constant, tolerance, seed or anchor list needed changing; the
  anchor search intervals (`_AR_TARGETS`, `_AR_GROUPS`, `_O_GROUPS`) are
  resolved against the database and were already generous enough.

## Pipeline default registration — decision

**Keep the default `ambient`, but note it is now materially active.** Rationale:

1. On the corrected axis the ambient Ar I NIR registration passes its ≥3-line
   gate on ~100 % of runs (≥10 clean anchors), so `ambient` now applies a real,
   small NIR correction (~−0.25 px) instead of the offset-era no-op.
2. **Element VIS/UV registration is now validated** (VIS run-to-run SD ≤ 12 pm,
   Fe→V VIS transfer 40 pm, flat shared slope; UV where ≥3 golden lines) — the
   offset-era reasons to withhold it (optically-thick displaced lines, 597 pm
   transfer, opposite slopes) were axis artefacts. So `apply_element` is now
   well-justified for VIS.
3. I did **not** promote `apply_element` to the default: it changes the
   inversion's wavelength input pipeline-wide, a high-blast-radius change that
   should be gated on a dedicated inversion-accuracy benchmark, not on
   registration self-consistency. The residual is small (~14 pm), so deferring
   costs little. This is recorded for a follow-up benchmark decision.

## Old (offset-era) vs corrected — headline numbers and verdicts

| quantity | 2026-09-23 (18 px too blue) | 2026-09-24 (corrected) |
|---|---|---|
| Native/API axis error | ~18 px (UV −1.6/VIS −2.3/NIR −3.3 nm) | fixed; residual ≈ −0.25 px (tens of pm) |
| Residual constant? | (n/a — dominated by 18 px) | per-run/per-cal, centred −0.25 px, −0.5…+0.25 |
| Ar I NIR anchors valid | only 696.5/772.4/801.5; 763/794/811/842 = Fe II | **all** anchors register as argon (±30 pm) |
| Ambient NIR shift | −0.19 nm, effectively **one line** (696.5) | −0.03…−0.05 nm, **10–12 clean lines/run** |
| VIS within-seg slope | Fe −5.8 / V +1.5 pm/nm (opposite) | Fe +0.1 / V +0.0 (flat, shared) |
| Fe→V VIS transfer | 597 pm (does not transfer) | **40 pm (transfers)** |
| Run-to-run SD (VIS/NIR) | 60–83 pm (V VIS 24) | 3–12 / 13–26 pm (all ≤ 30) |
| Post-registration residual | Fe 185 / V 141 pm | **Fe 14 / V 14 pm** |
| Warm-up drift verdict | not supported (NIR ±50 pm bound) | not supported; ~1 pm/h resolved, negligible |
| Gas Ar calibration | 0/26 runs (1 usable line) | **20/20 Fe, 87/87 V** |
| Gas O detection | 0 runs (purge) | 6/20 Fe, 72/87 V (tentative, 0 calibrated) |
| Pipeline default | `ambient` (a no-op on this data) | `ambient` (now applies real NIR correction) |
| Element VIS/UV apply | withheld (unvalidated) | validated; `apply_element` justified, default deferred to a benchmark |

## Limits and unverified

- Analyzer/spectrometer temperature is still not logged; the direct
  temperature-vs-shift regression cannot be run (only warm-up minutes).
- Absolute warm-up zero rests on the assumed +3 h analyzer clock offset; slope
  conclusions are offset-invariant, absolute intercepts are not.
- The two argon engines disagree by ~38 pm on V (gas −88 vs ambient −50 pm);
  both small, cause not chased here (`gas_cross_check` records it).
- O I detections are recorded as tentative evidence; no independent O truth
  exists, so accuracy / false-positive rate is unscored.
- NIR per-line residual (Fe ~170 pm) is real optical-thickness/displacement of
  strong pure-metal lines, unaffected by the axis fix; only the offset is used.
- `benchmark_gas_calibration.py` was NOT re-run: its pinned archive
  (`provenance/physical-triage-real-data-20260922.npz`) is offset-era data; its
  semi-synthetic recovery is axis-independent and still valid. Corrected-axis gas
  validation is folded into this study instead (table above).
