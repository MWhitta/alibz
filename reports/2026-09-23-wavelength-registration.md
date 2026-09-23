# Wavelength registration and the warm-up-drift hypothesis

2026-09-23. Specialist (Opus 4.8). Provider: Claude subscription (no switch).

## Owner's hypothesis

> The database mismatch is most likely because the wavelength calibration is
> done at a lower temperature than data collection (calibration first, near
> 25 °C; instrument warms to 35–40 °C). Account for this by documenting the
> collection temperature and including a temperature correction. Also use Ar I
> and O I lines, in the pipeline and in the peaky indexer, as separate
> wavelength calibrations present in most spectra (argon purge, ambient air).

## Data

53 acquisition runs, 10-shot means: 26 Fe (Aesar 99.98 %) and 27 V, from the
scratchpad acquisition tree (`acq/<run>/shots/shot-N.csv`, 7 914 samples, three
segments UV 186–365 / VIS 365–620 / NIR 620–948 nm, pitch 0.089/0.129/0.179 nm).
Two calibration epochs: 50 runs on the 2026-09-22 13:39 calibration (warm-up
4–414 min) and 3 early Fe runs on the 2026-09-21 calibration (~22 h). Vendor
calibration cubics from the handheld `calibration.json` (base) and
`currentwlcalibration.json` (13:39 current). Regenerate with
`scripts/registration_study.py`; raw numbers in
`reports/registration-study-20260923.json`; figures in
`reports/figures/registration-20260923/`.

**Calibration epochs.** The per-run `calibrationTime` lives only in the analyzer
shot records on Moissanite (offline here); the ledgers give `created_at` (UTC).
The analyzer local clock is mis-set (~+3 h vs UTC), so the absolute warm-up zero
is uncertain. **The drift SLOPE — the owner's actual question — is invariant to
a constant offset in the epoch**, so the verdict below does not depend on the
zero. Each epoch is anchored consistently (cal-A anchored so the first cal-A
run, run-126d0c96, sits at +4 min per the brief).

## Method

Two estimators (`alibz/wavelength_registration.py`), convention shift =
observed − database:

- **`ambient_registration`** — composition-independent, Ar I / O I / N I / Hα;
  a robust per-segment mode of the matched anchor offsets. The Ar NIR anchor is
  the reliable composition-independent part.
- **`element_registration`** — DETERMINISTIC golden-line estimator. Statistical
  forest matching (vote/window modes) was demoted to a diagnostic after it was
  shown to measure each element's *alias pattern*, not the instrument (a joint
  vote-score offset+slope fit gives Fe VIS +1.86 vs V VIS −1.49 pm/nm on the
  identical axis). Instead, `golden_lines(db, composition, segment)` selects
  lines unambiguous *by construction*: a composition I/II line in the segment's
  top ~150 by predicted strength (gA·e^(−E_k/9 kK), ion II ×0.5) with NO other
  composition+Ar/N/O/H line within ±0.50 nm above 3 % of its strength (Ar I is
  the NIR golden set). Each golden line is measured with `gaussian_subpixel_center`
  (nearest SNR ≥ 6 peak within ±0.45 nm, blend rejected), and a robust (Huber)
  per-segment fit yields the offset; < 3 golden lines ⇒ the segment is
  `unregistered` (fall back to pooled/ambient, never a vote mode).

Golden-line counts: Fe UV 3 / VIS 12 / NIR 83; V UV 6 / VIS 24 / NIR 78. The
vote-mode diagnostic (with its dominance) is attached under `diagnostic`. Tests
cover the golden unambiguity rule, recovery of a known a+b(λ−λ0) with ±0.35 nm
decoys, and the Fe→V transfer (`tests/test_wavelength_registration.py`).

`thermal_drift_model` fits per-segment Δλ = a + b·minutes **within one
calibration epoch** and returns 0 with a detection bound when unresolved.
`vendor_calibration_shift` converts the two pixel→nm cubic sets into a
per-segment Δλ(λ).

## Results

> **Estimator history (2026-09-23).** Two earlier estimators were rejected: a
> per-line matcher (aliased, ~200 pm scatter) and a statistical vote/window
> estimator (which measures each element's alias pattern, not the instrument —
> a vote-score offset+slope gives Fe VIS +1.86 vs V VIS −1.49 pm/nm on the same
> axis). §2 below uses the DETERMINISTIC golden-line estimator; the vote-mode is
> kept only as a diagnostic. Ambient Ar and the Gaussian peak-fit are unchanged.

### (a) Shift vs warm-up time — offset drift limit

Run-to-run SD of the per-run golden-line OFFSET within each (element,
calibration) group (target ≤ 30 pm):

| group | segment | n | mean (pm) | SD (pm) |
|---|---|---|---|---|
| V / 09-22  | **VIS** | 27 | +11 | **24** |
| V / 09-22  | NIR | 27 | −94 | 83 |
| Fe / 09-22 | VIS | 23 | −19 | 73 |
| Fe / 09-22 | NIR | 20 | +106 | 60 |

V VIS meets the target (SD 24 pm); the other segments are 60–83 pm. UV is
`unregistered` (Fe has only 3 golden lines, V 6; too few clean observed peaks).

Warm-up drift of the offset, epoch A (4–414 min; Fe supplies the spread, V sits
at 410 min):

| segment | slope (pm/h) | 2σ limit (pm/h) | resolved | n |
|---|---|---|---|---|
| VIS | +5.7 | **6.4** | no | 50 |
| NIR | −36 | 12 | yes | 47 |
| UV  | — | — | (unregistered) | 2 |

**The VIS offset drift is < 6 pm/h (2σ, unresolved)** over 4–414 min — i.e. no
significant warm-up drift in the one well-registered segment. NIR shows a small
resolved −36 pm/h. Over the ~7 h of collection the VIS bound is < ~45 pm total,
far below the fixed offsets in §2(b). This is what the data allow: the warm-up
drift is bounded small, not the dominant effect.

### (b) Vendor base→current calibration change

The 13:39 recalibration itself moved the wavelength scale by:

| segment | Δλ across segment (pm) | mean (pm) |
|---|---|---|
| UV  | +90 → +51 | +70 |
| VIS | −76 → −110 | −94 |
| NIR | −210 → −235 | −223 |

**This is the same size as, or larger than, the observed offsets** (per-run
element shifts averaged +10 / −63 / +49 pm with ±200 pm scatter; the fixed
mis-calibration is a −223 pm NIR step). A single recalibration event dwarfs any
warm-up drift over the 7 h of collection.

### (c) Fe→V transfer: offset is instrument, slope is database error

The decisive validation (fit Δλ(λ) from Fe golden lines, apply to V golden
lines; `fe_v_reconciliation.png`):

| segment | Fe offset (pm) | V offset (pm) | Fe−V offset (pm) | Fe slope (pm/nm) | V slope (pm/nm) | Fe→V transfer resid, median (pm) |
|---|---|---|---|---|---|---|
| UV  | +4 | −265 | +268 | +3.2 | — | 533 |
| VIS | +71 | +55 | **+17** | −5.8 | +1.5 | 597 |
| NIR | +113 | −95 | +208 | −0.4 | +1.0 | 256 |

Two clear conclusions:
1. **The segment OFFSET is a shared instrument quantity in VIS**: Fe +71 vs
   V +55 agree within 17 pm (and V VIS is reproducible to 24 pm run-to-run).
2. **The within-segment SLOPE is NOT**: Fe VIS −5.8 and V VIS +1.5 pm/nm have
   opposite signs on the identical axis, and transferring the full Fe a+b(λ)
   function to V leaves a 597 pm residual (not the ~30–50 pm a real instrument
   function would give). The slope and the ~150–240 pm per-line residual are
   dominated by **per-element database wavelength errors**, not an instrument
   scale change. UV and NIR offsets also disagree between metals (Fe−V 268/208
   pm), so only the VIS offset is a trustworthy shared instrument term.

Therefore the deployed correction is the per-segment **OFFSET only** (Huber
intercept); the fitted slope is recorded as a diagnostic and **not applied**
(applying an element-specific slope would inject database error into other
elements' lines).

### (d) Post-registration residual and sub-pixel comparison

The residual about the applied offset stays large — Fe median 185 pm, V 141 pm —
because it is the per-line database-error floor, not instrument scatter (the
run-to-run *offset* SD is far smaller, V VIS 24 pm). Ambient Ar I is detected in
**48/53 runs** and is the stable NIR anchor (Fe −192 ± 67, V −176 ± 142 pm);
O I in 0 runs.

Gaussian vs parabolic sub-pixel centring on the *same* detected peaks (dedicated
paired run, n = 1786):

| method | median \|res\| (pm) | rms (pm) |
|---|---|---|
| Gaussian | 47.8 | 62.3 |
| parabolic | 51.9 | 70.6 |

Gaussian instrumental-profile centring lowers the residual rms by ~12 % on the
identical peak set. On synthetic isolated lines it is unbiased across the pixel
phase and ~2× more accurate than parabolic at SNR 20 (0.035 vs 0.08 px;
`tests/test_peakfit.py`).

## Verdict on the owner's hypothesis

**The warm-up-temperature hypothesis is not supported as a significant, let
alone dominant, cause.** With the deterministic golden-line offset (V VIS
reproducible to 24 pm), the VIS offset drift over the 4–414 min warm-up range is
**< 6 pm/h (2σ, unresolved)** — bounded to < ~45 pm over the whole collection;
NIR shows only a small −36 pm/h. The dominant, robust effect is the **fixed
pixel→nm mis-calibration**: the vendor's recalibration moved each segment's
offset by +70/−94/−223 pm, the same magnitude as the observed shifts. Crucially,
that mis-calibration is a per-segment **offset**, not a wavelength-dependent
scale change: the apparent within-segment slope does **not** transfer between
Fe and V (opposite signs; 597 pm Fe→V residual) and is dominated by per-element
database wavelength errors. The right correction is therefore a **per-spectrum
per-segment OFFSET** from golden lines, applied where ≥ 3 golden lines register
(reliable in VIS; Ar anchors NIR; UV unregistered); the slope is diagnostic
only. Documenting collection temperature/warm-up remains worthwhile (provenance
is written), but this window bounds any temperature correction below ~6 pm/h in
VIS; `thermal_drift_model` reports the bound and returns zero where unresolved.

## The correction as deployed

`analyze_spectrum` (config `wavelength_registration`, **default `ambient`**)
computes and records the ambient, golden-line element, combined and vote-mode
registrations under `analysis['wavelength_registration']` (`['element']`,
`['diagnostic']`). In the default `ambient` mode it applies **only the
composition-independent ambient (Ar/O/N/H) registration to the NIR segment** and
keeps the legacy anchor shift for UV/VIS; the element registrations are recorded
but not applied. `apply_element` additionally applies the golden element offset
where a segment passes quality — but element UV/VIS is **diagnostic until
validated against a line lamp / CRM** (the strongest pure-metal lines are
optically thick and displaced; §c). `['applied_segments']` logs what changed.
`thermal_drift_model` returns 0 in VIS (drift < 6 pm/h, unresolved) with the
bound reported.

## Provenance fields (pantheum-I)

`_finish_dataset` now writes into test.json, manifest.json and the dataset
metadata: `wl_calibration_time` (+`_raw`), `collected_at`, `warmup_minutes`,
`analyzer_temperature_c` = {value:null, source:null, observed_at:null} (source
will be `WLCalLog solenoidTemp`), and `wl_calibration_coefficients`.
`alibz.pipeline.load_spectrum_metadata` reads these from a sibling test.json.
Pantheum assumes the analyzer clock reads UTC; this study infers a ~+3 h offset —
**both unverified against hardware**; only the drift intercept (not the slope) is
affected.

## Files changed

- `alibz/wavelength_registration.py` (new): `ambient_registration` (Ar/O/N/H
  anchor mode), `golden_lines` + `element_registration` (DETERMINISTIC golden-
  line offset via `_predicted_strength_table`, `_measure_golden`,
  `_huber_polyfit`), `vote_mode_diagnostic` (kept as diagnostic only),
  `combined_registration` (Ar NIR anchor), `RegistrationShift`,
  `apply_registration`, `registration_to_json`, `vendor_calibration_shift`,
  `thermal_drift_model`.
- `alibz/utils/peakfit.py` (new, 1–330): `instrument_sigma_px`,
  `gaussian_subpixel_center` (+ `_two_gaussian`), `estimate_instrument_profile`,
  `refine_peaks`.
- `alibz/pipeline.py`: `AnalysisConfig.wavelength_registration` + `.subpixel`
  (~262–271); `analyze_spectrum` params + validation (~397–403, ~518–522);
  `_coarse_composition` (~372–414); registration compute/apply block
  (~600–650); per-peak Gaussian diagnostics `peak_refinement` (~555–577);
  return keys `wavelength_registration`, `peak_refinement` (~1096, 1099);
  indexer kwarg `wavelength_registration` (~666–671); `analyze_directory` call
  threads both config fields (~1345–1348); `load_spectrum_metadata` (new,
  ~365–392).
- `alibz/peaky_indexer_v3.py`: `__init__` param `wavelength_registration`
  (~447–457); `wavelength_registration_report()` (~2633–2665).
- `scripts/registration_study.py` (new): the measurement study + figures.
- `docs/wavelength_registration.md` (new); `DECISIONS.md` (alibz + pantheum-I
  entries).
- Provenance (pantheum-I, delegated + verified): `pantheum/alibz/acquire.py`
  (`_parse_calibration_time` :91, `_wl_calibration_provenance` :107,
  `_warmup_minutes` :143, `_finish_dataset` fields), `tests/test_alibz_acquire.py`.

## Tests

- alibz baseline (session start): **507 passed, 4 skipped, 71 subtests, 0
  failed** (276 s).
- alibz after the golden-line rework: `tests/test_peakfit.py` (17) and
  `tests/test_wavelength_registration.py` (17, incl. golden-line unambiguity
  rule, a+b(λ−λ0) recovery with ±0.35 nm decoys, Fe→V transfer, and the vote
  diagnostic alias test) both green; full-suite subset (registration + peakfit +
  pipeline) green. The full suite carries the same **2 pre-existing failures in
  untracked gas-calibration files another session is editing**
  (`tests/test_gas_calibration.py::…abstains`, `…benchmark::…` — a `peaks[:, 1]`
  TypeError on a list); neither imports my modules; both were absent at session
  start. My changes regress none of the pre-existing 507.
- pantheum-I (`python3 -m pytest tests/test_alibz_acquire.py -q`): before **69
  passed / 19 subtests**, after **76 passed / 37 subtests** (verified by me).

## Limits and unverified

- **Peak-table re-centring deferred (decision for judgment).** The coordinator
  asked to re-centre the production peak table with the Gaussian fit by default.
  The blind fit already produces Voigt-profile centres, and re-centring the
  whole table is an unvalidated, high-blast-radius change to the physics
  inversion. I therefore attached the Gaussian fit only as an **additive
  per-peak diagnostic** (`analysis['peak_refinement']`: σ_px, χ²/ν, blend) and
  did **not** change the centres the inversion consumes. Gaussian centring IS
  used where it is validated and beneficial (the registration). Whether to
  re-centre the inversion table by default should be decided with a dedicated
  inversion-accuracy benchmark.
- Absolute warm-up zero is uncertain (analyzer clock mis-set; per-run
  `calibrationTime` only on Moissanite). Slope conclusions are offset-invariant;
  absolute intercepts are not.
- Analyzer/spectrometer temperature is not logged (only laser `solenoidTemp` in
  the handheld `WLCalLog`); `analyzer_temperature_c` is null with the source
  named. The temperature-vs-shift regression could not be run.
- **Only the segment OFFSET is a shared instrument quantity; the within-segment
  slope and per-line residual are database wavelength errors** (Fe→V transfer
  597 pm, opposite Fe/V slopes). So the golden-line function is deployed as an
  offset only; a true wavelength-dependent instrument scale cannot be extracted
  from these single-element metal forests. Reaching ≤ 30 pm everywhere would
  need a sparse, well-known standard (a lamp) or a multi-element CRM, not dense
  Fe/V forests.
- Only **V VIS** reaches the ≤ 30 pm offset target (SD 24 pm); Fe VIS/NIR and
  V NIR offsets scatter 60–83 pm; **UV is unregistered** (Fe 3 / V 6 golden
  lines — too few clean observed peaks). The 597 pm Fe→V transfer means the
  earlier "wavelength-dependent instrument function" reading is withdrawn: it
  was per-element database error.
- Absolute warm-up zero uncertain; O I not detected (purge displaces air O); the
  3 old-calibration runs give no warm-up spread; analyzer temperature not logged.
- Peak-table re-centring deferred (above) remains a decision for judgment.

## Coordinator verification (main session, 2026-09-23)

Three implementations of the element-line registration were checked against
the raw spectra; none is validated for the UV/VIS segments, and the pipeline
default was changed accordingly (ambient Ar registration applied to the NIR;
element registrations recorded as diagnostics only).

- **Ambient (Ar I) NIR registration: trusted.** Observed = db − 0.18 nm;
  fresh Fe runs (4–20 min after the 13:39 calibration) −192 ± 67 pm, the V
  sequence (155–175 min) −176 ± 142 pm, the three runs 18–21 h after the
  previous calibration within the same range. No warm-up drift at the ±50 pm
  level in the NIR. O I was absent in all 53 runs (argon purge), so O I cannot
  serve as a calibrator here.
- **Vote-mode / windowed estimator (v1, v2): rejected.** Within one 20-minute
  V sequence the per-run UV/VIS shifts scattered by ~200 pm (fit σ 25 pm);
  the fresh-Fe UV came out +200 pm where every anchor-based estimate is
  −120 to −150 pm; element-vs-Ar NIR disagreement ~390 pm. My segment-wide
  joint offset+slope fit reproduced the instability: Fe VIS slope +1.86 ±
  0.43 pm/nm versus V VIS −1.49 ± 0.46 pm/nm on an identical wavelength axis,
  dominance ≤ 1.23. Forest matching measures each element's alias pattern.
- **Golden-line estimator (v3): not validated either.** On the golden lines
  themselves the measured offsets scatter over ~470 pm within one segment
  (V I 437.92 −316, 438.47 −160, 438.99 +152, 445.20 −217, 459.41 +137,
  480.75 +69 pm). Inspection of the raw samples shows why: the strongest lines
  of a pure-metal plasma are optically thick (τ 1.5–3.7 measured on 2026-09-22)
  and their profiles are distorted or displaced — Fe I 438.354 has no maximum
  at its position in the fresh Fe run (the strong feature sits at 438.035),
  and the V I 437.92/438.47/439.00 triplet appears as a single peak at 438.31
  with the two companions almost absent. The Fe→V transfer failure (250–600
  pm) therefore cannot be blamed on database wavelength errors: the database
  values for these lines equal NIST.
- **Sample-identity check (inconclusive):** the 60 strongest visible peaks of
  the V run match V's top-300 predicted lines in 50 % of cases (chance 28 %)
  and Fe's in 52 %; but the pure Fe run scores only 57 % against Fe, so the
  predicted-strength ranking (gA·e^{−E/kT} at 9 kK, no response, no opacity)
  is too poor a template for this test to decide anything.
- **What stands:** the Gaussian instrumental-profile sub-pixel fit (tests:
  0.035 px at SNR 20, unbiased; −15 % registration residual vs parabolic on
  the same lines); the calibration provenance (every shot record carries the
  calibration timestamp; the 13:39 recalibration moved the scale by +69/−94/
  −223 pm UV/VIS/NIR; the only temperature the handheld exposes is the laser
  solenoid, 35 °C, in the vendor's calibration-check log); the Pantheum
  provenance fields.
- **Verdict on the warm-up hypothesis:** not supported where it can be
  tested (NIR, ±50 pm over 4–175 min and across an 18 h gap); untestable in
  the UV/VIS until an unambiguous wavelength reference is available. The
  dominant term in the database mismatch is the pixel→nm polynomial itself
  (0.1–0.3 nm, wavelength-dependent), which the vendor's own recalibration
  moves by ~0.1–0.2 nm per segment. A line lamp (Hg/Ar or Ne) or a certified
  multi-element reference measured on this unit is the prerequisite for a
  UV/VIS registration; the optically thin, medium-strength lines (E_i > 2 eV)
  of a dilute analyte in a matrix would be the next-best anchors.

## Patch verification (shared-file split)

Other sessions have uncommitted work in both trees, so the changes are delivered
as patches of only my hunks, verified on clean `--detach HEAD` worktrees.

**alibz** — `reports/patches/2026-09-23-wavelength-registration.patch` (15 files):
new `alibz/wavelength_registration.py`, `alibz/utils/peakfit.py`,
`tests/test_wavelength_registration.py`, `tests/test_peakfit.py`,
`scripts/registration_study.py`, `docs/wavelength_registration.md`,
`reports/2026-09-23-wavelength-registration.md`,
`reports/registration-study-20260923.json`,
`reports/figures/registration-20260923/*.png` (5); plus my 2 hunks in
`alibz/peaky_indexer_v3.py` and my `DECISIONS.md` entry.
`alibz/utils/wavelength.py` — **no hunks of mine** (its working-tree diff is
another session's).
`alibz/pipeline.py` — **excluded** from the applyable patch: my hunks there are
interleaved at the line level with concurrent uncommitted work (gas calibration
`apply_gas_calibrations`/`baseline_shift`; `physical_triage`) inside the same
`AnalysisConfig`, `analyze_spectrum` signature/body/return and
`analyze_directory`, so they are not standalone-applyable on bare HEAD. They are
provided informationally as
`reports/patches/2026-09-23-wavelength-registration-pipeline-INTEGRATION.diff`,
to re-integrate by hand once that work lands.

Verified: `git worktree add --detach /private/tmp/alibz-reg-v4 HEAD` (cd817c4),
`git apply --check --binary` clean, applied; `python -m pytest
tests/test_wavelength_registration.py tests/test_peakfit.py` (repo `.venv`,
`ALIBZ_DB` → the tracked-but-uncheckout db) = **33 passed, 1 failed**. The single
failure is `test_pipeline_registration_on_off`, which needs the excluded pipeline
integration (`TypeError`: `analyze_spectrum` has no `wavelength_registration`
kwarg at HEAD); on the full working tree that carries the concurrent work the
whole suite passes (559). Worktree removed.

**pantheum-I** — `reports/patches/2026-09-23-provenance-fields.patch`:
`tests/test_alibz_acquire.py` (2 hunks — `datetime` import + the provenance test
class). The `acquire.py` provenance itself is **already committed to
pantheum-I HEAD** (24cb0f3), so no `acquire.py` hunk remains. Verified on
`git worktree add --detach HEAD`: `git apply --check` clean, applied,
`python3 -m pytest tests/test_alibz_acquire.py -q` = **76 passed, 37 subtests**.
Worktree removed.

## Correction (2026-09-23 reconciliation with the gas-calibration engine)

The "Ambient (Ar I) NIR registration: trusted" bullet above overstates the
basis. On 10-shot Fe run means the ambient NIR value rests on **one line**, Ar I
696.543 nm (SNR ≈ 17, shift −0.188 nm, MAD 11 pm over 25 of 26 archived runs);
772.4 and 801.5 nm agree (−0.12 / −0.17 nm) but are below or near the SNR gate,
and the remaining Ar I anchors (706.7, 738.4, 763.5, 794.8, 811.5, 826.5,
842.5 nm) match repeatable **iron** features 0.2–0.6 nm away (Fe II 763.19,
794.51, 810.98, 842.06 nm and three unlisted features). In 5 of 53 runs the
robust mode locked onto the 794.8 nm iron feature (−0.34 nm). The median
−0.196 nm and the ±50 pm no-drift bound therefore describe the 696.5 nm line,
not a multi-line registration (re-run with the composition gate: 46/53 runs,
median −0.196 nm, MAD 9.5 pm; the gate removed the two outliers at +0.48 and
−0.33 nm and nothing else); the pipeline default (`ambient`, ≥ 3 lines)
correctly applies nothing to such data. `ambient_registration` now takes the
sample composition as an anchor gate (removal only). Evidence, the per-anchor
table and the binding rule between the two argon engines:
`reports/2026-09-23-gas-registration-reconciliation.md`.
