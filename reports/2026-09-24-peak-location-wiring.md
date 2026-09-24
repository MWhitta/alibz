# Per-segment peak-location wiring into the alibz pipeline — 2026-09-24

Specialist agent (Opus 4.8), Claude subscription, no provider switch. Repo
`alibz`, branch `fix/z300-pixel-offset`. No commit, no hardware. Implements the
owner decision (cal/reports/2026-09-24-empirical-lsf.md sec 3): ePSF for VIS+NIR
peak location, vendor spline / native parabola for UV, as a **per-segment switch
driven by the shape file**, plus the per-segment local-pitch fix
(cal/reports/2026-09-23-native-grid-method-audit.md A0/B1).

## 1. What was built (file:line)

### 1a. Packaged shapes + loader
- `alibz/data/epsf-z300-20260924.json` (NEW): copy of `cal/data/epsf-z300-20260924.json`
  with, per segment, a `location` block carrying a **`location_method`** string
  (**UV `spline`, VIS `epsf`, NIR `gaussian`**) plus `accuracy_pm` and `evidence`,
  a `packaging` provenance block, and `segments.NIR.unstable_sub_bands=[0]`
  (the 622-645 nm band). NIR carries a `gating_note` with the seam-bias numbers
  (§6). Every segment's ePSF shape is retained, so enabling a future UV/NIR ePSF
  is a one-field `location_method` edit. (The earlier boolean `use_for_location`
  was replaced by `location_method` per the coordinator decision, §6e.)
- `.gitignore:50-51`: added `!alibz/data/` so the packaged shape is versioned
  (the global `data/` rule would otherwise exclude it).
- `alibz/utils/epsf_shapes.py` (NEW): `Z300Shapes` + `load_z300_shapes()`
  (`@lru_cache`); `method_for(seg)` returns the configured `location_method`,
  `uses_epsf(seg)` the convenience bool; `epsf_at(wl)` interpolates sub-band
  coefficients in wavelength ("ePSF lambda"), dropping the unstable NIR band.

### 1b. Dispatcher
- `alibz/utils/peak_location.py` (NEW):
  - `locate(x, y, peak_indices, method="auto", instrument="auto", ...) ->
    (centres_nm, uncertainties_nm, methods)` — the specified interface. In
    `auto` it uses each segment's `location_method` (UV spline, VIS epsf, NIR
    gaussian); `"epsf"/"spline"/"parabola"/"gaussian"` force one.
  - `is_z300_native(x)` — multi-pitch + Z300 pitch-band gate (rejects uniform
    1/30 nm exports and synthetic grids).
  - ePSF fit via `epsf.fit_line` on per-segment integer pixel coordinates,
    centre bounded (±1 px), falling back to spline-root / gauss on
    non-convergence or blend; `method_used` records the actual method. The
    `gaussian` method uses the validated point-Gaussian (`gaussian_subpixel_center`).
  - `locate_records` / `locate_one_record` — dict adapters for the existing
    callers; **non-Z300 input delegates byte-identically** to
    `peakfit.refine_peaks` / `peakfit.gaussian_subpixel_center`.
  - Uncertainty for spline/parabola/gaussian comes from the covariance-based
    `gaussian_subpixel_center` 1-sigma (those estimators give none); ePSF uses
    its own `center_err × local pitch`.

### 1c. Per-segment local pitch (global-pitch defect)
- `alibz/utils/native_grid.py:+842..` (NEW): `segment_pitch_array(x)` returns a
  per-sample pitch array, the legacy global `inc=median(diff(x))`, and
  `is_native`; **byte-identical to `inc` on uniform/1-pitch grids** (only a
  genuine multi-pitch native grid gets per-segment values); dead 4th channel
  (>955 nm) excluded. `local_pitch_nm(x, at=None)` scalar accessor.
- `alibz/peaky_finder.py`: replaced the six global-pitch sites with per-peak
  local pitch (all byte-identical on uniform grids):
  - `find_background` `:497` — ARPLS `lam=(smooth_nm/pitch)**4` now computed
    per segment span.
  - `fit_peaks` `:886` — `_local_inc`/`_cap_at` closures; `_voigt_seed` min_inc,
    the width cap (`:990`, `:1028`) and the centre bound (`:1038`) are per-peak.
  - `_fwhm_cap` `:1069` — per-segment aware (kept for callers/tests).
  - `fit_shoulders` `:1113` — cap, `dx`, seed min_inc (`:1203`) and centre bound
    (`:1209`) per-peak.
  - `fit_all` `:1261` — cap, centre bound (`:1289`), window_span floor per-peak.
  - `fit_spectrum` `:1460` — the `inc/2` sigma/gamma zero-floors are per-peak.
- `alibz/relative_profiles.py:279` — line-power integral uses the per-sample
  pitch array.
- `alibz/peak_window_pca.py:278` — FWHM/area use the peak's local pitch.
- `alibz/session_calibration.py:43,51` — parabolic sub-pixel centre offset uses
  the peak's local pitch.
- `alibz/wavelength_registration.py:670,867` — the ~2 nm percentile-filter window
  size uses `local_pitch_nm(x)` (dead-channel-excluded); `_local_pitch` (:170)
  was already per-segment-local and unchanged.

### 1d. Callers routed through the dispatcher
- `alibz/pipeline.py:631` — the additive `peak_refinement` diagnostic now calls
  `locate_records`; label `method="auto"`, adds `method_used`. Still additive:
  the physics inversion keeps the blind Voigt centres.
- `alibz/wavelength_registration.py:321,684,848` — these three sub-pixel anchor
  sites were briefly routed through `locate_one_record` but **REVERTED to
  `gaussian_subpixel_center`** after the V-regression isolation (§6): the ePSF
  NIR fit carries a sub-band-correlated bias that inflates the registration
  drift-fit residual. Registration keeps the proven point-Gaussian; the ePSF
  stays the peak-LOCATION default for VIS/NIR via the dispatcher and flags
  (owner decision), which is a different consumer than the registration
  centroid. The window-size `local_pitch_nm` change (`:670,867`) is kept.

## 2. Tests

- Baseline (HEAD, before): **601 passed, 4 skipped, 71 subtests** in 551 s
  (605 collected). `.venv/bin/python -m pytest tests/ -q`.
- After: **614 passed, 4 skipped, 71 subtests** in 561 s (618 collected). No
  regressions: 601 pre-existing pass unchanged (incl. the one edited assertion),
  +13 new. Re-run after the §6 registration revert: **614 passed, 4 skipped, 71
  subtests** (unchanged). Third run after the §6e NIR gating: **614 passed, 4
  skipped, 71 subtests** (unchanged). `.venv/bin/python -m pytest tests/ -q`.
- New: `tests/test_peak_location.py` (13 tests, all pass in ~2 s): per-segment
  pitch byte-identity + native + dead-channel; `is_z300_native`; **per-segment
  dispatch asserts each segment resolves to its configured method** (UV spline,
  VIS epsf, NIR gaussian); forced methods; **synthetic sub-pixel truth** (forced
  ePSF recovers a known centre to <3 pm VIS / <4 pm NIR at three phases); ePSF
  blend fallback; **non-Z300 byte-identical delegation**; **real-data
  regression** (trimmed fixture, VIS Fe 438.35 -> epsf, NIR Ar I 811.53 ->
  gaussian; each within 0.1 nm of db and 0.03 nm of the Gaussian).
- Fixture: `tests/fixtures/z300_peakloc_windows.npz` (4 KB, trimmed ±3 nm
  windows from corrected Fe run-eae9d54f...).
- Changed test: `tests/test_wavelength_registration.py:346` — the additive
  diagnostic's `method` is now `"auto"` (was `"gaussian"`); this reflects the
  in-scope routing change, behaviour otherwise unchanged (synthetic grid ->
  delegated Gaussian).

## 3. End-to-end A/B on corrected data (3 Fe + 3 V runs)

`before` = HEAD worktree; `after` = this tree. `analyze_spectrum(..,
wavelength_registration="apply_element", subpixel="gaussian", n_calls=<N>)`.

n_calls=16 (bounded runtime). Per-segment registration residuals populate only
NIR here: UV/VIS golden/ambient anchoring is "unregistered" in this config for
both before and after (too few golden lines), so those cells are empty in BOTH
arms — the whole-spectrum effect shows in peaks-recovered and r_squared. Harness
+ raw JSON: scratchpad `ab_harness.py`, `before.json`, `after.json`.

### 3a. Per run (before -> after)
| run | fam | r_squared | peaks fit | NIR reg resid MAD (pm) | ambient global MAD (pm) | ambient anchors | runtime s |
|--|--|--|--|--|--|--|--|
| run-eae9d54f | Fe | 0.090 -> 0.227 | 40 -> 60 | 22.9 -> 16.2 | 28.4 -> 18.4 | 8 -> 8 | 32.1 -> 26.5 |
| run-126d0c96 | Fe | 0.340 -> 0.399 | 38 -> 58 | 18.0 -> 14.3 | 32.2 -> 16.5 | 9 -> 8 | 23.3 -> 25.2 |
| run-5291e643 | Fe | 0.015 -> 0.115 | 52 -> 75 | 17.0 ->  9.9 | 27.7 -> 11.1 | 9 -> 10 | 28.2 -> 28.8 |
| seq-645563aa | V  | 0.258 -> 0.144 | 97 -> 130 | 11.7 -> 16.4 | 16.6 -> 11.1 | 8 -> 8 | 55.1 -> 58.6 |
| seq-ab7e906c | V  | 0.295 -> 0.489 | 111 -> 138 |  6.9 -> 12.1 |  5.1 -> 20.3 | 6 -> 8 | 51.8 -> 58.8 |
| seq-5beff45e | V  | 0.394 -> 0.463 | 108 -> 145 |  2.9 -> 11.8 |  6.5 -> 16.8 | 6 -> 8 | 63.3 -> 51.5 |

Method actually used by the additive diagnostic (after), e.g. Fe run-5291:
`spline 64 (UV), epsf 9 (VIS/NIR), gaussian(epsf_fallback) 1`; V seq-5beff:
`spline 98, epsf 40, gaussian(epsf_fallback) 5`. So UV -> spline, VIS/NIR ->
ePSF, blends fall back — the per-segment switch fires as designed.

### 3b. Reading it — including what got worse
- **Fe (3/3): clearly better.** r_squared up on all three; NIR registration
  residual MAD down 6-7 pm (22.9->16.2, 18.0->14.3, 17.0->9.9); ambient global
  MAD down 10-17 pm (to 11-18 pm); ~+50 % peaks recovered (40->60, 52->75) — the
  per-segment pitch fix restores the NIR/VIS seed width, cap and centre bound so
  more real lines survive, and the ePSF/Ar-anchored registration tightens.
- **V (mixed).** More peaks recovered (97->130, 108->145) and more Ar anchors
  found (6->8 on two runs); r_squared up on 2/3 (0.30->0.49, 0.39->0.46) but
  **down on seq-645563 (0.26->0.14)**. **NIR registration residual MAD got WORSE
  on all three V runs** (11.7->16.4, 6.9->12.1, 2.9->11.8 pm) and ambient global
  MAD rose on 2/3 (5.1->20.3, 6.5->16.8). Not a clean comparison: the anchor set
  changed (n 6->8), so the MAD is over different lines; the two newly admitted
  Ar lines add scatter. It is also consistent with cal/reports/2026-09-24 sec 3,
  which found the NIR ePSF validated on **Ar I** (Ar-anchored registration) but
  the tiny **V I** NIR sample preferred the spline (unstable 622-645 nm band) and
  said to weight it low. Net for V: better peak recovery and inversion on most
  runs, but NIR registration scatter regressed and one r_squared fell. **NOTE:
  this §3 table is config (c) (pre-revert, ePSF in registration). Both V
  regressions were isolated and addressed in §6 — registration reverted to the
  point-Gaussian, which is non-regressive on all 6 runs.**
- **Runtime: unchanged** (median 42.0 -> 40.2 s at n_calls=16).
- Aggregate medians: r_squared 0.277 -> 0.313; peaks 74.5 -> 102.5; runtime
  41.95 -> 40.15 s.

## 4. UV data plan (from validate_epsf power analysis)

**Why UV fails today, not a sample-size problem alone.** The UV build already
had 901 windows / 31 db lines / 95 runs (more windows than VIS's 844) yet the
ePSF still LOST the accuracy test (7.6 pm vs spline 4.8, paired +4.0 [1.4,6.6]
pm). Two structural causes (cal/reports/2026-09-24-empirical-lsf.md sec 3, 6):
(i) UV lines are the best-sampled (~1.75 px FWHM) so the classic estimators
carry little phase bias — the ePSF has less to win; (ii) the UV reference is
83 % V II and its FWHM varies strongly within the segment (1.6-2.1 px across the
three sub-bands), so the shape itself is uncertain and class-contaminated. A
usable UV ePSF therefore needs *distinct, well-sampled, thin, multi-species*
lines with real sub-pixel-phase coverage, per sub-band — not just more V II.

**Acceptance tests and what drives their power** (validate_epsf):
- (a/b) cluster-aware / cross-validated phase uniformity: the null draws ONE
  uniform phase per DISTINCT line, so power scales with the number of distinct
  lines and with phase coverage, not window count. UV cluster p rose 0.01->0.11
  when distinct lines went 15->31; passing comfortably (p>~0.3) wants ~40-50.
- (c) accuracy: paired bootstrap over lines; the current UV CI half-width is
  ~2.5 pm on 15 distinct Fe lines / 12 runs. To resolve a true ~2 pm ePSF-vs-
  spline difference the CI must halve -> ~4x the lines, i.e. ~45-60 distinct
  thin lines with clean db wavelengths.
- (d) dither / (e) residual-vs-phase: `dither_structure`/`dither_test` need
  `min_per_group>=3`, `min_per_run>=5`, >=30 structured windows, and >=3 windows
  in each of ~8 phase bins per line. Per-run registration drift is only
  ~0.1-0.5 px, so filling the phase axis for a line needs ~10-20 exposures of
  that line (or a deliberately dithered wavelength solution).

**Estimate.** Target ~45 distinct, isolated, thin, high-lying UV lines spread so
each of the three UV sub-bands (242-274, 275-314, 324-362 nm) holds >=12-15,
each line seen in >=10 exposures (so >=~5 survive QC per run and the phase axis
fills from drift): ~45 lines x >=10 runs ~ >=450 clean UV windows with genuine
phase diversity, roughly half the current 901 but multi-species and phase-
covering rather than 83 % V II. If deliberate sub-pixel dithering of the
calibration is added, ~5-8 runs per line suffice.

**Pure standards to shoot (sparse, strong, isolated UV lines the Z300 can
excite), and why each:**
- **Al** — Al I 394.40/396.15 resonance doublet: very bright, isolated, anchors
  the 365+ sub-band; the known doublet separation cross-checks registration.
- **Cu** — Cu I 324.75/327.40 resonance doublet + 296-nm lines: bright and
  isolated, populates 324-362 (the weakest UV sub-band) and 275-314.
- **Mg** — Mg II 279.55/280.27 and Mg I 285.21: strong, isolated, fills the
  275-314 sub-band and adds an ION/neutral pair.
- **Si** — Si I 250.69/251.61 multiplet and 288.16: neutral UV lines in the
  242-274 sub-band that is otherwise line-poor.
- **Zn** — Zn I 328.23/330.26/334.50 triplet (+ Zn II 202/206 if reachable):
  isolated triplet with known spacings, more 324-362 coverage.
- **Sn** — Sn I 284.00/300.9/317.5: additional neutral UV lines spanning two
  sub-bands, different element to break class degeneracy.
- **Ti** — Ti II 323-338 dense manifold + Ti I ~365: supplies many distinct UV
  lines to reach the ~45 count and to populate 324-362 (use only the isolated
  members; the manifold is partly blended).
- **Mn** — Mn II 257-260 triplet: fills the blue 242-274 sub-band.

Choose thin, higher-E_i lines where possible (avoid folding physical width into
Psi), keep a NEUTRAL/ION and resonance/non-resonance mix across elements so no
single class dominates as V II does now, and acquire many exposures per standard
(or dither the calibration) to sample sub-pixel phase.

## 5. Unverified / limits

- The ePSF centre only enters CONSUMED numbers through wavelength registration
  (golden/vote/ambient anchors) and the additive `peak_refinement` diagnostic.
  The physics inversion still consumes the blind Voigt centres from
  `peaky_finder`; those are only *indirectly* improved via the per-segment pitch
  fix (seed width/cap/centre-bound). Re-centring the inversion table on ePSF
  centres was deliberately NOT done (out of scope; unvalidated).
- The per-segment pitch fix is proven byte-identical on uniform grids by the
  full suite (601 pre-existing tests unchanged) and by
  `test_pitch_uniform_is_byte_identical`; its effect on native data is exercised
  only through the A/B, not a golden-value test (no pre-existing native-grid
  fit fixture).
- `is_z300_native` is a heuristic (multi-pitch + pitch-band). A trimmed
  single-segment window cannot be auto-detected (the real-data fixture test
  forces `instrument="z300"`). Callers that pass whole native spectra detect
  correctly (verified on a real corrected run).
- The A/B used n_calls=16 (not the default 40) to bound runtime; r_squared is
  therefore lower than a production run and is used only for before/after
  comparison, not as an absolute quality number.
- ePSF uncertainties for the spline/parabola/gaussian methods are borrowed from
  the covariance-based Gaussian fit (those estimators have no native centre
  covariance); documented in `peak_location._gauss_diag`.
- UV plan counts are order-of-magnitude estimates from the observed CI/counts,
  not a formal power simulation (validate_epsf has no explicit power routine).

## 6. V regressions: isolated (follow-up)

Three A/B configs on the 6 runs (n_calls=16 unless noted): **(a)** HEAD;
**(b)** pitch fix only, registration on the point-Gaussian (peak_location NOT in
the registration path); **(c)** the full change (registration routed to ePSF).
Raw: scratchpad `before.json` (a), `afterb.json` (b), `after.json` (c),
`ar_isolate.py` (controlled anchor test), `run_one.py` (n40).

### 6a. NIR registration MAD (pm), per run: a / b / c
| run | fam | a HEAD | b pitch+gauss | c full ePSF |
|--|--|--|--|--|
| run-eae9d54f | Fe | 22.9 | 22.9 | 16.2 |
| run-126d0c96 | Fe | 18.0 | 18.0 | 14.3 |
| run-5291e643 | Fe | 17.0 | 16.5 |  9.9 |
| seq-645563aa | V  | 11.7 | 10.0 | 16.4 |
| seq-ab7e906c | V  |  6.9 |  6.2 | 12.1 |
| seq-5beff45e | V  |  2.9 |  2.9 | 11.8 |

**The ePSF registration (c) helps Fe NIR but hurts V NIR on every V run.**
Config (b) is non-regressive on NIR MAD for all 6 runs (each ≤ HEAD).

### 6b. Controlled cause — same 12-Ar-anchor set, three locators (resid MAD, pm)
`ar_isolate.py` locates the identical Ar I set (696.5-912.3 nm; none in the
unstable 622-645 band) with the point-Gaussian, the spline root and the ePSF,
per run, MAD about each locator's own median shift:
| run | gauss | ePSF | spline |
|--|--|--|--|
| Fe 126d | 11.1 | 13.6 | 16.9 |
| Fe 5291 | 11.4 | 13.5 | 19.1 |
| V 645563 | 6.3 | 10.1 | 11.6 |
| V ab7e | 6.0 | 7.7 | 7.8 |
| V 5beff | 5.5 | 8.8 | 7.9 |

The **point-Gaussian is the best NIR anchor locator on all runs**; the ePSF adds
a **sub-band-correlated red bias** (+14..+44 pm on 696-772 nm = NIR band 653-750,
~0 on 794-842 = band 752-852, negative at 912): a step at the sub-band seam that
the ePSF-lambda coefficient interpolation introduces and that a single-shift (or
low-order drift) registration does not absorb. It is **NOT** the unstable
622-645 nm band (no Ar anchor lies there) and only weakly tracks V I proximity.

**Reconciliation with cal/reports/2026-09-24 (ePSF wins on Ar I, 29 runs):** that
result used the leave-one-out dither-structure ePSF with a full **per-run linear**
registration and measured peak-location accuracy; the packaged segment/sub-band
ePSF used as a **registration centroid with a simple shift** exposes the seam
bias. Peak-location accuracy and registration-anchor centroiding are different
consumers. **Decision: NIR stays ePSF for peak LOCATION (owner flag unchanged),
but wavelength registration is reverted to the point-Gaussian.** No gating by
element is needed — Gaussian is best for registration on both Fe and V.

### 6c. seq-645563 r_squared drop: 0.258 -> ?
Cause is the **pitch fix, not the dispatcher**: r2 a/b/c = 0.258 / 0.097 / 0.144
(n16), and it PERSISTS at full n_calls=40: HEAD **0.269 (97 peaks)** ->
pitch-fix **0.124 (130 peaks)**; T 22.7->24.8 kK, ne ~17.3 (both physical). The
per-segment cap fix restores the correct (wider) VIS/NIR component widths and
recovers ~34 % more peaks in the dense V I forest; the CF-LIBS r_squared
(explained variance over the peak array) falls because the added weak forest
lines are harder for the limited forward model to explain. This is a
detection-completeness / explained-variance tradeoff, not a peak-location error,
and the pitch fix is audit-validated and raises r_squared on all 3 Fe runs and
2/3 V runs — so the evidence supports **keeping the pitch fix**; no code change.
(If desired, re-basing the CF-LIBS r_squared gate for the denser peak set is a
separate, out-of-scope task.)

### 6d. Net shipped state
Pitch fix + dispatcher + packaged shapes + additive pipeline diagnostic kept;
registration reverted to the point-Gaussian. Re-A/B (b vs HEAD): NIR MAD
non-regressive on all 6 runs; r_squared up on 3/3 Fe and 2/3 V; seq-645563
r_squared is the documented detection-completeness tradeoff above.

### 6e. Coordinator decision — NIR ePSF gated for peak LOCATION too
The coordinator reproduced the isolation (ar_isolate.py) and ruled that the
14-44 pm sub-band-seam bias is a LOCATION error, not only a registration one
(on V, NIR ePSF MAD 8.8 pm vs point-Gaussian 5.5 pm with each method's own
shift; the cal-report NIR "ePSF wins on Ar I" was vs the SPLINE and TIED the
Gaussian on 772 nm). Change applied:
- `alibz/data/epsf-z300-20260924.json`: the boolean `use_for_location` is
  replaced by a per-segment `location_method` string — **UV `spline`, VIS
  `epsf`, NIR `gaussian`** — with each segment's ePSF shape retained and a NIR
  `gating_note` (the seam-bias numbers above). Re-enabling a UV/NIR ePSF is a
  one-field edit once validated.
- `alibz/utils/epsf_shapes.py`: `method_for()` returns `location_method`
  (`uses_epsf()` convenience); `use_for_location` removed.
- `alibz/utils/peak_location.py`: the dispatcher honours `location_method` per
  segment, same bounded fallback chain, records `method_used`; the `gaussian`
  method is the point-Gaussian `gaussian_subpixel_center`.
- Registration stays on `gaussian_subpixel_center` (§6, unchanged).
- Tests updated: `test_shape_flags_drive_method`,
  `test_auto_method_per_segment_on_native` (asserts each segment dispatches
  `spline/epsf/gaussian`), `test_real_data_regression_*` (NIR expects gaussian).

## 7. Final shipped configuration

| segment | location_method | rationale |
|--|--|--|
| UV  | `spline`   | spline/parabola beat the ePSF on thin Fe lines (4.7-4.8 vs 7.6 pm); UV ePSF uncertain (83 % V II ref, FWHM 1.6-2.1 px). |
| VIS | `epsf`     | ePSF wins clearly and robustly (5.3 vs 10-12 pm), paired CIs exclude 0. |
| NIR | `gaussian` | NIR ePSF has a 14-44 pm sub-band-seam location bias (V MAD 8.8 vs point-Gaussian 5.5 pm on identical Ar anchors); point-Gaussian is the best NIR locator. ePSF shape retained; re-enable when the seam bias is fixed. |

- Wavelength **registration**: `gaussian_subpixel_center` (all segments).
- Pipeline **peak_refinement** diagnostic: additive, dispatched per the table
  (UV spline / VIS epsf / NIR gaussian), records `method_used`; the physics
  inversion still consumes the blind Voigt centres.
- Per-segment **pitch fix**: on (byte-identical on uniform grids); it is the
  net-positive lever (Fe r2 3/3 up, more peaks recovered) and the cause of the
  seq-645563 detection-completeness r2 tradeoff (§6c), kept per the evidence.
- Tests (shipped config, third full run): **614 passed, 4 skipped, 71 subtests**
  in 551 s (unchanged from before this follow-up).
- A/B shipped vs HEAD (n_calls=16), NIR registration resid MAD (pm):
  Fe 22.9->22.9, 18.0->18.0, 17.0->16.5; V 11.7->10.0, 6.9->6.2, 2.9->2.9 —
  **non-regressive on all 6 runs** (each <= HEAD). r_squared: Fe up 3/3
  (0.09->0.218, 0.34->0.399, 0.015->0.116), V up 2/3 (0.295->0.484,
  0.394->0.453); seq-645563 0.258->0.097 is the pitch-fix detection-completeness
  tradeoff (§6c, +34 % peaks, persists at n40). `method_used` confirms the
  dispatch: UV spline, VIS epsf (with gaussian/spline fallbacks on blends), NIR
  gaussian.
