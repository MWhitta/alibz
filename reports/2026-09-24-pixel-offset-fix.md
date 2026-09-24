# Z300 pixel→wavelength −18 px offset fix (step 1 of 3) — 2026-09-24

Prepared, NOT deployed / committed. No hardware actions. Owner's uncommitted
edits (pantheum-I optimization.py/reservation.py/ramanlab, alibz) left untouched.
Root cause: reports/2026-09-23-native-axis-18px-offset.md.

## What changed (file:line)

### pantheum-I (live path is acquire.retrieval=data_api, so these are the fix)
- `pantheum/alibz/z300_calibration.py`
  - Module docstring rewritten (was ~L27-38): the old "confirmed … landed on
    the CSV endpoints" claim is vacuous (the knot trim clamps endpoints for any
    offset); the −18 px offset had to be recovered by line cross-correlation.
  - `CALIBRATION_STATUS` (L89) now "verified:18px-axis-offset …; inactive
    4th-channel placeholder excluded".
  - New `PIXEL_OFFSET = -18` (L112) with full provenance/cross-check comment.
  - New `MIN_SEGMENT_SPAN_NM = 5.0` (L122) + `AR_I_NIR_STICKS` (L127, 17 Ar I
    air-nm lines).
  - `pixels_to_wavelength(..., pixel_offset=PIXEL_OFFSET)` (L190): evaluates
    `_polyval(coeffs, p + pixel_offset)` and excludes any segment whose evaluated
    span over all pixels < MIN_SEGMENT_SPAN_NM (the 960–961 nm stub) — L243-257.
  - New helpers `_median`/`_highpass`/`_pearson` and
    `argon_axis_offset_px(wavelength, intensity, …)` (L310-407): high-passes the
    620–948 nm segment, correlates a shifted Ar I Gaussian stick template over
    −30…+10 px in 0.25 px steps, returns `{offset_px, peak_corr,
    runner_up_corr(>3px away), ambiguous, dispersion_nm_per_px, n_points,
    window_nm, sticks_used}`; returns offset_px=None (never raises) on thin data.
- `pantheum/alibz/z300.py`
  - Import `PIXEL_OFFSET` (L28).
  - `synthetic_shot` (L324): intercept now `knots[s] - slope*PIXEL_OFFSET` so the
    fake maps stored pixel 0 → knots[s] and last pixel → knots[s+1] *under* the
    −18 offset (existing test intent preserved).
- `pantheum/alibz/acquire.py`
  - Import `argon_axis_offset_px` (L37).
  - `_sorted_unique_csv` refactored into `_native_xy` + `_sorted_unique_csv`;
    new `_average_axis_check` (L1046) and `_axis_check_warning` (L1060).
  - `_finish_dataset` records `axis_check` in provenance (test.json, manifest,
    dataset metadata) and now returns `(dataset_id, shot_count, axis_warning)` (L1126).
  - Live caller appends the axis warning to the batch `detail` when |offset|>2 px
    or the lock is ambiguous (never rejects) — caller L904/L1116-1123.

### alibz
- `scripts/z300_opal_ingest.py`
  - `legacy_spectrum` (L468): `index = np.arange(...) + decoder.PIXEL_OFFSET`
    (−18) for the legacy ZIP/gzip-JSON path (was offset 0); comment updated.
  - Manifest `pixel_offset` (L694) now `decoder.PIXEL_OFFSET` for ALL formats
    (was `… else 0.0`).
  - `LEGACY_CALIBRATION_NOTE` (L70) rewritten for the −18 offset.
  - `LEGACY_REFERENCE_SHA256` (L62) updated to the NEW z300_calibration.py hash
    (`449dfb49…`, was `8a6135d0…` = the pre-fix offset-0 module) so the "legacy
    math matches pantheum z300_calibration.py @ <hash>" provenance stays true.
    [decision — see Open questions; no test asserts this literal.]
- `scripts/deploy-alibz-pixel-offset.sh` — new deploy wrapper (below).

## Tests

### Test counts (run before AND after editing)
| Suite | Before | After |
|---|---|---|
| pantheum-I full `python3 -m unittest discover -s tests` | 966 OK (26 skip) | 979 OK (26 skip) |
| pantheum-I `tests.test_z300_calibration` | 14 OK | 24 OK |
| pantheum-I `tests.test_alibz_acquire` | 76 OK | 79 OK |
| alibz `pytest tests/test_z300_opal_ingest.py` | 84 passed | 84 passed |

(+13 pantheum tests = 24−14 plus 79−76; opal count unchanged, assertions updated.)
Before-baselines were taken by `git stash push` of only my files (owner edits
untouched), running the suite, then `git stash pop`.

### New tests (pantheum-I tests/test_z300_calibration.py)
- `PixelOffsetTest`: default is −18; offset shifts the evaluation point (raises
  every sample by 18 nm at 1 nm/px); the 960–961 nm 4th-channel placeholder is
  excluded; MIN_SEGMENT_SPAN_NM == 5.0.
- `FakeRoundTripTest`: synthetic_shot maps pixel 0→knots[s], last→knots[s+1]
  under the default −18 offset.
- `ArgonAxisOffsetTest` (REAL-DATA regression, fixture below): at −18 the three
  Ar I lines land within ±0.12 nm (measured ≤0.025 nm, parabola-refined peaks)
  and the guard reads |offset|≤0.5 px, not ambiguous; **at offset 0 the same
  checks fail** (lines 0.47–0.84 nm off; guard reads −18 px) — both asserted.
  Plus a thin-data guard test (returns offset_px=None).

### New tests (pantheum-I tests/test_alibz_acquire.py)
- `WavelengthCalibrationProvenanceTests.test_finish_dataset_records_axis_check_provenance`
  — axis_check present in test.json/manifest/dataset.
- `AxisCheckTests` — warning thresholds; real NIR average reads ~0 px, no warning.

### Existing tests updated (only where the old expectation WAS the bug)
- tests/test_z300_calibration.py: 6 synthetic-math calls now pass
  `pixel_offset=0` (they hand-check pure polynomial math with an implicit offset
  of 0, not the physical axis — behaviour preserved, not the bug).
- `RealFixtureTest.test_real_calibration_produces_monotonic_wavelength`:
  old assertion required the top wl within 1 nm of the 961 nm knot, i.e. it
  required the placeholder stub to be emitted. Now asserts every wl < 960 nm
  (placeholder excluded), which is the corrected behaviour.
- tests/test_alibz_acquire.py: `_finish` helper unpacks the new 3-tuple
  `(dataset_id, shot_count, axis_warning)`.
- alibz tests/test_z300_opal_ingest.py: `pixel_offset == 0` → `-18` (legacy);
  reference-math test renamed *_pixel_zero_math → *_offset_pixel_math and its
  independent scalar loop evaluates at `pixel + decoder.PIXEL_OFFSET`.

## Real-data regression fixture
`pantheum-I/tests/fixtures/z300_v_pure_run2_nir_trimmed.json` (39,996 bytes):
the NIR segment (index 2) — knots [620.0, 960.0], that segment's cubic and its
2,066 raw pixels — of V_pure_run2 shot-0
(moissanite run-7d9ce38658474a1f8dc54acc273c9b4b/raw/shot-0.json; local copy
under scratchpad/raw/…). At −18: Ar I 811.531→811.507 (Δ−0.024), 801.479→801.460
(Δ−0.019), 763.511→763.522 (Δ+0.011) nm; guard offset +0.25 px. At 0: same lines
land 0.47–0.84 nm too blue; guard −18.0 px.

## Axis-guard values on real runs (argon_axis_offset_px, shot--1 average)
Sample-independent (Ar flush lights the same lines regardless of Fe/V matrix);
the guard is NOT fooled by Fe/V NIR lines — every run reads ~−18 px uncorrected
and ~0 px corrected, none ambiguous. Residual −0.25 px on some = the genuine
calibration/registration shift noted in the root-cause report.

V_pure_run2 (local):
```
run-09f1642f  off0 -18.25(r0.56) -> off-18 +0.00(r0.57)
run-12577962  off0 -18.50(r0.46) -> off-18 -0.25(r0.47)
run-1c97f55e  off0 -18.25(r0.52) -> off-18 -0.25(r0.53)
run-1e18a9a4  off0 -18.25(r0.56) -> off-18 +0.00(r0.57)
run-2ce26190  off0 -18.25(r0.57) -> off-18 +0.00(r0.59)
run-2f32e7b1  off0 -18.25(r0.60) -> off-18 +0.00(r0.62)
```
Fe pure metal (moissanite):
```
run-11d07e55  off0 -18.25(r0.48) -> off-18 +0.00(r0.51)
run-126d0c96  off0 -18.00(r0.47) -> off-18 +0.00(r0.50)
run-2e03f73d  off0 -18.00(r0.55) -> off-18 +0.00(r0.58)
run-6e38b110  off0 -18.25(r0.58) -> off-18 +0.00(r0.62)
run-bb777df5  off0 -18.25(r0.45) -> off-18 -0.25(r0.48)
run-eae9d54f  off0 -18.00(r0.45) -> off-18 +0.00(r0.47)
```
(Fe→element mapping via optimization_batches⋈optimization_sessions on the live
alibz.sqlite.)

## Deploy wrapper (dry-run only; nothing deployed)
`alibz/scripts/deploy-alibz-pixel-offset.sh`, modelled on
`deploy-alibz-native-grid.sh`; uses reservation-locked, manifest-pinned
`deploy-alibz-fire-fix.py`. Deploys the 3 pantheum files to
moissanite:~/pantheum-I. EXPECTED_BEFORE = the LIVE moissanite hashes, confirmed
2026-09-24 to equal `git show HEAD:<path>` for all three:
- z300_calibration.py  BEFORE 8a6135d0… (live==HEAD)  AFTER 449dfb49…
- z300.py              BEFORE efc6adf7… (live==HEAD)  AFTER fb4e4f67…
- acquire.py           BEFORE 58ab146f… (live==HEAD)  AFTER d67efa0d…
The wrapper re-checks live==BEFORE before pinning and aborts on mismatch.

Opal producer: live retrieval is **data_api**, so z300_opal_ingest.py is NOT on
the live critical path (opal_retrieval is configured but inactive). It lives on
the Windows Opal box at `C:/InstrumentControl/alibz/z300-ingest-20260922`, staged
by `scripts/stage_z300_opal.py` (already dry-run/--apply, hashes files
dynamically). The wrapper's Phase B runs that stager in dry-run; re-stage with
`stage_z300_opal.py --apply` before switching to opal_database retrieval.

### Dry-run output (2026-09-24; runtime verified UNCHANGED afterward)
```
{"apply": false, "files": ["pantheum/alibz/z300_calibration.py", "pantheum/alibz/z300.py", "pantheum/alibz/acquire.py"], "active_work": 0, "configuration": "unchanged"}
dry run only; rerun with --apply to install and restart the alibz services
== Phase B: Opal ingest producer (z300_opal_ingest.py) ==
{"apply": false, "target": "C:/InstrumentControl/alibz/z300-ingest-20260922", "files": {"z300_opal_ingest.py": "912766af…", "z300_fb_decode.py": "3130e640… (unchanged)"}, "adb_source": "…", "new_services": 0}
```
Post-dry-run live hashes still 8a6135d0…/58ab146f…, config 5175c3e1… (unchanged).

## What the user must run (from alibz/)
1. Review the diffs in pantheum-I and alibz.
2. Deploy the live path:  `bash scripts/deploy-alibz-pixel-offset.sh` (dry run),
   then `bash scripts/deploy-alibz-pixel-offset.sh --apply` (installs to
   moissanite, backs up, restarts pantheum-alibz{,-worker}.service, verifies).
3. Commit both repos (not done here).
4. (Later, step 2/3) Re-derive stored native datasets and re-score sessions;
   `scripts/stage_z300_opal.py --apply` only when opal_database retrieval is used.

## Open questions / could not verify
- `LEGACY_REFERENCE_SHA256` update to the new module hash is a judgment call
  (in-scope provenance consistency; no test pins the literal). Flag if you'd
  rather leave it pinned to the pre-fix module.
- `--apply` path (service restart, backup, install) NOT exercised — dry run only.
- Fe average shots pulled read-only from moissanite; older Fe runs (2026-09-21)
  had empty raw/ dirs and were skipped.
