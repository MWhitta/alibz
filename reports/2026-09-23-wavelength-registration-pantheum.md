# Wavelength-calibration provenance fields — pantheum-I `acquire.py`

2026-09-23. Repo: `/Users/mwhittaker/Projects/github/pantheum-I`. No commit made
(per instructions). No network/hosts/instruments touched; ramanlab/camera/D455
paths untouched (verified below).

## Summary

`_finish_dataset` DOES have access to the shot records' wavelength-calibration
block at finish time — `average` (the shot-index `-1` record, same one used
for `average.csv`) carries `wlCalibrations[i]` with `calibrationTime` and
`pixToNm.coefficients`, exactly as documented in
`pantheum/alibz/z300_calibration.py`. So the "no data path" fallback in the
brief was not needed: the fields are populated from real data, not stubbed to
null-with-comment (except `analyzer_temperature_c`, which is null by design
per the brief).

## Changes (all in `pantheum/alibz/acquire.py` unless noted)

1. **`_parse_calibration_time(raw)`** — new module-level helper, inserted
   after `_atomic_write` (file lines 91–104 of the edited file). Parses
   `"%b %d, %Y %I:%M:%S %p"` (e.g. `"Sep 22, 2026 1:39:21 PM"`); returns
   `None` on any non-string/empty/malformed input, never raises.

2. **`_wl_calibration_provenance(shot)`** — new helper (lines ~106–137).
   Reads a shot dict's `wlCalibrations` list, returns
   `{wl_calibration_time, wl_calibration_time_raw, wl_calibration_coefficients}`.
   `wl_calibration_time_raw` is the raw `calibrationTime` value if the key was
   present on any segment (kept even if not a parseable string, per the brief:
   "keep raw if present"); `wl_calibration_time` is its ISO-8601 parse or
   `None`; `wl_calibration_coefficients` is a list of per-segment coefficient
   lists (from `pixToNm.coefficients`), or `None` if any segment's
   coefficients could not be read. Never raises on a malformed/absent block.

3. **`_warmup_minutes(collected_at_iso, calibration_dt)`** — new helper
   (lines ~139–159). `(collected_at - wl_calibration_time)` in minutes, or
   `None` if either is missing.
   **Documented assumption** (also in the code comment): the analyzer's
   `calibrationTime` string carries no timezone, and there is no verified way
   to convert it to UTC. The helper strips `collected_at`'s UTC tzinfo and
   subtracts the two naive timestamps directly — i.e. it assumes the
   analyzer's local clock reads the same wall-clock value as UTC. This is
   **only correct if the analyzer clock is actually set to UTC**, which is
   unverified against a real instrument. This is the single open assumption
   from the brief; everything else is exact.

4. **`_analyzer_temperature_placeholder()`** — new helper (lines ~161–167).
   Returns `{'value': None, 'source': None, 'observed_at': None}` with a
   comment naming the eventual source (`WLCalLog solenoidTemp`, not yet
   wired).

5. **`_finish_dataset` signature** — added `started_at=None` kwarg (old:
   `(self, run_id, run_mode, params, test_id, average, shots, work_dir,
   requested=None)`). Body now builds one `provenance` dict from
   `_wl_calibration_provenance(average)` + `_warmup_minutes` +
   `_analyzer_temperature_placeholder()`, and splices it (`**provenance`)
   into all three write sites:
   - `test.json` (was: `{'test_id', 'run_mode', 'grid', 'requested_shots',
     'dropped_frames'}`, now also has the 5 provenance keys)
   - the dataset's `meta['acquisition']` dict (was: `{'run_id', 'test_id',
     'params', 'instrument'}`, now also has the 5 provenance keys)
   - `manifest.json` (was: `{'run_id', 'run_mode', 'test_id', 'instrument',
     'calibration_status', 'params', 'shots', 'requested_shots',
     'dropped_frames', 'grid', 'files'}`, now also has the 5 provenance keys)

   Comment added at the top of the calibration block noting it reads the
   *same* `average.wlCalibrations` block `_sorted_unique_csv` already uses
   for pixel→nm conversion, so provenance matches the spectra it describes.

6. **Call site** (`run()`, the only caller of `_finish_dataset`): now passes
   `started_at=row['started_at']` — the existing `acquisitions.started_at`
   column, set to `_now()` (UTC ISO-8601) when the row transitions to
   `running`. This is the "run-start timestamp acquire already has" the
   brief asked for; no new timestamp source was introduced.

All additions are purely additive: no existing field, key, or write site was
removed or renamed; the new `started_at` kwarg defaults to `None` so no other
caller is affected (grep confirmed `_finish_dataset` has exactly one call
site, at `run()`).

## Tests — `tests/test_alibz_acquire.py`

Added `import datetime` and one new test class,
`WavelengthCalibrationProvenanceTests` (appended at end of file), reusing the
existing `Service`/`small_params`/`wait_for_terminal` harness and the
file's convention of calling private methods directly (precedent:
`Acquisition._sorted_unique_csv` is already called directly in
`test_shot_csv_keeps_the_native_detector_grid`).

- `test_parse_calibration_time_parses_the_observed_format` — parses the
  exact brief string to `datetime(2026, 9, 22, 13, 39, 21)`.
- `test_parse_calibration_time_is_none_safe_on_junk` — `None`, `''`, `'   '`,
  `'not a date'`, `12345`, `0`, a list, a truncated string, and an ISO string
  all return `None` (8 subtests).
- `test_finish_dataset_writes_calibration_provenance_everywhere` — a
  synthetic shot with a real calibrationTime string, `started_at` 7h30m
  later; asserts all 5 fields are correct and identical across `test.json`,
  `manifest.json`, and the dataset metadata (3 subtests): `wl_calibration_time
  == '2026-09-22T13:39:21'`, raw preserved, `collected_at` echoed,
  `warmup_minutes == 450.0`, temperature placeholder, 4 segments × 4
  coefficients.
- `test_finish_dataset_calibration_time_is_null_safe_when_absent` —
  `calibrationTime` deleted from every segment (pixToNm coefficients kept,
  since those are required for the spectrum itself and never optional on a
  real shot): `wl_calibration_time`/`_raw`/`warmup_minutes` all `None`,
  `collected_at` still populated, coefficients unaffected (3 subtests).
- `test_finish_dataset_collected_at_is_null_safe_when_started_at_missing` —
  `started_at=None`: `collected_at` and `warmup_minutes` both `None`,
  `wl_calibration_time` still parses (3 subtests).
- `test_finish_dataset_coefficients_null_when_calibration_block_missing` —
  direct unit test of `_wl_calibration_provenance({})` and `('not a dict')`
  → all-`None` dict.
- `test_live_end_to_end_run_carries_calibration_provenance` — full
  `start()`/`run()` path through simulate mode (the fake `synthetic_shot`
  calibrationTime is an int placeholder `0`, unrelated to this brief):
  confirms `wl_calibration_time` is `None`, raw is `0`, `collected_at` is
  set, coefficients present — i.e. the wiring survives the real code path,
  not just direct `_finish_dataset` calls.

## Test counts

- **Before**: `python3 -m pytest tests/test_alibz_acquire.py -q` →
  `69 passed, 19 subtests passed` (verified before any edit).
- **After**: same command → `76 passed, 37 subtests passed`.
- Net: +7 test methods, +18 subtests, 0 failures, 0 skips.

## Constraint verification

- No `git commit` run (`git status` still shows all changes unstaged/dirty).
- `git diff --stat -- pantheum/alibz/acquire.py tests/test_alibz_acquire.py`
  is the only diff attributable to this task; `git diff` on the full repo
  shows many other files already modified (ramanlab, camera, opal-camera,
  ui tests, docs, config) — these were **already dirty before this session
  started** (confirmed by inspecting `git diff -- pantheum/alibz/acquire.py`
  line-by-line: deletions include lines like `return self._public_run(row),
  True` that this task never touched, i.e. pre-existing uncommitted work from
  another session, exactly as the brief warned). This task touched only
  `pantheum/alibz/acquire.py` and `tests/test_alibz_acquire.py`.
- `pantheum/ramanlab/*`, `web/ramanlab/*`, `deploy/ramanlab/*`,
  `tests/test_ramanlab.py`, and everything camera/D455-related were not
  opened or edited by this task.

## What I could not verify

- The single-clock assumption in `_warmup_minutes` (analyzer local clock ==
  UTC) is unverified against a real Z300 instrument — no live/real hardware
  access in this task, consistent with the "no instruments" constraint. This
  is called out both in the code comment on `_warmup_minutes` and here, as
  instructed.
- `analyzer_temperature_c` is intentionally left fully null (per brief); the
  `WLCalLog solenoidTemp` source is named only in a comment, not read from
  anywhere, since no such read path exists yet.
