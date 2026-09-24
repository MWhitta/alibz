# Re-derive stored Z300 datasets for the -18 px offset + re-score sessions (step 2) — 2026-09-24

Step 2 of the Z300 pixel→wavelength fix (root cause:
reports/2026-09-23-native-axis-18px-offset.md; step 1 deploy:
reports/2026-09-24-pixel-offset-fix.md). **Prepared + dry-run only. Nothing was
written to live Pantheum state on moissanite** (SQLite mtime unchanged at
2026-09-24 01:03; no `rederive-backup-*` dirs in any run dir; services active).
The only files written on moissanite are in the staging bundle dir
`~/pantheum-rederive-bundle-20260924/` (tool copy, fixed-module copy, dry-run
JSON report).

## Tool
- `scripts/rederive-alibz-axis-offset.py` — dry run by default, `--apply` to execute.
- `scripts/rederive-alibz-axis-offset.sh` — stages the tool + the local fixed
  `z300_calibration.py` onto moissanite and runs it under `~/pantheum-I`.
- `tests/test_rederive_axis_offset.py` — 21 pure-logic tests (classification,
  refusal rules, idempotence, native_xy dedup/offset).

Modelled on `scripts/recover-alibz-awaiting-data.py --refetch` (re-store under a
NEW dataset id via `Acquisition._finish_dataset`, re-score the completed batch
with `optimization._analyze`, refresh `_next_proposal`) and on
`scripts/deploy-alibz-fire-fix.py` (reservation lock, active-work refusal,
SQLite+config backup, hash pinning).

### Single source of truth
`--apply` **refuses** unless the live
`~/pantheum-I/pantheum/alibz/z300_calibration.py` hashes to the fixed module
`449dfb4981ecfc993c07489b65ee8241b794dc87894e921be4d0371c16217beb`, and uses that
live module for the conversion. For a dry-run preview before deploy, the wrapper
stages the local fixed module and passes `--preview-module` (reporting only,
never written to live). Conversion at both offsets goes through the same fixed
`pixels_to_wavelength(..., pixel_offset=)`, so BEFORE (0) and AFTER (−18) are one
code path.

**Discrepancy vs the brief:** the brief expected the dry run to report the fix
"not yet deployed". In fact the live module ALREADY hashes to the fixed version
(`449dfb49…`), so the tool converts with the *live* module and `fix_deployed:
true`. Step 1 appears to have been applied to moissanite's `~/pantheum-I` since
the 2026-09-24 deploy report (which recorded live `8a6135d0…`). I could not
verify the alibz services were restarted onto the fixed code; the file is in
place. `--apply` is therefore already permitted by the hash gate.

## Inventory (dry run, 2026-09-24; live moissanite, read-only)

127 live acquisitions (113 succeeded, 3 failed, 11 uncertain). Classification of
every live run (113 re-derive + 14 report-only = 127):

| class | count | action | meaning |
|---|---|---|---|
| a_api_native | 107 | re-derive | native/API dataset built at offset 0 (no `axis_check` in any test.json) |
| b_opal_legacy | 6 | re-derive | Opal legacy ZIP/gzip-JSON; opal-manifest `pixel_offset` None/0 (note says "offset 0"); native raw JSON present → re-derived on the same path |
| d_no_raw | 14 | report only | no raw pixels on disk (11 uncertain + 3 failed); 2 have a `test_id` (refetchable while the test is still on the analyzer), 12 do not — **not refetched here** |
| e_already_correct | 0 | skip | none: no stored dataset yet carries the fix |
| c resampled (23,250-pt) | n/a | superseded | the old linear 1/30 nm datasets share the same wrong map; every affected run has a native re-derivation, which supersedes them. They are separate dataset rows, left untouched (never deleted). The tool operates per-acquisition and does not enumerate them separately (see Unverified). |

**Re-derive: 113 runs → 107 completed batches across 9 sessions** (2 closed, 7
ready/complete). Closed sessions affected:
`opt-b61065d72c7941fcbfa5317099265cb2`, `opt-e301e73a4cf94c36beb98b44feee0f70`.

### Ar I NIR axis check, BEFORE (offset 0) vs AFTER (−18), all 113 re-derive runs
`argon_axis_offset_px` on the average shot, run at both offsets:
- BEFORE `offset_px`: min −18.75, median **−18.25**, max −18.0.
- AFTER  `offset_px`: min −0.5, median **−0.25**, max +0.25 — none ambiguous.

This reproduces the root-cause finding exactly: every stored native/legacy
dataset is ~18 px too blue; the fix brings it to ~0 px (residual −0.25 px = the
genuine registration shift). The 6 opal-legacy runs all read −18.25 → ~0.

Per-run table (run_id, session(s), created_at, class, raw sources, axis
before/after) is in the JSON report
`~/pantheum-rederive-bundle-20260924/dryrun-report.json` (108 KB, one row per
run). Representative rows:

```
run-eae9d54f… Fe  a_api_native  raw_json  axis -18.00 -> 0.00
run-12577962… V   a_api_native  raw_json  axis -18.25 -> (V_pure_run2)
run-39998be9… Fe  b_opal_legacy raw_json  axis -18.25 -> 0.00
```

## Predicted score changes (dry-run preview; a sample of batches)
BEFORE = `_analyze` on the current on-disk offset-0 native CSVs; AFTER =
`_analyze` on shot CSVs re-derived at −18 into a temp dir (no live write).
Composition is taken from `optimization._composition_fields(session)` — the same
input Pantheum's optimizer scores with.

| run | element | delay/period | score BEFORE | score AFTER | eligible |
|---|---|---|---|---|---|
| run-12577962 (V_pure_run2) | V | 20/10 | 0.461 | **1.170** | both true |
| seq-645563aa (V) | V | 5/10 | 0.449 | **1.188** | both true |
| run-eae9d54f (Fe) | Fe | 10/25 | 0.422 | **1.467** | both true |

Scores rise ~2.5–3.5× because the reference windows now land on the lines
(BEFORE grid 7914 pts incl. the 960–961 nm placeholder, max step 12.2 nm; AFTER
grid 5848 pts, placeholder excluded, max step 0.198 nm). Stored DB metrics are
preserved in batch provenance on `--apply` so before/after stays comparable.

## Dry-run output (top-level)
```
{"apply": false, "fix_deployed": true,
 "conversion_module": "live …/z300_calibration.py (449dfb4981ec)",
 "counts_by_class": {"a_api_native": 107, "d_no_raw": 14, "b_opal_legacy": 6},
 "rederive_runs": [113 ids], "affected_sessions": [9 ids],
 "closed_sessions_affected": [opt-b61065…, opt-e301e7…],
 "report_path": "~/pantheum-rederive-bundle-20260924/dryrun-report.json"}
```

## What `--apply` does (safety + mechanics)
1. Refuse unless live calibration hash == fixed (single source of truth).
2. Refuse if any acquisition queued/running/cancelling, any job queued/running,
   or any hardware op pending/uncertain (`active_work_reasons`).
3. Hold `reservation.lock` (bounded 60 s wait), back up SQLite + config to
   `~/pantheum-rederive-backup-<ts>/`, stop both alibz services.
4. Per run classified `rederive`, idempotently (skips if the pointed dataset's
   metadata carries `rederived_from` or test.json already has an aligned
   `axis_check`):
   - copy old `average.csv` / `shots/*.csv` / `test.json` / `manifest.json` to
     `<run>/rederive-backup-<ts>/` (old + new hashes recorded in the report);
   - rebuild them in place from `raw/shot-*.json` via `_finish_dataset` with the
     live fixed module — this creates a **new** dataset id (content changed →
     new `import_bytes` identity; old dataset row kept) and **queues a `preview`
     analysis job** for it the normal way (config `auto_engines: ['preview']`);
   - link the new dataset `{rederived_from, reason, axis_check}` and move the
     acquisition's `dataset_ids` pointer to it (old ids retained in `detail`);
   - re-score each completed batch (`_analyze` → new `metrics`, old metrics kept
     in batch `detail` provenance), then refresh the session:
     ready/complete → recompute `proposal`+`best`+state; **closed → recompute
     `best` only and stay closed** (a finished study is not silently reopened).
   DB writes for one run are one `BEGIN IMMEDIATE` transaction.
5. Restart both services (finally). Write a JSON report of every change.

Idempotent: a second `--apply` re-derives identical −18 content → same dataset
id (no duplicate) and the run is skipped anyway.

## What the owner runs, in order (from alibz/)
1. `bash scripts/deploy-alibz-pixel-offset.sh` then `--apply` (step 1 — already
   appears applied: live hash == `449dfb49…`; confirm services restarted).
2. `bash scripts/rederive-alibz-axis-offset.sh` (step 2 dry run; review the JSON
   report on moissanite).
3. `bash scripts/rederive-alibz-axis-offset.sh --apply` (step 2 execute).
4. Commit alibz (this tool + test); re-run the cal/ analyses on the corrected
   datasets (step 3).

## Rollback
- SQLite/config: restore `~/pantheum-rederive-backup-<ts>/alibz.sqlite` and
  `alibz.json` over `~/.local/state/pantheum/alibz/alibz.sqlite` /
  `~/.config/pantheum/alibz.json` (services stopped).
- Per-run files: copy `<run>/rederive-backup-<ts>/*` back over
  `average.csv`/`shots/`/`test.json`/`manifest.json`.
- New datasets and their queued jobs are additive; the old datasets and the old
  acquisition pointer are what the SQLite restore brings back, so restoring the
  DB is sufficient to fully revert. New dataset rows/originals can be left
  (unreferenced) or deleted by id.

## Test counts
- `tests/test_rederive_axis_offset.py`: **21 passed** (new; before: 0).
- Repo total collected: **605** (was 584 before this change; +21). The full
  suite runs >120 s and was not run to completion; I added files only and edited
  no existing code, so existing counts are unchanged. New file runs green in 0.02 s.

## Unverified
- `--apply` NOT executed (dry run only): re-derivation, re-score, service
  stop/start, and the SQLite/file backups are exercised only by the pure-logic
  tests and by reading the modelled scripts, not run live.
- Class (c) 23,250-pt resampled dataset rows are not separately enumerated/counted;
  the tool supersedes them via native re-derivation and leaves them untouched.
  If the owner wants an explicit inventory of those rows, that is a small add.
- The live module hashes to the fixed version but I did not confirm the alibz
  services are running that code (step-1 restart).
- The `recover-alibz-awaiting-data.py` model calls `_analyze(run_dir,
  session['element'])`, which raises "composition must be a non-empty list"; this
  tool uses `_composition_fields(session)[0]` (the live optimizer's own call at
  optimization.py:1085) instead. Flag if the recover path needs the same fix.
- 2 `d_no_raw` runs have a `test_id` and may be refetchable from the analyzer
  while the test persists; not attempted here.
```
