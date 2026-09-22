# Deferred-retrieval runs: the data API works; recovery script ready — 2026-09-22

## Finding

The 2026-09-21 conclusion that "every /data/* path kills the analyzer's HTTP
server" was the epoch-millisecond `/data/tests` crash (fixed 09-21 16:28). With
the fixed cursor handling the data API is usable on Z300-0915 v2.19.7:

- `GET /data/shotspectrum?test=d45a1971…&shot=0`: HTTP 200 in 0.147 s, gzip
  13,736 B → 86,630 B JSON with `knots[5]`, `wlCalibrations[4].pixToNm.coefficients[4]`,
  `pixels[4][2066]`, exactly what `Acquisition._sorted_unique_csv` expects; it
  converts to a 23,250-line CSV from 186.06 to 960.99 nm. `/instrument/id`
  answered HTTP 200 immediately afterwards (and after 33 more fetches).
- `GET /data/tests?since=1790000000` (seconds) lists today's eight tests.
- The instrument DB (`/storage/sdcard0/libzdata/libzdb.cblite`, read with the
  device's own `/system/xbin/sqlite3` over root ADB) holds `type=test` docs for
  `d45a1971…`, `63f26855…` and the aborted `11c87a24…`, each with
  `config.rasterStart [134,76,70]`, delay 10, period 25, 10 shots, argon 300.

So `acquire.retrieval=deferred` is no longer necessary; `data_api` closes the
loop (fire → test list → 10 shots + average → dataset → offline Fe metrics →
next proposal) without any bench step.

## Recovery tooling

`scripts/recover-alibz-awaiting-data.py` (+ `.sh` wrapper that copies it to
`moissanite:~/pantheum-recover-20260922/recover.py` and runs it from `~/pantheum-I`):

- dry run (default): fetches average + all shots for every `awaiting_data` run
  and converts them with the deployed code; no writes, no service change.
- `--apply`: refuses if any acquisition is queued/running or a hardware op is
  pending; backs up config + SQLite to `~/pantheum-recover-backup-<ts>`; stops
  both alibz services; for each run writes `raw/`, `shots/shot-N.csv`,
  `average.csv`, `test.json`, `manifest.json` via `Acquisition._finish_dataset`
  (same as a live data_api run), imports the average as a dataset with
  `metadata.acquisition`, sets the run `succeeded` (test_id, dataset_ids, shots);
  re-queues any `awaiting_data` optimization batch, sets its session `acquiring`
  and calls `Optimization._reconcile`, which scores the ten shots and proposes
  the next condition; then restarts the services.
- `--enable-data-api` (with `--apply`): rewrites `acquire.retrieval` to
  `data_api` in `~/.config/pantheum/alibz.json` (backed up) before restart.

Dry run at 09:4x PDT: 
```
run-e825d9f5… test d45a1971… shots 10 csv_lines 23250 fetched
run-a50ac639… test afb5b8b9… shots 10 csv_lines 23251 fetched
run-eae9d54f… test 63f26855… shots 10 csv_lines 23251 fetched   (batch of opt-b177bed6…, to rescore)
```

## Not done / caveats

- `--apply` was not run by the agent (live DB/config change + service restart is
  the user's step). After it, the session `opt-b177bed6…` should read `ready`
  with a proposal, or `failed` if the ten shots are not eligible for scoring.
- In `data_api` mode the readiness probe calls `/data/tests?since=0` on each
  status refresh (~1.3 s) and the post-fire poll pages through the whole test
  list (~8 pages) before the new id appears; both are within the 600 s timeout
  but slower than deferred mode. Watch the first live batch's timing.
- Memory `z300-data-path` corrected (data API usable; ADB root read path).

## Applied by the user, 09:40:22 PDT

`scripts/recover-alibz-awaiting-data.sh --apply --enable-data-api` (the two
earlier runs had already been recovered separately, so only run-eae9d54f… was
awaiting):

- run-eae9d54f… → `succeeded`, 10 shots, dataset `0982aedefc2b7f0833adb766`.
- Batch → `completed`, score 0.2863987; session `opt-b177bed6…` → `ready`,
  proposal delay 5 / period 25 ("untested axis neighbor of the best measured
  condition 10/25").
- Config `acquire.retrieval` → `data_api`, sha256 `d183eb11…`; backup
  `~/pantheum-recover-backup-20260922T094022`; both services active.
- Post-check 10:0x PDT: `live_allowed=true`, `data_api_reachable=true`,
  all three fired tests are `succeeded` runs with datasets
  (`2679219a…`, `4f678d79…`, `0982aede…`). Hardware checkout had expired.
