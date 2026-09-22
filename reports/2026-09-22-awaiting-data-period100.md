# Run stuck in awaiting_data: the analyzer stored one shot at period 100 — 2026-09-22

## Answer

`run-ee6913a925c84e1f817f3142ab985cb6` (session `opt-981ee9c2…`
"Fe_Aesar_automated_cal_run1", batch delay 5 / period 100, dispatched 11:15:26 PDT,
analyzer test `8606e2f7…`) is stuck because the instrument recorded **one shot,
not ten**, and the Opal retrieval (deployed by the parallel ingest task at
10:05 PDT, `acquire.retrieval=opal_database`) correctly refuses to complete a
ten-shot run from a one-shot bundle — but it retries every 15 s forever and
nothing can end the wait.

## Evidence

- Analyzer log (pid 884, clock 52 s behind): 11:14:35.453 `Starting test
  TestConfig{… intergrationDelay=5, intergrationPeriod=100, dataPulsePeriod=100,
  numDataPerLocation=10 …}` → `Go` 11:14:35.921 → exactly one `shot:buffer
  [169:1]` at 11:14:37.281 → `Stop FPGA polling` 11:14:37.625 → `test finished`
  → `saved test: 8606e2f7…`. The five preceding tests (periods 10/25/50) each
  show ten `shot:buffer` groups in the same ~1.7 s window.
- Instrument DB doc for `8606e2f7…`: `shotTable {"all": "0b14d793…"}` (legacy
  ZIP form), 32,042 B vs 170,808 B for the ten-shot period-50 test. The ZIP
  holds exactly two members: `0` and `-1` (15,914 B each).
- Data API: `/data/shotspectrum?test=8606e2f7…&shot=-1` and `shot=0` → 200;
  shots 1–9 → 404.
- No test with `intergrationPeriod 100` exists anywhere in the instrument's
  history; the vendor UI never used it. Every period-10/25/50 test today stored
  ten shots. So period 100 is an instrument-side limit (one shot captured), not
  a transport problem.
- Retrieval: `retrieval_attempts` 19+ at 15 s cadence, `retrieval_error`
  "Opal retrieval command failed" (stderr is discarded by
  `retrieval.fetch_archive`). Running the configured command by hand:
  rc=3 `z300 ingestion: legacy ZIP does not contain the exact expected shots and
  average` (PendingData; `scripts/z300_opal_ingest.py:483`). My first manual run
  collided with a worker attempt and got `another ingestion owns this run`
  (the per-run lock works; nothing stale holds it).
- `pantheum/alibz/retrieval.py` has no attempt cap; `Acquisition.cancel` only
  accepts queued/running runs; the optimizer treats `awaiting_data` as active,
  so session `opt-981ee9c2…` stays `acquiring`.

## What it means

- The session's five completed batches (10/25, 5/25, 5/10, 10/10, 5/50; ten
  native shots each) are fine. The stalled batch will never complete.
- The grid `[10,25,50,100]` includes a period the instrument cannot execute; the
  search will keep proposing (x,100) neighbours and each will stall the same way.
- Once the run is marked terminal, current optimizer semantics set the session
  `blocked` (any non-succeeded terminal batch stops the search) and a new
  session would restart the grid from scratch.

## Options (not applied)

1. **Unblock only**: mark the run `failed` ("analyzer stored 1 of 10 shots") by
   a guarded DB update (`state='awaiting_data' AND retrieval_claim IS NULL`);
   the retrieval loop stops, the next session poll reconciles, session →
   `blocked`. Then create a new session without period 100 (loses the 5
   batches' search state, though their datasets/scores remain).
2. **Unblock and continue** (code change + deploy): in `Optimization._reconcile`
   treat a `failed` batch (physical outcome known) as "condition tested,
   unusable": mark the batch failed, recompute the proposal from the completed
   batches, session → `ready`; keep `uncertain`/`cancelled` → `blocked`. Also
   exclude the failed condition's period from further proposals or remove 100
   from the grid; and cap retrieval attempts (e.g. mark the run failed after
   N PendingData retries so it cannot spin forever). `retrieval.py` and the
   ingest script belong to the parallel task's deployment; touching them needs
   coordination.
3. Data-API retrieval (`retrieval=data_api`) would have ended this run as
   `failed` at the shot-list stage instead of spinning; it does not fix period 100.
