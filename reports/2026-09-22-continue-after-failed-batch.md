# Continue after a failed batch; cap retrieval retries; drop period 100 — 2026-09-22

## Outcome

Patched and tested in the Pantheum source; **deployment and the unblock are the
user's steps**. No physical command; no provider switch.

## Changes

- `pantheum/alibz/optimization.py` (live before `2c96f4a6…`): new
  `_fail_and_continue`; `_reconcile` routes acquisition `failed`, wrong shot
  count, and ineligible metrics through it: batch `failed` (detail carries the
  run's own reason), `total_shots` credited, `_next_proposal` recomputed from
  scored batches, session `ready`/`complete`. If no batch has scored, the
  session blocks (nothing to step from). `uncertain`/`cancelled` and offline
  analysis exceptions still block.
- `pantheum/alibz/retrieval.py` (live before `fbd2cd96…`): `max_attempts`
  (default 40, 1–10000) in `retrieval_config`; after that many failures the run
  is set `failed` with "Opal retrieval gave up after N attempts: …" (test id
  kept, claim cleared, no replay). `fetch_archive` now reads stderr (bounded 4 KB
  ring) and surfaces only lines prefixed `z300 ingestion: ` (≤200 chars) in the
  error, e.g. "Opal retrieval command failed: legacy ZIP does not contain the
  exact expected shots and average". Run detail shows "attempt k of N".
- `scripts/unblock-alibz-session.py` (+ `.sh`): refuses unless the deployed
  optimizer has the new rule; marks the run `failed` (guarded on
  `state='awaiting_data' AND retrieval_claim IS NULL`, retried through a
  worker attempt window), removes a period from the session grid, then GETs the
  session through the portal socket so the serve process reconciles, and prints
  the resulting state/proposal.
- `scripts/deploy-alibz-continue.sh`: manifest-pinned deploy of the two modules.
- Tests: `tests/test_alibz_optimization.py`
  `test_failed_batch_counts_as_tested_and_the_search_continues` (acquisition
  failure and ineligible metrics continue; the failed condition is excluded from
  proposals; an uncertain outcome still blocks);
  `tests/test_alibz_retrieval.py`
  `test_attempt_cap_ends_the_run_failed_with_the_ingest_reason` (reason
  surfaced, raw stderr not stored, run failed after the cap, config bound).
- `DECISIONS.md`, `docs/acquisition-optimization.md`.

## Verification

- Focused: optimization + retrieval + acquire + checkouts: 121 tests OK.
- Full suite: `reports/2026-09-22-continue-full-tests.log` **800 tests in 153.528 s, OK (skipped=26)**.
- Live hashes of both modules equal the source "before" pins (the parallel
  ingest task's 10:05 deployment is the base).

## Order of operations for the user

1. `scripts/deploy-alibz-continue.sh` then `--apply`.
2. `scripts/unblock-alibz-session.sh --run run-ee6913a925c84e1f817f3142ab985cb6 --session opt-981ee9c278c04b0a9aaa63db74b98bb6 --drop-period 100 --reason "Analyzer stored 1 of 10 shots: intergrationPeriod 100 is not executable on Z300-0915."`
   (dry run), then the same with `--apply`. Expected: run failed, session
   `ready`, periods `[10,25,50]`, proposal = an untested neighbour of the best
   of the five scored batches.
3. Continue the study from the portal.

## Deployed by the user; the cap and the continue rule acted on their own

At the user's unblock dry run (after `deploy-alibz-continue.sh --apply`), the
live worker had already ended run-ee6913a9… as `failed` at attempt 62 with
`retrieval_error` "Opal retrieval command failed: legacy ZIP does not contain the
exact expected shots and average", and the optimizer had continued: session
`opt-981ee9c2…` `ready`, proposal delay 10 / period 50 ("all current
best-condition axis neighbors were already tested"). Only the grid change
remained; `unblock-alibz-session.py` now applies just that (plus a reconcile)
when the run is already terminal. The current proposal (10/50) is inside the
reduced grid, so it stays valid.

Applied 12:1x PDT: periods now `[10,25,50]`; session `ready`; best = 5/10 @ 0.3845
(run-56a8de4a…); proposal 10/50; batch states 5 completed / 1 failed.
