# Verified allowlist, non-blocking refusals, retrieval hardening, data-API path, raster batches — 2026-09-22

## Outcome

Implemented and tested in the Pantheum source (`optimization.py`, `acquire.py`,
`retrieval.py`, `tools/z300_fb_decode.py`, tests, docs, DECISIONS). **Deploy and
the retrieval switch are the user's steps**; both restart the alibz services
and refuse while an acquisition is active. No physical command; no provider switch.

## What changed

1. **Verified allowlist** (`Optimization.verified_conditions`, `_require_verified`).
   Source of truth is the acquisition ledger: a succeeded live run whose stored
   `shots` equals its requested `numShotsPerLocation × numlocations`, keyed by
   delay/period/pulsePeriod with the largest shot count. Live delay/period
   sessions need every grid pair verified with ≥ 10 shots; live shot plans need
   their (delay, period) verified with ≥ `shots_per_location`. Checked at
   creation (ValueError listing the missing pairs and how to verify them: one
   live 10-shot Acquire-panel batch) and at dispatch (NotAllowed before any
   batch row exists, session stays `ready`). `optimization.require_verified_conditions=false`
   disables it; simulate sessions are exempt. `defaults.verified_conditions`
   exposes the list. Live ledger today: (5,10) (5,25) (5,50) (10,10) (10,25)
   (10,50) (20,25) at pulsePeriod 100. Period 100 and (20,10)/(20,50)/(50,x) are not.
2. **Refusals do not block.** In `next_batch`, a NotAllowed/Conflict/ValueError
   raised before an acquisition record exists deletes the persisted
   `dispatching` batch row and restores the session to `ready` with the same
   proposal; the reason ("hardware checkout expired; gantry is not idle", …)
   goes into the session detail. Refusals after a record exists keep the old
   terminal handling.
3. **Retrieval hardening.** `RetrievalError.refused` marks failures where the
   Opal producer itself classified the data (`z300 ingestion: …`); only those
   increment the new `acquisitions.retrieval_refusals` column and count toward
   `max_attempts`. Transport failures (SSH, timeout, size) back off
   exponentially (retry_seconds × 2ⁿ, capped 600 s) and never give up. Run
   detail shows "refusal k of N" or "transport failure n, retry in s".
   `recover-alibz-awaiting-data.py --refetch` now also re-fetches `failed` runs
   over the data API and re-scores their batches; `--apply --enable-data-api`
   works with no pending runs (config-only).
4. **API-path native grid.** The study is to run with `acquire.retrieval=data_api`
   (native grid straight from `/data/shotspectrum`, one producer). The Opal
   FlatBuffers decoder is archived at `pantheum-I/tools/z300_fb_decode.py`
   (byte-identical body to alibz `scripts/z300_fb_decode.py`, sha `3130e640…`).
5. **Raster batches.** `raster_sites(params)` enumerates the session window
   (`startLocation`→`endLocation` at `stepSize`, inclusive; ≤ 50 per axis) in
   generalized-Hilbert order (`_gilbert2d`, Červený's construction: every cell
   once, unit steps, any rectangle). Batch `position` p fires site p mod K with
   `numlocations 1`, `startLocation` = site, `endLocation` = site + step (the
   verified vendor shape); `location` records the site; details say "site k of
   K" and the lap once sites repeat. Vendor window 134..206 × 76..124 at 24 →
   4 × 3 = 12 sites. Same-spot shot plans stay on site 1.

## Verification

- Focused: optimization + retrieval + acquire + checkouts: **126 OK**
  (new: gilbert coverage/adjacency over 7 rectangles; 14-batch raster walk with
  wrap; verified gating at create and dispatch; refusal undo; transport backoff;
  refusal-only cap).
- Full suite: `reports/2026-09-22-raster-verified-full-tests.log` **820 tests in 159.155 s, OK (skipped=26)**.
- Live hashes of the three modules equal the git-HEAD "before" pins.

## Order of operations for the user (between batches)

1. `scripts/deploy-alibz-raster.sh` (dry run) then `--apply`.
2. `scripts/recover-alibz-awaiting-data.sh --apply --enable-data-api`
   (switches `acquire.retrieval` to `data_api`, backs up config, restarts).
3. Session `opt-b61065d7…` (delays 5/10/20, periods 10/25/50): proposals at
   (20,10) or (20,50) will be refused until each is verified by one live
   10-shot batch from the Acquire panel; the session stays `ready`.
4. New sessions raster automatically; existing sessions raster from their next
   batch (position continues, so site = position mod 12).

## Caveats

- Site-to-site heterogeneity of the Fe standard now enters the comparison in
  place of crater evolution; the metrics' repeatability CV is within a site.
- `endLocation = site + step` for the last column/row reaches 230 × 148, inside
  the stage's 550 × 550 travel; other windows are not checked against travel
  limits by the portal (the analyzer clamps or refuses).
- The verified query relies on SQLite JSON1 (3.37 on Moissanite; fine).
