# Spectrometer frame drops: batches tolerate ≥ 6 stored spectra — 2026-09-22

## Finding

From 12:16 PDT the analyzer's spectrometer readout logs `E/onyx: checksum
mismatch! computed/read …` followed by `error. event type: 7` during a shot; the
laser fires every shot (ten `shot:buffer` events) but the saved test holds
fewer spectra and the missing index answers 404 on `/data/shotspectrum`.
Tests 00537a8a… 8/10, 35c4b74b… 9/10, c665693c… 9/10 stored; 1721c588… 10/10.
None of the morning's five tests were affected. Solenoid 39 °C, uptime 3.5 h,
no kernel USB fault. Root cause open; a power-cycle and re-observation are the
next bench step. Every pipeline stage assumed exactly ten spectra, so these
batches failed and, under `data_api`, would have ended `uncertain` with the
hardware hold pinned.

## Change (user decision: "> 5 is ok")

- `acquire.py`: `MIN_STORED_SHOTS = 6`, configurable `acquire.min_shots`
  (1–600). `_run_live` fetches spectra until the first 404 (after at least one)
  and keeps the contiguous set; a run whose count is below the request and
  below the minimum ends `failed` with the test id and the hardware hold
  resolved; otherwise it succeeds with `shots` = stored, `requested_shots` and
  `dropped_frames` in test.json/manifest.json and the drop in the run detail.
- `optimization.py`: batches with `min_shots..10` spectra are scored and
  credited with their actual count; details say "(n dropped by the instrument)";
  fewer → failed and the search continues. Metrics `shot_count` must equal the
  stored count.
- `optimization_metrics.py`: `analyze_batch(min_shots=6)` accepts a contiguous
  `shot-0..shot-(n-1)` set (gaps rejected); a line needs ⌈0.8 n⌉ detections;
  result carries `shot_count` and `min_detected_shots`.
- `retrieval.py`: Opal manifests may carry `min(6, expected)..expected` shots.
- `z300.py` fake server: `stored_shots_limit` for tests.
- `scripts/recover-alibz-awaiting-data.py`: partial tests ≥ 6 are stored with
  `requested_shots`/`dropped_frames`; shorter ones are reported and skipped.
- A short test never verifies a condition (verification still requires a fully
  stored run). DECISIONS and docs updated.

## Verification

- Focused (acquire, metrics, optimization, retrieval, checkouts): **137 OK**,
  including: live run with 8/10 stored succeeds with dropped_frames 2 and 3/10
  fails with test id; 8-shot batch scored / 5-shot batch fails-and-continues;
  metrics accept 8, reject 5 and reject gaps; Opal bundle of 8/10 completes,
  5/10 refused.
- Full suite: `reports/2026-09-22-dropped-frames-full-tests.log` **824 tests in 161.770 s, OK (skipped=26)**.
- Live hashes of the five modules equal git HEAD; the wrapper pins them.

## Order of operations for the user (between batches)

1. `scripts/deploy-alibz-dropped-frames.sh` then `--apply`.
2. `scripts/recover-alibz-awaiting-data.sh --run run-bb777df5e62c4fc18c3e2c33cbb6c766 --apply`
   (9 stored spectra → succeeded, batch scored, session `opt-b61065d7…` continues).
   Optionally `--refetch --run run-8264fabf… --run run-d99401be…` for the two
   failed 8- and 9-shot tests (their sessions: one blocked, one already continued).
3. `scripts/recover-alibz-awaiting-data.sh --apply --enable-data-api` (switch
   retrieval to the analyzer data API; now safe with dropped frames).
4. Power-cycle the analyzer at the next natural pause and watch `E/onyx` in the log.
