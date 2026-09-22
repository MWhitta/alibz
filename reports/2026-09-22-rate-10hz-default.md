# Stepped study defaults to 10 Hz — 2026-09-22

## Outcome

Patched and tested in the Pantheum source; **deployment is the user's step**
(`scripts/deploy-alibz-rate-10hz.sh`, then `--apply`), because the auto-mode
classifier refuses the agent's copy/deploy commands. No physical command issued;
no provider switch.

## Why

The delay/period ("stepped acquisition") study hard-coded 100 Hz
(`RATE_HZ = 100`, `PULSE_PERIOD = 10`). Z300-0915 refuses `pulsePeriod 10` with
"Invalid laser config parameters" (abort code 8) before firing (09:12 PDT,
run-6d9b08b3…). Its `hardware.cfg` z300 profile defaults to `pulsePeriod 100`
(10 Hz), which fired at 08:58 (test d45a1971…); Profile Builder offers at most
50 Hz on this unit. User decision: default the study to 10 Hz.

## Changes

- `pantheum/alibz/optimization.py`: `RATE_HZ = 10`, `PULSE_PERIOD = 1000 // RATE_HZ`
  (100 ms, also used for `cleaningPulsePeriod`); new `RATE_100HZ_PULSE_PERIOD = 10`
  keeps the existing `rate_100hz_verified` gate for any live session that
  explicitly carries `pulsePeriod 10` (shot-plan studies pass the operator's
  value through). `defaults()`, session rows and export provenance now report
  `rate_hz 10`, `pulsePeriod 100`.
  live before `f4f0fced…` → after `be02da8d…`.
- `web/alibz/app.js`: session "Rate" fallback 100 → 10 Hz (display only).
  live before `f2f9bb51…` → after `f7252cf9…`.
- `tests/test_alibz_optimization.py`: default session asserts pulsePeriod 100 /
  rate 10; the rate-gate test now builds a 100 Hz shot-plan session (refused with
  the 100 Hz message, session stays `ready`) and a default 10 Hz live session
  (refused only by the acquisition gates of the test service, never by rate).
- `DECISIONS.md`: 2026-09-22 entry.

## Verification (Mac, unsandboxed)

- optimization + metrics + acquire + checkouts: 111 tests OK.
- `node --check web/alibz/app.js`; `node --test tests/test_alibz_ui.cjs`: 36 pass.
- Full `python3 -m unittest discover -s tests`: see
  `reports/2026-09-22-rate-10hz-full-tests.log` **783 tests in 156.352 s, OK (skipped=26)**.
- Live Moissanite hashes of both files equal the "before" pins; services active.

## Known limitation (pre-existing, not changed)

`next_batch` persists the batch row before calling `acquisition.start`; any
`NotAllowed`/`Conflict`/`ValueError` from the acquisition gates then marks the
batch `failed` and the session `blocked` (`optimization.py` `_set_terminal`).
So a transient readiness refusal (analyzer unreachable, gantry not idle) kills
the session and a new one must be created. Worth relaxing later.

## After deploy

Refresh alibz, reconcile any uncertain hold, create a NEW session (old ones stay
`blocked`); the study will fire at pulsePeriod 100 with the vendor geometry.
