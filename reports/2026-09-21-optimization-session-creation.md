# Optimization session creation without physical enablement

## Outcome

Live optimization session creation now persists a ready session even when the
private 100 Hz commissioning flag is false. Creation remains guarded by loaded,
fresh, idle status and an owned hardware checkout in the UI. All physical gates
remain on `next_batch`: live enablement, the private 100 Hz commissioning flag,
fresh operator confirmation, ownership, stale/busy checks, persistence-before-
dispatch, cancellation, and locks were not changed.

No provider switch occurred.

## Changes

- `pantheum/alibz/optimization.py:134-163`: removed the
  `optimization.rate_100hz_verified` refusal from `create`. Session validation,
  idempotency, locking, and persistence are otherwise unchanged.
- `pantheum/alibz/optimization.py:389-392`: audited the retained live-rate gate
  in `next_batch`; a false/missing commissioning flag still raises `NotAllowed`
  before a batch record or acquisition dispatch is created.
- `web/alibz/app.js:3121-3123`: `optimizationCreateBlocked` now applies only the
  shared loaded/stale/busy/status/owned-checkout guards. It no longer treats the
  live-rate commissioning flag as a blocker for inert session creation.
- `web/alibz/app.js:3124-3131`: limited the live-rate tooltip to an existing live
  session action. The Create button no longer presents a firing-gate explanation.
- `web/alibz/app.js:3114-3119`: audited the unchanged next-action gates. Existing
  live sessions still require both `acquireStatus.live_allowed` and
  `live_rate_encoding_enabled` before Next batch is enabled.
- `tests/test_alibz_optimization.py:102-111`: revised the existing live-rate
  regression to create a live session with the default false flag, assert there
  are zero acquisition runs, assert `next_batch` raises `NotAllowed`, assert
  there are still zero runs, and assert the session remains `ready`.
- `tests/test_alibz_ui.cjs:518-552`: revised the existing inert-creation UI
  regression to use a live session with owned hardware,
  `live_allowed=false`, and `live_rate_encoding_enabled=false`; it asserts Create
  is enabled, Next batch is disabled with the live-firing reason, creation sends
  exactly one session POST, and no next/acquisition POST is sent.
- `docs/acquisition-optimization.md:85-89`: clarified that session creation is
  inert and may occur while physical acquisition is disabled, while every live
  batch still requires the private rate commissioning flag.

## Verification

Baseline:

- `PYTHONPATH=. python3 -m unittest tests.test_alibz_optimization -q` — 17 tests
  discovered; the 16 non-HTTP tests passed and the single HTTP test could not
  bind localhost in the restricted sandbox (`PermissionError: [Errno 1]`).
- `node --check web/alibz/app.js` — passed.
- `node --test tests/test_alibz_ui.cjs` — 31 passed, 0 failed.

After the change:

- `PYTHONPATH=. python3 -m unittest tests.test_alibz_optimization.OptimizationTests -q`
  — 16 passed, 0 failed in 4.073 s. This includes the revised creation/dispatch
  regression.
- `node --check web/alibz/app.js` — passed.
- `node --test tests/test_alibz_ui.cjs` — 31 passed, 0 failed in 197 ms. This
  includes the revised Create-enabled/Next-disabled regression.

The focused before/after test count is unchanged: 16 backend unit tests and 31
Node UI tests pass. The initial full targeted Python module additionally contains
one localhost HTTP test that the sandbox could not execute.

## Scope and evidence

The runtime change removes three gate lines from `optimization.create` and
narrows two frontend expressions. The retained backend dispatch gate is visible at
`pantheum/alibz/optimization.py:389-392`; the retained frontend dispatch gates
are visible at `web/alibz/app.js:3114-3119`. No next-batch dispatch code,
physical enablement, confirmation handling, cancellation, reservation, or lock
code was changed. The checkout is a deployed-runtime snapshot without `.git`
metadata, so a repository-native `git diff` was unavailable; the exact touched
lines and SHA-256 hashes below provide the handoff evidence.

- `pantheum/alibz/optimization.py`: `8dcb62cf498600d35f1f54d2448e6ce82d730f1294e6350de28137b0e7a8fb51`
- `web/alibz/app.js`: `52f50ae81a7bbea087ca4c4ccf3b73932dad95384eb3e668701af5691f2f8ca3`
- `tests/test_alibz_optimization.py`: `6fbfcee2b504d892cc8199377b71db5513458314532718850a0ff8c8390edcc5`
- `tests/test_alibz_ui.cjs`: `8ac1771465c9ef893311f5a1a4c05ad5989e0bf6f5881fbbbc5cd60d94b62cf2`
- `docs/acquisition-optimization.md`: `8a54d68b822b11febc1430e808982d0fd518c9807abc93332e437c31c64a322e`

## Not verified here

- Browser interaction and deployment were intentionally left to the parent
  session, as specified in the brief.
- The repository full suite and pymatgen-enabled full suite were intentionally
  not run in this worker checkout; the parent owns those checks per the brief.
- The localhost HTTP route test was not rerun outside the restricted sandbox.
