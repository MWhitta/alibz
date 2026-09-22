# Raster-step fix for optimization sessions — 2026-09-22

## Outcome

Patched in the Pantheum source (`~/Projects/github/pantheum-I`), fully tested,
and **deployed by the user at 09:11:16 PDT** with `scripts/deploy-alibz-raster-step.sh
--apply` (the auto-mode classifier had refused the agent's own copy/deploy commands).
Backup `/home/mwhittaker/pantheum-fire-fix-backup-20260922T091116`; installed
hashes matched the manifest; both services active. Post-deploy check at 09:20 PDT:
portal `live_allowed=true`, no blockers, analyzer reachable, gantry Idle, checkout
held; `/api/acquire/status` presets and `/api/acquire/settings?mode=4` now serve
start [134,76,70], end [206,124,70], step 24, resetStage true.
No physical command was issued; no provider switch.

## Why

Analyzer log (root ADB via opal) proved `stepSize=0` throws
`ArithmeticException: divide by zero` in `LIBZLaserController.start:1108`
before firing, at both `[0,0,0]` (08:48:09) and `[134,76,70]` (08:52:54); the
analyzer never replies and the portal run ends `uncertain` after 90 s. The
analyzer's mode-4 default (`[134,76,70]`→`[206,124,70]`, step 24, one location,
resetStage true) fired from the Acquire run panel at 08:57:41 and finished in
2.5 s as test `d45a1971-68aa-4fa6-b511-e75f3e62fc0c` (run-e825d9f5…,
`awaiting_data`). Session creation then failed on the portal's own rule
"startLocation and endLocation must identify the same physical spot", and even a
compliant session would have sent the forced `stepSize 0.0`.

## Changes (source, `git diff` vs live "before" hashes)

- `pantheum/alibz/z300.py`: `validate_raster_params` rejects `stepSize <= 0`
  with the reason; both `DEFAULT_PRESETS` carry the analyzer default geometry.
  live before `a01ac059…` → after `eeb08470…`.
- `pantheum/alibz/optimization.py`: dropped the start == end rule (`:224`) and the
  forced `'resetStage': False, 'stepSize': 0.0` in both study types (`:242`, `:260`);
  `numlocations` is still forced to 1 (the single-spot guarantee).
  before `3f199210…` → after `f4f0fced…`.
- `web/alibz/app.js:2821`: form refuses step <= 0 ("the analyzer crashes on 0").
  before `a8afa004…` → after `f2f9bb51…`.
- Tests: `tests/test_alibz_optimization.py` (geometry passes through; step 0 and
  negative refused), `tests/test_alibz_acquire.py` (presets carry [134,76,70]/24;
  zero step rejected), `tests/test_alibz_ui.cjs` fixture uses the vendor geometry.
- `DECISIONS.md`: 2026-09-22 entry.

## Verification (Mac, unsandboxed)

- `python3 -m unittest tests.test_alibz_optimization tests.test_alibz_optimization_metrics`: 26 OK.
- `python3 -m unittest tests.test_alibz_acquire tests.test_alibz_checkouts`: 84 OK.
- `node --check web/alibz/app.js`; `node --test tests/test_alibz_ui.cjs`: 36 pass, 0 fail.
- `python3 -m unittest discover -s tests`: **782 tests, OK, 26 skipped** in 156 s
  (`reports/2026-09-22-raster-step-full-tests.log`; Moissanite's runtime showed
  11 skips on 2026-09-21, the extra skips are optional deps absent on the Mac).
- Live hashes of the three files on Moissanite equal the source "before" hashes
  (checked 09:0x PDT), so the manifest-pinned deploy will not clobber anything.

## Deploy

`scripts/deploy-alibz-raster-step.sh` copies the three files plus a manifest
(before/after/config SHA-256) and `deploy-alibz-fire-fix.py` to
`moissanite:~/pantheum-raster-step-bundle-20260922`, then runs the deploy script:
reservation lock, refuses on active acquisitions/jobs/unresolved hardware ops,
backs up files+config+SQLite, stops/starts `pantheum-alibz-worker` and
`pantheum-alibz`, verifies installed hashes. Note run-e825d9f5… is
`awaiting_data`, which is not an active state, so the check should pass.
After deploy: refresh alibz, create a new session (existing ones are `blocked`),
the form's default geometry is now the analyzer's. Untested: start == end with a
nonzero step; the vendor default (start != end, one location) is the known-good.

## 09:12 PDT first session batch after deploy: geometry accepted, rate rejected

Run `run-6d9b08b3d7444e8d8421528a13c9e908` (session `opt-ce76767253…`) sent the
vendor geometry (step 24) with `pulsePeriod 10` (the study's hard-coded 100 Hz).
Analyzer replied HTTP 500 in 0.77 s with body `"11c87a24-251f-4a09-ae0f-f2606fd74531"`.
Analyzer log (09:11:49.984 analyzer time): `Starting test … dataPulsePeriod=10,
numShotAvgFPGA=1, rasterStepSize=24 …` → `E/LIBZLaserController: test configuration
is invalid` → `TestSession.abort with code: 8` → `Invoking LIBZ Service halt` →
`java.lang.Exception: Invalid laser config parameters at
LIBZLaserController.executeTask:1452`; then `saved test: 11c87a24…` (an empty test
record). No laser-pump lines, so nothing fired. Run `uncertain`,
`hardware_resolved=0`: operator must reconcile before the next dispatch.

The only material differences from the 08:57 run that fired: `dataPulsePeriod`
10 vs 100 and the analyzer-derived `numShotAvgFPGA` 1 vs 0. The analyzer clamps
`cleaningPulsePeriod` 10→20 but passes data pulse 10 through and then rejects it.

Vendor UI evidence (pinned Profile Builder JAR, `RasterSettingPanel`): the test-rate
combo is built from `pulseOptions` gated by `enable50Hz`/`PULSE_50Hz` and
`Instrument.supports50HzData`; its error text is "Invalid Settings. Try adjusting
shot per location, cleaning shot per location, data rate, cleaning rate." The
class's constant pool holds 50 and no 100, and bytecode `bipush` operands include
10 and 50 but never 100. So **the vendor UI offers at most 50 Hz (pulsePeriod
20)**; the earlier finding that "100 Hz encodes pulsePeriod 10" was an encoding
fact, not evidence that 100 Hz is selectable or accepted. The private config has
`rate_100hz_verified: true`, which only unlocks the encoding; the analyzer itself
refuses it. `optimization.py` fixes `RATE_HZ=100`, `PULSE_PERIOD=10` for the
delay/period study; a shot-plan study uses the operator's pulsePeriod.

Untested and decision-pending: 50 Hz (pulsePeriod 20; the vendor has special
shot-averaging handling at 50 Hz) vs the proven 10 Hz (pulsePeriod 100).

### On-device configuration (read-only via root ADB, 09:35 PDT)

- `/storage/sdcard0/sciaps/xyzstage.json`: `startLocation [134,76,70]`,
  `endLocation [206,124,70]`, `rasterStepSize 24`, `maxTravel [550,550,374]`;
  the WL-calibration raster uses step 6. Confirms the geometry now in the presets
  is the instrument's own.
- `/storage/sdcard0/sciaps/hardware.cfg`: `mAnalyzerType z300`; the z300 profile's
  `mDefaultStandardPulsePeriod` is `100` (10 Hz); the only other value in any
  profile is `20` (50 Hz, z200/z200C). No profile uses 10 (100 Hz).
- `/storage/sdcard0/sciaps/laserconfig.json` (2023-04-13) holds spotlight and
  pump-time safety thresholds only (`pumptimeThreshold 500`, `dutyCycleBudget 1`);
  no rate limit. `laserconfigs_MW/` holds "Enabled"/"Disabled" variants of it
  from 2023-04-14.
- `instrumentid.txt`: `Z300-0915`.

Together with the Profile Builder combo evidence, the supported rates on this
unit are 10 Hz (pulsePeriod 100, proven) and at most 50 Hz (pulsePeriod 20,
untested); 100 Hz (pulsePeriod 10) is refused as "Invalid laser config parameters".
