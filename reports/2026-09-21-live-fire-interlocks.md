# Live firing refused by the Z300's own interlocks — 2026-09-21

## Answer

The portal, network, and analyzer API are all working. Every live fire since the
16:28 PDT API recovery reached the analyzer and was refused by the analyzer in
under 100 ms with a vendor interlock. The handheld's Android log gives the reason
for each refusal. No fire, motion, cancel, or configuration change was issued by
this diagnosis; all analyzer access was read-only (logcat, dumpsys, ps, prefs).
No provider switch.

## Attempts (portal DB `acquisitions`, moissanite) vs analyzer log (clock 52 s behind)

| Portal dispatch (PDT) | Run | Params | Analyzer reply | Analyzer log (TestController, pid 17030) |
|---|---|---|---|---|
| 16:32:38 | run-1d88f968… (Fe_test session batch, d10/p25, 10 shots, [134,76,70]) | real batch | HTTP 521 TriggerLocked | (no line retained) |
| 16:52:30 | run-c4590706… | all-zero quick test | HTTP 520 LaserNotArmed | 16:51:38.984 `laser not armed`; RemoteService launched `.libzhardwarecommon.LaserArmActivity` on the handheld screen |
| 16:53:06 | run-f58fb86b… | all-zero quick test | HTTP 521 TriggerLocked | 16:52:14.765 `Trigger is locked` |
| 16:56:38 | run-1a0632de… | all-zero quick test | HTTP 521 TriggerLocked | 16:55:46.414 `Trigger is locked` |

All five live runs ever recorded (including the 14:57 one) are `uncertain` with no
test ID; API firing has never yet produced a test on this instrument.

## What the handheld was doing

- 14:43 boot. 14:54:53 Geochem Pro (pid 19075) started; WL calibration succeeded
  14:55:11. The 14:57:47 fire attempt logged **no** interlock warning, i.e. it passed
  both the armed and trigger checks with Geochem Pro in the foreground; it then died
  to the epoch-ms `/data/tests` crash (already fixed).
- 16:00–16:03 operator went to Android Settings, then Home. **16:09:55 the Geochem Pro
  process died.** From then until 16:56 the handheld sat on the launcher with no test
  app running.
- 16:06 RemoteService (pid 17030) was started by ADB (API recovery).
- 16:51:38 remote fire → `laser not armed`; the RemoteService itself pushed the
  **Laser Arm screen** onto the handheld. 16:51:52 `LaserArmFragment` ran its arm task
  (operator interaction) and warned `error loading laser pin file
  /storage/sdcard0/sciaps/slpp.bin: ENOENT`, so the laser PIN must be typed, not
  auto-loaded.
- 16:52:14 and 16:55:46 remote fires → `Trigger is locked` (laser now armed).
- 16:56:01 operator went Home, 16:56:04 relaunched Geochem Pro (pid 17255);
  `GeochemProActivity` is the resumed activity at 17:02. No fire attempt since.

## Interpretation

1. **520 LaserNotArmed**: the laser disarms (at least) when the app restarts or
   the unit reboots. A remote `/data/firetest` while disarmed makes the analyzer
   display its Laser Arm (PIN) screen; the operator must arm it on the handheld.
   Profile Builder's only response is the dialog "Please arm the laser in the LIBZ
   unit and try again." There is no API arm call (pinned JAR, whole-JAR sweep).
2. **521 TriggerLocked**: correlation, not vendor code — the trigger check passed
   at 14:57 with Geochem Pro in the foreground and failed at 16:32/16:53/16:56 while
   Geochem Pro was dead and the launcher was showing. `TriggerLockService` (pid 878,
   foreground service, bound by both RemoteService and geoChemMode) most likely locks
   the trigger unless a test activity is active. Prefs show trigger mode
   `TOUCH_START_STOP_TRIGGER`; unchanged since 2026-09-09.
3. **The portal's `triggerLocked` gate is not protective.** `/instrument/id`
   reported `triggerLocked=0` immediately before every refused dispatch and at
   16:57:16. The identity field does not track the state TestController enforces,
   so the readiness card can show "trigger unlocked" while fires are refused.

## Recommended bench procedure (operator)

1. Geochem Pro open on its START screen on the handheld (it is, as of 17:02).
2. Start the batch from the portal. If the handheld shows the Laser Arm screen,
   arm the laser there (type the laser PIN); then start the batch again.
3. Use a fresh optimization session; `opt-16247fdab67b41c1b2ffc0616a62ba96` and
   `opt-78c196a5e31d4f8ea914c1344ded0c9a` are `blocked`.
4. Note the three 16:52–16:56 runs carried all-zero parameters (delay 0, period 0,
   1 shot at [0,0,0], pulsePeriod 1). Check which control produced them before
   relying on a "quick test" button.

## Software follow-ups (not done)

- Surface the vendor 520/521 reason in the run panel as an operator instruction
  ("arm the laser on the handheld" / "open Geochem Pro START screen"), and stop
  presenting `/instrument/id` `triggerLocked=0` as proof the trigger is unlocked.
- The deployed runtime is `~/pantheum-I` (acquire.py sha256 143b3d29…) with
  `acquire.retrieval=deferred`; this is the deferred-retrieval branch that
  STATUS.md said was not deployed. Someone deployed it at 16:42:08 PDT
  (backup `pantheum-I-acquisition-backup-20260921T164147`). STATUS.md corrected.

## Evidence / access

- Portal DB: `~/.local/state/pantheum/alibz/alibz.sqlite` on moissanite.
- Analyzer log via opal: `ssh opal` + PowerShell `-EncodedCommand` running
  `C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe
  -P 5038 -s 0123456789ABCDEF logcat -d -v time` (adb shell is root on this unit).
- Analyzer clock: `Mon Sep 21 17:02:02 PDT 2026` at ~17:02:54 local.
- Vendor JAR strings: `LibzScan$1` "Please arm the laser in the LIBZ unit and try
  again."; `LIBZHttpClient` "Laser is not armed." / "Trigger is locked.".

## 2026-09-22 08:49 PDT batch: interlocks passed, analyzer laser controller crashed

Run `run-4259c0f14de04848bfca9236d06fd596` (session `opt-c91afd1adef14eb694ecd9471a31e3bf`,
"Fe pure metal", d10/p25, 10 shots, 100 Hz) dispatched 15:49:01.636Z and ended
`uncertain` at 15:50:31Z with "Z300 request timed out: /data/firetest" (90 s cap).
`hardware_resolved=0`; the hardware hold is pinned until the operator reconciles.

Analyzer log (clock ~52 s behind; adb root via opal):

- 08:47:19.718 `TriggerLockService: Received start id 12: Intent { act=TOGGLE_TRIGGER_LOCK ... bnds=[0,45][272,100] }`
  then `TestController: mIsTriggerLocked = false`. **The trigger lock is the padlock
  toggle in the app's status bar**, not a foreground-app rule. This corrects the
  interpretation above.
- 08:48:09.070 `LIBZService: control reg: 0x289`; 08:48:09.132
  `LIBZLaserController: Starting test TestConfig{mFPGAVersion=3.20.14, useGating=true,
  argonPreflush=0, intergrationDelay=10, intergrationPeriod=25, dataPulsePeriod=10,
  numDataPerLocation=10, cleaningPulsePeriod=20, numCleaningPerLocation=0,
  numShotAvgFPGA=1, rasterStartLocation=[0, 0, 0], rasterEndLocation=[0, 0, 0],
  rasterStepSize=0, numLocations=1, testIndex=0, numTestsInSeries=0, ...}`
- 08:48:09.140 `E/TestController: java.lang.ArithmeticException: divide by zero
  at com.sciaps.libz.hardware.datacollection.LIBZLaserController.start(LIBZLaserController.java:1108)
  at com.sciaps.android.libzhardwarecommon.webserver.TestController.handleTakeTest(TestController.java:324)`
- No HTTP response was sent (portal waited the full 90 s). No laser lines followed.
  `/instrument/id` still answers HTTP 200 afterwards (08:51 PDT).

This is the first request ever to reach the laser controller. It died on integer
division before firing. The only zero-valued integer in the config that the request
controls and the vendor default does not zero is `rasterStepSize=0` (float 0.0 in
the request, int 0 on the analyzer). `cleaningPulsePeriod` was raised 10→20 by the
analyzer (50 Hz cleaning cap), showing it normalises some fields but not step.

Analyzer mode-4 defaults (`GET /instrument/defaultParams?mode=4`, 08:50 PDT):
`startLocation [134,76,70]`, `endLocation [206,124,70]`, `stepSize 24`,
`numlocations 1`, `resetStage true`, `pulsePeriod 100`, `cleaningPulsePeriod 100`,
`argonpreflush 300`, d10/p25, 10 shots. These coordinates are the **internal
beam-steering raster stage**, not the gantry.

Portal sources of the zeros: `pantheum/alibz/z300.py` `DEFAULT_PRESETS` (both presets
use start=end=[0,0,0], stepSize 0.0) and `pantheum/alibz/optimization.py:242,260`
(forces `stepSize: 0.0`, `resetStage: False`) plus `:224` (requires start == end).
Yesterday's Fe_test session used [134,76,70] with step 0 and never reached the
controller (trigger locked), so step 0 has not been observed to work anywhere.

### Bench test that needs no deploy

The Acquire run panel POSTs the operator-edited parameters (`web/alibz/app.js:3004`,
`acquire.py:401`). After reconciling the hold, run one live batch from that panel
with the analyzer's own defaults (start [134,76,70], end [206,124,70], step 24,
numlocations 1, resetStage true, argon 300) and the desired d10/p25, 10 shots,
pulsePeriod 10. If it fires, the divide-by-zero is the raster geometry; then fix
`DEFAULT_PRESETS` and the optimization module (allow a nonzero step / vendor
geometry for the single-spot case) and redeploy. If it still crashes with step 24
and start == end, the divisor is the raster length and start must differ from end.

### 2026-09-22 09:00 PDT: divide-by-zero confirmed as stepSize=0; first successful API fire

| Portal run (PDT) | Geometry sent | Analyzer result |
|---|---|---|
| 08:53:46 run-97299b49… | start=end=[134,76,70], step 0, resetStage false, pulse 10 | 08:52:54 `Starting test … rasterStepSize=0` → `ArithmeticException: divide by zero` at `LIBZLaserController.start:1108`; no reply; portal timeout; uncertain |
| 08:58:34 run-e825d9f5… | start [134,76,70], end [206,124,70], step 24, resetStage true, pulse 100, argon 300 | 08:57:41 `Starting test … rasterStepSize=24, numLocations=1` → laser pump, FPGA polling, 08:57:44 `test finished`; analyzer returned test id `d45a1971-68aa-4fa6-b511-e75f3e62fc0c`; portal state `awaiting_data` (deferred retrieval) |

So `stepSize=0` crashes the analyzer regardless of location; a nonzero step with the
vendor default geometry fires. start==end with a nonzero step is still untested.
Note the working run used `pulsePeriod 100` (10 Hz) and `numShotAvgFPGA=0`; the
100 Hz timing study uses `pulsePeriod 10`, which the analyzer accepted at 08:52:54
(it failed later, on step), so pulse period is not implicated.

The optimization session builder cannot produce a working batch as deployed:
`optimization.py:224` rejects start != end, and `:242`/`:260` overwrite `stepSize`
with 0.0. The "same physical spot" rule is a portal invariant, not a vendor one;
with `numlocations=1` the analyzer fires at one raster-stage location whatever
`endLocation` says. Required change: keep the operator's (nonzero) step, and either
drop the start==end rule or verify start==end+step>0 on the bench first. Also
replace the zero geometry in `z300.py DEFAULT_PRESETS`.
