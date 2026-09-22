# SciAps Z300-0915: device, API and pipeline notes

Everything below was verified on the bench on 2026-09-21/22 unless marked
*unverified*. Evidence and timelines live in `reports/2026-09-21-*.md` and
`reports/2026-09-22-*.md`; the Pantheum side is in
`../pantheum-I/docs/alibz-architecture.md`, `../pantheum-I/docs/acquisition-optimization.md`
and `../pantheum-I/DECISIONS.md`. The Opal database route is documented in
`docs/z300_opal_ingestion.md`.

## 1. Identity, network, access

| Item | Value |
|---|---|
| Instrument | `Z300-0915`, firmware `v2.19.7-0-gca24454`, versionCode 680, FPGA 3.20.14, Android 4.2.2 |
| Network | `192.168.60.65:9000` on the `lml_automation` Wi-Fi; from Moissanite use the proxy `192.168.50.112:19000`. `config/alibz.example.json`'s `192.168.50.65` is dead since the 2026-09-13 renumber. |
| HTTP server | `com.sciaps.android:remoteService` (devsmart miniweb inside the vendor app); no authentication; one thread pool |
| Vendor timeouts | Profile Builder uses 5 s connect / 90 s read; the portal caps live I/O at min(90, configured) |
| Clock | **The RTC (PCF8563, `/sys/class/rtc/rtc0`) loses time on every power-off**: its backup cell is dead, so a cold boot starts at 1970-01-02 (Android's floor) and logcat shows `01-01 16:02:xx` PST until someone sets the date. A warm reboot keeps the time. No NTP (`NetworkTimeUpdateService` has no reachable server). Before the fix the analyzer ran ~52 s behind Moissanite. Test `unixTime` and `/data/tests?since=` cursors are analyzer seconds. Check/set with `scripts/z300-clock.sh` (status is read-only; `--set` copies Opal's local time via root ADB `date -s YYYYMMDD.HHMMSS`). |
| USB | Opal sees `mtp,adb`. ADB is **root** (`uid=0`). Serial `0123456789ABCDEF`, host server port 5038. The recovery copy of platform-tools is in `C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\`; the ingest uses its own copy in `C:\InstrumentControl\alibz\z300-ingest-20260922\platform-tools\`. |
| Reaching Opal | `ssh opal` (user `whittaker`, port 52222 via Moissanite). Pass PowerShell as `-EncodedCommand` (UTF-16LE base64); plain quoting is mangled. From Moissanite's worker: `ssh -p 52222 whittaker@192.168.50.112`. |
| On-device tools | `/system/xbin/sqlite3` (3.7.11) reads `libzdb.cblite` in place; `logcat -d -v time` is the single most useful diagnostic; `dumpsys activity`, `dumpsys power`, `ps` work. No `head`, `which`. |
| MTP vs API | Enabling the remote service was believed to disable MTP; ADB and the API coexist. |

## 2. HTTP API, as actually behaved

| Endpoint | Behaviour |
|---|---|
| `GET /instrument/id` | `{id, version, versionCode, laserCoolingOn, triggerLocked, argonPSILevel, wlCalibrationNeededCode, supports50HzData, supports50HzCleaning}`. **`triggerLocked` does not reflect the enforced state**: it read 0 before every `TriggerLocked` refusal. `wlCalibrationNeededCode` 1 = ok (green), 0 = needed, −1 unknown. |
| `GET /instrument/defaultParams?mode=4` | The unit's own `xyzstage.json` geometry: start `[134,76,70]`, end `[206,124,70]`, step 24, one location, `resetStage true`, `pulsePeriod 100`, `cleaningPulsePeriod 100`, `argonpreflush 300`, delay 10 / period 25, 10 shots. |
| `POST /instrument/saveAcquisitionParams?mode=` | Writes the instrument's stored defaults. Not needed for firing (the fire body carries its own params); not enabled in the portal (`z300.save_params`). |
| `GET /instrument/getArgonPSI` | ~12.5 PSI observed. |
| `POST /data/firetest` (JSON RasterParams) | Returns the new test id (string). **520** `LaserNotArmed`, **521** `TriggerLocked` (empty bodies). **500 with the new test id as body** when the laser controller rejects the config ("Invalid laser config parameters", abort code 8) – an empty test record is still saved. **No reply at all** (client times out) when `stepSize` is 0 (ArithmeticException inside `LIBZLaserController.start`). |
| `GET /data/tests?since=<SECONDS>` | `{"data": [<=200 test ids, oldest first], "next": <cursor or null>}`; follow `next` until null. **Milliseconds crash the service** (`NumberFormatException`); after two crashes Android stops restarting it and port 9000 goes dead. `since=2147483647` → `{"data": [], "next": null}`. |
| `GET /data/shotspectrum?test=<id>&shot=<n>` | gzip (`application/x-spectrum`) JSON `{knots[5], wlCalibrations[4]{calibrationTime, pixToNm.coefficients[4]}, pixels[4][2066]}`; `shot=-1` is the average; 404 for shots the test did not store. ~0.15 s each. |
| `POST /data/cancel` | Vendor abort; untested here. |
| `/laser/raster`, screenshot | Exist in the vendor client; untested. Screenshot polling logs `main display 480x272` twice a second. |

## 3. Interlocks and operator steps (why fires were refused)

1. **Laser arming.** After a reboot or an app restart the laser is disarmed. A
   remote fire while disarmed makes the handheld *display* `LaserArmActivity`;
   the operator must arm it there (laser PIN; `/storage/sdcard0/sciaps/slpp.bin`
   is absent, so it is typed). Profile Builder's only reaction is "Please arm
   the laser in the LIBZ unit and try again." No API can arm it.
2. **Trigger lock.** The padlock toggle in the app's status bar
   (`TOGGLE_TRIGGER_LOCK` → `TriggerLockService`; `TestController: mIsTriggerLocked = false`
   when cleared). Fires were also refused as locked while the Geochem Pro
   process was dead and the launcher was showing (2026-09-21 16:32–16:56), so
   keep Geochem Pro open on its START screen as well.
3. **Calibration.** `wlCalibrationNeededCode 0` is advisory in the vendor UI
   (a confirm dialog); the portal does not gate on it. Last WL calibration
   2026-09-21 14:55, next wanted 24 h later.
5. **After a power cycle: clock, then launcher.** With the clock at 1970 the
   vendor launcher `com.sciaps.android.home` ("LIBZ Home", v2.18.1) shows a
   modal "Loading..." dialog and opens Settings > Date & time (twice, at
   +9 s and +23 s after boot). Setting the date and time there fixes the
   clock but the dialog never closes: the launcher sits idle behind it (no
   working thread, nothing logged) and Geochem Pro cannot be opened from the
   handheld. The HTTP API, `TriggerLockService` and `RemoteService` are
   separate processes and keep working, so data-API retrieval is unaffected;
   only new fires are blocked (no Geochem Pro START screen, laser disarmed).
   Recovery: `scripts/z300-clock.sh --set` (if the clock is still wrong) then
   `scripts/z300-clock.sh --restart-home` (force-stops the launcher; Android
   relaunches it against the corrected clock), or on the handheld: pull down
   the status bar → Settings → Apps → LIBZ Home → Force stop. Then open
   Geochem Pro, arm the laser (PIN), check the padlock.
4. **Laser safety** (`/storage/sdcard0/sciaps/laserconfig.json`): pump-time
   window 4 / threshold 500 µs / `pumptimeTemp 40 °C` / `dutyCycleBudget 1` /
   `pumpThresholdCost 300 s`. A LASER STOP event halts the laser and re-enables
   it when the cost buffer decays. Solenoid temperature was 35–39 °C during
   today's batches.

## 4. RasterParams rules learned the hard way

| Parameter | Rule |
|---|---|
| `startLocation`/`endLocation`/`stepSize` | The **internal beam-steering stage** (max travel `[550,550,374]`), not the gantry. `stepSize` must be > 0 (0 → divide-by-zero, no reply). Vendor geometry above fires; start = end with step > 0 is *unverified*. One location per batch = `numlocations 1`. |
| `pulsePeriod` (ms) | 100 (10 Hz) works; 20 (50 Hz) is the vendor maximum, *unverified*; **10 (100 Hz) is refused** ("Invalid laser config parameters"). `hardware.cfg` z300 default is 100. `cleaningPulsePeriod` 10 is silently clamped to 20. |
| `intergrationDelay`/`intergrationPeriod` (vendor units) | Verified to store ten shots: delay 5/10 with period 10/25/50. **Period 100 stores ONE shot** (test "finishes" normally, `saved test`, no error). Delays 20 and 50 are in the current grid and *unverified*. |
| `numShotsToAvg` | 1 → `numShotAvgFPGA 0` at 10 Hz. 50 Hz has vendor-side averaging rules ("Linear" shot grouping auto-set). |
| `numlocations × numShotsPerLocation` | ≤ 600 (vendor UI rule). |
| `argonpreflush` | 300 used throughout; the vendor UI validates ≥ 0. |

## 5. Data on the instrument

- Database: `/storage/sdcard0/libzdata/libzdb.cblite` (Couchbase Lite 1.x =
  SQLite `docs`/`revs`; `revs.current=1`; `docs.docid` = test id). A test doc:
  `{unixTime, onlyAvgSaved, standard, shotTable{all|all_fb: <sha1>}, config{rasterStart, gating, intergrationDelay, intergrationPeriod, numCleaningShotsPerLocation, numShotsPerLocation, numShotsToAvg, rasterNumLocations, argonPreflush}, type:"test", displayName, metadata}`.
- Bundles: `/storage/sdcard0/libzdata/spectrum/<xx>/<sha1>`. Today's API-fired
  tests used the **legacy ZIP** form (`shotTable.all`): members `-1`, `0`…`9`,
  each gzip JSON identical to the shot-spectrum API payload (~16 KB; 171 KB for
  ten shots). The FlatBuffers `all_fb` form is decoded by
  `scripts/z300_fb_decode.py` (PIXEL_OFFSET −18; segments stitched half-open at
  the knots). A rejected config still saves an empty test record.
- Native grid (from a real shot): 7,914 samples after knot trimming; UV 2,027 px
  at 0.078–0.096 nm, VIS 1,985 px at 0.113–0.142 nm, NIR 1,836 px at
  0.156–0.198 nm, plus a 960–961 nm stub. **NIR coverage ends ~12 nm below the
  960 nm knot**; the vendor's 1/30 nm export interpolates across that gap.
- SD card also holds `pumptime.csv` (28 MB, growing), `wlcalspectrum.csv`
  (278 MB), `xyzstage.json`, `hardware.cfg`, `laserconfig.json`.

## 6. Failure modes seen and their recovery

| Symptom | Cause | Recovery |
|---|---|---|
| Port 9000 refuses / resets | RemoteService crashed twice (ms cursor) and Android stopped restarting it | `adb -P 5038 -s 0123456789ABCDEF shell am startservice -n com.sciaps.android/.service.RemoteService` (root ADB via Opal). Reboot with USB unplugged also works. |
| 520 / 521 on firetest | Laser not armed / trigger locked | Arm on the handheld (PIN); unlock the padlock; Geochem Pro on START. |
| Fire times out, run `uncertain` | `stepSize 0` divide-by-zero | Never send 0 (validator now refuses); reconcile the hold. |
| 500 with a UUID body | Invalid laser config (e.g. 100 Hz) | Use pulsePeriod 100. |
| Run `awaiting_data` forever | Test stored fewer shots than requested (period 100) and retrieval kept pending | Cap now ends it `failed` after `max_attempts`; the optimizer continues. |
| Test "finishes" with 8–9 of 10 spectra | **Spectrometer-link frame drops**: `E/onyx: checksum mismatch! computed/read …` then `error. event type: 7` on the shot; the laser fired all ten (`shot:buffer` x10) but one spectrum is never stored (shot n → 404). Seen 4 times between 12:16 and 12:38 PDT on 2026-09-22, 3 of 4 tests affected, none in the morning's 5 tests. Solenoid 39 °C, uptime 3.5 h, no kernel USB fault. | Unknown root cause (thermal/EMI on the readout link?). Power-cycle and re-observe; make the pipeline tolerate ≥ 8 stored shots (decision pending). |
| Trigger "unlocked" in the portal but refused | identity field is cosmetic | Read the handheld log: `logcat` tag `TestController`. |
| Black/absent handheld screen | display asleep | wake on the device; there is no API. |
| Handheld stuck on "Loading..." over the app grid after a power cycle; logcat dated `01-01` | Dead RTC backup cell → clock at 1970 → launcher's date check opens Date settings and never dismisses its dialog | Set the clock (`scripts/z300-clock.sh --set`, or Settings > Date & time), then restart the launcher (`--restart-home` or Force stop LIBZ Home). API and retrieval keep working meanwhile. See §3 item 5. |

## 7. Pipeline state (2026-09-22 12:15 PDT)

- Live config: `acquire.retrieval=opal_database` (`max_attempts` 40, retry 15 s,
  timeout 240 s); `enabled_actions ["z300.firetest"]`; hardware checkout TTL
  900 s. Study defaults: 10 Hz, vendor geometry, native grid, `min_central_px`
  5 / `min_sideband_px` 3 metric windows, failed batches count as tested.
- Scripts (alibz `scripts/`): `check-optimization-readiness.py`;
  `deploy-alibz-*.sh` (manifest-pinned, reservation-locked; the auto-mode
  classifier refuses agent-run deploys, the user runs them);
  `recover-alibz-awaiting-data.{py,sh}` (data-API recovery / `--refetch`);
  `unblock-alibz-session.{py,sh}`; `z300_opal_ingest.py` (deployed to Opal);
  `z300_fb_decode.py`.
- Datasets on the old resampled grid still exist (`2679219a…`, `4f678d79…`,
  `0982aede…`); runs now reference native datasets.

### Before a delay/period study

`scripts/check-study-conditions.py [--delays 5,10,20 --periods 10,25,50 | --session opt-…]`
(read-only) lists which grid conditions have a fully stored live run
(the optimizer refuses the others), the Acquire-panel batches that would verify
them, active acquisitions, trigger lock and calibration flags. Exit 1 means
the study would be refused. The gilbert raster starts at the window origin,
which is the worn fixed spot of the 2026-09-22 sessions: shift the window
(startLocation) or the sample first.

## 8. Anticipated problems (for the owner to decide)

0. **Spectrometer frame drops (active, 2026-09-22 afternoon).** 3 of the last 4
   tests stored 8–9 of 10 spectra after `onyx` checksum errors. Every pipeline
   stage assumes exactly ten (retrieval validation, `_run_live` shot fetch →
   `uncertain` under `data_api`, optimizer `shots != 10` → failed, metrics
   "exactly ten shot spectra"). Until batches tolerate dropped frames, most
   batches will fail and `data_api` mode would leave runs `uncertain` with the
   hardware hold pinned. Decide: accept ≥ N stored shots (recommend 8) and
   record `dropped_frames`, or stop and service the instrument.

1. **Unvalidated grid points waste fires.** Period 100 stored one shot; delays
   20 and 50 in the current grid have never been tried. Each bad point now
   costs one fire and one failed batch. Option: a one-shot "envelope" test per
   grid point before a study, or a config allowlist of verified
   (delay, period, pulsePeriod) values enforced at session creation.
2. **A refused dispatch still blocks a session.** In `next_batch`, a refusal by
   the acquisition gates after the batch row is persisted (checkout expired,
   gantry not idle, analyzer unreachable) marks the batch `failed` via
   `_set_terminal` → session `blocked`, unlike the reconcile path. With a 900 s
   checkout TTL this will bite during long studies. Fix: undo the persisted
   batch on a pre-dispatch refusal instead of failing it.
3. **Retrieval cap vs. slow Opal.** 40 × 15 s ≈ 10 min. A genuinely slow or
   wedged ADB pull would end a good run `failed`; `recover-alibz-awaiting-data.py`
   covers `awaiting_data`/`succeeded` runs but not `failed` ones. Consider
   `--refetch` accepting `failed` runs whose test exists.
4. **Two native-grid producers.** `pixels_to_wavelength` (API path, inclusive
   knot trim) and `z300_fb_decode.stitch` (Opal path, half-open trim, pixel
   offset −18) should be cross-checked on one test before archives from both
   are pooled. Keep one retrieval path per session.
5. **ADB fragility.** The host ADB server on 5038 serves two platform-tools
   copies (one in a Windows Temp directory). A version mismatch restarts the
   server ("killing…") mid-pull; a Temp cleanup removes the recovery copy; an
   Opal or analyzer reboot can re-prompt "Allow USB debugging". Pin one copy
   under `C:\InstrumentControl` and keep the recovery command with it.
6. **RemoteService has no watchdog.** Two crashes and the API stays dead until
   someone runs `am startservice`. A scheduled Opal task that checks
   `/instrument/id` and restarts the service would remove a bench trip.
7. **Clock resets on every power-off (dead RTC cell).** Beyond the launcher
   hang (§3 item 5), a fire made while the clock reads 1970 gets a 1970
   `unixTime`, so the test lands at the *front* of the `/data/tests` list and
   provenance timestamps are wrong. The portal has no clock check because the
   HTTP API exposes no time; the ADB-based `scripts/z300-clock.sh` is the
   check. Options for the owner: (a) run `--set` after every power cycle and
   before the first batch (cheapest); (b) add an Opal scheduled task that
   syncs the clock whenever ADB sees uptime < 10 min; (c) replace the RTC
   backup cell (a bench repair on the analyzer's board). Until then, avoid
   power cycles: `--reboot` (warm) keeps the time.
8. **Laser thermal/duty limits.** Solenoid at 39 °C against `pumptimeTemp 40`
   and `dutyCycleBudget 1`: sustained batches can trigger LASER STOP events that
   look like short tests. Watch `LIBZLaserController` for "STOP" and pump times.
9. **Wavelength calibration cadence.** The unit wants WL calibration daily
   (`nextCalTime`); the portal ignores `wlCalibrationNeededCode`. Decide whether
   to gate live batches on it or schedule the calibration.
10. **One spot, cumulative craters.** Every batch fires at `[134,76,70]`;
    `total_shots` is now > 60 on that spot. The stage can raster (vendor default
    3 × 2 grid, step 24); choose whether batches should move.
11. **Storage growth on the SD card** (`pumptime.csv`, spectra, cblite) and
    argon use (300 preflush per batch, 12.5 PSI observed) are unmonitored.
12. **Concurrent deployers.** Two sessions deployed to the same runtime within
    an hour; manifest pins caught nothing only because edits did not overlap.
    Serialize deployments and keep `reports/STATUS.md` single-writer.
13. **`/data/tests` pagination cost** grows one page (~1 s) per 200 tests; in
    `data_api` mode the readiness probe also calls it every 5 s.
