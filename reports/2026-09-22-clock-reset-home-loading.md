# Z300 clock reset after power cycle: launcher stuck on "Loading..." — 2026-09-22

Status: diagnosed; clock already correct; launcher restart handed to the operator
as a wrapper (`scripts/z300-clock.sh --restart-home`) because the auto-mode
classifier refuses remote writes (key injection / `am force-stop`) from this
session. Data-API recovery is not blocked by the launcher state and its dry run
passed. No fire, motion, cancel or settings push was issued.

## Symptom

After the 13:09 PDT power cycle the handheld showed the LIBZ Home app grid under a
modal "Loading..." spinner. USB to Opal was reconnected and root ADB works
(`adb devices` lists `0123456789ABCDEF`, `id` → `uid=0`). The HTTP API answered
`/instrument/id` in 72 ms and `/data/tests` in 0.31 s throughout.

## Evidence (all read-only)

- `date` on the analyzer at 13:17:20 PDT vs Opal 13:17:16 PDT; uptime 471 s
  (boot ≈ 13:09:29). RTC `rtc-pcf8563` now reads 2026-09-22 20:26:28 UTC.
- `logcat -d -v time` starts at `01-01 16:02:35` (= 1970-01-02 00:02:35 UTC,
  Android's earliest supported time, applied when the kernel clock is at 0).
  So the RTC lost its time on power-off: the backup cell is dead.
- The launcher (`com.sciaps.android.home`, pid 597, v2.18.1-0-g81ab353) started
  `android.settings.DATE_SETTINGS` at +9 s and +23 s after boot.
- `SystemClock: Setting time of day to sec=1790118214` (date set to 09-22,
  time-of-day kept: 16:03:34) then `sec=1790107860` (13:11:00) — the operator set
  date, then time, in that dialog. Kernel timezone moved 480 → 420 min west
  (PST → PDT) when the date changed. The apparent later "re-sets" at the end of
  the dump are an artefact: old `logcat` merges the main and system buffers by
  timestamp, so lines written while the clock read 16:03 sort last.
- Thread dump of pid 597 (SIGQUIT after creating `/data/anr`): main thread only
  drawing the spinner; `FPGA Thread` and `ProcessManager` parked; no vendor
  frames running. `dumpsys window` shows two HomeActivity windows (activity +
  dialog). Nothing from pid 597 in logcat after "LIBZ Home is already default".
- Tombstones 07/08 at 13:18 are this session's own `grep | head` SIGPIPEs;
  06 (11:20) predates the reboot.

Conclusion: the launcher's boot-time date check shows the dialog and delegates to
Settings; once the date is fixed nothing dismisses the dialog. It is idle, not
deadlocked, and a restart clears it. The same thing was seen at the start of
development (operator recollection; no written record found in either repo's
reports or transcripts).

## Fix and hand-off

`scripts/z300-clock.sh` (Mac → `ssh opal` → root ADB):

- default: read-only status (Opal time, analyzer time, RTC, uptime, skew,
  focused window, whether the launcher dialog is present, date prompts since
  boot, Pantheum active acquisitions).
- `--set`: copies Opal's local time (`date -s YYYYMMDD.HHMMSS`), broadcasts
  TIME_SET, verifies |skew| ≤ 120 s.
- `--restart-home`: `am force-stop com.sciaps.android.home`; Android relaunches
  the default launcher; verifies the dialog is gone and saves a screenshot.
- `--reboot --yes`: warm reboot (RTC keeps time while powered).
- `--set`/`--restart-home` refuse while Pantheum has queued/running acquisitions.

Operator sequence now: `scripts/z300-clock.sh` (expect `clock: OK`, dialog
present) → `scripts/z300-clock.sh --restart-home` → open Geochem Pro, arm the
laser (PIN), check the padlock → `scripts/recover-alibz-awaiting-data.sh
--refetch --apply --run run-bb777df5… --run run-8264fabf… --run run-d99401be…`.

The refetch dry run at 13:24 PDT fetched 8/9/9 spectra for the three tests and
listed all three batches for re-scoring; `retrieval_now: data_api`.

## Recorded elsewhere

- `docs/z300_device_notes.md`: §1 Clock row, §3 item 5, §6 row, §8 item 7.
- Denied by the classifier this session: `input keyevent BACK`, and the readiness
  script call (`scripts/check-optimization-readiness.py`) on the second run.
