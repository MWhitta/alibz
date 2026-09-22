# Analyzer status Sleep/Wake controls — deployed 2026-09-22

Added and deployed explicit **Sleep display** and **Wake display** buttons in
Pantheum alibz’s Analyzer status panel, with independently observed display
state and checkout/activity explanations. Source changes are synchronized to
`../pantheum-I`. No provider switches; subscription throughout. No commit made.

## Behavior

The hardware-checkout holder can sleep a freshly observed awake display when
no physical acquisition, uncertainty, or live pending retrieval blocks it.
Wake is available to that holder during a recovery hold and when display state
is unknown. Commands are serialized, audited separately from acquisition
operations, and successful only after the requested state is verified. The UI
ignores stale pre-command status responses, prevents overlapping commands, and
keeps analyzer API errors separate from display state.

Production route: authenticated portal → Moissanite → existing Opal SSH bridge
→ USB ADB → Z300. Direct analyzer Wi-Fi/Bluetooth wake is not established. No
new Opal service or listener was installed.

`scripts/z300_opal_display.py` checks the exact Android 4.2.2 firmware and USB
serial. Binder transaction 6 wakes and 7 sleeps, with reason 0
(`GO_TO_SLEEP_REASON_USER`) and two signed i32 words for long uptime. No power
toggle key or replay; an already requested state is a no-op. Legacy ADB
CR-CR-LF parsing was corrected and regression tested.

The long battery-screen timeout is preserved. Installed helper:
`C:/InstrumentControl/opal/z300-display.py`; existing Python and temporary-
directory ADB are reused.

## Changes and deployment

Pantheum: new `pantheum/alibz/display.py`; service/HTTP wiring; controls in
`web/alibz/{app.js,index.html,styles.css}`; backend/UI tests; example config;
`docs/alibz-display.md`; appended `DECISIONS.md` entry. Local tooling: Opal
helper/tests and `scripts/deploy_display_controls.py`/tests.

Deployed **11:29:31 PDT**. The tested deployment script defaults to dry-run,
verifies baseline hashes, backs up source/config/SQLite, holds reservation.lock,
refuses active/uncertain physical work, and changes only six runtime files and
top-level display configuration. Only pantheum-alibz.service restarted; the
retrieval worker remained running.

Backup: `/home/mwhittaker/pantheum-display-backup-20260922T182931180544Z`.
Manifest: `provenance/display-controls-deployment-20260922.json`.
Unrelated config compared equal to backup excluding display. All six runtime
hashes and all three served asset hashes matched. Both services remained active.

## Verification

- Full backend suite: 813 tests run, OK, 26 optional skips. Log:
  `reports/2026-09-22-display-backend-tests.log`.
- Final affected backend subset: 15 passed; UI: 42 passed; JS syntax passed.
  Logs: `2026-09-22-display-focused-tests.log` and
  `2026-09-22-display-ui-tests.log` in this directory.
- Opal helper: 8 passed; deployment script: 2 passed.
- Actual helper through Moissanite/Opal/USB returned fresh Awake; production
  GET returned HTTP 200 and Awake.
- Production POST Wake while already awake returned HTTP 200 with fresh Awake
  at `2026-09-22T18:33:41.215403+00:00` and a completed display audit row. The
  helper’s already-awake path issues no Binder command. No deliberate physical
  sleep-to-wake cycle was performed.
- Chrome visual/accessibility verification confirmed buttons, Awake badge,
  and pending-retrieval explanation. Sleep was disabled while live run
  `run-ee6913a925c84e1f817f3142ab985cb6` awaited retrieval. Later database read
  found that run failed, finished `18:33:50.404128+00:00`; UI then enabled Sleep.
  This task made no acquisition mutations or recovery requests.
- All three cameras remained Live; D455 vertical stacking was preserved.

Original analysis tab untouched; verification used a separate Acquire tab.
No laser, motion, cancellation, replay, or display-sleep command was issued.
Physical sleep-to-wake remains unexercised on the device; command construction,
state confirmation, errors, and backend permission gates have automated tests.
