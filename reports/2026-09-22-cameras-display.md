# alibz live cameras and Z300 display — 2026-09-22

## Camera repair completed

The user requested working D455 color/depth live views arranged vertically, and
battery sleep prevention plus remote wake for the Z300. The user confirmed
vertical stacking (color above depth), matching the deployed layout. Images
retain their native orientation.

At approximately 09:38 PDT both Pantheum D455 sources had zero frames and
TimeoutError. Opal's existing OpalCamera scheduled task was Ready (stopped), no
TCP/8790 listener, last result 3221225786 (0xC000013A), last started Sep 14.
Last camera log entries Sep 21 14:31 showed successful HTTP frames followed by
normal idle shutdown of the capture pipeline; no crash cause was established.
The task is already allowed on battery, has no execution limit, and allows three
restart attempts at five-minute intervals. At 09:42, Start-ScheduledTask restored
the existing camera publisher. No install, network change, or device reset.

Verified D455 serial 220422301696, real realsense backend, both 640x480 streams.
At 09:44:59 all three Pantheum camera sources were live, had current timestamps,
and changed tokens across a three-second observation. Publisher approximately
6.6 color / 6.27 depth fps. Evidence: provenance/camera-live-20260922.json.
Signed-in browser visually showed actual color image and depth image plus the
Z300 screen, all marked Live. Operator acquisition was active concurrently;
no agent firing, cancellation, motion, sleep, or analyzer-app restart was used.

## Vertical layout deployed

Pantheum source /Users/mwhittaker/Projects/github/pantheum-I:
- web/alibz/app.js: stable source identity on each camera tile.
- web/alibz/styles.css: Z300 left; D455 color upper right and depth lower right.
  At <=680px all views stack in their existing reading order.

Static deployment 09:43:31 PDT, no service restart or configuration edit.
The hash-guarded existing deployment script was reused; unchanged index.html
was included in its fixed three-file manifest. Backup on Moissanite:
/home/mwhittaker/pantheum-panel-errors-backup-20260922T094331
Bundle: /private/tmp/pantheum-camera-20260922 (also staged on Moissanite at
~/pantheum-camera-layout-20260922).

Verified source/deployed SHA-256:
- app.js d04d9b5bddc42d3d51a35e74c4920d9458e75edcbb6f526b85e16d83041ab4a8
- styles.css 58cb625c99cd846b48e8c85ebfd7e515babe39a726c97aed2359ea95f935d8e8

Actual HTTP-served app.js and styles.css hashes also match source and the bundle
(provenance/camera-served-assets-20260922.json).

JavaScript syntax passed; existing UI suite 36/36 passed
(reports/2026-09-22-camera-ui-tests.log). Desktop browser verification passed.
Narrow layout follows the existing 680px breakpoint; separate narrow browser
verification not performed yet. Existing frontend/acquisition changes preserved.

## Z300 battery setting applied

Original and effective screen timeout were 600000 ms. Backed up and changed to
2147483647 ms (24.86 days), the maximum supported inactivity interval; this is
not an indefinite wake lock. Parent independently verified the configured and
effective timeout while powered=false, plug_type=0, screen_on=true. Original
plugged-in stay-awake value 3 and mtp,adb USB configuration were preserved.
Evidence: provenance/z300-display-power-backup-20260922.json and
provenance/z300-power-independent-status-20260922.json.

The network/USB wake helper and limitations are documented in
reports/2026-09-22-z300-power-wake.md. Direct analyzer Wi-Fi and Bluetooth wake
are unverified. Real off-to-on testing is not being performed during concurrent
operator acquisitions. Parent verified the Android4.2.2 AIDL and service CLI
against pinned official source; transaction6 wakeUp(long) requires two i32 words,
not unsupported i64 syntax. Source pins: provenance/z300-wake-aosp-20260922.json.
Five current helper tests pass (reports/2026-09-22-z300-power-tests.log). Camera launcher audit
found no evidence supporting a launcher patch; the exploratory recommendation
was withdrawn after checking official PowerShell semantics. No task-settings or
launcher changes were deployed (reports/2026-09-22-camera-reliability.md).

No provider switch. Subscription used throughout.

Parent independently reran the final wake helper: exit0, exact normal Binder
reply, screen remained on and maximum timeout effective on battery. Evidence:
provenance/z300-wake-independent-20260922.json. This verifies command acceptance
and awake no-op behavior, not a deliberate off-to-on transition.
