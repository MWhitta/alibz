# Z300 API recovered without reboot — 2026-09-21

Status: complete. API recovered without reboot; compatibility fix deployed;
original acquisition enablement restored and live readiness verified.

## Recovery and cause

USB/Bluetooth tethering was not required. Opal already exposed an authorized
USB ADB interface for serial `0123456789ABCDEF`, Android 4.2.2, USB configuration
`mtp,adb`. Portable Google platform-tools 37.0.1 were staged at
`C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921`.
The host ADB server used isolated port 5038. No driver, firmware, APK, or
persistent service was installed. The downloaded ZIP SHA-256 was
`45f4d63113e895ebde0c90f194099a4676b6ac653bd28d54314a9e022bbc1a99`.

Android startup logs identified `com.sciaps.android/.service.RemoteService`.
At 14:59:02 and 15:02:44, that process crashed with
`NumberFormatException: Invalid int: "1790027760000"`. Android restarted it
once, then stopped trying after the second crash. The old epoch-millisecond
query therefore crashes this firmware. The earlier inference from Java Date
logging was incorrect; it did not establish the server's query units.

Before waking the service, acquisition polling was disabled under the hardware
reservation lock after confirming no active acquisitions or unresolved physical
operations. The original private configuration is backed up on Moissanite at
`~/.config/pantheum/alibz.json.api-recovery-20260921`.
Original SHA-256: `37e26ff18ff5b9b82ea949ea2f8c6678dbf9dd84de807fb487471676f9587728`.
Disabled SHA-256: `2d10dea71fdb10b960968df48279031d92b54c975be3ee2ee83da2e93b4c3ae4`.
Only `acquire.enabled` changed. It was restored after deployment; the final
configuration SHA-256 exactly matches the original.

The recovery command was:

```text
adb -P 5038 -s 0123456789ABCDEF shell am startservice -n com.sciaps.android/.service.RemoteService
```

The service started as process 17030, with device uptime 4972.97 seconds.
Identity then returned HTTP 200 for Z300-0915, firmware v2.19.7-0-gca24454,
triggerLocked=0, wlCalibrationNeededCode=1. The API still returned HTTP 200
at 23:10:59Z. Native LIBZService, trigger lock, and analyzer app processes were
not stopped. No laser, motion, cancellation, or replay was issued.

## Approved test-list diagnostic

The user explicitly approved inspection of the historical test-list response,
retaining only types and a small structural sample, without spectrum endpoints.
`GET /data/tests?since=0` returned HTTP 200 in 1.335 seconds, 7828 JSON bytes.
Its shape was `{"data": [200 UUID strings], "next": 1613659041}`.
Only structural metadata and two example IDs were printed; no full payload was
saved. Pinned Profile Builder bytecode independently declares
`getTestsSince(J): List<String>`, correcting the earlier List<LIBZTest> claim.

A single first-page query cannot find a new test when historical data exceed
one page. The fix must validate cursors and account for pagination before exact
acknowledged-test-ID correlation. The pinned vendor UI caller divides Java `Date.getTime()` by 1000, proving
epoch seconds. The bounded pagination implementation is now deployed and tested.

## Approval review and concurrent work

Automatic review rejected an APK export. No APK was copied; Android startup
logs provided the required service name instead. It also initially rejected the
historical test-list inspection; the user then explicitly approved that read.

Concurrent edits appeared in the real Pantheum acquisition module introducing
`retrieval=deferred` and `awaiting_data`. They are outside the isolated query-fix
stage and must be preserved during source integration. They must not be included
in the deployed fix without separate review and validation.

No provider switch occurred. The original failed run remains uncertain and its
optimization session remains blocked; it will not be replayed automatically.

## Additional structural checks

Following the returned cursor once (`since=1613659041`) returned HTTP 200
in 0.893 seconds, 200 string IDs, and `next=1636459228`. Only types, count,
and cursor were printed. A future query at the largest signed 32-bit integer
(`since=2147483647`) returned HTTP 200 with `data=[]` and `next=null`.
No IDs or spectra were retained from these checks. This confirms both forward
pagination and the end-of-list shape without firing a test.

## Deployment and final verification

At 16:28:23 PDT, only the reviewed `pantheum/alibz/acquire.py` and `z300.py`
from `/private/tmp/pantheum-fire-diagnostics-20260921` were deployed.
The reservation-locked deployment verified no active acquisitions/jobs or
unresolved hardware operations, backed up runtime/configuration/SQLite, and
verified both installed hashes. The runtime backup is
`/home/mwhittaker/pantheum-fire-fix-backup-20260921T162823`.

Installed runtime SHA-256:
- acquire.py: `fd6b373b93c8024f27adbc70cb09355b0f55a3ede3b3f1c44cc792324c50dde9`
- z300.py: `a01ac05967253592d9c7a8c521e7ab50740ba3f0679f227e2b1604453b8d658c`

The fix validates outgoing and returned cursors before I/O, accepts vendor
string test IDs, checks readiness with `since=0`, and searches at most 64
strictly advancing pages for the exact acknowledged test ID. Each page's read
timeout is capped to the remaining polling deadline. Malformed data or a missing
ID retains uncertainty and the hardware hold; no fire is replayed. Socket timeouts
are not an absolute wall-clock guarantee against a peer that trickles bytes.

The pre-recovery private configuration was restored byte-for-byte under the
reservation lock only after runtime verification. Both portal services are active.
At 23:28:54Z, the portal reported `live_allowed=true`, no readiness blockers,
identity and test-list API healthy, trigger unlocked, and gantry connected/Idle.
Hardware checkout remained held by the user with no blocked reasons.
Evidence: `reports/2026-09-21-api-recovery-readiness.json`.

Verification:
- Isolated deployed runtime: 774 backend tests in 176.987 seconds; OK, 11 optional skips.
- Agent focused acquisition/checkout suites: 79 tests; OK.
- Merged source acquisition suite: 65 tests in 37.352 seconds; OK.
- Changed Python files parsed as Python 3.10; restore helper compiled.
- Deployment and restoration dry runs both passed; runtime hashes and restored
  configuration hash verified; both services active.

Concurrent deferred-retrieval code, appended tests, and documentation were
preserved in the real Pantheum source. They were excluded from deployment. The
source acquisition file therefore intentionally differs from the deployed one.
The architecture document records why fire acknowledgement alone is not proof
of stopped equipment. Source merge backup:
`/Users/mwhittaker/Projects/github/pantheum-query-fix-source-backup-20260921T162743`.
Source provenance: `../pantheum-I/provenance/query-fix-source-20260921.json`.

The original uncertain run and blocked optimization session remain unchanged.
No real batch was used as a deployment test. The operator can create a new
optimization session and explicitly start its next batch; live shot acquisition
and spectrum recovery still require that operator-confirmed trial.
