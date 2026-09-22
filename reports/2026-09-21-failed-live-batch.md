# Failed live batch: adapter corrections deployed — 2026-09-21

## Outcome

Deployed at 15:40:13 PDT. Both alibz services are active, configuration is
unchanged, and the hardware hold is reconciled. At 22:40:35Z the gantry was
connected and freshly Idle, but the Z300 API was unreachable. Live firing is
correctly blocked. User was asked to reboot with USB disconnected; no further
reply has arrived. No firing, movement, cancel, settings push, or retry was
issued by this work. No provider switch or commit.

## Original batch and reconciliation

The user initiated run-ca9c938cc88c42e98b07493803992d00 in live session
opt-78c196a5e31d4f8ea914c1344ded0c9a for Fe_Aesar_99.98%_18823.
Dispatch began 21:57:46.972896Z and uncertainty was recorded 21:57:51.985560Z,
about 5.013 seconds later. No test ID, datasets, or shots were saved. The original
exception was discarded, so timing strongly suggests but does not prove a timeout.

The user explicitly observed no shots and a START screen. A read-only screenshot
independently showed Geochem Pro START without a dialog. At 22:01:45Z that bench
confirmation plus a fresh Idle gantry observation allowed reconciliation via the
existing API. Audit ID: e52188c4edb746109cb7b8371cccdbdb. The original acquisition
remains uncertain with hardware_resolved=1 and the optimization session remains
blocked. No scientific outcome was rewritten; nothing was replayed. Live database
verification after deployment found one acquisition and zero unresolved operations.

## Diagnosis and corrections

The pinned Profile Builder JAR SHA256 is
c8164f159eb610b0c7b154df7876010e9699332da5d9cf0704138c59c5814dd7.
The main session independently re-read the bytecode for the vendor's 5-second
connection / 90-second read timeouts, POST endpoint, JSON content type, HTTP
520/521 interlocks, and the test-list data envelope.

- Live adapter I/O now allows min(90, configured acquisition timeout) seconds.
  urllib applies the same timeout to connection and socket-read operations;
  short readiness probes remain bounded to three seconds per request.
- Test listings accept the vendor {data: [...]} envelope and the previous bare-list
  shape. Invalid shapes are rejected. The fake server now models the envelope.
- Live status and fresh start both require a valid read-only recent-test response,
  before any new acquisition or physical operation is recorded.
- Underlying bounded Z300 error diagnostics are retained on uncertain runs and
  operations; an acknowledged test ID is saved before later result reads.
- Vendor 520/521 responses retain useful interlock labels, even with empty bodies.
  No automatic retry or weakening of exclusivity/uncertainty gates was introduced.

Initially identity/default-parameter endpoints worked while /data/tests returned
an empty HTTP response in 0.017746 seconds. Additional timestamp queries reset,
and subsequently /instrument/id also timed out/reset. The analyzer-side cause is
unverified. Current calibration code 1 (before API loss) means okay/green in the
vendor UI; code 0 means calibration needed. This remains advisory information,
not a newly invented hard gate. Existing stage coordinates and acquisition
parameters were preserved.

## Verification

Main full backend run: 769 tests in 175.250 seconds, OK, 11 skipped.
Focused agent runs: 56 acquisition tests and 18 checkout tests passed. Main reviewed
all six changed files, verified Python 3.10 parsing and diff whitespace, and
independently verified the vendor artifact. A delayed fake acknowledgement at
5.2 seconds completed with exactly one fire request and retained exclusivity.
Invalid data preflight produced zero acquisitions, operations, or fire requests.
Post-ack failure retained its known test ID and uncertainty hold.

Initial sandbox execution could not bind local test sockets; the approved
unsandboxed suite above is the meaningful full verification. No UI files changed.
Physical acquisition remains unverified because the analyzer API is unavailable.

Deployment used a nonblocking reservation-file lock, checked absence of active
jobs/acquisitions/unresolved operations, backed up files/config/database, installed
only two manifest-pinned runtime files, and restarted both services. Dry-run and
apply succeeded. Deployed hashes matched; configuration SHA256 remained
37e26ff18ff5b9b82ea949ea2f8c6678dbf9dd84de807fb487471676f9587728.

## Files and backups

- Source backup: /Users/mwhittaker/Projects/github/pantheum-fire-fix-source-backup-20260921T153917
- Live backup: /home/mwhittaker/pantheum-fire-fix-backup-20260921T154013
- Live bundle: /home/mwhittaker/pantheum-fire-fix-bundle-20260921
- Runtime manifest: Pantheum provenance/fire-diagnostics-runtime-20260921.json
- Source manifest: Pantheum provenance/fire-diagnostics-source-20260921.json
- Live evidence: reports/2026-09-21-fire-fix-live-readiness.json
- Main suite log: reports/2026-09-21-fire-diagnostics-full-tests.log
- Detailed source audit: reports/2026-09-21-fire-timeout-audit.md
- Bytecode evidence: reports/2026-09-21-fire-timeout-bytecode.txt
- Implementation: reports/2026-09-21-fire-diagnostics-implementation.md
