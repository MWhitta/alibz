# Z300 fire diagnostics implementation

## Outcome

Implemented and focused-tested the bounded acquisition readiness/diagnostics fix in the isolated checkout `/private/tmp/pantheum-fire-diagnostics-20260921`, seeded from `/Users/mwhittaker/Projects/github/pantheum-I`. The source checkout was read only during this task; no deployment, live analyzer access, physical action, commit, or provider switch occurred.

## Changes

- `pantheum/alibz/z300.py:52,99-106` maps vendor HTTP 520 (`LaserNotArmed`) and 521 (`TriggerLocked`) to `Z300Error(code='interlock')`, retaining a useful bounded label even when the response body is empty.
- `pantheum/alibz/z300.py:176-184` accepts the pinned vendor `{ "data": [...] }` recent-test envelope plus the existing bare-list compatibility shape, and rejects missing/non-list `data` and non-object list items.
- `pantheum/alibz/z300.py:295-300,350-354,377-391` makes `FakeZ300Server` emit the vendor envelope and adds deterministic fire counts, delayed acknowledgements, alternate test-list payloads, and empty-body 520/521 responses for regression tests only.
- `pantheum/alibz/acquire.py:166-199` extends the bounded read-only readiness probe with a recent Unix-ms `/data/tests` request. It reports identity `reachable` independently from `data_api_reachable`, exposes bounded `data_api_detail`, and carries `wlCalibrationNeededCode` without gating on it.
- `pantheum/alibz/acquire.py:220-242,374-403` makes both status and the fresh live-start gate fail closed when the recent-test endpoint fails or returns an invalid shape. The refusal happens before inserting an acquisition or beginning a physical operation.
- `pantheum/alibz/acquire.py:482-502,546-550` retains bounded Z300 error code/message in both uncertain run and hardware-operation detail. Unexpected exceptions expose only the exception class, without a private message or traceback. Existing uncertainty, locks, and no-retry behavior remain intact.
- `pantheum/alibz/acquire.py:552-572` changes only the live operation client timeout to `min(90.0, timeout_seconds)` and records an acknowledged test id before later list/spectrum reads. Status/probe timeouts remain capped at 3 s. `urllib` has one timeout value applied to connection/socket operations, unlike the pinned vendor client's separate 5 s connect and 90 s socket/read timeouts; no custom transport or retry was added.
- `tests/test_alibz_acquire.py:100-114,132-141` covers vendor envelope, legacy list, invalid shapes, and empty-body 520/521 interlock diagnostics.
- `tests/test_alibz_acquire.py:426-521` verifies healthy status fields; identity-success/data-list-failure refusal with zero acquisitions, operations, and fire requests; a >5 s delayed fire acknowledgement completing once under the 90 s cap while exclusion remains active; smaller configured timeout propagation; post-ack data failure retaining test id, diagnostic, uncertainty, and hardware hold; and unexpected exception class-only detail.
- `tests/test_alibz_checkouts.py:382-383` keeps the pending-motion-operation test focused on its intended blocker by marking its mocked data API probe healthy.
- `docs/alibz-architecture.md:201-234` documents the added data-path preflight, split reachability fields, advisory calibration code, timeout model, immediate test-id persistence, and no-retry uncertainty behavior.
- `DECISIONS.md:81-94` records why identity alone is insufficient and why data-list readiness, envelope validation, timeout cap, and evidence-preserving uncertainty are required.

## Verification

Baseline, before edits:

- `python3 -m unittest tests.test_alibz_acquire -q` inside the sandbox: 48 tests attempted, 28 errors because managed sandbox policy denied loopback socket bind. This was an environment failure, not a code failure.
- Same command with approved loopback execution outside the sandbox: **48 passed, 0 failed** in 18.600 s.

After edits:

- `python3 -m unittest tests.test_alibz_acquire -q`: **56 passed, 0 failed** in 27.869 s.
- `python3 -m unittest tests.test_alibz_checkouts -q`: **18 passed, 0 failed** in 1.384 s.
- Focused total: **74 passed, 0 failed**.
- `python3 -m py_compile pantheum/alibz/acquire.py pantheum/alibz/z300.py tests/test_alibz_acquire.py`: passed.
- `diff -q` against the source checkout showed differences only for the six intended implementation/test/documentation files; `AGENTS.md` remained identical. The source checkout was never a write target.

## Evidence and limits

The delayed-ack regression uses the loopback fake server with a 5.2 s delayed POST acknowledgement and observes exactly one fire request. The failed-preflight regression keeps identity healthy, supplies an invalid recent-test shape, and observes no acquisition rows, unresolved physical operations, or fire requests. The post-ack regression observes an acknowledged test id on an `uncertain` run and the same bounded diagnostic on its unresolved hardware operation.

Only focused tests were requested for this implementation agent; the main agent is running the repository-wide suite. No real Z300, instrument LAN endpoint, private deployment configuration, UI, or physical interlock was exercised, so bench behavior remains unverified. No provider usage-limit evidence appeared, so no provider switch was made.
