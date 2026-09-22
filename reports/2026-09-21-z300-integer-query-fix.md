# Z300 integer query and pagination fix

## Outcome

Implemented and focused-tested the corrected `/data/tests` query contract in the existing isolated checkout `/private/tmp/pantheum-fire-diagnostics-20260921`. No real analyzer request, historical-data query, APK copy, deployment, physical action, commit, or provider switch was performed by this implementation agent.

The initial Unix-millisecond assumption was unsafe: Android logs showed the remote service crashing in `Integer.parseInt`, the pinned caller bytecode divides `Date.getTime()` by 1000, and live read-only diagnostics performed by the main agent established a 200-ID page with a signed-32-bit `next` cursor. The implementation starts from zero to avoid local clock skew, follows only validated server-returned cursors, and matches only the exact acknowledged fire ID.

## Changes

- `pantheum/alibz/z300.py:177-209` validates every outgoing `since` and returned `next` as a non-boolean integer in `0..2147483647`. Invalid caller values fail before `_json` or network I/O. Vendor string IDs are normalized to `{id: ...}`; compatible object entries remain supported, with strict non-empty string IDs. `test_page()` preserves the validated server cursor while `tests_since()` keeps the established list-only interface.
- `pantheum/alibz/z300.py:327-328,379-386` makes the loopback fake emit paged vendor string-ID envelopes, records every safe cursor, and exposes a configurable page size for deterministic tests.
- `pantheum/alibz/acquire.py:189` makes readiness validate one bounded page from `since=0` without assuming cursor units or scanning history.
- `pantheum/alibz/acquire.py:573-597` removes the Unix-ms time filter. Post-fire polling starts at zero, walks at most 64 strictly advancing server-returned cursors, rejects cycles/non-progress, caps each page read to the remaining monotonic polling deadline, and succeeds only on the exact acknowledged test ID. Failure remains uncertain with the existing hardware hold and no retry.
- `tests/test_alibz_acquire.py:100-132` covers vendor string envelopes and object compatibility, invalid shapes/IDs/cursors, 13-digit and overflow caller values rejected before network I/O, and invalid server cursors rejected before any follow-up request.
- `tests/test_alibz_acquire.py:475-543` covers non-advancing cursors, the 64-page cap, historical IDs never matching a different acknowledged ID, safe cursor traversal through pages, a delayed fire acknowledgement completing once, and exclusive operation preservation.
- `tests/test_alibz_checkouts.py:333-361` updates existing direct `_run_live` concurrency tests to mock the page interface while preserving their cancellation/serialization assertions.
- `docs/alibz-architecture.md:204-240` corrects the epoch-ms assumption, documents seconds evidence, zero-start clock-skew avoidance, integer bounds, server-cursor pagination, the page cap, shared deadline, and exact-ID correlation.
- `DECISIONS.md:81-98` records the corrected query shape and the safety rationale for zero-start bounded pagination.

## Verification

Baseline before this follow-up:

- `python3 -m unittest tests.test_alibz_acquire -q`: **56 passed, 0 failed** in 27.928 s.

After the integer and pagination changes:

- `python3 -m py_compile pantheum/alibz/acquire.py pantheum/alibz/z300.py tests/test_alibz_acquire.py tests/test_alibz_checkouts.py`: passed.
- `python3 -m unittest tests.test_alibz_acquire tests.test_alibz_checkouts -q`: **79 passed, 0 failed** in 32.042 s. One existing Python 3.13 `fork()` deprecation warning was emitted; it did not fail the suite.

The regressions prove that an unsafe 13-digit value and every other invalid caller cursor invoke zero `_json` calls; an overflow server cursor invokes exactly the first page request and no follow-up; a non-advancing cursor and a 65th required page leave the run uncertain after exactly one fire; a historical ID never substitutes for the acknowledged ID; and the delayed-ack path follows safe cursors `0`, `1`, and `2` to the exact new ID without replay.

## Integration and limits

Concurrent edits appeared in the real source checkout after the isolated stage was created, including deferred-retrieval runtime/tests/docs. The main agent was notified not to copy this stage wholesale and will merge only the reviewed query fragments while preserving those concurrent changes. The stage intentionally does not contain or validate that separate deferred-retrieval work.

Repository-wide tests and deployment are owned by the main agent. This implementation did not contact the real Z300; live structural evidence cited above came from the main agent's separately authorized read-only diagnostics. No provider usage-limit evidence appeared, so no provider switch was made.
