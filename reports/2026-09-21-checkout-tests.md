# Dedicated alibz checkout regression tests

## Outcome

Added 18 backend regression tests in `tests/test_alibz_checkouts.py`. The suite exercises the checkout contract through public reservation methods, HTTP routes, two `Service` instances, real forked local processes, and mocked Z300/motion clients. It uses only temporary directories, localhost, simulation, and fakes; it never contacts hardware or a remote service.

No runtime or existing test file was changed. No provider switch occurred.

## Test coverage and file references

- `tests/test_alibz_checkouts.py:26-70` supplies isolated raster parameters, temporary `Service` construction, acquisition-ledger insertion, generation lookup, and fork-process helpers.
- `tests/test_alibz_checkouts.py:73-158` provides an ephemeral localhost HTTP fixture. `test_split_cross_user_http_authorization` proves independent hardware/analysis ownership, opposite-resource denial, and propagation of the actual HTTP owner plus durable hardware generation to both acquisition start and optimization next. `test_reconcile_is_owner_only_at_http_boundary` proves a different portal user cannot reconcile and the owner can do so only through the recovery-gated route.
- `tests/test_alibz_checkouts.py:160-263` covers expiry with active/uncertain live work, pending/uncertain hardware-operation pinning without analysis interference, legacy migration to hardware only, malformed JSON and naive timestamp fail-closed behavior, legacy JSON `null` as free, the non-bypassable hardware gate when `reservation.enabled` is false, cross-`Service` guard serialization, and exactly one winner across simultaneous forked hardware reservations.
- `tests/test_alibz_checkouts.py:265-449` covers queued cancellation without constructing a vendor client, cancel-before-dispatch with no fire, cancellation ordered after an in-flight fire submission, a failed cancel remaining scientifically uncertain after a late analyzer success, live start and settings push blocked by pending motion, motion blocked by active/uncertain live acquisition, reconciliation preserving the scientific `uncertain` state while resolving only its hardware blocker, durable reconciliation audit evidence, and refusal without fresh verified Idle state or while a live run remains active.
- `tests/test_alibz_checkouts.py:452-482` proves `serve_lock` excludes a second serving process and that the worker lifecycle recovers analysis jobs without invoking acquisition recovery.

## Runtime sources inspected

- `pantheum/alibz/reservation.py:278-690`: SQLite resource leases, legacy migration, global file guard, safety holds, physical-operation ledger, dispatch fencing, and reconciliation.
- `pantheum/alibz/acquire.py:83-616`: acquisition schema, motion conflict checks, owner/generation persistence, dispatch ordering, cancellation, uncertainty, and startup recovery.
- `pantheum/alibz/motion.py:352-510`: motion/acquisition conflict rules, physical-operation recording, completion evidence, abort, and reconcile routing.
- `pantheum/alibz/__main__.py:29-61,128-131,305-384,451-496`: serve/worker locks, resource-aware HTTP guards, owner/generation forwarding, and process lifecycle.
- Existing fixtures and expectations were read from `tests/test_alibz.py`, `tests/test_alibz_acquire.py`, `tests/test_alibz_motion.py`, `tests/test_alibz_optimization.py`, and `tests/test_alibz_reservation.py`; none was edited.

## Verification

- Before changes, `tests/test_alibz_checkouts.py` did not exist, so it had no baseline test count.
- Initial targeted run after drafting 19 tests: `PYTHONPATH=/private/tmp/pantheum-live-socket-test python3 -m unittest tests.test_alibz_checkouts -v` — 19 passed, 0 failed in 1.911 seconds.
- The final suite was consolidated to the requested approximate 12–18 range and extended with the late-success/failed-cancel regression.
- Final targeted run with the exact requested command: `PYTHONPATH=/private/tmp/pantheum-live-socket-test python3 -m unittest tests.test_alibz_checkouts -v` — 18 passed, 0 failed in 1.484 seconds.
- `python3 -m py_compile tests/test_alibz_checkouts.py` — passed.
- The unittest commands required elevated sandbox permission only to bind the ephemeral `127.0.0.1` test server and exercise local process locks.

## Coordination and limitations

The backend owner supplied the final cancellation-lock interface while the suite was being written. The cancellation-order and late-success tests passed against that implementation; no discrepancy was found to send back. The backend owner was notified of the 18/18 result.

Per the task boundary, the full project suite, deployment, browser workflows, and real hardware behavior were not run here. These tests prove application-level exclusion and ordering against local fakes and OS locks; they do not claim exclusion of vendor GUIs, pendants, bench controls, or a physical controller/analyzer.
