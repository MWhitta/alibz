# Split checkout backend implementation

## Outcome

Implemented independent `hardware` and `analysis` leases with fresh SQLite authority and a cross-process advisory lock. Hardware is always exclusive; the legacy `reservation.enabled=false` switch only disables analysis checkout enforcement. Missing API resource remains the legacy-compatible `hardware` default.

Hardware ownership is fenced across complete HTTP mutations. Live analyzer and gantry submissions persist intent before vendor mutation, serialize with cancellation across processes, and retain durable uncertainty after transport loss or restart. Expired hardware leases remain pinned during live or unresolved physical work. Recovery remains owner-only and requires `confirm: {"hardware_idle": true}`, no active live executor, and a fresh verified Idle gantry observation; reconciliation records a durable audit without replay and leaves scientific acquisition outcome `uncertain`.

No provider switch occurred.

## Changed files

- `pantheum/alibz/reservation.py:78` replaces cached JSON authority with two SQLite leases, strict legacy migration to hardware only, corrupt-state fail-closed behavior, status fields `resource`, `can_release`, `can_reconcile`, and `blocked_reasons`, safety holds, cross-process guards, generation fencing, physical-operation ledger, and verified reconciliation (`:172`, `:278`, `:293`, `:420`, `:428`, `:434`, `:471`).
- `pantheum/alibz/__main__.py:46` adds a single serve-process authority lock acquired before recovery/socket replacement. `:130` adds GET resource selection; `:305-371` maps analysis and hardware mutations to the correct guarded lease and propagates authenticated owner/generation; `:481` stops the analysis worker from recovering the physical ledger.
- `pantheum/alibz/acquire.py:90` adds transactional additive schema migration for ownership, hardware resolution, operation ID, and `dispatch_started_at`. `:193-234` exposes motion/unresolved gates in status. `:266` persists settings-push intent. `:340-425` persists live fire intent and blocks unresolved conflicts. `:428-516` claims queued work atomically and serializes terminal completion against cancellation. `:518` persists the final dispatch marker immediately before fire under the shared lock. `:580` distinguishes predispatch cancel (no vendor request) from dispatched cancel, keeping failed cancellation active until the executor exits. `:640` leaves restarted physical work uncertain.
- `pantheum/alibz/motion.py:317` reports active acquisition/unresolved conflicts. `:364` persists owner/generation-bound motion intent, blocks movement/connect/set-zero/clear-alarm during live work, preserves abort/reconcile recovery, and serializes submissions. `:501` accepts completion only from a strictly newer verified controller observation (target position plus Idle where applicable). `:651` captures the observation marker atomically with emit and never retries ambiguous commands; `:740` advances the marker on status observations.
- `pantheum/alibz/optimization.py:369` propagates hardware lease owner/generation through optimizer acquisition dispatch, including simulated optimization.
- `tests/test_alibz.py:286` updates the mixed HTTP fixture to hold both independent resources.
- `tests/test_alibz_reservation.py:36` updates intentionally changed legacy expectations; `:198` adds resource independence, process contention, guarded mutation/release, schema-init concurrency, expiry safety-hold, migration, and corrupt timestamp tests.
- `tests/test_alibz_acquire.py:253` adds queued/running cancel, ambiguous settings, and late-result cancellation regressions; `:295` proves cancellation uncertainty cannot be overwritten by late success.
- `tests/test_alibz_checkouts.py:73` (parallel test contribution) adds 18 checkout HTTP, lease safety, physical ordering, reconciliation, and process-lifecycle regressions used in final verification.

`pantheum/alibz/service.py` was read and used as the existing transactional/storage boundary but required no source change.

## API contract

- `GET /api/reservation?resource=hardware|analysis`; missing resource means hardware.
- `POST /api/reservation` with `{resource, action, note?}`; missing resource means hardware. Actions remain `reserve`, `renew`, `release`, `override`.
- Status retains existing lease fields and adds `resource`, `can_release`, `can_reconcile`, and `blocked_reasons`.
- `POST /api/motion/command` with `{"action":"reconcile","confirm":{"hardware_idle":true}}` performs owner-only conservative reconciliation. It never replays a command.

## Verification

Required pre-change suite:

- `PYTHONPATH=src python3 -m pytest tests/ -q` -> **530 passed, 139 failed, 37 errors, 26 skipped, 199 subtests passed**. All captured failures/errors were dominated by the sandbox refusing loopback/Unix socket binds with `PermissionError: Operation not permitted`.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q` -> `.venv/bin/python` was absent, then the command ran with the same effective path and produced **530 passed, 139 failed, 37 errors, 26 skipped, 199 subtests passed** with the same socket sandbox failures.

Focused post-change verification:

- `python3 -m py_compile pantheum/alibz/{reservation.py,__main__.py,service.py,acquire.py,motion.py,optimization.py` -> success.
- `PYTHONPATH=src:/private/tmp/pantheum-live-socket-test python3 -m pytest ... -q -k 'not CheckoutHTTPTests'` -> **100 passed, 2 deselected, 11 subtests passed**.
- Final cancellation/lease subset after the audit ordering fix -> **35 passed, 4 subtests passed**.
- Simulated durable motion smoke: commanded X=2.5, observed post-dispatch position `2.5`, and verified zero unresolved operations.
- Simulated reconciliation smoke: `can_reconcile=true`, one uncertain operation reconciled, and zero unresolved operations afterward.
- Parent escalated full suite before the final fixture/cancellation fixes: **742 run, 3 failures, 6 errors, 11 skipped**; all nine failures were analysis HTTP fixtures that still held only hardware and were corrected at `tests/test_alibz.py:286` / `tests/test_alibz_reservation.py:345`.
- Final parent escalated full suite: **761 tests passed, 11 skipped** in 167.582 s.
- Final backend-focused rerun after the last exception-finalization lock fix: **135 passed** in 26.346 s.
- Parent frontend validation: **31 Node tests passed** and changed-script syntax checks passed.

Physical Z300/OpenBuilds hardware was not contacted; simulator, fake-server, persistence, concurrency, and browser fixtures provide the available evidence. No deployment was performed here.
