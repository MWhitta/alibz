# alibz split-checkout and physical-control concurrency audit

Date: 2026-09-21  
Source reviewed: `/Users/mwhittaker/Projects/github/pantheum-I` (read only,
including its uncommitted acquisition/optimization and gantry changes).  
Provider switches: none.

## Conclusion

The current single `reservation.json` reservation is a portal UI convention,
not an execution fence. `Handler.do_POST()` calls `Reservation.require()` and
then releases its lock before it calls the mutating method. The lease can
therefore expire, be released, or be overridden between the check and a Z300 or
gantry submission. A live acquisition and a gantry command also have no common
interlock, no durable owner/generation, and no rule that prevents a reservation
transfer while equipment is still active or its outcome is uncertain.

The smallest robust design is two independent leases, `hardware` and
`analysis`, stored in SQLite and read transactionally by every `Service`
instance. A physical operation must be durably registered against the current
hardware lease holder and generation in the same transaction that authorizes
it. Active or uncertain physical operations block hardware release, expiry
takeover, and override. The lease transaction must end before network I/O; the
persisted operation, rather than an in-memory lock held across I/O, supplies the
fence.

This remains Pantheum-only coordination. The SciAps vendor UI and unauthenticated
port-9000 API, OpenBuildsCONTROL UI, the older AECP frontend/adapter, pendant or
other local input, direct scripts, and a person at the bench are outside this
ledger. The repository-wide Python search found no additional Pantheum physical
emitter outside `pantheum/alibz`, but that does not exclude those external
authorities.

## Current mutation and control paths

### Analysis checkout

These HTTP requests are user-initiated analysis/catalog mutations and should
require the `analysis` lease:

| Route | Mutation |
|---|---|
| `POST /api/upload` | Writes an original/spectrum, dataset row, and automatic analysis jobs via `Service.import_bytes()` (`__main__.py:285-290`, `service.py:89-131`). |
| `POST /api/sync` | Starts a daemon thread that downloads/imports Drive sources and rewrites sync status (`__main__.py:291-295`, `service.py:390-414`). |
| `POST /api/datasets/<id>/metadata` | Replaces dataset metadata (`__main__.py:296-298`, `service.py:210-242`). |
| `POST /api/datasets/<id>/analyze` | Enqueues an analysis job (`__main__.py:299-302`, `service.py:244-265`). |
| `POST /api/example` | Imports the synthetic example and automatic jobs (`__main__.py:303-307`). |

The worker's `recover_jobs()` and `run_one()` (`service.py:267-306`) and the
automatic acquisition dataset import (`acquire.py:449-473`) are continuations
of already accepted work. They should not need a browser lease to finish. They
must retain their existing durable job/acquisition identity. `forget` is a
destructive CLI-only mutation (`__main__.py:429-433`, `service.py:308-358`) and
does not pass through any reservation; it remains a separate administrator
authority that the UI must not imply is excluded.

### Hardware checkout

These requests should require the `hardware` lease. Treat the complete Acquire
and optimizer workflow, including simulations and locally saved acquisition
settings, as one coherent hardware workspace:

| Route | Mutation / physical effect |
|---|---|
| `POST /api/acquire/settings` | Writes local settings; with `push:true`, POSTs parameters to the analyzer (`__main__.py:308-312`, `acquire.py:235-265`, `z300.py:141-142`). |
| `POST /api/acquire/runs` | Creates a simulated or live run; live mode later POSTs `/data/firetest` from a daemon thread (`__main__.py:313-316`, `acquire.py:294-436`). |
| `POST /api/acquire/runs/<id>/cancel` | Changes the run ledger and may POST `/data/cancel` (`__main__.py:330-336`, `acquire.py:475-495`, `z300.py:164-171`). |
| `POST /api/optimization` | Creates an operator-stepped acquisition session (`__main__.py:317-320`, `optimization.py:134-163`). |
| `POST /api/optimization/<id>/next` | Persists a batch and delegates to `Acquisition.start()`; a live session fires (`__main__.py:321-324`, `optimization.py:369-447`). |
| `POST /api/optimization/<id>/close` | Mutates optimizer workflow state (`__main__.py:325-329`, `optimization.py:449-463`). |
| `POST /api/motion/command` | Emits `connectTo`, G-code, set-zero, stop, or clear-alarm (`__main__.py:337-340`, `motion.py:345-417`). |

Read routes are not mutations. `GET /api/acquire/settings?refresh=1` and status
probes do contact the analyzer, and `GET /api/motion/status?touch=1` opens a
Socket.IO observation connection, but they do not open the gantry serial port
or issue a physical command (`acquire.py:117-174,208-233`,
`motion.py:298-343`). Camera polling is also observational.

### Background and process authorities

The deployed topology has a `serve` process, a separate analysis `worker`, and
a timer/CLI `sync` process. Each constructs its own `Service`
(`__main__.py:417-468`; the three systemd units under `deploy/alibz/`). Direct
Python callers can also invoke `Acquisition`, `Optimization`, `MotionHub`, and
`Service` methods without the HTTP reservation because enforcement exists only
in `Handler.do_POST()`.

The database supplies useful cross-process serialization for acquisition
creation and optimizer batches (`BEGIN IMMEDIATE` in `acquire.py:338-354` and
`optimization.py:381-411`). In contrast, `Reservation._lock`,
`MotionHub._command_lock`, and `Service.sync_lock` are process-local. The
reservation is loaded from JSON once during construction and never refreshed
(`reservation.py:43-61`), so two `Service` objects can make decisions from
different snapshots and overwrite one another. The sync timer and HTTP sync
can likewise overlap despite `sync_lock` (`service.py:390-414`).

There is no exclusive `serve` process lock. For a Unix socket a second serve
process unlinks the existing pathname before binding (`__main__.py:407-414`),
so the old process and its daemon threads can remain alive while the gateway
connects to the new process.

## Concrete races and unsafe terminal states

1. **Check then act.** Every gated route calls `require()` separately from its
   mutation (`__main__.py:285-340`). `require()` copies `_state` under a lock and
   returns (`reservation.py:202-214`). A concurrent release, override, or TTL
   expiry can occur before the physical or data mutation starts.

2. **Lease lifetime is unrelated to equipment lifetime.** `release()` clears
   the lease without checking an acquisition or gantry operation
   (`reservation.py:166-177`). Expired leases are immediately reservable and
   admins may immediately override (`reservation.py:131-143,179-198`).
   Acquisition rows contain no user or lease generation (`acquire.py:90-100`),
   and motion has no durable row at all. A new holder can cancel the previous
   user's run, push parameters, or move the gantry.

3. **Acquisition and motion can overlap.** The live-fire gate checks Z300
   configuration, confirmation, reachability, and trigger lock only
   (`acquire.py:315-336`). It does not require fresh verified gantry `Idle`.
   Motion gating checks controller status and its own cached `runStatus`, but
   not `Acquisition._busy()` (`motion.py:502-534`). The same user can move or
   set zero during firing, and parameter pushes have no common device lock with
   a run.

4. **The motion lock ends at submission.** `_command_lock` is released as soon
   as `Client.emit()` returns (`motion.py:345-417`). OpenBuildsCONTROL provides
   no Socket.IO acknowledgement (`motion.py:96-106,151-157`). Cached status can
   still say `Idle`, so the next request can emit another move before the first
   appears in status. A successful `emit` is recorded nowhere and proves
   neither receipt nor completion. On an emit exception the in-memory adapter
   says the outcome is uncertain, but restart or lease transfer loses that fact
   (`motion.py:536-555`). `clear_alarm` also clears the local alarm immediately,
   before upstream evidence (`motion.py:408-414`).

5. **The motion observer can disappear during motion.** Its poller stops after
   `idle_seconds` without a browser touch, clears all controller state, and
   closes the transport (`motion.py:640-659`). It has no persisted operation to
   finish or reconcile after reconnect/restart.

6. **Z300 POST failures are misclassified as certain.** A timeout or connection
   loss during `fire_test()` can occur after the analyzer accepted the test,
   but `Acquisition.run()` records `failed`, not `uncertain`
   (`acquire.py:385-397`, `z300.py:78-113,151-162`). A parameter-push timeout is
   similarly reduced to `ValueError` with no durable blocker
   (`acquire.py:244-258`). If the test-list poll times out after a known fire,
   the run is also `failed` (`acquire.py:419-435`). Those states allow another
   run even though physical state needs inspection.

7. **Cancellation is not proof of stopped equipment.** `cancel()` marks the
   run `cancelling`, swallows any Z300 cancellation failure, and returns
   (`acquire.py:475-495`). If `_run_live()` subsequently times out,
   `run()` converts the run to `cancelled` merely because the row said
   `cancelling` (`acquire.py:390-393`). This contradicts the repository rule
   that cancellation requests are not proof of a stop; an unverified stop must
   remain a hardware blocker.

8. **The worker can invalidate a healthy serve-owned run.** Worker startup
   always calls `Acquisition.recover()` (`__main__.py:438-445`). That method
   changes every active acquisition to `uncertain` without checking whether a
   healthy `serve` process still owns its daemon thread (`acquire.py:512-521`).
   `_busy()` ignores `uncertain` (`acquire.py:176-180`), so a worker restart can
   permit a second acquisition while the first thread and analyzer continue.
   The first thread may later overwrite the same row with `succeeded` or
   `failed`. A second serve process causes the same class of error.

9. **Analysis serialization is incomplete across processes.** The worker lock
   protects queue ownership (`__main__.py:28-42,438-451`), but `sync_lock` does
   not protect timer and portal sync from each other. `forget()` checks for
   active jobs before a later deletion transaction, leaving a time-of-check
   gap in which a job can be enqueued (`service.py:318-337`). These are analysis
   consistency limits rather than physical-control hazards, but the analysis
   checkout must not claim to exclude CLI/timer activity.

## Minimal robust implementation contract

### Durable leases

Use a SQLite `reservations` table keyed by resource (`hardware`, `analysis`)
with holder, note, acquired/expires timestamps, and a monotonically increasing
generation. All reserve/renew/release/override/require decisions must load the
row inside `BEGIN IMMEDIATE`; no cached authoritative state. Two different users
may hold the two resources concurrently, and one user may hold both.

`GET /api/reservation?resource=hardware|analysis` and POST bodies containing
`resource` are the explicit API. For compatibility, omission means `hardware`.
Reject unknown resources. One-time migration of a valid legacy
`reservation.json` imports it as `hardware` only; `analysis` starts free. The
migration must itself be transactional and idempotent across processes.

`reservation.enabled:false` may waive the user-lease requirement as the current
operator escape hatch, but it must not disable physical-operation
serialization, cross-device interlocks, uncertain blockers, or durable recovery.

### Atomic authorization and operation fencing

Do not hold a SQLite transaction or Python mutex during network I/O. Instead,
the physical method must atomically:

1. verify a live hardware holder and generation;
2. verify no conflicting active/uncertain operation;
3. persist an operation intent with action ID, holder, lease generation,
   request digest, target, and `prepared` state; and
4. commit before dispatch.

After commit, transition through `submitted` and a terminal state without ever
automatically replaying a prepared/submitted request after an ambiguous return
or restart. Any process that continues an operation verifies its stored owner
and generation; it never silently adopts a later lease. Active and uncertain
operations make hardware release, expiry takeover, and admin override return a
conflict that identifies the blocking operation. Renewal by the same holder is
allowed. A restart changes unfinished physical operations to `uncertain`, not
`failed` or free.

For queued analysis work, the lease authorizes enqueue/import/metadata intent;
the worker may finish after lease expiry. For HTTP sync, either register an
analysis operation until the thread completes or acquire a cross-process sync
lock; at minimum, no second sync should begin from the timer or another process.

All hardware entry points must accept the authenticated user or an authorization
context; route-only checks are insufficient. Store holder/generation on live
acquisitions and optimization batches. Simulated acquisitions and optimizer
sessions use the hardware checkout for workflow coherence but do not create a
physical blocker after their purely local work is accepted.

### Cross-device rules

- A live Z300 start requires a fresh verified gantry observation with
  `runStatus == "Idle"` and no active/uncertain motion. If gantry observation is
  unavailable, refuse live firing; simulation can proceed.
- While a live acquisition is queued, running, cancelling, or uncertain, refuse
  `connect`, `jog`, `goto`, `set_zero`, and `clear_alarm`. Keep `abort` available
  as a recovery action when the controller identity is verified. A cancel
  request remains available to the owner/admin recovery path, but it does not
  clear the blocker until stopped state is established.
- `save_settings(push=true)` conflicts with any live acquisition. Local settings
  changes are serialized under the hardware lease as well.
- An `uncertain` acquisition or motion stays blocking until an explicit,
  authenticated reconciliation records the observed safe state and operator,
  or a documented administrator recovery procedure does so. A new run ID is not
  reconciliation.

The analysis checkout stays independent: analysis uploads, metadata, and queued
analysis can proceed while hardware is reserved or active. Acquisition-produced
datasets and optimizer metrics are append-only continuations of the hardware
operation and may finish without taking the analysis lease.

### Motion completion evidence

OpenBuildsCONTROL cannot correlate a command to an acknowledgement. Persist a
pre-submit observation marker and accept only a strictly newer status from the
same verified controller:

| Action | Minimal completion evidence |
|---|---|
| `connect` | Expected active port and baud, GRBL firmware identity, fresh nonzero connection state. |
| `jog` / `goto` | Fresh `Idle` plus the commanded axis at the expected target within configured tolerance. A non-Idle transition is useful but cannot be mandatory because a short move may occur between 100 ms status broadcasts. |
| `set_zero` | Fresh `Idle` plus the selected work-coordinate axis near zero. |
| `abort` | Fresh verified controller state showing `Idle`/stopped after submission. |
| `clear_alarm` | Fresh status no longer in alarm and no new alarm indication; do not clear the cached alarm at emit time. |

If this evidence does not arrive by a bounded deadline, the connection drops,
or the process restarts, mark the operation `uncertain` and retain the hardware
blocker. Position/status evidence is still observational, not cryptographic
correlation: an external GUI or pendant can issue a command that happens to
satisfy it. The UI and docs must continue to state that limit.

### Process ownership

Only `serve` owns acquisition threads and the gantry connection. Add a
cross-process serve-owner lock/epoch before recovery and socket binding. The
analysis worker must not call `Acquisition.recover()` or physical optimizer
recovery; it cannot distinguish a dead serve owner from a healthy one. On a
verified serve takeover/restart, unfinished physical rows become uncertain once.
Do not unlink and replace a live Unix socket owned by another process.

## Required verification

- Two independent `Service` instances on the same data directory contend
  correctly for each lease; hardware and analysis can be held by different
  users; legacy state migrates to hardware only.
- A barrier-controlled release/override race cannot pass between authorization
  and physical intent persistence. Release, expiry takeover, and admin override
  fail while a live/uncertain operation exists and succeed after proven
  completion or reconciliation.
- Every HTTP route maps to the intended resource, including settings, simulated
  runs, all optimizer routes, cancellation, and motion. Holding the other lease
  is insufficient.
- A live fire is refused unless gantry state is fresh and Idle; motion is
  refused through acquisition queued/running/cancelling/uncertain states. Abort
  and cancel recovery behavior is tested separately.
- OpenBuilds fake-server tests omit acknowledgements. A command remains active
  until a newer matching status arrives, a timeout becomes uncertain, no
  automatic retry occurs, and restart preserves the blocker.
- Z300 tests inject timeout/disconnect after POST acceptance for fire, settings,
  and cancel. These outcomes become uncertain and cannot free the hardware
  checkout.
- Starting/restarting the analysis worker during a deliberately blocked live
  acquisition does not mutate its row. A second serve owner is refused.
- Portal and timer sync cannot overlap across processes. Analysis worker jobs
  accepted under a lease continue after that lease is released.

Existing tests cover single-instance lease CRUD, per-method gate matrices,
idempotent acquisition IDs, and some cross-`Service` optimizer transactions.
They do not cover the require/action race, lease transfer during physical work,
motion completion, cross-device exclusion, ambiguous Z300 writes, worker versus
healthy serve recovery, or cross-process reservation state.
