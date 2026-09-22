# Final acquisition cancellation and reconciliation review

Date: 2026-09-21  
Reviewed frozen staging runtime:
`/private/tmp/pantheum-checkouts-20260921/pantheum/alibz/acquire.py` and the
uncertain-acquisition reconciliation path in `reservation.py`. No runtime source
was edited.

## Outcome

No remaining safe-handoff gap was found in the bounded scope. The current code
serializes pre-dispatch cancellation, fire submission, cancellation, terminal
completion, exception finalization, and reconciliation through the same
cross-process operation lock. Reconciliation cannot clear an acquisition while
its executor remains in the active `queued`, `running`, or `cancelling` states.

## Verified ordering

- `run()` claims a queued row with a conditional update (`acquire.py:435-442`).
- `_run_live()` holds the dispatch guard, rechecks `running`, and persists
  `dispatch_started_at` immediately before `fire_test()` (`acquire.py:529-545`).
  Thus a cancel that wins before this point can prove that this run sent no
  vendor request.
- `cancel()` holds `operation_lock()` for the complete decision and vendor
  cancellation submission (`acquire.py:598-652`). A queued or running row with
  no `dispatch_started_at` becomes `cancelled` without contacting the vendor
  (`acquire.py:610-621`). A dispatched live run becomes `cancelling`.
- A failed/ambiguous vendor cancel leaves the row `cancelling`, rather than
  prematurely making it reconcilable, until the run executor observes results
  or exits its polling path (`acquire.py:627-649`).
- If results arrive after cancellation, normal completion takes the same lock,
  sees `cancelling`, and records `uncertain`; it cannot overwrite cancellation
  with success (`acquire.py:487-507`).
- `_CancelledBeforeDispatch`, `Z300Error`, generic exception, and normal
  completion finalization all hold `operation_lock()` across the acquisition
  state and hardware-operation state writes (`acquire.py:455-520`). This closes
  the previous window in which reconciliation could clear the two records
  between commits and hand off hardware before the executor restored a hold.
- `reconcile_operations()` now takes `operation_lock()` before its fresh-idle
  observation and transaction, and explicitly refuses any live
  queued/running/cancelling acquisition (`reservation.py:471-508`).

## Adversarial checks

A controlled pre-dispatch race returned `cancelled`, ended `cancelled`, completed
the prepared operation, and invoked neither vendor `fire_test` nor vendor
`cancel`.

A controlled failed cancel while a run executor was blocked in its polling path
returned `cancelling`; reconciliation raised `Conflict` while that executor was
alive. After the executor exited it finalized `uncertain`, and reconciliation
then cleared exactly one acquisition and one operation.

A controlled exception-finalization race paused inside `finish_operation()`
after the acquisition became uncertain. A concurrent reconciliation thread
remained blocked on `operation_lock()` until both finalization writes completed,
then reconciled the stable records.

Targeted repository tests also passed: 21/21 across
`SimulateRunTests` and `SplitReservationSafetyTests`. These include queued live
cancel, running live cancel, late results after failed cancel, two-process lease
contention, guarded release, and expired-acquisition safety holds.

The checks used fake clients and temporary data directories. They establish the
software ordering contract; they do not constitute a physical Z300 deployment
test.
