# Split-checkout lease review

Date: 2026-09-21  
Reviewed staging snapshot:
`/private/tmp/pantheum-checkouts-20260921/pantheum/alibz/reservation.py` and
`__main__.py`. Motion and acquisition behavior were out of scope except where
`__main__.py` must propagate lease authorization. No source was edited.

## Result

The shared advisory lock plus SQLite lease rows fix the original cross-instance
stale-JSON race. An adversarial fork test with two `Service` instances in
separate processes produced exactly one successful hardware reserve and one
`Conflict`. A guard held by one instance also made `release()` in a second
instance wait until the guarded side effect returned. The explicit route map is
correct: upload/sync/metadata/analyze/example use `analysis`; Acquire,
optimization, cancel, and motion use `hardware`; omitted resource defaults to
`hardware`.

Four safety/correctness gaps remained in the reviewed snapshot.

## Findings

### High: an expired safety-hold owner still passes the normal hardware guard

`_status_locked()` maps an expired lease with blockers to `state='safety_hold'`
and retains `held_by_you` (`reservation.py:411-431`). `_require_locked()` rejects
only `free` and `expired`; it accepts `safety_hold` for the old holder
(`reservation.py:543-554`).

Reproduction: reserve hardware with `ttl_seconds=0`, insert a pending hardware
operation, then call `guard()` as the old holder. Status reported
`safety_hold`, zero seconds remaining, yet the guard yielded successfully. This
permits ordinary new starts, settings pushes, optimization batches, and motion
after expiry without the explicit re-checkout that the max-TTL design requires.

Normal authorization must test the lease timestamp directly and reject every
expired lease, whether or not blockers exist. A safety hold should authorize
only narrow recovery/reconciliation paths. The same holder can explicitly
reserve again if policy permits; that transition is auditable and distinct from
silently retaining full authority.

### High: live acquisition authorization is not propagated from the HTTP guard

`POST /api/acquire/runs` and `POST /api/optimization/<id>/next` enter a hardware
guard but discard the returned lease (`__main__.py:319-330`). The called methods
accept `owner` and `lease_generation`, but omission uses `local operator` and
`legacy`. Consequently live acquisition and physical-operation rows are not
bound to the authenticated checkout generation.

Both handlers must bind `as lease` and pass
`owner=user, lease_generation=lease['generation']`. Settings push and motion
already do this (`__main__.py:313-318,344-349`).

### High: malformed timestamps fail open as expired

`_expired()` returns `True` for missing, malformed, or timezone-incompatible
timestamps (`reservation.py:318-323`). `reserve()` and `override()` then treat
the row as an ordinary expired lease. I replaced a current hardware row's
`expires_at` with each of `garbage` and the naive future timestamp
`2099-01-01T00:00:00`; status offered reserve/override and a second user
successfully replaced the hardware holder in both cases.

An invalid hardware timestamp is corrupt state, not evidence of expiry. Parse
and require a timezone-aware timestamp before every authorization/transfer; on
failure return the existing fail-closed corrupt status. Legacy migration must
apply the same timezone-aware validation. It currently accepts the naive future
timestamp, imports it, then immediately exposes it as expired.

### Medium: same-owner explicit reserve does not start a fresh max-TTL window

With no blockers, a second explicit `reserve()` by the current holder updates
only note and expiry (`reservation.py:464-466`). An adversarial check confirmed
both `acquired_at` and `generation` remained unchanged. After the old
`acquired_at + max_ttl_seconds` cap, the next `renew()` clamps back to the old
past cap, so the newly reserved checkout expires immediately.

When blockers exist, retaining the generation is necessary because accepted
work is tied to it. When no blocker exists, explicit reserve should create a
fresh `acquired_at`, expiry, and generation, matching the original documented
contract. A dedicated test should cover re-checkout after the old max-TTL cap.

### Medium: valid-shape legacy JSON can crash migration through unchecked fields

Migration validates only holder and expiry shape before binding `note`,
`acquired_at`, and `overridden_from` (`reservation.py:341-374`). A legacy object
with an object-valued `note` raised `sqlite3.ProgrammingError` during `Service`
construction instead of producing the fail-closed corrupt status. A malformed
string `acquired_at` was stored and later causes `renew()` to reset its cap from
the current time.

Validate all bound fields: bounded string note, optional string
`overridden_from`, and timezone-aware `acquired_at`; distinguish an absent
`acquired_at` (a documented fallback may be used) from a malformed one (mark
corrupt). The reviewed code correctly distinguishes parsed JSON `null` from a
JSON parse/read failure: absent and `null` migrated free, while invalid JSON
reported corrupt.

## Locking notes

`_exclusive_file_lock()` combines one process-wide `RLock` per resolved service
root with `flock`, and `guard()` holds it across the authorized side effect
(`reservation.py:219-259,561-567`). Every HTTP mutation and every lease transfer
uses that guard/lock, so the tested release-versus-guard race is closed. Status
reads need no advisory lock because SQLite commits expose lease rows atomically.

`require()` remains a check-only public method without the advisory lock held
after return (`reservation.py:556-559`). Production HTTP code no longer uses it.
It should be documented as diagnostic/deprecated or removed once compatibility
tests are updated so a later caller does not recreate the original check-then-act
race.

The old `tests/test_alibz_reservation.py` has not been adapted to split-resource
semantics. Its unit portion produced 14 passes, one assertion failure, and three
expected semantic errors; five HTTP fixtures could not bind loopback in this
sandbox. New tests are needed for two-process contention, guarded release,
expired safety holds, timestamp corruption, migration field types, fresh
same-owner generations, and owner/generation propagation from both live-start
routes.
