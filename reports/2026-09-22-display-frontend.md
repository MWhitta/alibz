# Analyzer display controls frontend

Implemented the staged Acquire-panel display controls in
`/private/tmp/pantheum-display-controls-20260922`.

- Added independent display status rendering with Awake, Asleep, Unknown, and Disabled states.
- Added `Sleep display` and `Wake display` actions gated by the display API contract.
- Integrated visible Acquire-tab polling with an in-flight guard; display failures become Unknown without erasing analyzer-independent controls.
- Preserved acquisition poll authentication/error handling, discarded stale display GET responses around commands, and skipped display polling while hidden, outside Acquire, or during a command.
- Applied immediate hardware-checkout gating, including same-holder safety-hold recovery for Wake.
- POST command failures report through `NOTICE_SCOPE.acquireStatus`; successful commands refresh display status.
- Added UI harness coverage for gating, blocked live firing with wake recovery, successful sleep, command errors, analyzer endpoint failure, stale GET ordering, and checkout loss.

Verification:

- `node --check web/alibz/app.js` — passed.
- `node --test tests/test_alibz_ui.cjs` — 42 passed, 0 failed.
- No live browser actions or deployment performed.
