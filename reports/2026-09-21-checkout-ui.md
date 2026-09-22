# Separate hardware and analysis checkout UI

## Outcome

Implemented two independent checkout surfaces and gates for the alibz UI. Hardware checkout controls every Acquire, optimizer, laser, simulation, and gantry mutation; analysis checkout controls upload, sync, example creation, metadata writes, and analysis enqueueing. Inspection and other read-only views remain available. Unknown and expired checkout state fails closed. A hardware `safety_hold` owner retains only the recovery actions needed to cancel acquisition or abort motion.

The shared cards display holder/status, countdown, notes, `blocked_reasons`, release eligibility, and backend-authorized override/reconciliation actions. Hardware reconciliation requires a checked operator acknowledgement and sends exactly `POST api/motion/command {action:"reconcile", confirm:{hardware_idle:true}}`; it is never automatic.

No provider switch occurred.

## Changes

- `web/alibz/index.html:35-57` adds compact Hardware checkout and Analysis checkout cards outside both tabs, with resource-specific reserve/release/override controls and the exceptional hardware reconciliation entry point. `web/alibz/index.html:207-208` removes the former Analyze-only global reservation panel while retaining the existing workspace. `web/alibz/index.html:283` adds the explicit idle-inspection reconciliation dialog.
- `web/alibz/styles.css:207-212` styles the shared checkout grid, compact cards, actions, and blocked-reason list. `web/alibz/styles.css:312` stacks the cards on narrow screens.
- `web/alibz/app.js:20-24` replaces global reservation state with independent hardware/analysis state, busy flags, polling, renewal timers, and stable renewal keys.
- `web/alibz/app.js:1432-1446` gates analysis mutations only with the analysis checkout.
- `web/alibz/app.js:1728` refreshes both resource checkouts when either tab is selected.
- `web/alibz/app.js:1775-1965` implements fail-closed resource gates, independent rendering/actions/polling/renewal, `blocked_reasons`, `can_release`, `can_override`, and `can_reconcile`. Renewal timers survive ordinary 30-second refreshes and restart only when ownership/lease/TTL changes. Reconciliation remains available to an expired/safety-hold owner only when the backend returns `can_reconcile:true`, requires explicit acknowledgement, and becomes retryable after a refusal.
- `web/alibz/app.js:2835-2841` gates acquisition parameter writes with hardware ownership. `web/alibz/app.js:2895-2903` gates simulated and live acquisition starts. `web/alibz/app.js:3116-3128` gates optimizer mutations. `web/alibz/app.js:3394-3398` permits cancellation as a narrowly scoped safety-hold recovery action. `web/alibz/app.js:3460-3488` gates all motion mutations while retaining owner Abort during a safety hold. `web/alibz/app.js:3618` refreshes both checkout resources on visibility return. `web/alibz/app.js:3644-3658` exposes checkout helpers only for focused Node tests. `web/alibz/app.js:3661` starts resource polling without auto-checkout.
- `tests/test_alibz_ui.cjs:78-105` makes the fake timers injectable for renewal testing. `tests/test_alibz_ui.cjs:144-151` adds a checkout fixture. `tests/test_alibz_ui.cjs:537-610` adapts existing optimizer tests to explicit hardware ownership without changing their assertions. `tests/test_alibz_ui.cjs:618-665` proves independence in both directions. `tests/test_alibz_ui.cjs:668-725` proves unknown/expired/disabled-hardware fail-closed behavior, safety-hold recovery scoping, and stable renewal timers. `tests/test_alibz_ui.cjs:727-804` proves blocked release reasons, reconciliation eligibility and payload, explicit acknowledgement, and retry after refusal.

## Verification

Before changes:

- `node --check web/alibz/app.js && node --test tests/test_alibz_ui.cjs`: 22 passed, 0 failed.
- `PYTHONPATH=src python3 -m pytest tests/ -q`: 530 passed, 139 failed, 37 errors, 26 skipped, 199 subtests passed. The broad failures/errors were present before this UI work and primarily came from sandbox-denied socket binding (`PermissionError: [Errno 1] Operation not permitted`).
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`: `.venv/bin/python` was absent in this staging copy; the shell still ran pytest with the remaining `PYTHONPATH` and produced the same 530 passed, 139 failed, 37 errors, 26 skipped, 199 subtests passed.

After changes:

- `node --check web/alibz/app.js && node --test tests/test_alibz_ui.cjs`: 31 passed, 0 failed. All 22 pre-existing UI tests still pass, providing the regression evidence for optimizer, motion-adjacent rendering, and analysis inspection behavior; 9 focused checkout tests were added.
- `PYTHONPATH=src python3 -m pytest tests/ -q`: 526 passed, 143 failed, 37 errors, 26 skipped, 199 subtests passed. This shared staging tree received concurrent backend checkout edits/tests between the before and after runs; failures remain dominated by the same sandbox socket restriction, plus backend work outside this UI task.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`: `.venv/bin/python` remained absent; fallback pytest result was 526 passed, 143 failed, 37 errors, 26 skipped, 199 subtests passed.
- Parent-session static-browser QA rendered and visually inspected both checkout cards in their unknown-backend state; both were visible, laid out correctly, showed unavailable status, and hid checkout actions.

## Unverified

- The staging copy has no `.git` directory, so a Git diff/check was unavailable. Scope was limited to the four authorized UI/test files plus this report, and all pre-existing Node UI tests remain green.
- The pymatgen-enabled Python invocation could not actually add a site-packages path because this staging copy has no `.venv/bin/python`.
- End-to-end browser behavior against the completed backend and deployed service was left to the main session as requested. Static browser QA and mocked Node interaction tests do not verify deployment, authentication, real renewal timing, or live hardware state.
