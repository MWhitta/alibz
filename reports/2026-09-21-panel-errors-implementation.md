# Panel-scoped alibz notices implementation

Date: 2026-09-21

## Outcome

Implemented the UI fix only in `/private/tmp/alibz-panel-errors-20260921`; no live or sibling source was edited and no provider switch occurred. Errors, progress, and local success messages now use a scope captured before each asynchronous operation. There is no mutable “last clicked panel” state. The original global `#notice` remains the fallback for initial/page-wide status and for mixed deployments where a requested local slot is absent.

The final parent review added a dialog-to-parent mapping at `web/alibz/app.js:76-99`: dialog notices are mirrored to their launching panel, so a retryable error is visible in an open modal and persists at the panel after the modal closes. Message text still uses `textContent`; the sign-in link remains a separately created anchor, preserving escaped error text.

## Changed staged files

- `web/alibz/app.js:9-18` defines immutable scope IDs; `web/alibz/app.js:76-107` selects scoped nodes, falls back to the global notice if a slot is absent, mirrors dialog notices to the parent panel, unhides the slot, and preserves the authentication link. `web/alibz/app.js:165-168` passes scope through `showAcquireError`.
- `web/alibz/app.js:1493-1545` scopes selected-spectrum loading and makes `loadStatus(noticeScope = null)` retain global initial-load behavior while supporting nested action scope.
- `web/alibz/app.js:1584-1725` scopes analysis, metadata, sync, example import, upload validation/progress/failure, and follow-up status loads. Upload errors stay in the open upload dialog; accepted uploads report in import/sync.
- `web/alibz/app.js:1780-1802` scopes analyzer-status background authentication failures.
- `web/alibz/app.js:1916-1993` scopes checkout polling, renewal, actions, and hardware reconciliation, including its open dialog.
- `web/alibz/app.js:2467-2488` and `web/alibz/app.js:2611-2616` scope camera-stream and camera-status authentication failures.
- `web/alibz/app.js:2883-2910` scopes acquisition-parameter validation, progress, success, and errors.
- `web/alibz/app.js:2993-3025` scopes run validation/progress to the active fire/simulation dialog and successful acceptance to the run panel. The dialog-parent mirror covers both retryable and terminal errors without changing retry IDs or close behavior.
- `web/alibz/app.js:3092-3140` and `web/alibz/app.js:3340-3434` scope optimization polling, validation, create/refresh/next/close actions, and next-batch dialog errors.
- `web/alibz/app.js:3448-3456` scopes run-cancellation notices to recent acquisitions.
- `web/alibz/app.js:3656-3695` scopes motion polling authentication, command failures, and numeric-position validation.
- `web/alibz/index.html:37,49,63,69,77,130,180,190,218,227,236,243,256,262,280` adds hidden accessible `role=status`, `aria-live=polite` slots as the first child of each relevant panel.
- `web/alibz/index.html:303,305-306,308,310` adds equivalent slots inside upload, simulation, reconciliation, live-fire, and optimization-next dialogs.
- `web/alibz/styles.css:44` makes local notices span metadata/grid layouts and remain visible with `position: sticky; top: 12px` while a tall panel is scrolled, without scrolling or moving focus.
- `tests/test_alibz_ui.cjs:873-897` verifies two simultaneous checkout errors remain in independent scopes and that an authentication notice includes `/portal/login`.
- `tests/test_alibz_ui.cjs:899-918` verifies a retryable simulated-run failure remains visible in its still-open confirmation dialog.

## Scope IDs

Panel slots: `notice-hardware-checkout`, `notice-analysis-checkout`, `notice-acquire-status`, `notice-cameras`, `notice-motion`, `notice-acquire-params`, `notice-acquire-run`, `notice-optimization`, `notice-acquire-runs`, `notice-datasets`, `notice-spectrum`, `notice-analysis`, `notice-results`, `notice-metadata`, and `notice-import-sync`.

Dialog slots: `notice-upload`, `notice-hardware-reconcile-dialog`, `notice-acquire-simulate-dialog`, `notice-acquire-fire-dialog`, and `notice-optimization-next-dialog`.

## Verification

Baseline, before staged changes:

- `node --test tests/test_alibz_ui.cjs`: 33 passed, 0 failed.
- `PYTHONPATH=src python3 -m pytest tests/ -q`: collection stopped with 30 errors because host Python lacks SciPy (`ModuleNotFoundError: scipy`).
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`: the host repository was unchanged by this staged task; the captured final run below establishes the same suite state before and after.

After staged changes, before the parent's final dialog-parent helper adjustment:

- `node --check web/alibz/app.js`: passed.
- `node --test tests/test_alibz_ui.cjs`: 35 passed, 0 failed.

Repository-wide after check (the staged UI files are outside the host repository, so this also represents the unchanged baseline host tree):

- `PYTHONPATH=src python3 -m pytest tests/ -q`: 30 collection errors, same missing-SciPy limitation as baseline.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`: 326 passed, 4 skipped, 55 subtests passed, 1 warning, in 274.19 seconds.

## Limits and handoff

The parent agent requested no further staged edits after adding the final dialog-parent mirror at `web/alibz/app.js:76-99`; therefore this agent did not rerun Node checks after that final helper-only adjustment. The parent explicitly owns that rerun, the three-way merge against the live baseline, browser geometry/interaction verification, and deployment. No real browser, live service, hardware, or deployment was exercised by this agent.
