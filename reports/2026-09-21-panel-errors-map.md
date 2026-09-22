# Panel notice/error scope map

Inspected only `/private/tmp/alibz-panel-errors-20260921/web/alibz/{app.js,index.html,styles.css}` and `/private/tmp/alibz-panel-errors-20260921/tests/test_alibz_ui.cjs`. No files in that worktree were changed and no tests were run, per brief.

## Recommended scope contract

Give each existing card an explicit `id` and place/resolve a notice region inside it. Pass the scope explicitly through `showNotice(message, type, login, scopeId)`, `showError(error, suffix, scopeId)`, and `showAcquireError(error, suffix, scopeId)`. Keeping scope as an argument is safer than consulting `document.activeElement` after an `await`, after a dialog closes, or during polling. A global `workspace-scope` remains useful for startup/status operations spanning several cards.

| Proposed scope ID | Existing boundary / heading |
|---|---|
| `workspace-scope` | global notice currently before all panels (`index.html:33`); startup/status spans header, datasets, queue, engines, result |
| `hardware-checkout-scope` | Hardware checkout article (`index.html:36-46`, heading `hardware-checkout-heading`) |
| `analysis-checkout-scope` | Analysis checkout article (`index.html:47-56`, heading `analysis-checkout-heading`) |
| `acquire-status-scope` | Analyzer status article (`index.html:60-63`, heading `acquire-status-heading`) |
| `camera-scope` | Cameras article (`index.html:65-70`, heading `acquire-live-heading`) |
| `motion-scope` | Motion control article (`index.html:72-122`, heading `motion-heading`) |
| `acquire-params-scope` | Acquisition parameters article (`index.html:124-171`, heading `acquire-params-heading`); its existing inline validation alert is `acquire-params-error` (`index.html:163`) |
| `acquire-run-scope` | Trigger a test article (`index.html:173-180`, heading `acquire-run-heading`) |
| `optimization-scope` | Stepped acquisition study article (`index.html:182-207`, heading `optimization-heading`) |
| `acquire-runs-scope` | Recent acquisitions article (`index.html:209-212`, heading `acquire-runs-heading`) |
| `datasets-scope` | Spectra library aside (`index.html:217-222`, heading `datasets-heading`) |
| `spectrum-scope` | selected spectrum article (`index.html:225-229`, heading `spectrum-heading`) |
| `actions-scope` | Run analysis article (`index.html:231-241`, heading `actions-heading`) |
| `result-scope` | alibz result article (`index.html:243-246`, heading `result-heading`) |
| `metadata-scope` | Spectrum metadata section (`index.html:248-260`, heading `metadata-heading`) |
| `queue-scope` | Recent jobs article (`index.html:263`, heading `queue-heading`) |
| `engines-scope` | Engine readiness/sync/upload/example article (`index.html:265-279`, heading `engines-heading`) |

Dialogs should inherit their launching card's scope instead of becoming durable scopes: upload -> `engines-scope`; simulate/fire -> `acquire-run-scope`; hardware reconcile -> `hardware-checkout-scope`; optimization-next -> `optimization-scope`; motion alarm -> `motion-scope` (`index.html:288-297`). This ensures the result remains visible after the dialog closes.

## Complete call-site routing

Helper forwarding must preserve scope: `showError`'s call to `showNotice` (`app.js:81-82`), and all three branches of `showAcquireError` (`app.js:140-143`). Authentication should retain the caller's scope; only callers explicitly assigned `workspace-scope` should become global.

| Scope | `showNotice` / `showError` / `showAcquireError` call sites |
|---|---|
| `workspace-scope` | startup/status loading, success, failure (`app.js:1501,1510,1519`); background job-poll authentication (`app.js:1550`) |
| `spectrum-scope` | dataset detail loading, success, failure (`app.js:1482,1490,1494`) |
| `actions-scope` | analysis submit, accepted, failure (`app.js:1564,1568,1574`) |
| `metadata-scope` | metadata saving, success, failure (`app.js:1599,1608,1610`) |
| `engines-scope` | sync progress/success/failure (`app.js:1637,1640,1643`); example creation/success/failure (`app.js:1653,1660,1662`); upload oversize/progress/success/failure (`app.js:1681,1684,1691,1694`) |
| `acquire-status-scope` | acquisition poll and explicit refresh authentication (`app.js:1756,1769`) |
| dynamic checkout scope | renewal/refresh authentication (`app.js:1888,1900`) and action progress/success/failure (`app.js:1918,1921,1923`): choose `hardware-checkout-scope` when `resource === "hardware"`, otherwise `analysis-checkout-scope` |
| `hardware-checkout-scope` | reconcile progress/success/failure (`app.js:1949,1953,1956`) |
| `camera-scope` | frame-stream authentication and camera-status authentication (`app.js:2450,2577`) |
| `acquire-params-scope` | load mode validation/progress/success/failure (`app.js:2845-2852`); save mode/parameter validation, progress, success, failure (`app.js:2858-2869`) |
| `acquire-run-scope` | run validation/progress/success/failure (`app.js:2955,2960,2969,2981`) |
| `optimization-scope` | session-load authentication (`app.js:3074`); refresh progress/success/failure (`app.js:3079,3093,3096`); create input/grid/parameter/mode errors, success, failure (`app.js:3299,3306,3310,3312,3323,3325`); next-batch success/failure (`app.js:3373,3379`); close success/failure (`app.js:3386-3387`) |
| `acquire-runs-scope` | cancellation progress/success/failure (`app.js:3402,3405,3408`) |
| `motion-scope` | motion-status authentication, command failure, invalid position (`app.js:3612,3632,3645`) |

No current call site naturally belongs to `datasets-scope`, `result-scope`, or `queue-scope`; those cards are updated by render functions rather than directly initiated operations. They still merit IDs for a consistent panel boundary and future errors.

## Async paths requiring early scope capture

If the implementation derives scope from the initiating element, capture it synchronously before the first `await` and reuse that captured ID for every later message/error:

- `selectDataset` (`app.js:1467-1495`): selection can change during the request; retain `spectrum-scope` alongside the existing epoch.
- `analyze` (`app.js:1557-1580`) and metadata submit (`app.js:1586-1615`): selection/focus may change while requests run; retain `actions-scope` / `metadata-scope`.
- sync, example, upload handlers (`app.js:1634-1698`): nested `loadStatus`/`selectDataset` emit their own messages, and upload closes its dialog before completion; retain `engines-scope` for the outer operation. Nested calls should not overwrite another scope's notice.
- acquisition poll/refresh (`app.js:1748-1773`), checkout renewal/refresh/action (`app.js:1882-1929`), and reconcile (`app.js:1944-1962`): capture the fixed or resource-derived scope before requests; reconcile closes its dialog before posting success.
- camera frame/status requests (`app.js:2430-2453,2572-2580`): background async errors must be pinned to `camera-scope`, independent of current focus/tab.
- parameter load/save (`app.js:2843-2873`): retain `acquire-params-scope`; `saveAcquireSettings` awaits a status refresh after its success.
- run submit (`app.js:2951-2986`): capture `acquire-run-scope` before request; both success and terminal failure may close the modal.
- optimization session/refresh/create/next/close (`app.js:3049-3099,3295-3388`): pin all messages to `optimization-scope`; refresh and next execute nested awaits, while next closes the dialog before success.
- cancellation (`app.js:3400-3410`) and motion command/status (`app.js:3607-3635`): pin to `acquire-runs-scope` / `motion-scope`; both await follow-up refreshes.

For interval/background callers (`pollTick`, checkout renewals, acquisition/camera/optimization/motion polling), use a literal scope ID rather than deriving from focus. The initiating UI element no longer exists as a meaningful source when the timer fires.

## CSS and test evidence

The only notice styling is global `.notice`, `.notice.error`, and `.notice.success` (`styles.css:41-43`); no panel-local positioning or hidden/empty state exists. The markup has one global live region (`index.html:33`) plus the acquisition-parameter validation alert (`index.html:163`). `tests/test_alibz_ui.cjs` contains no assertions against `#notice`, notice text/classes, or `showNotice`; relevant checkout and async behavior tests exist (for example `tests/test_alibz_ui.cjs:583` and `tests/test_alibz_ui.cjs:791`), but panel-scope isolation and post-await routing are currently unverified.

Suggested tests: one error in each of two scopes does not overwrite the other; dynamic hardware/analysis checkout routes correctly; an upload/run/optimization dialog can close before resolution and its message still lands in the launching card; a delayed response after focus/tab changes retains its captured scope; background polling authentication lands in its fixed panel scope; login links and `role=status`/`aria-live` semantics survive local routing.
