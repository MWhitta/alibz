# alibz errors beside their source panel — 2026-09-21

Deployed at 16:57:34 PDT. Errors and action feedback now appear at the top of
the relevant subpanel, with a sticky notice while that panel is in view. Open
confirmation dialogs have their own visible notice; dialog messages are also
retained in the launching panel so closing a dialog does not hide the failure.

## Behavior and review

Twenty accessible, initially hidden notice slots cover checkouts, analyzer,
cameras, motion, parameters, runs, optimization, spectra, analysis, metadata,
import/sync, and dialogs. Asynchronous handlers pass an explicit scope rather
than consulting the most recent click or current focus. Independent errors do
not overwrite each other. Authentication links and text-only rendering remain.
A missing local slot falls back to the page notice for mixed asset versions.
Only initial whole-workspace status remains global; refreshes caused by imports
or sync inherit their initiating panel.

Main review caught a terminal refusal that closed a run dialog before writing
to its now-hidden notice. A dialog-to-panel mapping fixes this and preserves
errors after a dialog closes. Tests cover both retryable 503 and refused 403
outcomes. Hardware confirmation, request IDs, API payloads, and retry behavior
were unchanged. No backend, configuration, service, or physical action changed
as part of this deployment.

## Verification

- Final source and deployed variant: `node --check` passed; 36/36 UI tests passed.
- Whitespace check passed for changed frontend files and the UI test file.
- Browser baseline reproduced the bug: a gantry error at y=-762.375 with a
  720-pixel viewport. After the fix the same error was visible at y=12.
- A separate optimization error appeared inside its own panel without replacing
  the gantry message or changing the global status notice.
- Parameter authentication failure retained its local sign-in link at y=12.
- A retryable run error remained visible inside its open confirmation dialog,
  and the launching panel retained the same message.
- The labeled example-data view and disabled/unknown engine states rendered.
- Final browser check was repeated against the updated live study UI: its
  optimization error was visible at y=178.922 with no global overwrite.
- All three served assets returned HTTP 200 and matched the tested SHA-256 hashes.

Browser checks used `output/playwright/panel-error-preview.py` on localhost.
All POST requests were synthetic failures and physical adapters were disabled.
Screenshots are `output/playwright/panel-errors-{motion,optimization,dialog,final,example}.png`.
The local browser and preview server were shut down afterward. Production
sign-in was not exercised; served content was verified through the authenticated
local portal socket. No real hardware action was issued.

## Concurrent work and deployment

The first deployment dry run correctly refused a concurrent live frontend edit,
before writing any file. The new live assets exactly matched the original source
baseline containing the separate shot-plan UI. The final notice fix was applied
to that updated baseline, preserving the concurrent deployment. All 36 tests and
the final browser check passed on this composition.

Source backup:
`/Users/mwhittaker/Projects/github/pantheum-panel-errors-source-backup-20260921T165416`.
Live backup:
`/home/mwhittaker/pantheum-panel-errors-backup-20260921T165734`.
Runtime bundle:
`/home/mwhittaker/alibz-panel-errors-deploy-20260921`.
Only `web/alibz/app.js`, `index.html`, and `styles.css` were installed.
No service restart was needed. Private configuration remained unchanged during
this deployment (SHA-256 `c4e64e1a31b3c2699aafdcbbe4c85261948e859b087fb0365a1db74f852d4fa3`).
That configuration had changed in the other deployment; this task did not
inspect or modify it and makes no new claim about acquisition readiness.

Source provenance: `../pantheum-I/provenance/panel-errors-source-20260921.json`.
Runtime provenance: `provenance/panel-errors-runtime-20260921.json`.
No provider switch. No commit was created.
