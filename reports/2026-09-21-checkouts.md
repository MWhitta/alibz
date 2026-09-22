# Independent alibz hardware and analysis checkouts

Deployed to Moissanite on 2026-09-21 at **14:15:07 PDT**, under the existing
explicit deployment approval. Both services are active; health is HTTP200.
Eight runtime hashes and all three served frontend hashes match the tested
bundle. No movement, serial-connect, or laser command was issued. No provider
switch occurred.

## Delivered behavior

- Independent exclusive hardware and analysis checkouts. One operator can hold
  hardware while a different analyst imports data and submits analysis.
- Hardware ownership covers gantry, acquisition settings, firing/cancellation,
  and the acquisition optimizer. Reading and acquisition-generated imports remain
  available without the analysis checkout.
- SQLite ownership, a cross-process lock and dispatch generations enforce portal
  exclusivity. Active or unresolved physical operations prevent release,
  expiry-based handoff and admin takeover. Hardware enforcement cannot be disabled
  by the legacy reservation configuration switch.
- Cancellation before dispatch sends no vendor request. After dispatch, failed
  cancellation stays active until the executor exits, then remains uncertain.
  All completion, failure and reconciliation transitions share the operation lock.
- Reconciliation requires the holder's inspection acknowledgement, fresh verified
  Idle gantry status, and no active live acquisition. It records the inspection
  without replaying commands or changing the scientific outcome.
- The prior reservation migrated into hardware only, preserving holder and expiry.
  At verification it belonged to mwhittaker@lbl.gov and was expired, with no
  blockers; analysis was free. Both resource enforcement paths are enabled.

API: GET `/api/reservation?resource=hardware|analysis`; POST `/api/reservation`
with resource/action. Omission selects hardware, never both resources. Recovery:
POST `/api/motion/command` with `{"action":"reconcile","confirm":{"hardware_idle":true}}`.

## Verified by the main session

- Required full backend suite: **761 run, 11 skipped, no failures**, 167.582s.
  An exception-finalization locking edit landed seven seconds after that process
  started, so the complete affected subset was rerun against the frozen final
  runtime: **135 passed**, 26.346s (acquire, checkouts, reservation and motion).
  Exact live Socket.IO dependencies were supplied using
  `PYTHONPATH=/private/tmp/pantheum-live-socket-test`.
- Node UI tests: **31 passed**; JavaScript syntax and Python3.10 parsing passed.
  Both repositories' `git diff --check` passed.
- Real Chrome: unknown-backend cards fail closed; a hardware holder and separate
  analyst coexist; the analyst imported the synthetic example and completed two
  preview jobs. Competing hardware checkout returned409, hardware controls stayed
  disabled for the analyst, and zero physical operations/acquisitions were created.
- Independent process contention produced exactly one hardware winner. Guarded
  release waited for the in-flight mutation; corrupt persisted time failed closed.
- Final bounded audit exercised predispatch cancel, failed cancel during polling,
  reconciliation refusal until executor exit, late results remaining uncertain,
  and locked exception finalization. Main inspected its code findings and reran
  the affected tests. See the final-review report for the adversarial evidence.
- Initial full suite exposed nine old HTTP fixtures expecting a single checkout;
  those now explicitly reserve analysis. Final tests retain opposite-resource
  denial assertions. Earlier sandbox-limited agent pytest runs were not used as
  deployment evidence.

## Deployment evidence and limits

Runtime source: `/Users/mwhittaker/Projects/github/pantheum-I`.
Only the explicit hashed file list was installed; unrelated Raman and scientific
engine edits were preserved. Source and runtime manifests are
`provenance/checkouts-source-20260921.json` and
`provenance/checkouts-runtime-20260921.json` in Pantheum.

Private configuration was byte-for-byte unchanged (SHA256
`5946184c259c2994448ad354defe1421e78f73bc82b1d04acfb53fadb015f517`).
Laser actions remain empty; the separate100Hz commissioning gate remains false.
At 21:15:25Z the freshly observed gantry was verified connected on COM3/115200
and reported **Running**, not Idle. No portal physical operation existed. The
normal idle gate therefore continues to block conflicting alibz commands.

The checkout coordinates **alibz control paths**; OpenBuildsCONTROL, Profile
Builder, pendants and direct scripts do not participate in its ledger. Physical
cancellation or movement was not tested. Browser interaction was verified against
an isolated local service, while live deployment was verified through the
existing Unix-socket upstream; production sign-in was not retested.

Backups:

- Source: `/Users/mwhittaker/Projects/github/pantheum-I-acquisition-backup-20260921T141426`
- Live files: `/home/mwhittaker/pantheum-I-acquisition-backup-20260921T141507`
- SQLite, legacy reservation and config: `/home/mwhittaker/pantheum-checkouts-state-backup-20260921T141507`
- Deployed bundle: `/home/mwhittaker/pantheum-checkouts-stage-20260921`

Evidence: `2026-09-21-checkout-browser.json/.png`,
`2026-09-21-checkout-deployed.json`, backend/UI/test reports, and
`2026-09-21-checkout-final-review.md`. Refresh alibz to load both checkout cards.
