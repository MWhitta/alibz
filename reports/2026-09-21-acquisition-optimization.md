# Fe acquisition optimization and gantry connection

Date: 2026-09-21. Provider: current subscription; no provider switch.

## Result

Implemented in Pantheum's `pantheum/alibz` and `web/alibz` workspace:

- Explicit reserved **Connect controller** opens the identified OpenBuilds USB
  connection, eliminating the vendor-GUI connection step. Status distinguishes
  network connection, pending serial connection and verified controller readiness.
  No home, jog, unlock, alarm clearing or laser action accompanies connection.
- Fe session baseline delay10/period25, 100Hz encoded as pulsePeriod10ms,
  ten individual shots, one analyzer location, zero cleaning shots; freeze all
  other settings and require one explicit confirmation per batch.
- Durable sessions/batches, stable caller IDs, no uncertain replay, serialization
  across Service instances, offline shot analysis before data-informed next proposal.
- Per-shot/per-line metrics, raw ZIP links, session export, atomic source hashes,
  oscillator strengths, shared-upper-level relative-response diagnostics and QC.
  No response correction or absolute calibration is asserted or applied.
- Private live 100Hz commissioning gate remains default-off; existing laser
  configuration is preserved. Operator must keep the physical gantry spot fixed.

## Verification by main agent

- Original workspace baseline: `python3 -m unittest discover -s tests`:
  **697 tests run, 26 skipped, no failures**.
- Final staging suite, exact live Socket.IO5.16.4/Engine.IO4.14.0/client1.9.2
  packages installed only under `/private/tmp/pantheum-live-socket-test`:
  `PYTHONPATH=/private/tmp/pantheum-live-socket-test python3 -m unittest discover -s tests -v`:
  **732 tests run, 11 skipped, no failures**,166.373s. Optional skips remain;
  changed gantry transport tests ran against a real loopback Socket.IO server.
- `node --check web/alibz/app.js` passed; `node --test tests/test_alibz_ui.cjs`:
  **22 passed, zero failed** (baseline18).
- Six direct scientific-metric tests check known relative response, variability,
  incomplete/nonfinite/grid-mismatched files, saturation, missing lines, blend and
  unknown-ceiling refusal. Installer test verifies dry-run, source hashes,
  preservation of laser gates/limits, and refusal while acquisitions are active.
- Headless Chrome against an isolated loopback service with injected synthetic
  Fe spectra: creation dispatched nothing; first explicit batch recorded ten
  shots and proposed5/25 from10/25; second explicit batch recorded twenty total
  and proposed20/25. No batch starts automatically. Screenshot:
  `reports/2026-09-21-acquisition-browser.png`. Generic built-in ramp simulation
  appropriately fails the Fe quality gate; the richer Fe fixture was for QA only.
- Reviewed code, delegated reports, exact OpenBuilds v1.0.388 connect/emit source,
  and original-file hashes. Unrelated Raman and scientific-engine edits preserved.

## Science and hardware limits

- Integration delay/period units remain vendor units; actual100Hz timing and
  firmware acceptance were not measured. No real laser shots or gantry motion
  were issued during development.
- 96 initial Fe reference windows span216.68–495.76nm. Full raw spectra are
  retained. Legacy Fe per-line citations/uncertainties are absent and disclosed;
  this is not a calibrated whole-detector reference. Unknown saturation ceiling
  withholds response pairs. Crater evolution is flagged for same-spot comparisons.
- Socket.IO emit means submitted, not physical-motion completion. The new
  connection path was exercised against mocks/local transports; live initial
  disconnected-to-connected operation remains a bench commissioning check.
- The physical controller was already connected when read-only live status was
  inspected; no forced disconnect was performed merely to demonstrate connection.

## Live acquisition approval constraint

After vendor bytecode directly established1000/Hz encoding, automatic approval
review rejected removing the newly introduced `optimization.rate_100hz_verified`
gate as exposing a live hardware-acquisition API without explicit approval, and
prohibited achieving that through an indirect workaround. The final code keeps
that gate at creation and every dispatch. No deployment step enables it or adds
`z300.firetest`. Explicit user approval is required before enabling live100Hz
optimization; this is separate from the requested gantry connection capability.

## Deployment

Local source installation completed:32 reviewed files copied into
`/Users/mwhittaker/Projects/github/pantheum-I`, hashes verified; backups under
`/Users/mwhittaker/Projects/github/pantheum-I-acquisition-backup-20260921T130028`.
The original unrelated Raman edits and the scientific alibz engine edits remain.
Final isolated gantry/helper recheck against exact live Socket.IO versions:
42 tests passed, zero skipped. Source pins are in
`provenance/acquisition-module-20260921.json`; the nine runtime files are pinned in
`provenance/acquisition-runtime-20260921.json`.

**DEPLOYED on Moissanite, 2026-09-21 at 13:15:58 PDT.** The user explicitly
approved deployment after reporting the old gantry “not reachable / command was
not confirmed” error. The earlier automatic-review rejection of the remote
upload was resolved by this authorization. The prepared nine-file bundle was
staged at `~/pantheum-acquisition-stage-20260921-1301/`, checked in dry-run mode,
and installed with the tested installer. Both installer passes found no active
acquisitions or analysis jobs. All installed hashes matched the manifest.

- Runtime backup: `/home/mwhittaker/pantheum-I-acquisition-backup-20260921T131558`.
- Configuration backup: `~/.config/pantheum/alibz.json.acquisition-backup-20260921T131558`.
- Only the identified USB controller configuration and `gantry.connect` were
  added to the private configuration. `acquire.enabled_actions` remains empty;
  `optimization.rate_100hz_verified` remains absent/default false.
- Both user services restarted and reported active. `/health` returned HTTP200.
  The served JS, HTML and CSS matched their exact runtime manifest hashes.
- `/api/optimization` returned the Fe defaults, delay10/period25, ten shots at
  one location, pulsePeriod10ms/100Hz, and `live_rate_encoding_enabled:false`.
- Read-only `/api/motion/status?touch=1` verified the controller at
  2026-09-21T20:16:46.723476Z: network and serial connected, COM3 at115200,
  FTDI0403:6001 serialA10OF9X9, grblHAL1.1f, stateIdle, no alarm, no gate reasons,
  `live_allowed:true`. Position was X102.041/Y10.004/Z-12.867mm.
- The existing user reservation was preserved. No gantry command, forced
  disconnect, laser command, or physical acquisition was issued during deployment.
  Initial disconnected-to-connected commissioning remains untested on hardware
  because the controller was already connected.

Refresh the alibz page to load the new UI. The Acquire panel now supports
**Connect controller** directly when the controller is disconnected. Live100Hz
optimization still requires its separately approved configuration change.
