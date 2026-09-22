# Argon and camera measurement precision — deployed 2026-09-22

User clarified: round pressure and camera measurements to one significant
figure; no camera layout change requested.

Changed only display formatting in ../pantheum-I/web/alibz/app.js. Shared
formatMeasurement handles Argon PSI, depth colorbar labels, depth sigma/time/
valid percentage, and calibration pitch/range/tilt/scale/residual/depth-check
readouts. Non-numeric values retain unknown or em dash. Frame counts, detected
grid counts, timestamps, camera geometry and stored measurements are unchanged.

Source synchronized; hash-guarded static deployment at 12:22:14 PDT. No service
restart, config change, or hardware command. Existing static deploy script
reused with dry-run first. Backup:
/home/mwhittaker/pantheum-panel-errors-backup-20260922T122214
Bundle: /private/tmp/pantheum-readout-precision-20260922.
App.js installed and HTTP-served SHA-256:
ed52029a4f68af335ee83af85b36dac91bf930343a9b4eedf26963250d0e3846

JavaScript syntax passed; existing UI suite 42/42 passed (readout-precision UI
log in this directory). Chrome live verification showed Argon PSI 10; camera
range 0.9 m; tilt 0.1 deg; scale 0.5 px/mm and 2 mm/px; RMS 0.1 px; lattice
residual 0.2 px; depth sigma 0.8 mm/2 mm; duration 1 s and valid 80%. Missing
depth cross-check remained unknown. All three camera tiles reported Live.
The global status request initially showed an unreadable-response notice while
acquisition, camera, and display endpoints rendered live values; no broad
service-health claim is made. The formatting change touches none of those APIs.
Original user tab unchanged; checked a separate tab. No provider switch.

## Documentation and source control

Pantheum commit `eb999ef32cf019a5dd8c5af138532ab81a413a04` contains the
formatter and operator documentation in `docs/alibz-display.md` and
`docs/alibz-rgb-calibration.md`, with its rationale in `DECISIONS.md`.
Earlier camera/display implementation and deployment records were already
committed. The alibz follow-up also includes the two previously untracked
display-helper test files; all 15 helper/deployment tests were independently
rerun successfully before commit. See `2026-09-22-display-commit-tests.md`.
