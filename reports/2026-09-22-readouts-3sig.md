# Three significant figures — deployed 2026-09-22

User revised Argon pressure and calibration measurement precision to three
significant figures. The shared frontend formatter now uses toPrecision(3),
retaining significant trailing zeros. Camera depth readouts use the same helper;
nominal board-pitch fallback now does too. Underlying data and geometry retain
full precision. Missing values remain unknown. Operator docs and DECISIONS.md
updated in ../pantheum-I, superseding the earlier one-figure display decision.

Deployed 12:39:52 PDT with existing hash-guarded static deploy script after dry
run. No service restart, configuration change, acquisition or hardware action.
Backup: /home/mwhittaker/pantheum-panel-errors-backup-20260922T123952.
Manifest: provenance/readouts-3sig-20260922.json.
Source, deployed, and HTTP-served app.js SHA-256:
7518f5ad48ff5b3fcf53eae7728bcd3462a7bd172aeb06c3302f416f86d79ee1

JavaScript syntax and existing 42 UI tests passed. Chrome verified Argon PSI
12.8, board pitch17.5 mm/0.688 in, range0.887 m, tilt0.127 deg, scale0.459 px/mm
and2.18 mm/px, RMS0.110 px, residual0.248 px; unknown depth cross-check retained.
Depth legend1.53/1.02/0.518 m, sigma0.781/1.86 mm, time1.26 s, valid77.2%.
All three camera sources Live; global status loaded successfully. Original user
tab untouched. Existing unrelated RamanLab edits excluded. No provider switch.

Pantheum source/documentation commit: 8ba3c39.
