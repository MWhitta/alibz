---
key: eppler1996matrix
title: "Matrix Effects in the Detection of Pb and Ba in Soils Using Laser-Induced Breakdown Spectroscopy"
year: 1996
topic: cflibs
relevance: 2
---

**Method/claim.** Matrix effects on Pb and Ba in sand/soil: cylindrical lens improved precision; speciation and bulk matrix change emission. Ba(II) emission correlates inversely with n_e, itself set by sand fraction (n_e 3.0e16 for pure sand to 6.9e16 cm^-3 for pure soil).

**Adoptable specifics.** LOD Pb/Ba: 17/76 ppm in sand, 57/42 ppm in soil; precision <= 10% RSD (5 replicates). Emission order Ba: carbonate > oxide > sulfate > chloride > nitrate; Pb: oxide > carbonate > chloride > sulfate > nitrate. n_e from Saha-Boltzmann of Ti lines.

**Where it would enter alibz.** Supports the n_e-from-phase-assemblage prior: ion/neutral ratio shifts with matrix-dependent n_e, so stage_disagreement partly reflects phase mix. Use 10% RSD as a replicate-precision floor for comparing shot means. No code change.

**Fit-quality test.** Real spectra: correlate fitted n_e with phase class (e.g. quartz vs carbonate) and check ion-stage ratio shifts as Eppler's inverse relation (expected sign: more ionized species falls as n_e rises).

**Caveats.** 1990s lab system, 1064 nm in air, ppm Pb/Ba rather than major elements.
