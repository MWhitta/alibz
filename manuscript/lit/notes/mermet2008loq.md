---
key: mermet2008loq
title: "Limit of quantitation in atomic spectrometry: An unambiguous concept?"
year: 2008
topic: cflibs
relevance: 2
---

**Method/claim.** Tutorial on limit of quantitation: 10-sigma definition, RSDnet, method limit, lowest point of the calibration graph, and uncertainty approaches, with ICP-AES, ICP-MS and LIBS examples. IUPAC: LOD is the smallest detectable concentration; LOQ is where the value becomes reasonably reliable. [Provided text = abstract + first page.]

**Adoptable specifics.** LOQ conventions named (10 sigma_blank; RSD-based). Standard practice (background knowledge): LOD = 3 sigma_blank / slope, LOQ = 10 sigma_blank / slope.

**Where it would enter alibz.** `detections.py`: report per-element LOD (3 sigma) and LOQ (10 sigma) in composition units alongside the existing detectability gate; define which one the gate corresponds to. Not queued.

**Fit-quality test.** Suite S trace cases: spurious mass and false-negative rate at fractions near LOD/LOQ; metric detection probability vs true fraction.

**Caveats.** Calibration-curve LOD does not map directly to a calibration-free engine; see villasboas2023lod.
