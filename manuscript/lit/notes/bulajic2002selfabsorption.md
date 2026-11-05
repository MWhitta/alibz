---
key: bulajic2002selfabsorption
title: "A procedure for correcting self-absorption in calibration-free LIBS"
year: 2002
topic: theory
relevance: 2
---

**Method/claim.** Curve-of-growth based correction applied inside CF-LIBS; on three NIST steels and three Au-Ag-Cu alloys it improves precision and accuracy by roughly an order of magnitude (abstract).

**Adoptable specifics.** Uses a reference thin line per element plus COG ratio to rescale thick lines before the Boltzmann plot; no thresholds extracted.

**Where it would enter alibz.** Already covered by alibz doublet-anchored tau (`invert_doublet_tau`); only the multi-line COG per element is new in `refinement.py`.

**Fit-quality test.** Fe I Boltzmann T consistency on Z300 Fe means after COG rescaling of resonance lines; synthetic SA round trip majors error.

**Caveats.** Alloys at ~1 atm; mineral matrices and 0.1-0.2 nm pixels blur resonance cores.
