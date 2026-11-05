---
key: herrera2009cflibs
title: "Semi-quantitative analysis of metal alloys, brass and soil samples by calibration-free laser-induced breakdown spectroscopy: recent results "
year: 2009
topic: cflibs
relevance: 3
---

**Method/claim.** CF-LIBS applied to Al, Zn, Cu-Ni, Ti, Ni alloys, brass and soils (Nd:YAG, ICCD, 220-700 nm) at 1000/100/0.1 mbar to classify accuracy as quantitative, semi-quantitative or qualitative. Uses Boltzmann + Saha with n_e from H-alpha (10% uncertainty) and delays 0.5-2 us.

**Adoptable specifics.** Abstract: matrix element ~5% relative error; other components usually semi-quantitative (<200%). Al alloys: major Al 0.3-6%, others 30-200% (Harvey scale: semi-quant 30-200%). Cites: Mars-condition soils Si <5%, others to 80%; Salle 2006 stand-off rocks: O 4-63%, others up to 1700% (external normalisation: O <32%). Dominant error: all-element detection, transition probabilities, partition functions, intensity fluctuations; a simulated 10% perturbation propagates to ~10% in minor components (detail unverified).

**Where it would enter alibz.** Sets the accuracy bar for 'what good looks like' (see report benchmark table). Argues for reporting relative-to-matrix-element ratios (`peaky_indexer_v3.py` output) rather than absolute fractions. No code change.

**Fit-quality test.** Compare suite-S and Z300 results against the 5% (matrix element) / 30-200% (others) scale; metric median relative error on majors.

**Caveats.** Metals in lab ICCD systems; minerals and handheld resolution are harder. Quoted soil/rock numbers are from other papers.
