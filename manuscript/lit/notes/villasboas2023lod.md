---
key: villasboas2023lod
title: "A statistical definition of limit of detection for calibration-free laser-induced breakdown spectroscopy"
year: 2023
topic: cflibs
relevance: 3
---

**Method/claim.** Statistical LOD for CF-LIBS: propagates the uncertainty of line intensities (estimated from the standard deviation of the background signal) through the CF equations; LOD depends on plasma temperature and line properties (E_upper, g A, partition function) and on emitter density. Validated for C and Ca in NaCl, comparable to calibration-curve LOD. [Abstract + intro provided; final formula not in text.]

**Adoptable specifics.** Idea: LOD_i = concentration at which the line peak = k sigma_bg propagated via I = F n g A exp(-E/kT)/U(T); k = 3 by convention. Depends on T, so LOD varies with plasma state and per line.

**Where it would enter alibz.** `detections.py` / `synthetic.py`: compute a per-element, per-run LOD from the local residual sigma and fitted T and use it to (a) gate detections and (b) report limits for non-detected elements. Not queued (detectability gate exists with a different criterion).

**Fit-quality test.** Suite S: inject target at 0.5-5x predicted LOD, measure detection probability (expect ~50% near LOD) and spurious-mass at LOD; real: reproducibility on Z300 Fe standard of trace-Mn/Cr detections. Paper: LOD ~ calibration-curve LOD for C, Ca in NaCl.

**Caveats.** Derived for optically thin lines in NaCl at lab conditions; handheld pixels and continuum structure inflate sigma_bg correlated noise (export rho(1)=0.92).
