---
key: dyar2016univariate
title: "Comparison of univariate and multivariate models for prediction of major and minor elements from laser-induced breakdown spectra with and wi"
year: 2016
topic: cflibs
relevance: 1
---

**Method/claim.** 1356 spectra / 452 geologically diverse samples: univariate single-line models are the least accurate for Si, Al, Ti, Fe, Mg, Ca, Na, K, Ni, Mn, Cr, Co, Zn; optimising the region gives 35% (majors) / 13% (minors) improvement; PLS on same regions improves 63% (majors) / 3% (minors); masking to element regions hurts majors, helps low-signal minors.

**Adoptable specifics.** Numbers above (abstract). Best univariate wavelength region is training-set specific and does not generalise.

**Where it would enter alibz.** Justification for whole-pattern fitting (`peaky_indexer_v3.py`) over single-line ratios; no change.

**Fit-quality test.** Benchmark context: relative improvement figures, not absolute errors.

**Caveats.** Statistical regression; says nothing about physics-based inversion.
