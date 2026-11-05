---
key: yaroshchyk2012pls
title: "Comparison of PCR, PLS, MB-PLS and serial PLS for Fe in iron ore using LIBS"
year: 2012
topic: ml
relevance: 1
---

**Method/claim**: Compares PCR, PLS, multi-block PLS and serial PLS for Fe (12.6-67.9 %) in iron ore from two spectrometers (208-222 nm, 300-855 nm).

**Adoptable specifics**: PLS and PCR gave similar accuracy, PLS with fewer latent variables; MB-PLS and S-PLS (separate UV and VIS blocks) were worse. Reported RMSEP about 2.2-4.5 wt% Fe, validation R2 0.87-0.97 (Sec. 3-4).

**Where it would enter alibz**: No solver change. Supports treating all segments in one joint fit (as alibz does) rather than per-spectrometer sub-models; a PLS baseline for comparing relative error on majors.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py)/data/remote_samples/*.csv: relative error on majors vs WNNLS; PLS expected worse out of distribution.

**Caveats**: Single matrix (iron ore), univariate target; empirical calibration, no transfer.
