---
key: li2021annreview
title: "Review of ANN chemometrics in LIBS"
year: 2021
topic: ml
relevance: 1
---

**Method/claim**: Only the first pages were retrieved. Reviews BPNN, RBFNN, CNN, SOM for classification and quantification; discusses variable selection, data-set use, overfitting and interpretability.

**Adoptable specifics**: Good practices: choose inputs by line areas or selected variables rather than full spectra; split by sample; report both calibration and prediction errors.

**Where it would enter alibz**: Supports using peak areas (alibz peak list) as model inputs rather than pixels.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): compare pixel-input vs peak-input learned model.

**Caveats**: Truncated text; no numbers extracted.
