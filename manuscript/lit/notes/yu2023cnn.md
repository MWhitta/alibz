---
key: yu2023cnn
title: "When Convolutional Neural Networks Meet Laser-Induced Breakdown Spectroscopy: End-to-End Quantitative Analysis Modeling of ChemCam Spectral Data for Major Elements Based on Ensemble Convolutional Neural Networks"
year: 2023
topic: ml
relevance: 1
---

**Method/claim.** Ensemble of 1D CNNs maps the raw 6144-channel ChemCam spectrum directly to major-oxide wt% with no dimensionality reduction. RMSE is 54% lower than PLS and 73% lower than ELM (abstract).

**Adoptable specifics.** Data: 69 standards, 240 valid spectra (200 train / 40 test, SPXY split); expanded set 345 standards, 1722 spectra (1435 / 287). Stride-based convolution without pooling (Sec. 3.2), weighted-average ensemble with weights in [0,1] normalised to 1 (Sec. 3.3). Per-element Rp^2 and RMSEP reported; relative error reaches 0.1% order for ECNN vs ~10% for PLS/ELM in individual samples (~line 1031). Errors blow up when true concentration is ~0.5% or less.

**Where it would enter alibz.** None in the physics path. Possible use: a learned prior or sanity cross-check on major-oxide fractions in `peaky_indexer_v3.py`, only with a Z300 labelled corpus.

**Fit-quality test.** Real spectra: held-out Fe-standard Fe fraction / non-Fe mass vs the physics result; metric RMSEP, lower is better. Requires hundreds of labelled spectra alibz lacks.

**Caveats.** ChemCam geometry, vacuum/Mars atmosphere, oxides only; black-box, not interpretable per line; random split of repeat spectra from the same sample risks leakage; conflicts with the physics-based goal.
