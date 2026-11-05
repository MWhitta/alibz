---
key: castorena2021deepcnn
title: "Deep spectral CNN for LIBS"
year: 2021
topic: ml
relevance: 2
---

**Method/claim**: Residual spectral CNN that jointly pre-processes (dark, continuum, wavelength shift, range, instrument response) and calibrates ChemCam-like spectra in one forward pass.

**Adoptable specifics**: Learns without dark, response or range side information; outperforms the MSL pipeline (background subtraction, wavelet denoising, match-filter shift, PLS/ICA-SM-PLS on 408 standards) per abstract. Numbers not in retrieved text.

**Where it would enter alibz**: Model for transformer_plan: nuisances (shift, gain, range) learned; combine with domain randomisation.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz: peak-location residual (pm) and line-matching fraction of the learned model.

**Caveats**: Real standards; no synthetic-to-real test.
