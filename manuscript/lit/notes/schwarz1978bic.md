---
key: schwarz1978bic
title: "Estimating the dimension of a model"
year: 1978
topic: ml
relevance: 3
---

**Method/claim**: Laplace approximation to the marginal likelihood gives BIC = -2 ln L + k ln n, consistent for model dimension.

**Adoptable specifics**: Eq. (BIC) with n = number of independent observations. alibz uses n_peaks (basin BIC) and noise-rescaled chi^2 in refinement; both are valid only if residuals are whitened and uncorrelated. Export grid has rho(1)~0.92 pixel correlation, so effective n < n_pixels: use n_eff=n(1-rho)/(1+rho) or native pixel count.

**Where it would enter alibz**: peaky_indexer_v3.py:2506 _select_basin_bic and refinement.py:119 _bic: replace n by n_eff for pixel-level cases; use n_peaks only for peak-area rows. Keep BIC_AMBIGUITY=6 (Kass-Raftery 'positive'), margins 10/30 for blends.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): false-blend rate and basin_ambiguous accuracy; data/remote_samples/*.csv: stability of basin choice. Expected fewer false blends.

**Caveats**: Asymptotic; with k~10 and n_peaks~100 it over-penalises relative to AICc for small n. Wrong if noise model misspecified (sigma_eff).
