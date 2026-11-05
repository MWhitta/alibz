---
key: wold2001pls
title: "PLS-regression: a basic tool of chemometrics"
year: 2001
topic: ml
relevance: 2
---

**Method/claim**: PLSR extracts latent variables maximising covariance of X and Y; standard multivariate calibration with diagnostics (VIP, Q2, residual leverage).

**Adoptable specifics**: Sec. 3: NIPALS, choose components by cross-validated Q2>0.0975; VIP>1 variable importance; DModX outlier detection.

**Where it would enter alibz**: Not for the physics solve. Useful as an ML baseline and for calibration transfer on Z300 vs lab spectrometer (see lepore2024calsharing): PLS on peak areas of the S suite as comparison.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py)/data/remote_samples/*.csv: relative error on majors vs WNNLS at equal data; expected PLS worse on out-of-distribution matrix.

**Caveats**: Matrix-effect sensitivity; not extrapolative.
