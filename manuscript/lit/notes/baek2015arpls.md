---
key: baek2015arpls
title: "Baseline correction using asymmetrically reweighted penalized least squares smoothing"
year: 2015
topic: ml
relevance: 3
---

**Method/claim**: arPLS: weights from a logistic function of negative residuals, w=1/(1+exp(2(d-(2s-m))/s)) with m,s the mean/std of negative residuals d; removes ALS's p and airPLS's hard zeroing.

**Adoptable specifics**: Weight update eq. (7); termination eq. (8) threshold 0.001 on the relative weight change (alibz ratio_tol=1e-3 matches); lambda and ratio are the only parameters. ALS asymmetry p recommended 0.001-0.1 (Sec. 2).

**Where it would enter alibz**: Already arPLS in peaky_finder._arpls (l.327; lam=(smooth_nm/pitch)^4 per segment, max_iter 20). Possible change: zero-weight database line windows on the first iteration so dense Fe/V forests are not absorbed; test looser ratio for speed.

**Fit-quality test**: data/remote_samples/*.csv: residual r^2, spurious-element count; Fe standard fe_aesar_9998_means_20260922.npz: Fe I Boltzmann T consistency. Runtime: iterations per spectrum.

**Caveats**: Underestimates baseline under dense line forests; noise std estimated including peaks.
