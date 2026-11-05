---
key: efron1979bootstrap
title: "Bootstrap methods: another look at the jackknife"
year: 1979
topic: ml
relevance: 2
---

**Method/claim**: Nonparametric bootstrap: resample data, recompute estimator, take empirical sampling distribution for standard errors and CIs.

**Adoptable specifics**: Sec. 2: B=50-200 replicates suffice for standard error, >=1000 for percentile CIs (Efron 1979 remarks). Applies to residual or paired resampling. For WNNLS use parametric bootstrap: y* = A c_hat + sigma_eff * eps*.

**Where it would enter alibz**: Already queued (bootstrap). Concretely in peaky_indexer_v3 after solve_at: B=100 parametric draws of peak areas at fixed (T,n_e), re-solve WNNLS (cheap: fixed dictionary), report percentile CIs per element and fraction of replicates with c_j>0 as presence stability.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): CI coverage at 68%/95% on majors and T error; expected coverage 0.9-0.95; runtime +B x NNLS.

**Caveats**: Conditional on (T, n_e): underestimates uncertainty from the outer search; blind to model misspecification. NNLS boundary makes percentile CIs for trace elements biased.
