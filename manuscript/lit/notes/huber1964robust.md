---
key: huber1964robust
title: "Robust estimation of a location parameter"
year: 1964
topic: ml
relevance: 2
---

**Method/claim**: Minimax estimator for epsilon-contaminated normals: quadratic loss for |r|<k, linear beyond; k tied to contamination.

**Adoptable specifics**: rho(r)=r^2/2 (|r|<=k), k|r|-k^2/2 otherwise; k=1.345 sigma gives 95% efficiency at the normal (standard; not in 1964 text, which gives k via epsilon). Scale must be estimated robustly (MAD/0.6745). IRLS weight w=min(1,k/|r|).

**Where it would enter alibz**: Already queued (Tukey IRLS in dev guide). Huber is the safer first step: convex, unique solution in WNNLS IRLS (reweight rows by min(1,k/|r_i|) of the sigma_eff-whitened residual). Contrast soft_l1 already in refinement.py.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py) with injected outlier lines: median relative error on majors, spurious mass; data/remote_samples/*.csv: r^2. Expected: spurious mass down, majors error down 10-30% if outliers present.

**Caveats**: Needs robust scale; with sigma_eff already inflated for large amp rows, IRLS may double-count downweighting.
