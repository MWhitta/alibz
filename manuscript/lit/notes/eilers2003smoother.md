---
key: eilers2003smoother
title: "A perfect smoother"
year: 2003
topic: ml
relevance: 3
---

**Method/claim**: Whittaker smoother: z = (W + lambda D^T D)^-1 W y, a penalised least squares fit with banded system solved in O(n); basis of ALS/airPLS/arPLS baselines.

**Adoptable specifics**: Eqs. in Matlab listing (l.121, 222): Cholesky of banded (W+lambda D'D), second-difference D for baselines. Cross-validation of lambda via leave-one-out in closed form (hat-matrix diagonal). Stiffness scales as pitch^-4: lambda=(smooth_nm/pitch)^4 is exactly this scaling.

**Where it would enter alibz**: peaky_finder.py:327/490 arPLS already uses lam=(smooth_nm/pitch)^4 per segment. Improvement: use scipy.linalg.solveh_banded (or sparse Cholesky) once per iteration, and choose smooth_nm by closed-form LOO-CV on peak-masked pixels rather than the fixed 1.9 nm.

**Fit-quality test**: Baseline-induced bias: residual r^2 on data/remote_samples/*.csv, majors error on synthetic suite S (bench/synth_cases.py); runtime per spectrum.

**Caveats**: CV on peak-contaminated data favours too-flexible baselines; compute CV only on pixels with arPLS weight>0.5.
