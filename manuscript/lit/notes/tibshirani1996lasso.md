---
key: tibshirani1996lasso
title: "Regression shrinkage and selection via the lasso"
year: 1996
topic: ml
relevance: 3
---

**Method/claim**: L1-penalised least squares gives sparse coefficients; solution via QP or (later) LARS/coordinate descent.

**Adoptable specifics**: Eq. (1): min ||y-Xb||^2 s.t. sum|b|<=t; equivalently +lambda||b||_1. Orthonormal case soft-thresholds b by lambda/2. Choose t by CV or Stein's risk. For NN constraint the penalty is lambda*sum(b): a linear term, so NNLS stays NNLS with b' = A^T b - lambda/2.

**Where it would enter alibz**: Already queued (NN-LASSO trace suppression). Exact variant: in WNNLS augment the normal-equation rhs (Gram form, FNNLS) with -lambda*w_j, w_j = 1/(initial coefficient) (adaptive lasso) so trace elements with only coincident lines drop to zero; lambda chosen by ΔBIC.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py) spurious mass, majors missed; data/remote_samples/*.csv spurious-element count. Expected spurious count down, trace recall slightly down.

**Caveats**: Shrinks true minor elements (bias); lasso is unstable for correlated columns (use elastic net, zou2005elasticnet). Needs concentration-scale normalisation.
