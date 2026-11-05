---
key: zou2005elasticnet
title: "Regularization and variable selection via the elastic net"
year: 2005
topic: ml
relevance: 3
---

**Method/claim**: Adds ridge to lasso: grouping effect keeps correlated predictors together; naive EN rescaled by (1+lambda2) removes double shrinkage.

**Adoptable specifics**: Eq. (3): (1-alpha)||b||^2/2 ... naive EN; corrected b = (1+lambda2)*b_naive (Sec. 3.3). Solved by augmented data: X* = (1+lambda2)^-1/2 [X; sqrt(lambda2) I], y* = [y;0], then lasso (Sec. 2.3). LARS-EN; p>>n handled.

**Where it would enter alibz**: WNNLS with augmented rows [A; sqrt(lambda2) I] is trivial since NNLS accepts it, plus L1 linear term. Fits alibz where element columns for neighbouring species (Fe/Mn/Cr, ions of one element) are collinear: grouping keeps ion stages of one element together, lasso alone picks one.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): majors missed, relative error on majors, spurious mass; data/remote_samples/*.csv: spurious-element count. Expected improvement over plain NN-LASSO on collinear sets.

**Caveats**: Two hyperparameters (lambda1, lambda2); must be set per-spectrum by BIC, adding runtime. Ridge part biases big majors.
