---
key: golub2003variable
title: "Separable nonlinear least squares: the variable projection method and its applications"
year: 2003
topic: ml
relevance: 3
---

**Method/claim**: Survey of VP (inverse problems 19 R1): VP converges in fewer iterations, is better conditioned, and often needs no starting values for linear parameters; covers constrained variants and Kaufman's simplified Jacobian.

**Adoptable specifics**: Secs. on Kaufman approximation (drop second term of D Phi^+; ~ halves per-iteration cost with similar convergence) and on VP with constraints/regularisation (augment Phi with sqrt(lambda)L rows). Mentions separable bound handling by projected LM.

**Where it would enter alibz**: peaky_indexer_v3 outer loop: after grid+GP, run scipy least_squares(method='trf', bounds) on the reduced residual vector (whitened NNLS residual) with finite-difference Jacobian over 4-5 outer params; ridge rows (evidence ridges) already fit the 'augment with sqrt(lambda)' form.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py) and data/remote_samples/*.csv: r^2 and T/log n_e error; expected modest gain; compare nfev vs gp_minimize. Check n_e-at-bound count does not rise.

**Caveats**: Reduced functional has kinks where NNLS support changes, so finite-difference Jacobians can mislead; polish should be gated by basin ΔBIC.
