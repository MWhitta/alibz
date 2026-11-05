---
key: golub1973variable
title: "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate"
year: 1973
topic: ml
relevance: 3
---

**Method/claim**: Original variable projection (VARPRO): for models y=Phi(alpha)c, eliminate the linear c by pseudo-inverse and minimise the reduced functional ||(I-Phi Phi^+) y|| over the nonlinear alpha only.

**Adoptable specifics**: Eq. (reduced functional r2(alpha)=||P_perp(alpha) y||^2) and the Kaufman-style derivative of Phi^+ (Sec. 3). Reduces dimension of the search and removes spurious local minima created by linear-parameter initialisation.

**Where it would enter alibz**: alibz already is a VARPRO structure: outer (T,n_e,sigma,gamma[,tau]) by gp_minimize, inner linear concentrations by WNNLS (peaky_indexer_v3). Difference: nonnegativity makes it a constrained projection, so GP/gradient-free is used, not Gauss-Newton. Option: Kaufman Jacobian inside active set to run a local LM polish from the GP optimum (see golub2003variable).

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): T error and log n_e error after polish; expect lower error, fewer gp evaluations at equal accuracy. Runtime per spectrum.

**Caveats**: Derivative theory assumes constant rank; NNLS active-set changes make the reduced functional only piecewise smooth.
