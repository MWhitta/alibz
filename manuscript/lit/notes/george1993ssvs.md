---
key: george1993ssvs
title: "Variable selection via Gibbs sampling"
year: 1993
topic: ml
relevance: 2
---

**Method/claim**: SSVS: Gibbs sampler over mixture prior with small-variance (tau) 'spike' and large-variance (c*tau) 'slab', avoiding a point mass.

**Adoptable specifics**: Eqs. (2.5)-(2.7): b_j | gamma ~ N(0, (gamma_j c_j tau_j)^2); suggest tau_j small relative to meaningful effect, c in 10-100; indicators gamma_j ~ Bernoulli(p_j). Posterior mean of gamma_j = inclusion probability.

**Where it would enter alibz**: Sampling is too slow per spectrum (S/R runtime). Use as a one-time offline calibration of pi_j on the S suite, or as the precise form of the Bayesian presence layer; online approximate with the ΔBIC odds rule.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): AUC of presence probability vs truth; Brier score. Expected better than the detection z alone.

**Caveats**: MCMC mixing when columns collinear; pseudo-prior choices matter.
