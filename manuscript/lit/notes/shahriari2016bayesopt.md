---
key: shahriari2016bayesopt
title: "Taking the human out of the loop: a review of Bayesian optimization"
year: 2016
topic: ml
relevance: 2
---

**Method/claim**: Review of BO: surrogate (GP, ARD Matern), acquisition (PI, EI, UCB, Thompson, entropy search), practical issues (hyperparameter marginalisation, constraints, high dimensions).

**Adoptable specifics**: Sec. on kernels: Matern-5/2 ARD preferred over squared-exponential; Sec. acquisition: EI/GP-UCB with kappa~2; hyperparameter marginalisation (MCMC) avoids poor point estimates with <50 evaluations. skopt default gp_hedge mixes EI/PI/LCB.

**Where it would enter alibz**: peaky_indexer_v3.py gp_minimize call (l.2443): set base_estimator=GaussianProcessRegressor(Matern(nu=2.5, length_scale per dim)), acq_func='EI', noise='gaussian' small; normalise outer dims to [0,1] (T in K vs log n_e vs sigma span orders).

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): T error / log n_e error, fraction of runs reaching true basin; data/remote_samples/*.csv: n_e-at-bound count. Expected: fewer basin misses at equal n_calls.

**Caveats**: Gains are empirical; with grid seeds the surrogate matters less. Confounded with dimension scaling, test them separately.
