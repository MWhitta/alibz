---
key: mitchell1988spikeslab
title: "Bayesian variable selection in linear regression"
year: 1988
topic: ml
relevance: 2
---

**Method/claim**: Introduces the spike-and-slab prior: coefficient is zero with probability 1-pi or drawn from a diffuse slab; posterior inclusion probabilities select variables.

**Adoptable specifics**: Sec. 2: prior mixture point-mass at 0 + uniform slab, spike/slab height ratio h; Sec. 3 search over models with branch-and-bound / posterior probability ranking. Hyperparameter pi_j = prior presence per element.

**Where it would enter alibz**: Matches the 'Bayesian presence model (proposed)' in the dev guide. Cheap version: for each candidate element compare marginal likelihood with/without it (BIC approx), posterior odds = exp(-ΔBIC/2)*prior_odds; prior_odds from geology (rare elements 1e-2).

**Fit-quality test**: data/remote_samples/*.csv/synthetic suite S (bench/synth_cases.py): spurious-element count and missed trace at fixed false-positive rate; expected calibrated presence probability (Brier).

**Caveats**: Prior-sensitive (Lindley paradox); needs defensible pi_j; model space 2^N must be pruned.
