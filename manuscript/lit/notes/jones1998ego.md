---
key: jones1998ego
title: "Efficient global optimization of expensive black-box functions"
year: 1998
topic: ml
relevance: 3
---

**Method/claim**: EGO: fit a kriging (GP) surrogate to few evaluations and choose the next point by maximum Expected Improvement; stopping when max EI is small relative to the incumbent.

**Adoptable specifics**: Eq. EI(x)=(f_min-yhat)Phi(z)+s phi(z), z=(f_min-yhat)/s; initial design ~10k points (k=dimension) by Latin hypercube (Sec. 'initial design'); stop when max EI < 1% of |f_min|; also log-transform the response when it spans decades (Sec. on transformations).

**Where it would enter alibz**: peaky_indexer_v3.py:2413-2450: n_initial_points=max(10,n_calls//4) with 15x7 grid seeds is already above 10k for k=4-5. Add: optimise log(cost) (costs span decades), and an EI-based early stop to cut n_calls=40.

**Fit-quality test**: Runtime per spectrum (R, S) with no loss in synthetic suite S (bench/synth_cases.py) T/log n_e error; expected 20-40% fewer evaluations.

**Caveats**: Kriging with stationary kernel is poor where the cost has cliffs (basin collapse). Early stopping on EI can lock in the wrong basin; keep the grid seeds and ΔBIC check.
