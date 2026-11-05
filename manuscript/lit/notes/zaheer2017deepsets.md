---
key: zaheer2017deepsets
title: "Deep Sets"
year: 2017
topic: ml
relevance: 2
---

**Method/claim**: Characterises permutation-invariant functions as rho(sum phi(x_i)); sum/mean pooling suffices.

**Adoptable specifics**: Thm. 2: f(X)=rho(sum_x phi(x)); practical: MLP phi, MLP rho, sum pooling; permutation-equivariant layer variant.

**Where it would enter alibz**: Cheapest baseline for the peak-set model in transformer_plan: phi on (lambda, area/sigma, width), rho to element logits; compare against Set Transformer before building it.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): majors missed / relative error; training cost. Expected a few % worse than attention but far cheaper.

**Caveats**: Sum pooling dilutes weak informative peaks among strong ones.
