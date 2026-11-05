---
key: storn1997de
title: "Differential evolution: a simple and efficient heuristic for global optimization"
year: 1997
topic: ml
relevance: 1
---

**Method/claim**: DE: population-based stochastic optimiser using vector differences; control parameters NP, F, CR.

**Adoptable specifics**: Rules of thumb (Sec. 3): NP=5-10 x D, F=0.5 (0.4-1), CR=0.1 (separable) or 0.9 (non-separable); strategy DE/rand/1/bin.

**Where it would enter alibz**: Alternative to gp_minimize for the 4-5-D outer problem (scipy.optimize.differential_evolution, polish=True). Needs ~200-500 evaluations, 5-10x the GP budget: not competitive at 40 calls.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): T error and runtime; expected equal accuracy at much higher runtime.

**Caveats**: Cost per evaluation dominates; not recommended except as an offline cross-check for global-optimum quality on hard S cases.
