---
key: dyar2012lasso
title: "PLS and lasso regression for LIBS of geological samples"
year: 2012
topic: ml
relevance: 2
---

**Method/claim**: 100 rocks at 9 m with 6144 channels; PLS1, PLS2 and lasso give comparable accuracy but lasso uses fewer than 24 non-zero channels per element.

**Adoptable specifics**: Lasso interpretable: selected channels coincide with emission lines; unscaled, uncentered data (Sec. 2).

**Where it would enter alibz**: Evidence that L1 selects physically meaningful lines; supports NN-LASSO already queued.

**Fit-quality test**: data/remote_samples/*.csv: spurious-element count after NN-LASSO.

**Caveats**: Held-out set of 100; accuracy comparable only.
