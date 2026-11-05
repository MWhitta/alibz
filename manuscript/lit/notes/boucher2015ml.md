---
key: boucher2015ml
title: "ML regression methods for major elemental analysis of rocks by LIBS"
year: 2015
topic: ml
relevance: 2
---

**Method/claim**: Nine models compared on 100 samples (50 shots x 5 locations): PLS1/2, PCR, lasso, elastic net, SVR-Lin vs K-PCR, SVR-Py, kNN.

**Adoptable specifics**: Sparse models best for MgO, SiO2, Fe2O3, CaO, MnO; worse for Na2O, K2O, TiO2, P2O5; nonlinear methods overfit; Wilcoxon on PRESS, most models indistinguishable except kNN.

**Where it would enter alibz**: Supports sparse/elastic-net element selection; weak lines respond less.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): major error and trace recall.

**Caveats**: Small sample count.
