---
key: li2020cnn
title: "LIBS multi-component quantitative analysis based on a deep CNN"
year: 2020
topic: ml
relevance: 2
---

**Method/claim**: CNN on more than 1400 MarSCoDe spectra of 23 national standards; compared with BPNN, PLS1, PLS2; CNN best overall RMSE, advantage larger without baseline removal.

**Adoptable specifics**: Abstract only retrieved. Result: CNN beats BPNN/PLS in overall RMSE and relative error; benefit greater on spectra without baseline removal.

**Where it would enter alibz**: Suggests a learned model could absorb baseline; alibz keeps explicit arPLS. Nothing to adopt now.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz: learned model with and without baseline input.

**Caveats**: 23 standards only; authors call results preliminary.
