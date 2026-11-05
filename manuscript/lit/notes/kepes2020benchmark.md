---
key: kepes2020benchmark
title: "Benchmark classification dataset for LIBS"
year: 2020
topic: ml
relevance: 1
---

**Method/claim**: 138 soil samples in 12 classes, composition with uncertainties; intended for pre-training and benchmarking classifiers.

**Adoptable specifics**: Constant 10 % uncertainty assumed in combined uncertainty; contest top accuracy about 90 %, two other approaches above 80 % (l.444).

**Where it would enter alibz**: Possible pre-training corpus for the learned model; element truth is of limited use for alibz.

**Fit-quality test**: data/remote_samples/*.csv: class separability of model embeddings.

**Caveats**: Different instrument; classification, not elemental quantification.
