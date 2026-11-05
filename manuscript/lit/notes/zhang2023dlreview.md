---
key: zhang2023dlreview
title: "Application of deep learning in LIBS: review"
year: 2023
topic: ml
relevance: 2
---

**Method/claim**: First review of DL for LIBS: CNN, autoencoder, transfer learning, data augmentation, challenges.

**Adoptable specifics**: Notes a 3-class rock classification improved from 64 to 85 % with DL (l.317); recommends augmentation and transfer learning for small datasets; flags interpretability and lack of public benchmarks.

**Where it would enter alibz**: Supports domain randomisation (tobin2017domainrand) and synthetic pretraining in transformer_plan.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz: accuracy of a synthetic-trained model with and without augmentation.

**Caveats**: Review; numbers not comparable across datasets.
