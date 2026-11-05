---
key: hao2024mlreview
title: "Machine learning in LIBS: review"
year: 2024
topic: ml
relevance: 2
---

**Method/claim**: Review of ML for LIBS classification and quantification (RF, SVM, ANN, CNN) with preprocessing and feature selection practices.

**Adoptable specifics**: Section tables list classification accuracies mostly 95-100 % (e.g. RF 98.03 %, 98-100 % in per-study table) on closed class sets; random train/test splits across shots of the same sample dominate.

**Where it would enter alibz**: Evidence for the transformer_plan caveats: report splits by sample, not by shot. No solver change.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz: evaluate any learned model with sample-level splits; line-matching fraction.

**Caveats**: Accuracies are leaky (shot-level splits) and not comparable to element-wise quantification.
