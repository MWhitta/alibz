---
key: lee2019settransformer
title: "Set Transformer: a framework for attention-based permutation-invariant neural networks"
year: 2019
topic: ml
relevance: 2
---

**Method/claim**: Set Transformer with induced set attention blocks (ISAB) and pooling by multihead attention; O(nm) with m inducing points.

**Adoptable specifics**: Sec. 3: SAB, ISAB with m=16-32 inducing points, PMA with k seed vectors. Handles variable-size sets, permutation invariance.

**Where it would enter alibz**: Peak list (center, area, width) per spectrum as a set; ISAB m~32 encoder, PMA decoding to element presence/log-concentration. Natural fit with peaky_finder outputs; avoids pixel-grid shifts altogether.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz/synthetic suite S (bench/synth_cases.py): majors missed, spurious mass of the learned model vs alibz WNNLS.

**Caveats**: Needs good peak detection; missed peaks are invisible to the model. Not proved on LIBS.
