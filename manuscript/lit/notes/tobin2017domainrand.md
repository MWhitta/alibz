---
key: tobin2017domainrand
title: "Domain randomization for transferring deep neural networks from simulation to the real world"
year: 2017
topic: ml
relevance: 2
---

**Method/claim**: Training only on simulated images with wide random nuisance (textures, lighting, camera) transfers to real world with 1.5 cm detector accuracy.

**Adoptable specifics**: Sec. III: randomise camera pose, texture, lighting, distractors; no real data used; ~100k images; error 1.5 cm (abstract). Principle: make real data one more draw from the training distribution.

**Where it would enter alibz**: synthetic.py / transformer_plan: randomise nuisances that alibz already models as unknowns: wavelength offset (+-0.25 px), per-segment gain step (~5x at 620 nm), LSF width, baseline shape, vacuum/air shift, T and n_e ranges, NIR PIXEL_OFFSET error, saturated lines, self-absorption. Widest ranges first, narrow later.

**Fit-quality test**: Model's accuracy on Fe standard fe_aesar_9998_means_20260922.npz (line-matching fraction, T consistency) trained with vs without each randomised nuisance; expected gain concentrated on registration/gain axes.

**Caveats**: Over-wide randomisation hurts precision; it cannot add physics the generator lacks (non-LTE, matrix effects).
