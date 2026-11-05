---
key: pan2010transfer
title: "A survey on transfer learning"
year: 2010
topic: ml
relevance: 2
---

**Method/claim**: Taxonomy: inductive, transductive (domain adaptation), unsupervised transfer; instance, feature, parameter, relational approaches; negative transfer warning.

**Adoptable specifics**: Sec. 2-3: sim-to-real is transductive (same task, different marginal P(X)); feature-representation and instance re-weighting (importance weights p_t/p_s) fit best. Negative transfer when domains differ too much.

**Where it would enter alibz**: docs/transformer_plan.md: synthetic training is source, real Z300 spectra target. Use unlabeled real spectra for domain-adversarial or moment alignment; instance weights not needed when physics is explicit. Fine-tune on pseudo-labelled real spectra only if gated by alibz fit quality.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz and data/remote_samples/*.csv: line-matching fraction and Fe T consistency of model output vs zero-shot; expected improvement; guard with negative-transfer check on S.

**Caveats**: Survey; no quantitative LIBS numbers. Kepes 2022 shows TL can hurt (kepes2022transfer).
