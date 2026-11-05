---
key: benjamini1995fdr
title: "Controlling the false discovery rate"
year: 1995
topic: ml
relevance: 3
---

**Method/claim**: BH step-up procedure controls expected proportion of false rejections at q for independent p-values.

**Adoptable specifics**: Alg. (l.230): sort p_(1)<=..<=p_(m); k=max{i: p_(i)<=i q/m}; reject H_(1..k). Controls FDR at q*m0/m for independent tests (later extended to PRDS; use Benjamini-Yekutieli with sum 1/i if arbitrary dependence).

**Where it would enter alibz**: detections.py / evidence rows: convert each element's detection z (one-sided) into p and apply BH at q=0.05-0.1 over candidate elements instead of a fixed z threshold. Elements share lines (dependence): use BY or permutation null.

**Fit-quality test**: data/remote_samples/*.csv: spurious-element count (down) vs trace recall; synthetic suite S (bench/synth_cases.py) spurious mass; expected spurious count down at similar recall.

**Caveats**: Independence fails for coincident lines; null of z may not be N(0,1) given non-Gaussian noise: calibrate empirical null on blank/negative-control regions.
