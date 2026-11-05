---
key: forni2013ica
title: "ICA classification of LIBS spectra"
year: 2013
topic: ml
relevance: 1
---

**Method/claim**: ICA components plus Manhattan or Mahalanobis distance classify unknown rocks; Mahalanobis easier to threshold.

**Adoptable specifics**: Distance in ICA space with cut-off; performance improves with more Martian spectra.

**Where it would enter alibz**: Mahalanobis distance of residual vectors as an outlier flag for spectra alibz cannot explain.

**Fit-quality test**: data/remote_samples/*.csv: flagged-spectrum overlap with low r^2.

**Caveats**: Classification, not element-level.
