---
key: kepes2022transfer
title: "Improving LIBS regression models via transfer learning"
year: 2022
topic: ml
relevance: 3
---

**Method/claim**: CNN trained on ChemCam transformed to SuperCam-like spectra extends SuperCam training data using overlapping targets.

**Adoptable specifics**: TL lowered RMSEP of TiO2, CaO, K2O by 16, 11, 22 % on the full test set, no effect on SiO2, Na2O, worse for the rest; after outlier filtering up to 28 % lower RMSEP (CaO). Training reproducibility degraded with TL (Sec. 3, Table 3).

**Where it would enter alibz**: Lesson for transformer_plan: test for negative transfer per element; use paired overlap targets as the calibration-transfer set.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz: per-line matching vs zero-shot; accept TL only if no element worsens.

**Caveats**: Gains inconsistent; reproducibility lower.
