---
key: he2022mae
title: "Masked autoencoders are scalable vision learners"
year: 2022
topic: ml
relevance: 2
---

**Method/claim**: MAE: mask 75% of patches, encode only visible ones, lightweight decoder reconstructs pixels; scalable self-supervised pretraining.

**Adoptable specifics**: Sec. 3: mask ratio 75% optimal, asymmetric encoder/decoder (decoder 8 blocks, 512 d), normalised pixel targets help. Pretraining 800-1600 epochs.

**Where it would enter alibz**: transformer_plan: self-supervised pretraining on the 4909 unlabeled real CSVs (moissanite) with 1D patches (e.g. 16 pixels) and 50-75% masking, then fine-tune on synthetic labels: domain adaptation without labels.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz/data/remote_samples/*.csv: line-matching fraction and masked-reconstruction error; expected better real-data robustness vs synthetic-only.

**Caveats**: Spectral masking may be trivially interpolated across smooth baseline; mask peak-centred patches.
