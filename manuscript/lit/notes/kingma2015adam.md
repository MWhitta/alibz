---
key: kingma2015adam
title: "Adam: a method for stochastic optimization"
year: 2015
topic: ml
relevance: 1
---

**Method/claim**: Adam: first/second moment adaptive step; defaults alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8.

**Adoptable specifics**: Alg. 1 with bias correction; AdamW decoupled weight decay is the later fix. Training only.

**Where it would enter alibz**: docs/transformer_plan.md synthetic-trained model: use AdamW lr 3e-4 (peak, cosine, 5% warmup) and gradient clip 1.0; nothing to change in the physics solver.

**Fit-quality test**: Validation loss on held-out synthetic set; then S-suite majors error of model output. Effect depends on architecture.

**Caveats**: Training detail only; no LIBS evidence.
