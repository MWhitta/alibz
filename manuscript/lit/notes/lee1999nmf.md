---
key: lee1999nmf
title: "Learning the parts of objects by non-negative matrix factorization"
year: 1999
topic: ml
relevance: 2
---

**Method/claim**: NMF: V~WH with W,H>=0 yields additive parts; multiplicative update rules (Lee & Seung 2001 Alg.).

**Adoptable specifics**: Updates H<-H*(W^T V)/(W^T W H); W<-W*(V H^T)/(W H H^T); parts-based sparse representation. NNDSVD init is later work.

**Where it would enter alibz**: Concentration matrix over many spectra at shared (T,n_e): NMF would learn element pseudo-spectra W; but alibz fixes W from the database. Use NMF on residual spectra of the fleet to discover unmodelled components (instrument fixed pattern, missing species) feeding peaky_pca.

**Fit-quality test**: data/remote_samples/*.csv: residual r^2 gain after adding k learned components; stage_disagreement down.

**Caveats**: Non-unique factorisation; risk of absorbing real elements into nuisance components.
