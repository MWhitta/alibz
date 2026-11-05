---
key: branch1999trf
title: "A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems"
year: 1999
topic: ml
relevance: 2
---

**Method/claim**: Extends Coleman-Li with 2-D subspace and PCG for large problems; basis of tr_solver='lsmr' in SciPy.

**Adoptable specifics**: Sec. 3: 2-D subspace of gradient direction and (approximate) Newton direction; PCG for large systems; Sec. 5 numerical results on bound-constrained sets.

**Where it would enter alibz**: peaky_finder.fit_all with hundreds of peaks: pass tr_solver='lsmr' and jac_sparsity (block diagonal: each peak touches only its own window). Currently dense Jacobian.

**Fit-quality test**: Runtime per spectrum; fit residual unchanged on data/remote_samples/*.csv (r^2 equal to 1e-6).

**Caveats**: Pure speed item; no accuracy effect expected.
