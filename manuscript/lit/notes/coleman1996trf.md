---
key: coleman1996trf
title: "An interior trust region approach for nonlinear minimization subject to bounds"
year: 1996
topic: ml
relevance: 2
---

**Method/claim**: Reflective/interior trust-region Newton method for bound-constrained NLS; the algorithm behind scipy least_squares(method='trf').

**Adoptable specifics**: Sec. 2: affine scaling D(x) with distances to bounds, strictly feasible iterates, reflective steps; two-dimensional subspace solved per iteration (Branch 1999).

**Where it would enter alibz**: refinement.py _fit and peaky_finder fits: already trf. Specific advice: set bounds that exclude the boundary (strictly inside) because TRF steps never sit on a bound; supply x_scale and jac sparsity for block-coordinate fit_all (jac_sparsity with lsmr).

**Fit-quality test**: fit_all runtime per spectrum and Fe standard fe_aesar_9998_means_20260922.npz peak-location residual; expected 2-10x faster fit_all with sparse Jacobian, same residuals.

**Caveats**: Interior iterates make parameters pinned at a bound (n_e-at-bound) converge slowly; this is diagnostic, not a bug.
