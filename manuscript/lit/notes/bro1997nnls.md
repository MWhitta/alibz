---
key: bro1997nnls
title: "A fast non-negativity-constrained least squares algorithm"
year: 1997
topic: ml
relevance: 2
---

**Method/claim**: FNNLS: precompute A^T A and A^T b once and run the Lawson-Hanson loop on normal equations; large speed-ups when A has many rows and few columns.

**Adoptable specifics**: Sec. 'Modified NNLS: Fast NNLS' (l.218): work with X^T X (n_cols^2) and X^T y; reuse across repeated solves with the same A. Speedup grows with rows/cols ratio (pixels >> elements). Caveat: squaring the condition number.

**Where it would enter alibz**: peaky_indexer_v3.py outer objective calls NNLS ~40-200x per spectrum with a cached dictionary when only (T,n_e) move; for the fixed-dictionary re-solve (solve_at) precompute the Gram matrix of whitened columns and call a Gram-based NNLS (scipy has none; implement the 30-line FNNLS). Only worthwhile if profiling shows NNLS dominates.

**Fit-quality test**: Runtime per spectrum (R and S): expected 1.5-5x faster solve step; solution must match scipy nnls to 1e-8 relative (unit test). Concentrations unchanged.

**Caveats**: Normal equations square the condition number; alibz dictionaries with near-collinear Fe/V columns may then differ at 1e-6. Gain is bounded by non-NNLS costs (line synthesis).
