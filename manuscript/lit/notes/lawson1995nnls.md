---
key: lawson1995nnls
title: "Solving Least Squares Problems"
year: 1995
topic: ml
relevance: 3
---

**Method/claim**: Chapter 23 gives the active-set NNLS algorithm (Algorithm NNLS) that solves min ||Ax-b|| s.t. x>=0 exactly in finitely many steps via Kuhn-Tucker conditions.

**Adoptable specifics**: Alg. NNLS (Ch.23): passive set P, dual w=A^T(b-Ax); add index with max w>tol, solve unconstrained LS on P, step back to the boundary if any x_P<=0. Termination tol is on w, not on x. Weighted variant = row-scale A and b by 1/sigma (Ch.1 weighted LS), which is exactly the whitened solve alibz already runs.

**Where it would enter alibz**: peaky_indexer_v3.py WNNLS solve (~l.1748, 2084): already uses scipy.optimize.nnls (a Lawson-Hanson port) on rows whitened by sigma_eff. Nothing to adopt except: report the dual w of excluded elements as a free detection statistic (w_j/se_j = KKT z for absent elements).

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): spurious mass and majors missed; expected small gain (diagnostic only). Runtime unchanged.

**Caveats**: scipy's nnls caps iterations at 3n; ill-conditioned dictionaries (collinear element columns) can hit the cap silently. No uncertainty output.
