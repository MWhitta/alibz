---
key: nocedal2006numopt
title: "Numerical Optimization (2nd ed.)"
year: 2006
topic: ml
relevance: 1
---

**Method/claim**: Textbook; only the front matter was retrieved in full text (1.48 MB extraction).

**Adoptable specifics**: Ch.4 trust-region (Delta update rho thresholds 0.25/0.75), Ch.10 nonlinear least squares (Gauss-Newton, LM), Ch.16-17 active-set/QP, Ch.18 SQP. Section numbers only; not verified from the extracted text.

**Where it would enter alibz**: Reference for refinement.py/least_squares choices; for an NN-LASSO with equality constraints (sum c=1 closure) use Ch.16 active-set QP rather than hand-rolled projections.

**Fit-quality test**: Not testable directly; supports golub2003 and coleman1996 experiments.

**Caveats**: Not read in detail; relevance 1.
