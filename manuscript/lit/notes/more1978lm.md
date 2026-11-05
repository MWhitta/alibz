---
key: more1978lm
title: "The Levenberg-Marquardt algorithm: implementation and theory"
year: 1978
topic: ml
relevance: 2
---

**Method/claim**: Trust-region reading of LM (MINPACK): choose lambda so the step fits a trust radius; scaling by column norms of J.

**Adoptable specifics**: Sec. 4-5: scaled step ||D p||<=Delta with D=diag of Jacobian column norms (adaptive), acceptance ratio rho>0.0001, Delta shrink 0.25 / grow 2 when rho>0.75. scipy method='lm' is this code (no bounds).

**Where it would enter alibz**: peaky_finder.py Voigt fits / refinement.py use least_squares(trf, bounds). LM has no bounds so only for the unbounded inner sub-fits; x_scale='jac' (this paper's D) should be set on trf calls with parameters of very different scale (center nm vs width pm).

**Fit-quality test**: Peak-location residual (pm) on Fe standard fe_aesar_9998_means_20260922.npz and fit_all runtime; expected equal or lower residual with x_scale='jac'.

**Caveats**: No bounds: fit_all has unbounded widths blow-ups (sim-to-real notes); keep trf.
