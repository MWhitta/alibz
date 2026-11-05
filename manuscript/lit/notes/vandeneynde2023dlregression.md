---
key: vandeneynde2023dlregression
title: "Deep learning regression for quantitative LIBS"
year: 2023
topic: ml
relevance: 2
---

**Method/claim**: Deep regression for aluminium alloys (27 references, 733 scrap pieces with XRF truth); one loss per element improves all metrics on scrap.

**Adoptable specifics**: Multi-loss (per-element weighted) and excluding the base element from learning; average RMSE 0.02 wt% Al, Si and at most 0.01 wt% Fe, Cu, Mn, Mg, Zn.

**Where it would enter alibz**: For transformer_plan: per-element loss with log-scale or relative error weighting so trace elements are not ignored.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py): median relative error on majors and minors with per-element loss.

**Caveats**: Narrow alloy matrix, ground truth by XRF.
