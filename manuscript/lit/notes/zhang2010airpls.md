---
key: zhang2010airpls
title: "Baseline correction using adaptive iteratively reweighted penalized least squares"
year: 2010
topic: ml
relevance: 2
---

**Method/claim**: airPLS: w_i = exp(t|d_i|/|d^-|) for negative residuals, zero for positive; adaptive, one parameter (lambda), few iterations.

**Adoptable specifics**: Sec. 2: weights from SSE-based rule; termination |d^-| < 0.001|y| (l.296); lambda 10-10^9 over grid (Fig.). Positive-residual points weighted zero, which biases the baseline down in noisy data.

**Where it would enter alibz**: Alternative to arPLS in peaky_finder._arpls (swap weight rule only; the banded solve is shared). Compare, not adopt: airPLS is known to oversubtract in noise-dominated regions.

**Fit-quality test**: data/remote_samples/*.csv residual r^2 and baseline negativity fraction; synthetic suite S (bench/synth_cases.py) majors error. Expected: equal or worse than arPLS.

**Caveats**: Zero weight on all positive residuals makes baseline sit under the noise floor; arPLS was designed to fix this.
