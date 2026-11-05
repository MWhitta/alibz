---
key: gornushkin2022montecarlo
title: "Intrinsic Performance of Monte Carlo Calibration-Free Algorithm for Laser-Induced Breakdown Spectroscopy"
year: 2022
topic: cflibs
relevance: 2
---

**Method/claim.** Forward-model CF-LIBS: a GPU Monte Carlo draws 1e5-1e6 configurations of (T, plasma size, species concentrations) for a uniform LTE plasma, synthesises spectra and minimises a cost function against a simulated slag spectrum (8 elements). Intrinsic accuracy ~1% on all eight elements (abstract).

**Adoptable specifics.** Test spectrum: resolution 84 pix/nm, Gaussian noise 0.5% of max, NIST lines, Griem Stark widths (Sec. 3, ~line 396-400). Cost function weighted per element (w_k = sum w_n p_n S / sum S) so minors count equally to majors (Sec. 2). Multi-start convergence: 11 random starts reached cost 0.0023 +/- 0.0001. RSD below 10%, mostly ~1% (Table 3); accuracy degrades toward low concentrations only when noise rises. Own line width from self-absorbed profiles solved via Eq. 5 when Stark data are missing.

**Where it would enter alibz.** `synthetic.py` / S suite: report 1% as the noise-free ceiling and add the 0.5% noise, 84 pix/nm configuration as a calibration case; per-element cost weighting in the pixel-level estimator.

**Fit-quality test.** Synthetic suite S round trips: median relative error per element vs this ~1% ideal bound; multi-start spread of cost as a basin-collapse metric.

**Caveats.** Idealised simulated data only; authors state real spectra will do worse. Heavy compute; no handheld or real-spectrum evidence.
