---
key: humlicek1982voigt
title: "Optimized computation of the Voigt and complex probability functions"
year: 1982
topic: ml
relevance: 1
---

**Method/claim**: w4 rational approximations to Faddeeva w(z) in four regions of the (x,y) plane, relative accuracy ~1e-4.

**Adoptable specifics**: Region boundaries s=|x|+y: >15 (2-term), 5.5-15, y>=0.195|x|-0.176, else (Sec. 3). Superseded by Zaghloul/Johnson (Faddeeva) at 1e-13.

**Where it would enter alibz**: alibz uses scipy.special.voigt_profile (Faddeeva, ~1e-13) in peaky_finder, inspection, minor_lines: no adoption. Only relevant for gpu.py voigt_profile_gpu, where a Humlicek-W4 kernel could replace a custom form; 1e-4 relative error suffices for 10^3-count lines.

**Fit-quality test**: Runtime of gpu.py profile evaluation; max-abs diff to scipy < 1e-4 relative (unit test). Fit residual unchanged.

**Caveats**: 1e-4 error is a systematic bias for high-S/N peaks (peak-location residual unaffected, area weights slightly).
