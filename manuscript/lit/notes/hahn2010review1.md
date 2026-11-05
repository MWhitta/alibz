---
key: hahn2010review1
title: "Laser-Induced Breakdown Spectroscopy (LIBS), Part I: Review of Basic Diagnostics and Plasma—Particle Interactions: Still-Challenging Issues "
year: 2010
topic: cflibs
relevance: 3
---

**Method/claim.** Comprehensive LIBS review (Part I): plasma physics, LTE, line shapes, Stark n_e, SA, detectors, delay/gate. Key point: McWhirter n_e >= 1.6e12 T^1/2 (dE)^3 is necessary, not sufficient; additional temporal (T4.2) and spatial conditions are needed; the criterion alone should be abandoned as proof of LTE.

**Adoptable specifics.** Eq. T4.1 McWhirter (already queued). T4.2/T4.3: |T(t+tau_rel)-T(t)|/T << 1 and same for n_e. Practical LTE checks (Fig. 7): line-to-continuum ratio and ion-to-neutral ratio as independent checks; start by measuring n_e from Stark widths using several lines (avoid adding H), compute with Eq. T4.22/T4.24; LTE is better established at shorter delays (Hahn): at longer delay n_e falls below the McWhirter limit. Eq. T4.12: SA factor with stimulated-emission-corrected absorption coefficient k*.

**Where it would enter alibz.** `pipeline.py` QC: extend the queued McWhirter diagnostic with a line-to-continuum and ion-to-neutral consistency flag (alibz already estimates the continuum in the background model and `stage_disagreement`). Stimulated-emission note matches the dev1 item that was A/B refused. McWhirter: already queued; extras not queued.

**Fit-quality test.** Real spectra: n_e-at-bound flag and QC flag rate; expected direction: flags correlate with high stage_disagreement. Diagnostic only; no accuracy effect size.

**Caveats.** Delay/gate on Z300 are acquisition parameters, and the exact plasma time window is not reported; criteria are for a single-time, single-zone plasma.
