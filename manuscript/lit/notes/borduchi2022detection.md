---
key: borduchi2022detection
title: "Study of the effects of detection times in laser-induced breakdown spectroscopy and missed variation of plasma parameters with gate width"
year: 2022
topic: cflibs
relevance: 2
---

**Method/claim.** Effect of delay and gate width on T, n_e (Saha-Boltzmann plot with OPC) and SNR in NaCl matrices (Ca 1-3 wt%, TiO2/CuSO4 spiking). Result: measured plasma parameters did NOT depend on gate width, SNR of Ca I 643.9 nm rose with gate width; LTE held even for gates >= 20 us except at 4 us delay. Plasma cools >50,000 K in tens of us.

**Adoptable specifics.** Abstract numbers above. Wider gate buys SNR without changing the apparent T, n_e in this system. Notes: more samples/pressures needed.

**Where it would enter alibz.** Acquisition guidance for Z300 (delay is the lever, period is exposure; see Z300 notes): test whether a longer gate raises Fe II/trace SNR without shifting the fitted T. Not a code change; could change `bench` acquisition presets.

**Fit-quality test.** Z300 Fe standard: Fe I Boltzmann T consistency and resolved-fit count vs gate width (expect resolved-fit count up, T unchanged). Paper: no T change across gate widths.

**Caveats.** Salt matrix and lab ICCD; Z300 gate options are limited and T drifts quickly at 7-12 kK, so the invariance may not hold.
