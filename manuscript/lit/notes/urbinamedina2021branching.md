---
key: urbinamedina2021branching
title: "Branching Ratio Method for Assessing Optically Thin Conditions in Laser-Induced Plasmas"
year: 2021
topic: cflibs
relevance: 3
---

**Method/claim.** Warns that the branching-ratio (intensity-ratio vs optically thin) test can pass even with strong self-absorption. With I/I0 = SA^0.46 and Delta_lambda/Delta_lambda0 = SA^-0.54 (Lorentzian, El Sherbini), the measured pair ratio equals the thin ratio times gamma = SA2/SA1; gamma can be ~1 for two strongly absorbed lines. Pb I in Pb-Sn alloy: SA ~0.25 (tau ~4) yet ratio within 25%.

**Adoptable specifics.** Eq. 2 thin ratio I2/I1 = (g2A2/g1A1) exp(-(E2-E1)/kT); Eq. 3 (same upper level/E2-E1 << kT) = g2A2/g1A1. Eq. 8 I/I0 = SA^0.46; Eq. 11 SA^0.54 = Delta_lambda0/Delta_lambda (sign lost in extraction; verify), Delta_lambda0 = 2 w_s n_e (Eq. 12). SA=0.25: peak 4x lower, integral ~2x lower. Discriminant only if the lines' tau differ strongly.

**Where it would enter alibz.** `utils/absorption.py` / `pipeline.py` QC: when using any pair/doublet thin-ratio check, report gamma (expected SA2/SA1 from tau) and flag the check as non-discriminating if |gamma-1| is small. K I 766/770 shares a lower level with a fixed 2:1 f ratio so it IS discriminating. Also applies to the planned branching-ratio corpus response (A9). Branching-ratio corpus method is planned; gamma criterion is new.

**Fit-quality test.** Synthetic SA round trips: inject tau ~1-4 on a Pb-like pair and confirm the QC flags non-discriminating pairs; real: K 766/770 ratio check unchanged. Metric: false-pass rate of the thin-ratio check (expect lower).

**Caveats.** Needs lines with distinct lower levels; Z300 pixels blur close pairs.
