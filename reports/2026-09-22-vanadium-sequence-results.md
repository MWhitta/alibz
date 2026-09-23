# First replicated delay/period study on vanadium — 2026-09-22

Session `opt-9d384296` ("V_pure_run1", composition V, live, 10 Hz), run as a
one-click sequence: grid delays [5, 10, 20] × periods [10, 25, 50], three
replicates each, 27 batches of 10 shots on 12 raster sites (laps 1–3),
16:12–16:33 PDT, no operator action between batches. Score is the per-element
window-quality metric (`composition-window-quality-v1`, V reference,
96 windows, native grid). Data read from Pantheum at 16:40 PDT.

## Per condition (mean ± SD of 3 replicates)

| delay/period | score | median SNR | median CV | usable lines | shots stored |
|---|---|---|---|---|---|
| 10/10 | **0.500 ± 0.010** | 18.7 | 0.200 | 19.3 | 9/9/10 |
| 20/10 | **0.497 ± 0.036** | 19.5 | 0.223 | 19.3 | 10/10/10 |
| 10/25 | 0.472 ± 0.016 | 17.5 | 0.203 | 18.7 | 8/10/9 |
| 20/25 | 0.469 ± 0.035 | 17.4 | 0.180 | 18.3 | 10/10/10 |
| 5/10  | 0.457 ± 0.016 | 17.5 | 0.196 | 18.0 | 10/9/9 |
| 5/25  | 0.438 ± 0.005 | 18.0 | 0.236 | 17.7 | 10/10/10 |
| 10/50 | 0.426 ± 0.017 | 14.8 | 0.190 | 17.7 | 10/10/9 |
| 5/50  | 0.414 ± 0.024 | 14.4 | 0.168 | 17.0 | 10/10/10 |
| 20/50 | 0.402 ± 0.008 | 13.5 | 0.225 | 17.7 | 10/10/6 |

Pooled replicate SD 0.021 (SE of a condition mean 0.012; of a factor mean over
nine batches 0.007).

## Factor means

| factor | level | mean score |
|---|---|---|
| period | 10 | 0.485 |
| period | 25 | 0.460 |
| period | 50 | 0.414 |
| delay | 5 | 0.436 |
| delay | 10 | 0.466 |
| delay | 20 | 0.456 |

## Reading

- **Period dominates and is resolved.** 10 → 50 costs 0.071 (≈ 10 SE) and
  the median SNR falls from ~19 to ~14: the longer gate adds continuum and
  noise after the line emission has decayed. Every delay row shows the same
  monotonic order.
- **Delay matters less.** 5 is lowest (0.030 below 10, ≈ 4 SE); 10 and 20
  are indistinguishable. Caveat: conditions ran in grid order, so delay 20 fell
  entirely on lap-2/3 sites (10–20 prior shots each) while delay 5 ran on
  fresh sites; the delay contrast is partly confounded with crater depth, the
  period contrast is not (each row's periods sat on adjacent sites of one lap).
- **Best cell: 10/10 (0.500 ± 0.010)**, with 20/10 equal within error. The
  ranking the Fe study could not make with one batch per condition is made
  here with three, because the replicate SD (0.021) is half the Fe worn-spot
  repeat SD (0.045): fresh sites plus replicates are what made the grid readable.
- **Vanadium scores higher than iron** (0.40–0.50 vs 0.29–0.42) mainly through
  more usable lines (17–19 of 96 windows vs 12–18) at similar SNR.
- **Frame drops persist** (10 of 27 batches short; the last batch stored only
  6 of 10). The 6-shot 20/50 replicate is included; excluding it moves nothing.

## Next

Extend the period axis downward (5, and the instrument minimum) at delays
10–20, each new point verified once from the Acquire panel; randomise
condition order or interleave replicates across laps so delay is no longer
confounded with crater depth; keep three replicates per condition.
