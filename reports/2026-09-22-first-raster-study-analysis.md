# First complete stepped (raster) delay/period study: analysis — 2026-09-22

Session `opt-b61065d72c7941fcbfa5317099265cb2`, sample "Fe pure metal", grid
delays [5, 10, 20] × periods [10, 25, 50] at pulsePeriod 100 (10 Hz), 10 shots
per batch, one gilbert raster site per batch (window [134,76]–[206,124] step 24,
12 sites). Six batches completed (13:38–13:46 PDT), then the optimizer's next
proposal, delay 20 / period 10, was refused as an unverified condition and the
operator closed the session. Data source: Pantheum SQLite
(`optimization_batches.metrics`, `acquisitions`), read 14:05 PDT.

## 1. Results per condition

Score = ln(1 + median SNR) × usable lines / 96 reference windows / (1 + median CV)
(`Fe-window-quality-v1`, native grid). "Signal 1st/last" is the ratio of total
positive signal in the first and last shot of the batch (crater/coupling drift).

| delay/period | site (raster #) | shots | usable lines | median SNR | median CV | score | signal 1st/last | max counts (median) |
|---|---|---|---|---|---|---|---|---|
| 5/10  | [158,100] (3) | 10 | 18 | 12.2 | 0.208 | **0.400** | 1.02 | 14,964 |
| 10/50 | [158,124] (6) | 10 | 17 | 11.0 | 0.181 | 0.373 | 0.94 | 13,523 |
| 10/10 | [134,100] (4) | 9 (1 dropped) | 16 | 11.8 | 0.174 | 0.362 | 1.11 | 11,841 |
| 5/50  | [134,124] (5) | 10 | 17 | 9.7  | 0.171 | 0.358 | 0.88 | 15,859 |
| 5/25  | [158,76] (2)  | 10 | 17 | 11.1 | 0.292 | 0.342 | 1.02 | 13,157 |
| 10/25 | [134,76] (1)  | 9 (1 dropped) | 14 | 14.7 | 0.274 | 0.315 | **0.51** | **9,063** |

Not tested: 20/10, 20/25, 20/50 (delay 20 never reached; see §4).

Factorial means (one observation per cell, so indicative only):

| factor | level | mean score |
|---|---|---|
| delay | 5 | 0.367 |
| delay | 10 | 0.350 |
| period | 10 | 0.381 |
| period | 25 | 0.329 |
| period | 50 | 0.366 |

## 2. How large is a real difference?

The 10/25 baseline has been run six times today across sessions, all at the
same spot [134,76,70]:

| batch | time | shots | lines | SNR | CV | score |
|---|---|---|---|---|---|---|
| run-eae9d54f | 09:31 | 10 | 18 | 12.0 | 0.140 | 0.422 |
| run-126d0c96 | 09:41 | 10 | 12 | 14.9 | 0.176 | 0.294 |
| run-b95ffbb4 | 11:12 | 10 | 14 | 16.0 | 0.248 | 0.331 |
| run-8264fabf | 12:17 | 8  | 16 | 12.4 | 0.254 | 0.345 |
| run-b17cc043 | 12:33 | 10 | 15 | 14.7 | 0.164 | 0.370 |
| run-bb777df5 | 12:38 | 9  | 14 | 14.7 | 0.274 | 0.315 |

Mean 0.346, SD 0.045, range 0.294–0.422. The whole spread of the six study
conditions (0.315–0.400, range 0.085) is under two baseline SDs, and the
factor contrasts (0.02–0.05) are about one SD. **One batch per condition cannot
rank these conditions.** The only defensible statements are qualitative:
period 25 is not better than 10 or 50, and delay 5 is at least as good as 10.

## 3. The site-1 confound

Raster site 1 is [134,76,70], the fixed spot used by every earlier session
today: at least 60 shots had already been fired there before this study's
first batch. That batch (10/25) shows the signature of a deep crater: signal
halved from first to last shot (0.51; every fresh site is 0.88–1.11), the
lowest peak counts (9,063 vs 11,841–15,859), fewer usable lines (14), and a
different line pattern (Fe-8178 SNR 118 and Fe-11271 23 vs 50–100 and 7–10
elsewhere; several other windows down to SNR 6–8). Its score is the worst of
the six, and its CV the second worst. So 10/25's poor showing here is at least
partly the spot, not the timing. The per-line SNR table for the 12 lines usable
in all six cells is in the appendix.

The gilbert raster starts at the window origin, so **every new session's first
batch (always the 10/25 baseline) lands on the worn spot** until the window or
the sample is moved.

## 4. Why the study stopped

The next proposal was delay 20 / period 10. The verified-condition rule (live
runs must have fully stored a 10-shot batch at that delay/period/pulsePeriod)
refused it: the ledger has full runs for 5/10, 5/25, 5/50, 10/10, 10/25, 10/50
and 20/25, none for 20/10 or 20/50. This session was created before the rule
was deployed, so its grid was never checked at creation; new sessions are
checked at creation with the same message.

`scripts/check-study-conditions.py` (new) is the pre-start check: it lists
verified/unverified conditions for a grid or an existing session, the
Acquire-panel runs that would verify them, active acquisitions, trigger lock
and calibration flags, and exits 1 if the study would be refused. Today:

```
20/10  NO   20/50  NO   (all seven others verified with 10/10 shots)
BLOCKER: 2 unverified condition(s): 20/10, 20/50
```

## 5. Recommendations

1. **Verify 20/10 and 20/50** with one 10-shot Acquire-panel batch each
   (fresh spot, pulsePeriod 100), then re-run the check; a 3×3 study can then
   run to completion. Alternative: set `optimization.require_verified_conditions`
   to false, since a failed batch now counts as "tested, unusable" and the
   session continues; that costs one wasted batch per bad condition.
2. **Start the next raster on fresh metal.** Move the window (e.g. startLocation
   y = 100, or a new x range) or reposition the sample. Longer term the
   optimizer should keep a per-site shot ledger and skip sites with prior
   shots (elevated below).
3. **Replicate.** With SD ≈ 0.045 per batch, ranking conditions 0.03 apart
   needs ≥ 3 batches per condition. The optimizer currently explores the grid
   once; a `replicates` setting (or re-running the grid on fresh sites) is
   needed before the score can choose a condition.
4. Keep watching `dropped_frames`: 2 of 6 batches lost one spectrum
   (spectrometer checksum errors continue after the reboot).

## Elevated for the owner

- Fresh-site rule in the optimizer (skip raster sites with prior shots, from the
  acquisition ledger) and a visible per-site shot count in the portal.
- Replicates per condition in delay/period studies.
- Whether to relax verification to "≥ acquire.min_shots stored" so a
  frame-dropped 9/10 Acquire-panel run still verifies a condition (today it does
  not; a fully stored run is required).

## Appendix: per-line median SNR (lines usable in all six cells)

| line | 5/10 | 5/25 | 5/50 | 10/10 | 10/25 (worn) | 10/50 |
|---|---|---|---|---|---|---|
| Fe-7992  | 8.3 | 10.4 | 7.7 | 7.9 | 14.1 | 9.0 |
| Fe-8178  | 60.9 | 66.8 | 49.7 | 53.3 | 117.9 | 99.2 |
| Fe-8940  | 10.5 | 11.1 | 9.3 | 9.0 | 7.2 | 11.1 |
| Fe-9062  | 7.4 | 7.7 | 7.2 | 7.7 | 15.8 | 8.0 |
| Fe-9782  | 7.8 | 7.9 | 7.5 | 8.6 | 12.8 | 7.9 |
| Fe-9804  | 28.2 | 34.3 | 35.1 | 33.7 | 41.7 | 37.7 |
| Fe-10563 | 20.1 | 18.1 | 13.4 | 23.8 | 8.3 | 24.6 |
| Fe-11271 | 6.7 | 7.9 | 9.9 | 6.9 | 23.3 | 9.4 |
| Fe-11503 | 12.5 | 12.9 | 10.3 | 12.8 | 15.3 | 14.1 |
| Fe-11533 | 21.5 | 21.0 | 28.3 | 41.3 | 21.5 | 29.4 |
| Fe-11687 | 10.1 | 8.6 | 9.7 | 10.3 | 18.6 | 9.4 |
| Fe-12385 | 14.0 | 20.2 | 6.6 | 16.6 | 5.7 | 11.0 |

Line ids are the reference-window ids in `optimization_metrics.py`; the
`atomic_uncertainties_unavailable`, `timing_units_unverified` and
`saturation_limit_unknown` flags apply to every batch.
