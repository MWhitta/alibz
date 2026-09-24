# Coordinator verification of the V_pure_run2 timing analysis — 2026-09-23

Independent checks (peak height minus side-band median, run means of per-shot native CSVs,
shift −0.14 nm) against cal/reports/2026-09-23-vanadium-timing-study.md (agent draft 1).

CONFIRMED
- Line signal flat with period: V I 411.2/437.9, V II 311.1 peak heights change <10 % P1→P50 at every delay.
- Frame drops 23/59; confound (delay row = raster lap; period = fixed 3-site set).

REFUTED / CORRECTED
1. "Continuum doubles with period" is not plasma continuum. The P50−P1 difference spectrum is the
   same at delays 10/20/50 (r = 0.94/0.96 vs d50) although line signal differs 2× between them;
   NIR (880–900 nm) 10th-percentile level goes NEGATIVE with period, delay-independent:
   run2 ≈ +20/−10/−100/−240 counts at P1/10/25/50 for every delay; run1 identical
   (+15/−70/−220 at P10/25/50). ⇒ period-scaled instrument background / dark fixed pattern with
   imperfect vendor subtraction (sign differs by segment). Period behaves like detector exposure
   longer than the plasma; it is not a gate on the plasma timescale.
2. Reading (a) "period is the gate and emission is complete within 1 unit" is internally
   inconsistent: d50/P1 still carries ~40–50 % of d1 line height (V I 411: 1114→500).
3. Delay IS a real plasma-timing effect, not only depth. run1 (0–20 prior shots/site): log-linear fit
   of V II 311 height on delay + prior shots: −0.0217 ± 0.0048 /unit (4.5σ) vs depth −0.0013 ± 0.0047;
   V I 411: −0.0053 ± 0.0054 (n.s.). The draft's "V II/V I fall is depth" is wrong.
4. "Long gate adds T-mixing apparent SA" (478.65/475.39 pair) is a fixed-pattern artefact: h475 gains
   an additive +40–55 counts P1→P50 at every delay; the P50−P1 difference has a +110–120 bump at
   475.2 nm at d1, d20 and d50 alike. The T-independent 417.94/415.97 peak-height ratio is flat with
   period (d1: 0.91/0.92/0.92/0.90); the draft's τ falling 1.70→0.47 with period is a measurement
   artefact. §4.4 synthetic-integral test is invalid (P1 and P50 at the same delay hold the same line
   photons).
5. Owner hypothesis (short integration reduces apparent opacity by excluding changing-T plasma):
   NOT testable with the period knob; not supported. Real SA (417/415 compressed at every period)
   is period-independent. Only delay changes the temporal mix (V II/V I changes).
OPEN: 417/415 peak ratio falls with delay/lap (0.91 at d1 → 0.49 at d50) — more opacity late/deep,
or a blend; not separated.
