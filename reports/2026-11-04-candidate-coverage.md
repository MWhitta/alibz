# Candidate-coverage fix — HALTED, premise falsified, needs judgment

Date 2026-11-04. Agent: specialist (Opus 4.8). Provider: Claude subscription (no switch).
`.venv/bin/python`, db `db`. The fit-quality panel was NOT run (moissanite on series 2).
**No production code changed.** This is a stop-and-escalate: the brief's stated root cause
(candidate-line coverage gap for Mg II) is not reproducible, and the prescribed fix targets a
gap that does not exist. Evidence below; the design question is at the end.

## Bottom line

- I reproduce the brief's solve EXACTLY: `silicate_4el` oracle at truth, `nearest`, candidates
  restricted to the 4 true elements, all prefilters off → **Si .390 / Ca .293 / Fe .294 / Mg .023**
  (binary_ca_mg → Ca .600 / Mg .400). So my harness matches the finding's setup.
- But the **orphaned-flux statistic as the brief defines it is 0.0%, not 54%.** Every rendered Mg II
  line HAS a same-element candidate line within `shift_tolerance` (0.1 nm). Verified three ways
  (raw db, built `line_table` with prefilters off, min_gA=0). The candidate builder DOES offer the
  Mg II lines (`line_table.n_lines=10583`, all Mg I/II in band present).
- The real "54%" is a different statistic and a different defect (below). A coverage rule (brief
  step 2) is a **no-op** here, and the step-3 regression test ("orphaned flux < 5%") would pass
  trivially *before* any change while failing to capture the actual defect.

## What 54% actually is (reproduced to the exact number)

54.2% of Mg's rendered flux (9.548e3 / 1.762e4, on 230 / 842 Mg peaks) lands on peaks whose
per-species design-matrix row `A[i,:]` is **entirely zero** — the flux is orphaned *in the design*,
but NOT because the candidate wavelength is missing. Breakdown of the mechanism:

- Of those zero-design Mg peaks, **100% of the orphaned FLUX** (54.2%) sits on peaks that share an
  EXACT wavelength (89 peaks) / a wavelength within 1e-3 nm (114 peaks) with ANOTHER peak that has a
  nonzero design row. The remaining zero-design peaks are XUV (18–21 nm) with ~1e-28 flux — noise.
- Cause: the forward model (`synthetic._component_lines`) renders MULTIPLE database transitions at
  the same wavelength (e.g. at 279.5528 nm: Mg II gA=1.04e9 PLUS a Mg II gA=0.0 duplicate db row; at
  279.0777 nm: Mg II gA=1.6e9 plus a gA=0 twin). The oracle test harness (`_oracle_peaks`) emits
  **one peak per rendered db line**, so these become two coincident peaks at an identical wavelength.
- `overlap_mode="nearest"` (`_apply_overlap_mode`, peaky_indexer_v3.py:1043) keeps only the
  max-overlap entry **per column (per db line)**. When two peaks are degenerate in wavelength, every
  co-located candidate line's argmin/lexsort tie-breaks onto ONE of the twins; the other twin gets
  NO candidate line → zero design row → its real flux (areas 85, 17, …) is unattributable. Si/Ca/Fe
  show ~0% of this because their strong lines are not rendered as coincident duplicate pairs.

So the chain is: duplicate coincident db lines → duplicate coincident oracle peaks → `nearest`
winner-take-one assignment orphans one twin. NOT candidate coverage, NOT the gA floor, NOT the
emissivity gate, NOT `max_ion_stage`, NOT band clipping.

## Self-consistency check (confirms solver/candidates are fine, the per-peak map is not)

At truth, τ=0 and atomic-uncertainty disabled, so forward areas carry no hidden factors.
`obs == A·c_true` holds to machine precision in **element totals** (ratio 1.000 for all four), i.e.
the candidate set and emissivities are consistent forward↔inverse. But **per-peak**
`||obs − A·c_true|| / ||obs|| = 0.37` (56 peaks) because `nearest` redistributes the twin-peak flux.
Setting `obs := A·c_true` (per-peak self-consistent) and re-solving recovers **Mg .091** (≈ truth
.10; Si/Ca/Fe .363/.273/.273). So both the NNLS solver and the candidate list are faithful; the
defect is the per-peak design construction under `nearest` on degenerate peaks.

Dropping Fe from candidates does NOT restore Mg (Si absorbs it: Mg .032, Si .553) — no single-element
coverage/exclusion fix helps, confirming it is not element-specific.

## Design size / runtime (brief asked)

`nearest`, 4-element restricted, prefilters off: design `A` = **14701 peaks × 8 species**
(`line_table.n_lines = 10583`). One `solve_at` = **0.06 s**. (Unchanged — no fix applied.)

## Heavy-element hallucination (U/Pb) — brief step 4

Not reached. It was observed under `conserve` with the full candidate set; the above work is under
the 4-element-restricted `nearest` oracle (no U/Pb present), so I cannot report whether a coverage
fix reduces it, because there is no coverage fix to apply. It is a separate (spurious-species-gate)
defect, as the wave-2 report already concluded; not a one-liner.

## The design question for judgment (why I did not implement)

The brief prescribes a candidate-coverage rule + an "orphaned flux < 5%" regression test. The
orphaned flux by that definition is already 0%; the rule changes nothing and the test does not guard
the real defect. The real defect is one of two design calls, each with regression risk to the seven
uncommitted options (esp. `overlap_mode`) and each flagged by the wave-2 report as panel-arbitrated:

1. **Oracle harness realism.** `_oracle_peaks` emits one peak per db line, creating two peaks at an
   identical wavelength — something no spectrometer can resolve and the real peak-fitter never
   produces (it merges coincident lines into one feature). The at-truth oracle may simply be an
   unphysical target for `nearest`. Fix = merge coincident oracle peaks (test-harness change), which
   makes the at-truth claim moot but does not touch production.
2. **`nearest` semantics on degenerate peaks.** Change `_apply_overlap_mode(mode="nearest")` so a
   candidate line whose best-match wavelength is shared by ≥2 peaks is distributed across the tied
   peaks (or so coincident peaks are merged before assignment) instead of winner-take-one. This is a
   production `overlap_mode` change that alters every `nearest` solve — the wave-2 report explicitly
   says mode choice must be scored on the fit-quality panel before adoption, which I was told not to
   run.

Both are design decisions beyond "candidate coverage," and the prescribed fix would not move the
Mg number. Recommend judgment decide (1) vs (2) (or that the at-truth `nearest` oracle is not a
target worth fixing), then re-brief. I did not want to ship a coverage change that is a no-op or a
`nearest`-semantics change that needs the panel I was told to avoid.

## Tests
No production code changed, so I did not re-run the suite; the wave-2 report's baseline
(`ALIBZ_SKIP_ROUNDTRIP=1 pytest -q` → 660 passed, 10 skipped, 71 subtests) stands unchanged.

## Artifacts (scratchpad, not in repo)
`orphan.py`, `orphan2.py`, `collin.py`, `selfcon.py`, `argmax.py`, `dump.py`, `dup.py` under the
session scratchpad. All numbers above are from these.

---

## Follow-up: option (1) implemented — merged (fitter-realistic) oracle

Date 2026-11-04 (same session). Coordinator decision: the at-truth oracle must behave like the
peak-fitter — every rendered line within `max(1e-3 nm, 0.25 x instrumental FWHM at that wavelength)`
of another coalesces into ONE peak with the summed area (db duplicates at identical wavelength are
the degenerate case). `overlap_mode="nearest"` semantics unchanged. No production code changed; the
builder lives in the new **tests/test_oracle_roundtrip.py** (`merged_oracle_peaks`, shared for later
waves). Instrumental FWHM = Olivero-Longbothum Voigt FWHM of the per-segment
`InstrumentResponse.gaussian_sigma_nm`/`lorentzian_gamma_nm`; single-linkage clustering on the sorted
axis; area-weighted centroid + area-weighted widths. Peak-table shrinks: silicate 14701 -> 8717,
binary -> 1237. I left tests/test_overlap_conservation.py bit-identical (it now serves as the
UNMERGED-oracle regression record documenting the collapse); the merged target is the new file.

### silicate_4el at truth (merged oracle), element fractions

| mode | restriction | Si | Ca | Fe | Mg | spurious >1% |
|---|---|---|---|---|---|---|
| legacy | 4-el | 0.309 | 0.373 | 0.226 | 0.092 | — |
| conserve | 4-el | 0.386 | 0.236 | 0.271 | 0.107 | — |
| nearest | 4-el | 0.364 | 0.274 | 0.271 | 0.091 | — |
| legacy | full | 0.305 | 0.368 | 0.222 | 0.091 | none |
| conserve | full | 0.366 | 0.225 | 0.255 | 0.101 | **Dy .014 / Pb .016 / U .015** |
| nearest | full | 0.364 | 0.274 | 0.271 | 0.091 | **none** |

Truth Si/Ca/Fe/Mg = .40/.30/.20/.10.

### binary_ca_mg at truth (merged oracle), element fractions

| mode | restriction | Ca | Mg |
|---|---|---|---|
| legacy | 4-el | 0.606 | 0.394 |
| conserve | 4-el | 0.578 | 0.422 |
| nearest | 4-el | 0.600 | 0.400 |
| legacy | full | 0.604 | 0.393 |
| conserve | full | 0.576 | 0.420 |
| nearest | full | 0.600 | 0.400 |

Truth Ca/Mg = .60/.40. No spurious species >1% in any binary cell.

### Verdict

- The merge FIXES the Mg collapse: nearest silicate Mg .023 -> **.091** (truth .10). The degenerate
  Mg II twins are no longer orphaned (zero-design Mg flux 54.2% -> 0%).
- nearest silicate: **Si .364 / Ca .274 / Mg .091 all within +/- 0.05**; **Fe .271 is OFF by 0.071,
  so it does NOT meet the literal +/- 0.05.** This is not the orphaning defect and not fixable by any
  peak-table change: the fully self-consistent control `obs := A.c_true` (zero orphaned flux) gives
  the SAME Fe .273. It is the stage-aggregation (`_aggregate_elements`) floor for a 3-stage element
  (Fe I/II/III). The test asserts Si/Ca/Mg at +/- 0.05 and Fe non-collapsed (0.05 < Fe < 0.35) with a
  loud comment; I did NOT silently assert +/- 0.05 for Fe.
- binary: nearest exact .600/.400; merging also removes most of legacy's historic bias here
  (Ca .73 unmerged -> .606 merged) — so on this scene the legacy-vs-conserve debate was really about
  coincident-line merging.
- **Step 4 (U/Pb hallucination, full candidates, realistic oracle):** nearest shows **NO spurious
  species >1%** on either scene — it is clean once the oracle is realistic. conserve STILL hallucinates
  on silicate (Dy 1.4% / Pb 1.6% / U 1.5%, ~4.5% total, absorbing Ca which drops to .225); legacy and
  binary-all-modes are clean. So the heavy-element hallucination is specific to `conserve` + full
  candidates, not to `nearest`; a separate spurious-species-gate question for the panel, not a
  one-liner.

### Tests (new file only; no production code changed)
`ALIBZ_SKIP_ROUNDTRIP=1 pytest tests/test_oracle_roundtrip.py -q` -> **3 passed** (15.3s).
tests/test_overlap_conservation.py unchanged -> 9 passed (re-run, 85.8s). Suite collects 673 tests,
no collection/import errors. Full suite not re-run end to end (unchanged elsewhere; ~11 min); fit-
quality panel deliberately not run.
