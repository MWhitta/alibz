# w3 thermometer options (PARTIAL, handed back before variant results)

Status: implementation + 5 tests done; w3t-base panel equal to cpi-s3 on all 36 items. Variants w3t-offset and w3t-lte were still RUNNING on moissanite (~/alibz-w3t/reports/fit-panel/w3t-{offset,lte}.json, logs alongside); w3t-both, production reruns, panel compares, delta distribution and the full-suite count are NOT done. Full suite was running locally: log at scratchpad/w3t-full.log (scratchpad of this session).

Implementation (tree scratchpad/w3t, alibz/peaky_indexer_v3.py): _last_state recorded in _solve_concentrations_thin; _stage_tie_cost now dispatches (OFF -> _stage_tie_cost_core = the original body, untouched); _species_delta_e, _ne_crit, _lte_flagged_mask, _stage_tie_cost_options added; result keys stage_temperature_offset_k / lte_flagged_species only when the option is on. Also scripts/fit_quality_panel.py _plasma records both keys when present.
Offset search: 5-point grid on [0,max] plus one parabolic refine (<=6 tie evaluations; delta=0 always included so tie(delta) <= tie(0)). Tied fit: ion columns (ion>=2) rebuilt from line weights at T+delta (Saha fraction, Boltzmann, partition all at T+delta, same ne); neutral columns and the free fit unchanged.
LTE: dE_s = smallest nonzero Ei/Ek among species' table lines; flagged species skip the whole element tie term.

dE_s / ne_crit(8 kK), unfiltered table: H I 10.20 eV 1.5e17; He I 19.8 1.1e18; N I 10.33 1.6e17; N II 0.006 3e7; O I 9.15 1.1e17; Ne I 16.6 6.6e17; Mg I 2.71 2.9e15; Mg II 4.42 1.2e16; Ar I 11.55 2.2e17; Ar II 16.4 6.3e17; Ca I 1.88 9.5e14; Ca II 1.69 6.9e14; Fe I 0.052 2e10; Fe II 0.048 1.6e10. Caveat: for Fe and N II the proxy hits fine-structure levels, so Fe is never flagged. Ca I is flagged at 12 kK / log ne 15 (crit 1.2e15), so the "never flagged 15-18" test is restricted to T<=8 kK.

Tests tests/test_stage_thermometer_options.py: 5 passed. OFF bit-identical to value recorded on main tree (447015.68272867054; Mg/Ca terms equal); single-T oracle delta=0.0, tie unchanged; two-T scene (neutrals rendered at T, ions at T+1500) tie 4.64e7 -> 0.0, delta=1500 exactly; Ar I flagged at log ne 16, not 17.5; Ca never flagged (T<=8kK); flagged element skipped.
Base equality: compare cpi-s3 vs w3t-base: 36 items, 0 changed, all aggregates identical.
Unverified: all variant panel outcomes, delta distribution, full-suite count.

## Update: variant results (w3t-both NOT finished when written)
Full suite: 673 passed, 10 skipped (668 + 5 new), 13:32.

Headline means (base -> variant), compares in reports/fit-panel/w3t-offset.md and w3t-lte.md:
| metric | base | offset(2000) | lte |
|---|---|---|---|
| synthetic median_rel_err | 0.1559 | 0.1177 (2 better/2 worse) | 0.1559 |
| synthetic abs_T_rel_err | 0.0686 | 0.1382 (1 better/3 worse) | 0.0686 |
| synthetic spurious_mass | 0 | 0.00016 | 0 |
| synthetic dominant_correct | 1 | 1 | 1 |
| real r_squared | 0.4782 | 0.4887 (2 better/3 worse) | 0.4782 |
| real median_stage_disagreement | 0.4997 | 0.6848 (5/5 worse) | 0.4997 |
| fe fe_fraction | 0.5456 | 0.5233 (1 better/8 worse) | 0.5456 |
| fe n_non_fe | 1.154 | 1.154 | 1.154 |
LTE: all 36 items identical to base.

delta (offset, max 2000): synthetic binary_ca_mg 2000, silicate_4el 2000, feldspar_k_na 0, iron_rich_low_t 0, binary_argon_ambient 2000 (truth is single-T, so delta should be ~0: it rails at the bound on 3/5 -- finding; delta absorbs a wrong outer state/model error, not a real second temperature). Real: REE_01 2000, REE_44 1000, argon_noAr 0, scan9x9 2000, user_spec1 2000. Fe (26): min 0, median 0, max 2000.
LTE flagged species: synthetic none; real none; Fe 6/26 items flag only H_1 (single-stage element, so no tie term removed); 20 none.

Verdict: offset -> REJECT (synthetic abs_T_rel_err worse 3/5, fe_fraction worse 8/26, real stage disagreement worse, delta rails on single-T truth). LTE -> neutral/no-op on this panel (identical metrics): adopt only as harmless diagnostic if wanted, no accuracy evidence; REVISE the proxy (fine-structure levels make Fe/N II never flaggable; gases are the only flaggable species and have no multi-stage support). Not run: w3t-both, production reruns.

## Compare tables (coordinator-appended from reports/fit-panel/w3t-{offset,lte}.md, 2026-11-04)

### offset (stage_temperature_offset_max_k=2000) vs w3t-base

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 2 | 2 | 1 | 0 | 0.1559 | 0.1177 |
| synthetic | spurious_mass | 0 | 1 | 4 | 0 | 0 | 0.0001624 |
| synthetic | abs_T_rel_err | 1 | 3 | 1 | 0 | 0.06862 | 0.1382 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 1 | 1 |
| real | r_squared | 2 | 3 | 0 | 0 | 0.4782 | 0.4887 |
| real | median_stage_disagreement | 0 | 5 | 0 | 0 | 0.4997 | 0.6848 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.6 |
| real | ne_at_bound | 1 | 0 | 4 | 0 | 1 | 0.8 |
| fe | fe_fraction | 1 | 8 | 17 | 0 | 0.5456 | 0.5233 |
| fe | non_fe_mass | 1 | 8 | 17 | 0 | 0.4544 | 0.4767 |
| fe | n_non_fe | 0 | 0 | 26 | 0 | 1.154 | 1.154 |
| fe | abs_T_dev_10kK | 5 | 4 | 17 | 0 | 1.276 | 1.229 |

## Verdict

- **synthetic**: regressed (3 better vs 6 worse metric-comparisons)
- **real**: regressed (3 better vs 8 worse metric-comparisons)
- **fe**: regressed (7 better vs 20 worse metric-comparisons)

### lte (lte_gap_flag=true) vs w3t-base

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 0 | 0 | 5 | 0 | 0.1559 | 0.1559 |
| synthetic | spurious_mass | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 0 | 0 | 5 | 0 | 0.06862 | 0.06862 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 1 | 1 |
| real | r_squared | 0 | 0 | 5 | 0 | 0.4782 | 0.4782 |
| real | median_stage_disagreement | 0 | 0 | 5 | 0 | 0.4997 | 0.4997 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.8 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 0 | 0 | 26 | 0 | 0.5456 | 0.5456 |
| fe | non_fe_mass | 0 | 0 | 26 | 0 | 0.4544 | 0.4544 |
| fe | n_non_fe | 0 | 0 | 26 | 0 | 1.154 | 1.154 |
| fe | abs_T_dev_10kK | 0 | 0 | 26 | 0 | 1.276 | 1.276 |

## Verdict

- **synthetic**: no net change (0 better vs 0 worse metric-comparisons)
- **real**: no net change (0 better vs 0 worse metric-comparisons)
- **fe**: no net change (0 better vs 0 worse metric-comparisons)

### both (offset 2000 K + lte flag) vs w3t-base — coordinator, 2026-11-04 09:50

Headline means identical to the offset-only variant (syn 0.1177 / |T err| 0.1382 / real r² 0.4887 / stage dis. 0.6849 /
Fe 0.5233 / n_non_fe 1.154); the LTE flag again removes no tie term. File: reports/fit-panel/w3t-both.{json,md}.
Verdict unchanged: offset REJECT, LTE flag NO-OP (revise the ΔE proxy before any re-test).
Item-level: both equals offset on 35/36 items; real/user_spec1 differs at the 4th decimal (T 7182→7184 K, r² 0.91978→0.91978,
stage dis. 0.5911→0.5912) with no species flagged at the final state — a transient flag at some low-nₑ trial nudged the
search path. Negligible; consistent with the no-op verdict.
