# A/B: tabulated partition functions (Barklem 2016) and Debye IPD

Date 2026-11-03. No commits. Both options default OFF (`partition_source="lines"`, `ipd=None`).

## Changes (file:line)
- `alibz/utils/sahaboltzmann.py`: imports/constants 1-33; `_roman` 23; `debye_length_m` 35; `debye_ipd_eV` 41; `__init__` options + validation + `partition_extrapolated` 89-104; `reset_partition_flags` 205; `_barklem_table` 208; `_barklem_partition` (ln U linear in T; >1e4 K log-linear extrapolation from last two nodes sets `partition_extrapolated`; absent species -> line sum, one WARNING per run, rest DEBUG) 219-247; table substitution inside `partition()` 302-306; Debye IPD inside `ionization_distribution` Saha exponent only 364-371 (`Ei -= ion_label * e^2/(4 pi eps0 lambda_D)`, label 1 = neutral so z+1 = label; no partition-sum truncation).
- `alibz/pipeline.py`: `AnalysisConfig.partition_source/ipd` 265-267; `_get_sb(dbpath, partition_source, ipd)` keyed per option pair (defaults keep the old key) 429-440; `analyze_spectrum` signature 557-558, `sb = _get_sb(...)` + `reset_partition_flags()` 810-811, result dict keys `partition_source`, `ipd`, `partition_extrapolated` 1253-1254; config -> analyze_spectrum 1510; `analyze_directory` signature 1569-1570 and cfg build 1628. NOTE: pipeline.py and peaky_indexer_v3.py carry concurrent uncommitted edits by another worker (sparsity/robust_loss); I touched only the lines above and did not edit peaky_indexer_v3.py (the indexer receives the configured `sb` via the existing injection). CLI flags not added.
- `scripts/build_partition_barklem2016.py` (new): TSV -> npz builder. `db/partition_barklem2016.npz` (new, 277 species x 42 T). `db/atomic_line_sources.json`: new key `partition_barklem2016`. `.gitignore` exception line 7; `pyproject.toml` data-files entry after `observed_lines_nist.tsv`.
- `tests/test_partition_ipd.py` (new).

## Provenance of the table (full table WAS downloaded)
- CDS ftp mirrors `cdsarc.cds.unistra.fr` (ftp, viz-bin, ReadMe) and `cdsarc.u-strasbg.fr`: timed out at 60 s. `vizier.cfa.harvard.edu/ftp/cats/...`: 404. Worked: VizieR-CfA ASU service.
- URL: `http://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/A%2bA/588/A96/table8&-out.max=unlimited&-out.all=2&-out.meta=hud`, retrieved 2026-11-03, TSV sha256 `5391d67c8599a0a4f5ae3d406e75750b32837d28ed5bda8143495c03d72d2fef`; npz sha256 `4c48381651459dbbba60dd837cb4b065cb25295c8f377045338d67543e7ccd4f`. T grid (42 pts, 1e-5..1e4 K) read from the ASU column metadata. 284 table rows: 277 neutral/ion species kept, 7 negative ions skipped. Raw TSV kept in session scratchpad only (not in repo); URL reproduces it.
- The paper's printed Table 8 sample has only H/D, so test node values (Fe I 24.613 @4000 K; Fe II 61.4794 @9000; Ca I 5.69578 @1e4; Ca II 2.07348 @4000; Mg I 1.37994 @9000; Mg II 2.03571 @1e4) were copied from the downloaded table, not from the paper text.

## Tests
- New: `tests/test_partition_ipd.py`, 16 tests (6 table nodes to 1 %, interpolation, continuity at 10 kK + extrapolation flag + reset, species fallback + once-per-run warning, lambda_D = 19.5 nm / dchi = 0.074 eV (2 %), Saha-ratio factor exp(dchi/kT)=1.11 (2 %) and (z+1) scaling, IPD leaves partition untouched, bad options rejected, defaults-OFF bit-identical x3 elements x3 T: `np.array_equal` on partition, ion fractions and `line_emissivity`, against an independent re-derivation of the HEAD formulas).
- Extra identity check: golden arrays (partition, ion fractions, line_emissivity for Fe/Ca/Mg at 6/8/12 kK) from a pre-edit copy of the tree vs. post-edit tree: 15/15 arrays `array_equal` True.
- Full suite `ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest -q`: BEFORE (pre-edit tree copy, includes other worker's WIP): 620 passed, 9 skipped, 0 failed (890 s). AFTER: 636 passed, 9 skipped, 0 failed (800 s). Failing set before: empty; added: none. (pymatgen-enabled invocation not run separately; the change does not touch pymatgen paths.)

## Panel (n_calls 8, draws 2, seed 11, --workers 2; baseline = reports/fit-panel/baseline-20261103.json, default Fe offset 18)
Files: reports/fit-panel/{partition,ipd,partition-ipd}-20261103.json, compare-<variant>.md.
Caveats: baseline's synthetic iron_rich_low_t errored (abundances sum != 1) so baseline synthetic means use 4 items, while ipd and partition-ipd have 5 (not paired; partition has 4). The synthetic generator (`alibz/synthetic.py:693`) builds truth with the LINE-SUM partition, so the barklem2016 inverse model is inconsistent with synthetic truth by construction; synthetic T/ne comparisons for `partition` cannot show benefit. n=4-5 synthetic items, 5 real: differences are small-sample. The extrapolation flag is in the analysis dict but not recorded by the panel JSON.

| variant | syn abs T_rel_err (mean) | syn median_rel_err | real median_stage_disagreement | real r^2 | Fe fe_fraction (mean) | Fe abs_T_dev_10kK (mean) |
|---|---|---|---|---|---|---|
| baseline | 0.0830 (n4) | 0.2729 | 0.4827 | 0.4695 | 0.4189 | 1.307 |
| partition | 0.1098 (n4) | 0.2659 | 0.4788 | 0.4689 | 0.5438 | 1.277 |
| ipd | 0.0784 (n5) | 0.1988 | 0.5351 | 0.4706 | 0.4523 | 1.308 |
| partition-ipd | 0.0933 (n5) | 0.2383 | 0.5163 | 0.4727 | 0.5467 | 1.230 |

Panel verdict lines (from `compare`): partition: synthetic no net change, real no net change, fe improved (43 vs 35). ipd: synthetic no net change, real improved (6 vs 4), fe improved (39 vs 20). partition-ipd: synthetic regressed (4 vs 6), real no net change, fe improved (46 vs 39). Fe-panel fe_stage_disagreement is present for only 5-8 items per run (0.985-0.988 mean, unchanged); Fe T (T_rec) mean 23.1 kK baseline vs 22.8 / 23.1 / 22.3 kK, i.e. the Fe panel T is not constrained by either option.

One-line verdicts
- partition: tiny real-stage-disagreement gain (0.4827 -> 0.4788), synthetic T error worse (0.083 -> 0.110, but synthetic truth uses the line-sum partition), Fe composition changes are large and mixed (fe_fraction up on many runs): inconclusive, no evidence to enable.
- ipd: most favourable: synthetic median_rel_err 0.273 -> 0.199 (all 4 paired items better), T error not worse in aggregate (0.078, but item sets differ; 4/4 paired items have larger abs T_rel_err), real r^2 up on 4/5 but real stage_disagreement worse on 3/5 (0.483 -> 0.535); worth a larger/paired run, still default OFF.
- partition-ipd: Fe T deviation from 10 kK improved on 17/26 runs (1.307 -> 1.230) but synthetic regressed per compare verdict and real stage_disagreement worse (0.516): no clear win.

## Per-item rows changed by >10 % (selected metrics)

### partition: items with any metric changed by >10% (relative; abs >0.02 for near-zero)
| item | metric | base | variant | change |
|---|---|---|---|---|
| synthetic/binary_ca_mg | max_rel_err | 0.2041 | 0.1606 | -21% |
| synthetic/binary_ca_mg | median_rel_err | 0.1701 | 0.1338 | -21% |
| synthetic/feldspar_k_na | T_rel_err | 0.02691 | 0.1311 | +387% |
| synthetic/feldspar_k_na | dominant_correct | 1 | 0 | -100% |
| synthetic/feldspar_k_na | max_rel_err | 0.6121 | 0.9383 | +53% |
| synthetic/feldspar_k_na | median_rel_err | 0.1781 | 0.2197 | +23% |
| synthetic/feldspar_k_na | r_squared | 0.8493 | 0.6749 | -21% |
| synthetic/binary_argon_a | dominant_correct | 0 | 1 | +nan% |
| fe/run-e825d9f567 | fe_fraction | 0.4138 | 0.9343 | +126% |
| fe/run-126d0c96e1 | fe_fraction | 0.2624 | 0.9117 | +247% |
| fe/run-6e38b11021 | fe_fraction | 0.07643 | 0.09652 | +26% |
| fe/run-b95ffbb4f6 | fe_fraction | 0.2858 | 0.8406 | +194% |
| fe/run-674f4feb6d | fe_fraction | 0.0126 | 0.2499 | +1883% |
| fe/run-56a8de4a62 | abs_T_dev_10kK | 1.5 | 1.289 | -14% |
| fe/run-8264fabf6e | r_squared | 0.2544 | 0.1743 | -31% |
| fe/run-b17cc0430a | r_squared | 0.3754 | 0.2728 | -27% |
| fe/run-d99401be15 | fe_fraction | 0.3222 | 0.08082 | -75% |
| fe/run-d99401be15 | r_squared | 0.4228 | 0.38 | -10% |
| fe/run-bb777df5e6 | abs_T_dev_10kK | 1.421 | 1.255 | -12% |
| fe/run-07c25af6dd | fe_fraction | 0.3409 | 0.9315 | +173% |
| fe/run-6cd149cca8 | fe_fraction | 0.02626 | 0.341 | +1198% |
| fe/run-6cd149cca8 | r_squared | 0.1881 | 0.2308 | +23% |
| fe/run-2e03f73db1 | r_squared | 0.3384 | 0.2424 | -28% |
| fe/run-468a150cbd | abs_T_dev_10kK | 1.278 | 1.121 | -12% |
| fe/run-fd1e6d48a1 | abs_T_dev_10kK | 1.309 | 1.171 | -11% |
| fe/run-fd1e6d48a1 | fe_fraction | 0.4001 | 0.6208 | +55% |
| fe/run-fd1e6d48a1 | r_squared | 0.2617 | 0.3067 | +17% |
| fe/run-01105eaef0 | fe_fraction | 0.8398 | 0.9802 | +17% |
| fe/run-90143c6b30 | fe_fraction | 0.7933 | 0.977 | +23% |
| fe/run-74c9332760 | abs_T_dev_10kK | 1.437 | 1.2 | -16% |
| fe/run-74c9332760 | fe_fraction | 0.5631 | 0.9273 | +65% |
(31 rows)

### ipd: items with any metric changed by >10% (relative; abs >0.02 for near-zero)
| item | metric | base | variant | change |
|---|---|---|---|---|
| synthetic/binary_ca_mg | max_rel_err | 0.2041 | 0.1769 | -13% |
| synthetic/binary_ca_mg | median_rel_err | 0.1701 | 0.1479 | -13% |
| synthetic/feldspar_k_na | max_rel_err | 0.6121 | 0.2308 | -62% |
| synthetic/feldspar_k_na | median_rel_err | 0.1781 | 0.1512 | -15% |
| synthetic/iron_rich_low_ | status | error | ok | |
| synthetic/binary_argon_a | dominant_correct | 0 | 1 | +nan% |
| synthetic/binary_argon_a | max_rel_err | 0.2607 | 0.1115 | -57% |
| synthetic/binary_argon_a | median_rel_err | 0.2173 | 0.09291 | -57% |
| real/REE_44 | median_stage_disagreement | 0.4279 | 0.5658 | +32% |
| real/user_spec1 | median_stage_disagreement | 0.5082 | 0.6194 | +22% |
| fe/run-6e38b11021 | fe_fraction | 0.07643 | 0.04914 | -36% |
| fe/run-b95ffbb4f6 | fe_fraction | 0.2858 | 0.332 | +16% |
| fe/run-674f4feb6d | fe_fraction | 0.0126 | 0.09794 | +677% |
| fe/run-674f4feb6d | r_squared | 0.3751 | 0.3269 | -13% |
| fe/run-b17cc0430a | fe_fraction | 1 | 0.7149 | -29% |
| fe/run-b17cc0430a | r_squared | 0.3754 | 0.2734 | -27% |
| fe/run-07c25af6dd | abs_T_dev_10kK | 1.23 | 1.391 | +13% |
| fe/run-07c25af6dd | fe_fraction | 0.3409 | 0.4484 | +32% |
| fe/run-2e03f73db1 | fe_fraction | 0.06125 | 1 | +1533% |
| fe/run-2e03f73db1 | r_squared | 0.3384 | 0.2387 | -29% |
| fe/run-fd1e6d48a1 | abs_T_dev_10kK | 1.309 | 1.171 | -11% |
| fe/run-fd1e6d48a1 | fe_fraction | 0.4001 | 0.2959 | -26% |
| fe/run-fd1e6d48a1 | r_squared | 0.2617 | 0.302 | +15% |
(22 rows)

### partition-ipd: items with any metric changed by >10% (relative; abs >0.02 for near-zero)
| item | metric | base | variant | change |
|---|---|---|---|---|
| synthetic/binary_ca_mg | max_rel_err | 0.2041 | 0.1689 | -17% |
| synthetic/binary_ca_mg | median_rel_err | 0.1701 | 0.1407 | -17% |
| synthetic/silicate_4el | max_rel_err | 0.636 | 0.7369 | +16% |
| synthetic/feldspar_k_na | T_rel_err | 0.02691 | 0.09881 | +267% |
| synthetic/feldspar_k_na | dominant_correct | 1 | 0 | -100% |
| synthetic/feldspar_k_na | max_rel_err | 0.6121 | 1.034 | +69% |
| synthetic/feldspar_k_na | median_rel_err | 0.1781 | 0.3464 | +94% |
| synthetic/feldspar_k_na | r_squared | 0.8493 | 0.7159 | -16% |
| synthetic/iron_rich_low_ | status | error | ok | |
| synthetic/binary_argon_a | dominant_correct | 0 | 1 | +nan% |
| synthetic/binary_argon_a | max_rel_err | 0.2607 | 0.1286 | -51% |
| synthetic/binary_argon_a | median_rel_err | 0.2173 | 0.1072 | -51% |
| real/argon_noAr | median_stage_disagreement | 0.3714 | 0.4358 | +17% |
| real/user_spec1 | median_stage_disagreement | 0.5082 | 0.6202 | +22% |
| fe/run-e825d9f567 | fe_fraction | 0.4138 | 0.9344 | +126% |
| fe/run-126d0c96e1 | fe_fraction | 0.2624 | 0.9208 | +251% |
| fe/run-5291e64308 | fe_fraction | 0.3022 | 0.09088 | -70% |
| fe/run-6e38b11021 | abs_T_dev_10kK | 1.391 | 1.192 | -14% |
| fe/run-6e38b11021 | fe_fraction | 0.07643 | 0.04137 | -46% |
| fe/run-b95ffbb4f6 | fe_fraction | 0.2858 | 0.8508 | +198% |
| fe/run-674f4feb6d | fe_fraction | 0.0126 | 0.1523 | +1108% |
| fe/run-56a8de4a62 | abs_T_dev_10kK | 1.5 | 1.268 | -15% |
| fe/run-da01e553ae | T_rec | 2.03e+04 | 9524 | -53% |
| fe/run-da01e553ae | abs_T_dev_10kK | 1.03 | 0.0476 | -95% |
| fe/run-8264fabf6e | r_squared | 0.2544 | 0.1757 | -31% |
| fe/run-b17cc0430a | r_squared | 0.3754 | 0.2735 | -27% |
| fe/run-d99401be15 | fe_fraction | 0.3222 | 0.9154 | +184% |
| fe/run-bb777df5e6 | abs_T_dev_10kK | 1.421 | 1.243 | -13% |
| fe/run-07c25af6dd | fe_fraction | 0.3409 | 0.904 | +165% |
| fe/run-07c25af6dd | r_squared | 0.2953 | 0.2631 | -11% |
| fe/run-6cd149cca8 | fe_fraction | 0.02626 | 0.2282 | +769% |
| fe/run-2e03f73db1 | r_squared | 0.3384 | 0.2592 | -23% |
| fe/run-6ce8e9b394 | fe_fraction | 0.04406 | 0 | -100% |
| fe/run-6ce8e9b394 | r_squared | 0.1369 | -1.159 | -946% |
| fe/run-468a150cbd | abs_T_dev_10kK | 1.278 | 1.105 | -14% |
| fe/run-fd1e6d48a1 | abs_T_dev_10kK | 1.309 | 1.171 | -11% |
| fe/run-fd1e6d48a1 | fe_fraction | 0.4001 | 0.6144 | +54% |
| fe/run-fd1e6d48a1 | r_squared | 0.2617 | 0.3012 | +15% |
| fe/run-01105eaef0 | fe_fraction | 0.8398 | 0.9849 | +17% |
| fe/run-01105eaef0 | r_squared | 0.1551 | 0.07974 | -49% |
| fe/run-90143c6b30 | fe_fraction | 0.7933 | 0.9754 | +23% |
| fe/run-74c9332760 | abs_T_dev_10kK | 1.437 | 1.214 | -16% |
| fe/run-74c9332760 | fe_fraction | 0.5631 | 0.6415 | +14% |
(42 rows)

## Not verified
- pymatgen-enabled test invocation not separately run. Not checked: CLI flags, fit_inspector live-script plumbing (not added), other consumers of `SahaBoltzmann` (peaky_maker, synthetic) are unchanged and use the default options. Barklem species with label >= 40 or absent in table fall back silently (DEBUG). Panel runs were done under heavy machine load (other workers); runtime numbers are not comparable.

## Follow-up: consistent (non-inverse-crime) synthetic path
- `alibz/synthetic.py`: `SyntheticSpectrumGenerator(..., partition_source="lines")` (ctor) and `_stage_partition` uses the Barklem table via a `SahaBoltzmann` when set (falls back to line sum if a stage is absent). `bench/synth_cases.py`: `make_scene(..., partition_source, ipd)` builds the Saha stage split with `SahaBoltzmann(partition_source, ipd)`; `render_case` builds the generator with `partition_source`; both recorded in `scene.metadata` (`partition_source`, `ipd`). No peaky_maker change (suite S does not use it).
- `scripts/fit_quality_panel.py`: `run --render-config-json` (default: the `partition_source`/`ipd` keys of `--config-json`, `RENDER_KEYS`); stored as `render_config` in the output JSON and passed to synthetic items only (real/Fe never see it); `partition_extrapolated` now recorded in every item's metrics via `_plasma`. Note: this changes `ipd`/`partition` runs from now on (synthetic is rendered with the option); use `--render-config-json '{}'` to keep the old inversion-only behaviour. Panel NOT run.
- Tests added (tests/test_partition_ipd.py): render defaults OFF identical to explicit OFF and metadata; tables render differs; slow round trip (opt-in `ALIBZ_RUN_SLOW_ROUNDTRIP=1`, ~4 min). Render-OFF output bit-identical to HEAD verified against a HEAD copy of bench/ + synthetic (binary_ca_mg seed 3 intensity arrays equal).
- Result of the round trip (run once): FAILS the "at least as well" criterion: median_rel_err tables/tables 0.216 vs lines/lines 0.084 (binary_ca_mg, seed 11, n_calls 8, draws 2). Marked xfail(strict=False) with that reason; cause not isolated (candidates: indexer paths not using `sb` partition, T/ne trade-off with larger tabulated U for Ca I near 10 kK, single seed). Not evidence for enabling `partition_source`.
