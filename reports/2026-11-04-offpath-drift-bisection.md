# Default-path drift bisection (2026-11-04)

Outcome: the drift is caused by TWO unguarded default-path changes, both in alibz/peaky_indexer_v3.py. None of the opt-in groups (overlap_mode, bo_settings, bic_effective_n, detection_fdr, sparsity/robust_loss, partition_source/ipd, generator plumbing, detections ladder) changes the default numbers on these items.

Method: no production file was touched (`git diff | md5` = 229c14118b74bed279f19741af656325 before and after). Instead each variant is a COPY of the current tree (alibz/, bench/, scripts/ copied, db/data symlinked) under the scratchpad v_<name>/, with one group reverted; panel run locally (`.venv/bin/python -m scripts.fit_quality_panel run --items binary_ca_mg,silicate_4el --workers 2`, defaults n_calls 8, draws 2, seed 11). Scripts: scratchpad mk.sh, run.sh, cmp.py.

Results (median_rel_err / T_rec / log_ne_rec / r_squared):
- ref (current tree): ca_mg 0.14737 / 8925.4 / 16.5 / 0.63061 ; silicate 0.38016 / 10040.4 / 17.0844 / 0.64115
- syn (HEAD synthetic.py + bench/synth_cases.py): identical to ref to all digits
- det (HEAD detections.py): identical to ref
- head (ALL changed alibz/bench/scripts files at HEAD): ca_mg 0.17007 / 8908.1 / 16.5 / 0.62720 ; silicate 0.52619 / 9042.9 / 16.5 / 0.59581  (= the old local baseline-20261103.json 0.1701/0.5262, bit-for-bit)
- noinner (ref, INNER_STAGE_COUPLING_WEIGHT 0.15 -> 0): ca_mg 0.10580 ; silicate 0.37637 / 10000 / 17.0 (differs from ref)
- nostark (ref, run() default stark_width_weight 0.05 -> 0.0): ca_mg 0.14377 ; silicate 0.51136 / 8993.6 / 16.5 (differs from ref)
- both reverted: bit-identical to head (0.17007/0.52619, T 8908.1/9042.9). So reverting just these two reproduces the pre-change numbers exactly; every other group is bit-neutral (default OFF) on these items.

Mechanisms:
1. peaky_indexer_v3.py:152 `INNER_STAGE_COUPLING_WEIGHT = 0.15` plus the block at ~2113-2155 (in `_solve_concentrations_thin`): adds soft stage-tie rows to A_aug whenever an element has >=2 active stages (lam_inner>0 by default, scaled by stage_consistency_weight). Changes the NNLS system, hence concentrations and the T/ne basin. Not an option; no flag.
2. peaky_indexer_v3.py:3267 `PeakyIndexerV3.run(stark_width_weight=0.0 -> 0.05)`. pipeline.py does not pass it, so the Stark-width ne coupling (cost at ~2515-2542) is now ON by default. This is what flips silicate_4el between log_ne 16.5 (off) and 17.08 (on).
Neither is floating-point reordering, RNG, or a tolerance: they are deliberate model/default changes (the stage-coupling comment says it belongs to the stage-consistency work; stark weight default is a threshold that moved). They are not part of the A/B workers' option groups, so the "bit-identical when OFF" checks never covered them.

Synthetic tolerance (synthetic.py rtol 1e-8 -> 1e-6) and render_case plumbing: render_case sha256 of x and y compared, current vs HEAD synthetic.py+synth_cases.py, seed 11, defaults: binary_ca_mg, silicate_4el, feldspar_k_na, binary_argon_ambient are bit-identical. iron_rich_low_t cannot render at HEAD (ValueError: stage abundances sum 0.99999998884), so only the tolerance affects it (it newly renders; no baseline comparison possible).

Caveats / unverified:
- Local (macOS, workers 2) numbers differ from the moissanite remote numbers on the same tree (remote baseline2 0.092/0.3254; local ref 0.1474/0.3802): platform BLAS/skopt differences. The local series baseline(17:05/17:21) -> ref reproduces the same direction (0.5262->0.3802 on silicate), and the remote series 1->2 shows the same direction (0.5025->0.3254), so the attribution is consistent but I did not run the two reverts on moissanite.
- Only 2 of 36 items checked; the real/Fe items were not rerun, though both changes are in the shared indexer path and would act on them.
- Quick inspection only of refinement._bic (rho=0 path is arithmetically identical to HEAD); not separately run, but covered by the 'both == head' identity.

Note on tree integrity: I modified no tracked file. The `git diff` md5 changed during the session (229c14... -> 57e88f...) because another concurrent agent edited alibz/pipeline.py after my variant copies were taken (variants were built from the pipeline.py as of the start; the later pipeline.py edit is not reflected in any run above).
That later edit flips AnalysisConfig defaults (pipeline.py:265-272: partition_source "lines"->"barklem2016", ipd None->"debye", overlap_mode "legacy"->"conserve"); it post-dates every run here and would itself move default numbers.
