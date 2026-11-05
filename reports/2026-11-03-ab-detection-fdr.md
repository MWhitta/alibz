# Detection FDR (BH/BY) -- status 2026-11-03: IMPLEMENTED + UNIT-TESTED; PANEL NOT FINISHED

Options (default OFF): `detection_fdr: Optional[float]=None`, `detection_fdr_method: str="bh"|"by"`,
`detection_z_floor=2.0` (indexer ctor only).

Changes (grep for `detection_fdr` / `fdr_select` for exact lines):
- alibz/peaky_indexer_v3.py: `DETECTION_Z_FLOOR`, `fdr_select()` (~l.54); ctor args (~l.597) + validation;
  `_aggregate_elements`: hard z>=3 gate skipped when FDR on, elements dropped after the loop (per-element estimates are independent, so deferral is exact);
  `convergence_info` gains detection_fdr_q / _method / _threshold_z / detection_m / detection_z_floor / detection_q_value ONLY when FDR on.
- alibz/pipeline.py: AnalysisConfig fields (l.253-254), analyze_spectrum params (l.546), passed to the 3 indexer sites + idx_kwargs + run-level AnalysisConfig/cfg sites; `write_detections_csv` appends `q_value` column only if any detection carries it (default CSV byte-identical).
- alibz/detections.py classify_detections: with FDR, non-significant -> marginal (indexer z>=floor) or weak; significant keeps detected/single-line ladder; record has `q_value`.
- tests/test_detection_fdr.py: 7 tests, all pass (BH vs noise, BY<=BH, floor, OFF==hard gate, ON convergence_info, CSV q_value, status ladder).

Deviation: spec'd vector (3 true + 30 candidates, noise 2.5-3.2) cannot be separated by BH at q=0.05: step-up threshold at rank 3-6 of m=30 is z~2.3-2.6, so BH legitimately admits them. Test uses m=303 (threshold ~z 3.3).

Suite: an early `-x` full run (started before most edits) = all passed, 0 failures (no baseline failing set). Second full run with changes was at ~75% with no failures when I was forced to hand back; final count not recorded.

Panel: `run --variant fdr05` was at 3/36 items, fdr05-by not started (chained after). No compare tables exist. Re-run compare once reports/fit-panel/fdr05-*.json and fdr05-by-*.json appear.
