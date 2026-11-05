# W4 — manuscript figures from real spectra

2026-11-04. Branch `fix/z300-pixel-offset`. Specialist. Produces five
publication figures (F1–F5) from the production `analyze_spectrum` chain and
wires them into the LaTeX manuscript. Nothing committed; `pipeline.py` and
`peaky_indexer_v3.py` not edited.

## Commands

    # analyses (6 keys, spawn pool of 6; OMP/OPENBLAS/MKL=1 set in-script)
    .venv/bin/python scripts/manuscript_figures.py analyze --workers 6
    .venv/bin/python scripts/manuscript_figures.py resummarize   # rewrite JSON from pkl
    .venv/bin/python scripts/manuscript_figures.py fig all        # render from cache
    cd manuscript && make && make check

`scripts/manuscript_figures.py` (new): `analyze` fills
`manuscript/figures/cache/<key>.pkl` + `.json`; `fig F1..F5|all` renders
`manuscript/figures/F<n>-<slug>.{pdf,png}` at dpi 200, width 6.5 in (`figure*`
= full textwidth in this one-column `article`), font ≥ 7 pt, `tab10`/element
colours, deterministic (`svg.hashsalt`, fixed seeds, no timestamps).
`manuscript/figures/cache/` added to `.gitignore`.

Every analysis used `analyze_spectrum(x, y, "db", capture_stages=True)` with
adopted defaults (n_calls 40, grid, gp_seed 42, overlap conserve, partition
barklem2016, ipd debye). Fe runs pass no extra kwargs (the Fe panel passes only
n_calls/draws); the 18-px native-axis correction is applied by
`showcase_data.load` → `fit_quality_panel.load_fe(..., pixel_offset=18)`.

## Per-key runtime / final state (from cache JSON)

| key | runtime | T (K) | log ne | r² | top fractions |
|---|---|---|---|---|---|
| user_spec1 | 79.5 s | 7654 | 16.31 | 0.914 | K .69, Si .13, Rb .08 |
| mw2_1000 | 135.2 s | 8587 | 16.94 | 0.624 | K .22, Si .20, Al .12 |
| jc_mdd017_24_15_2nd_1 | 47.8 s | 8614 | 16.44 | 0.788 | Li .59, Si .16, Na .09 |
| jc_alaskire | 175.6 s | 9461 | 16.84 | 0.790 | K .47, Si .19, Na .16 |
| fe_fresh_d5p25 | 61.2 s | 22601 | 17.69 | 0.346 | Fe .932, H .068 |
| fe_reused_fail | 51.3 s | 25000 | 17.01 | 0.364 | B .833, Fe .166 |

(jc_alaskire cached for completeness; it is not used by any of the five figures.)

## Figures, window choices, verification (each PNG viewed with Read)

- **F1 overview** `user_spec1` → `F1-overview.{pdf,png}`. Three shared-x panels
  (raw+bg, data−bg+model+wavelength-tinted peak rug, residual/σ with ±2σ),
  dashed UV/VIS/NIR seams at 365/620 nm. Reads correctly; residual excursions
  are at the Ca II/K I resonance lines (real self-absorption). Caption: 221
  peaks, T 7654 K, log ne 16.31, r² 0.914 — all match JSON.
- **F2 walkthrough** `mw2_1000` → `F2-walkthrough.{pdf,png}`. Windows chosen
  automatically: (1) **392.5–397.5 nm** (max lines added blind→final, Ca-rich),
  (2) **249.2–254.2 nm** (a refine_data split/merge site, line-dense UV). Three
  stage columns (blind / 3a+3b / final) × 2 windows with residual/σ strips;
  right block = stacked composition per solve (P1,P2,D3.0,D2.0,SA) + T/log ne
  per solve; n-peaks-per-stage footer. Nested gridspec; no overlaps. Caption
  303→341 peaks, T 8563→8587 K, K .22/Si .20/Al .12/C .11, r² 0.62 — match.
- **F3 refinement** `mw2_1000` → `F3-refinement.{pdf,png}`. Windows K I
  766.49 and Li I 670.78 (resonance). S (symmetric), A (`sa_voigt`), final,
  residual strip; box prints action/verdict/ΔBIC/RMS. Caption K ΔBIC(S−A)=76,
  RMS 28.3→19.4; Li ΔBIC=83, RMS 92.3→58.9 — match `f3_refinement` block.
- **F4 confounder** `jc_mdd017_24_15_2nd_1` → `F4-confounder.{pdf,png}`. Window
  278.5–286 nm; data−bg+model with per-peak dominant-species stems (Mg claims
  the Mg II doublet + Mg I 285.2; Mn claims one minor peak); as-fit vs resolved
  bars + status text. Caption Mn 0.00 (upper-limit), Mg 0.022, T 8614, r² 0.79
  — match.
- **F5 standard** `fe_fresh_d5p25` (+`fe_reused_fail`) → `F5-standard.{pdf,png}`.
  (a) Saha–Boltzmann Y = ln[A_obs·λ·Z_s(T)/(hc·gA·f_s(T,ne))] vs E_k, slope
  −1/kBT at fitted T; (b) composition bars with detection z-scores + reused
  run. Caption T 22601 K, 112 Fe II / 0 Fe I, Fe 0.93 (z=315) / H 0.07 (z=23),
  reused B 0.83 / Fe 0.17 — match.

Automated cross-check (`manuscript_figures` JSONs): 16/16 caption numbers agree.

### Route used for per-species physics (F5)
Preferred route per brief: used `result.peak_assignments` (element/ion/observed/
contribution/total_predicted) + final peak wavelengths from
`final["sorted_parameter_array"]`. Per-line gA/Ek/λ read directly from the
`Database` Fe line table (`db.lines("Fe")` with `_COL_*` indices); Z_s and f_s
from the pipeline's shared `_get_sb("db","barklem2016","debye")`; emissivity via
`line_emissivity`. Each Fe peak mapped to the strongest-gA Fe line of its
assigned ion within 0.15 nm of the registration-shifted peak centre. The full
indexer was **not** rebuilt. Expression stated in the F5 caption.

## LaTeX wiring

- `04-workflow.tex`: F1 (`fig:overview`) after the stage table; F3
  (`fig:refinement`) at the end of the refinement subsection; F2
  (`fig:walkthrough`) after the whole-pattern-solve subsection.
- `07-results.tex`: F4 (`fig:confounder`) after "Confounder resolution"; F5
  (`fig:standard`) under "Pure-element standards on the Z300".
- Each has one `Figure~\ref{...}` sentence in surrounding prose (no existing
  paragraph rewritten) and a `% src: manuscript/figures/cache/<key>.json`
  comment on the quoted-number line. All five via
  `\includegraphics[width=\textwidth]{F<n>-slug}` (PDF, graphicspath set).
- `cd manuscript && make` → exit 0, 54 pages, all five PDFs embedded
  (grep `F[1-5]-*.pdf` in main.log); `main.aux` defines fig:overview..standard
  as figures 1–5. `make check`: 0 TODOs, no `undefined`/`Warning--` lines.

## Discrepancies with existing prose (NOT edited — for the coordinator)

1. **F4 vs 07-results "Confounder resolution".** The prose describes the
   hot-Mn samples where Mn was fitted at 40–53 % and resolution moved Mn
   53 %→0 % and Mg 8 %→31 %. On the selected spectrum (`jc_mdd017_24_15_2nd_1`)
   under the current adopted defaults, Mn is already an **upper limit** (as-fit
   fraction 0.00); Mg is detected at 0.022 with Mn only flagged as a potential
   confounder, and as-fit == resolved — there is no Mn mass to reattribute. The
   dramatic contest the text quotes is the 2026-07 summary.csv outcome, not the
   current one. F4 shows and states the current outcome (brief anticipated this).
2. **F5 vs 07-results summary / memory.** The text's "roughly 7–14 kK" and the
   memory note "T ≈ 7–10 kK from Fe I VIS lines" conflict with this run's fitted
   **T = 22.6 kK** on `fe_fresh_d5p25` at the production defaults, where **all
   112 Fe-assigned peaks are Fe II and no Fe I is assigned** (so the intended
   Fe I/Fe II Saha stage-tie is not exercised) and r² = 0.35. The figure shows
   the actual state; this looks like a hot/ionized basin on the native-grid
   default fit and may warrant a judgment look (possible basin or settings
   issue), but the brief forbids changing pipeline settings, so it is reported
   only. `fe_reused_fail` fitted T railed at 25 000 K (an upper bound).
3. **F3 vs 04-workflow spectrum-0 story.** The text describes K I 766.5 as a
   phantom split that is *merged*. On spectrum 1000 the decision is a
   self-absorption **tag** (verdict asymmetric, action sa-tag), not a merge;
   F3's caption says so. (Consistent with the brief's "never force the story".)

## Unverified / notes

- Determinism across machines not re-run here (fixed seeds + Agg + hashsalt set;
  single render verified).
- F2 `tight_layout` emits a benign "Axes not compatible" warning (nested
  gridspec); output is correct.
- `make` ran the full LaTeX build; I did not independently audit non-figure
  pages.
- Cache pkls carry the full analysis dicts (FitResult + stages); figures
  regenerate from `--from-cache` without recomputation.

## Second pass (coordinator rework)

All five figures regenerated from cache (only the new F4 sample was analyzed);
`make` clean (54 pp, 0 undefined refs, make check clean); 18/18 caption numbers
re-cross-checked against the cache JSON. Every PNG viewed again.

Commands added: `scripts/manuscript_figures.py resummarize [--keys ...]`
(rewrites JSON from pkl without re-analysis); `showcase_data.py --build/--stage/
--check` re-run with the new entry (7/7 OK, six originals unchanged).

**F1** — panel-(a) legend moved to lower right (no longer collides with the NIR
label); caption now names Li\,I \SI{670.8}{} as the largest residual excursion
($\approx-7\sigma$), plus Ca\,II H&K and K\,I 766/770.

**F2** — data now drawn as a thick light-grey line, stage model in a saturated
per-column colour; pink ticks mark peaks new to each column. Windows chosen by
visible change (see scores below); per-column residual RMS printed in each strip.
Plasma panel: T y-range mean$\pm$\SI{1000}{\kelvin}, log\,$n_e$ $\pm0.5$ dex, T
values annotated, $r^2$ per solve shown. Footer replaced by a peaks-per-stage
step line (labels rotated 90°). x-labels only on the bottom row;
`constrained_layout` (no tight_layout warning). Caption now quotes the max
element-fraction change across solves = 0.019 (Al). **Window scores** (5-nm,
reject if per-column RMS spread < 5%): chosen 392.5–397.5 nm (score 5 = 3 added
>5σ + 2 split/merge, RMS spread 46.5%) and 420.0–425.0 nm (score 3 = 3 added,
spread 58.7%); both passed, no fallback.

**F3** — K window widened to 765–771 nm (both 766.49 and 769.90 in view), full
peak height (no clipping), S/A summed over both in-window sa-tag decisions. The
residual strip now plots S, A and final as three labelled lines ("data − model
[counts]"); text box upper-left, legend upper-right (no overlap). Cache + caption
updated to the window-matched RMS: K 24.9 (S) / 18.5 (A) / 24.9 (final), ΔBIC 76;
Li 92.3 / 58.9 / 88.6, ΔBIC 83. Caption states the tag keeps the symmetric
envelope (final tracks S) while A is the preferred-but-recorded shape.

**F4** — new sample selected from the 38-spectrum current-defaults survey
(`data/jchristensen_20261104/`, rc=0). Rule applied: the `confounded` spectrum
with the largest |fraction − fraction_resolved| → **MDD017-29-2c #1 Kaleb** (Mn
confounded, 0.102 → 0.000, cs 1.00). Added to `showcase.json` (key
`jc_mdd017_29_2c_1`, sha256 + stats), staged, `--check` OK, analyzed with
`capture_stages=True`. Left panel now uses the `support` table: per-element
predicted (stem) and observed (bar marker) amplitudes, areas converted to peak
height; Mn's sole support line at 279.76 nm predicts ≈1/3 of the observed
amplitude there, Mg accounts for the doublet + Mg\,I 285.2. **Under-fit check:**
the earlier 279.55 under-shoot was specific to the old sample; on this sample
the model matches the doublet (279.49: model 955 vs obs 912). Right panel: as-fit
vs resolved with cs printed per element. **Survey numbers** (in JSON
`f4_survey`): 38 spectra, Mn `confounded` on 6, Mn upper-limit on 10, 6 spectra
with any `confounded` status, median $r^2$ 0.78. `summary.csv`/`detections.csv`
copied to `reports/jchristensen-20261104/`.

**F5** — reframed as the two-thermometer discrepancy. Pipeline Fe\,II points
coloured by detector segment, y clipped to [−12, 2] (1 point at ln≈−20 dropped,
stated). The Fe\,I visible Boltzmann fit was computed on the same spectrum with
`build_thermo_set`/`measure_thermo`/`robust_boltzmann` **copied verbatim** from
`cal/scripts/fe_plasma_analysis.py` (importing runs argparse/heavy code), using
`build_thermo_set(db,1,420,620,0.8,60)` and the VIS registration shift from the
pipeline (+9.8 pm). **Outcome:** it fits (10 of 60 candidate lines, $T_B=11.5\pm
\SI{0.4}{\kilo\kelvin}$, $r^2=0.99$) but the cal `resolved` flag is **False**
because the $E_k$ span is only 1.25 eV (< 2.4 eV threshold) — a narrow lever arm,
stated in the caption. The Fe\,I points carry a display-only vertical offset
(only the slope is comparable), stated. Panel (b): consistent element colours,
fresh solid / reused hatched, z-scores. Caption contrasts pipeline 22.6 kK (all
Fe\,II, $n_e$ at the Hα bound) with Fe\,I 11.5 kK (≈ campaign median 10.8 kK).

Discrepancies with prose (unchanged, still reported): F5 thermometer tension is
now the figure's subject rather than a caveat. F4 confounder story now matches
the "Confounder resolution" prose (Mn reattributed to Mg) on the newly chosen
spectrum. F3 sa-tag vs spectrum-0 merge note retained.

Unverified: cross-machine determinism; non-figure manuscript pages; the Fe\,I
thermometer's narrow $E_k$ span means $T_B$ should be read as indicative.
