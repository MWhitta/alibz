# Literature ingest: ML / statistics / numerical methods (2026-11-03)

## Coverage
55 keys in manuscript/bib/ml.bib. 53 notes written to manuscript/lit/notes/<key>.md (<=250 words each).
No fulltext: huang2023mlsoil, keys1981cubic.
Unusable text: yu2023cnn (journal front matter; stub note, relevance 0).
Truncated to abstract/first page (notes say so; no numbers invented): anderson2022supercam, noel2025alias, li2021annreview, li2020cnn, castorena2021deepcnn, vandeneynde2023dlregression, lepore2024calsharing, dyar2014chemcam, forni2013ica, nocedal2006numopt (front matter only).
Page/section pointers for classic methods partly come from background knowledge of the papers, not verified in the extracted text (e.g. Huber k=1.345, Chernoff boundary mixture, Kass-Raftery 6, EGO 10k design). Treat as to-verify.
Nothing was run; no code or tests touched.

## Ranked top-10
| # | Idea | Source keys | Module | Metric + direction | Cost | Risk |
|---|------|-------------|--------|--------------------|------|------|
| 1 | Elastic-net NN-LASSO (augmented rows + linear L1 term, adaptive weights, lambda by dBIC); extends queued NN-LASSO | tibshirani1996lasso, zou2005elasticnet, dyar2012lasso, boucher2015ml | peaky_indexer_v3 WNNLS | S spurious mass down, majors missed flat; R spurious-element count down | M | trace shrinkage |
| 2 | BH/BY FDR over element detection z; one-sided null (z>1.645) and empirical null from blanks | benjamini1995fdr, wilks1938lrt | detections.py | R spurious-element count down at equal recall | S | dependence via coincident lines |
| 3 | Huber IRLS on sigma_eff-whitened rows before Tukey | huber1964robust | peaky_indexer_v3 solve | S spurious mass, majors error down; R r^2 | S | double downweighting with sigma_eff |
| 4 | Effective n (pixel correlation rho=0.92) in BIC | schwarz1978bic | _select_basin_bic, refinement._bic | S false-blend rate, basin_ambiguous accuracy | S | thresholds retune |
| 5 | Parametric bootstrap CIs at fixed (T,ne) using cheap WNNLS re-solve (queued) | efron1979bootstrap | peaky_indexer_v3 | S CI coverage 0.68/0.95 | M | conditional on outer fit |
| 6 | BO settings: Matern-5/2, normalised dims, log-cost, EI early stop | jones1998ego, shahriari2016bayesopt | gp_minimize call l.2443 | S T and log ne error, runtime down 20-40 % | S | early stop locks wrong basin |
| 7 | LM/TRF polish on reduced (VARPRO) functional after GP, gated by dBIC | golub1973variable, golub2003variable, coleman1996trf | peaky_indexer_v3 | S T/log ne error down; ne-at-bound not up | M | kinks at support changes |
| 8 | Domain randomisation of registration/gain/LSF/baseline nuisances in synthetic.py for transformer_plan, plus MAE pretraining on 4909 real CSVs | tobin2017domainrand, he2022mae, pan2010transfer, kepes2022transfer | synthetic.py, transformer_plan | Z line-matching fraction up, Fe T consistency | L | negative transfer |
| 9 | Spike-and-slab-style presence probability via dBIC odds with element priors (the queued Bayesian presence model) | mitchell1988spikeslab, george1993ssvs | detections.py | R/S Brier and AUC of presence | M | prior sensitivity |
| 10 | fit_all speed: jac_sparsity + tr_solver lsmr + x_scale='jac'; Gram-form NNLS for fixed-dictionary re-solves | branch1999trf, more1978lm, bro1997nnls | peaky_finder.fit_all, solve_at | runtime 2-10x down, residuals equal | S | none for accuracy |

## Reported ML-for-LIBS accuracies
| Paper | Task | Data | Metric | Number |
|---|---|---|---|---|
| kepes2022transfer | oxide regression, CNN + transfer | SuperCam soil | RMSEP change | TiO2 -16 %, CaO -11 %, K2O -22 %; SiO2, Na2O 0; others worse; after outlier filter up to -28 % (CaO) |
| vandeneynde2023dlregression | Al alloy regression | 27 refs, 733 scrap (XRF truth) | RMSE | 0.02 wt% Al, Si; <=0.01 wt% Fe, Cu, Mn, Mg, Zn |
| li2020cnn | multi-component CNN | 1400+ spectra, 23 standards | RMSE | CNN best vs BPNN, PLS1/2 (no number in text) |
| boucher2015ml | 9 regression models | 100 rocks | PRESS | sparse best for MgO, SiO2, Fe2O3, CaO, MnO; worse Na2O, K2O, TiO2, P2O5 |
| dyar2012lasso | PLS vs lasso | 100 rocks, 6144 channels | accuracy | comparable; lasso <24 nonzero channels per element |
| yaroshchyk2012pls | PCR/PLS Fe in ore | Fe 12.6-67.9 % | RMSEP | 2.2-4.5 wt%; R2 val 0.87-0.97 |
| sirven2006ann | NN Cr in soil | 3 soils | accuracy/precision | about 5 % with line-area inputs |
| kepes2020benchmark | soil classification contest | 138 samples, 12 classes | accuracy | about 90 % best |
| hao2024mlreview / dehbozorgi2025mldl | classification (reviews) | various | accuracy | 90.4-100 %, mostly shot-level splits |
| zhang2023dlreview | rock classification, 3 classes | review | accuracy | 64 to 85 % with DL |
| dyar2014chemcam | ChemCam | MSL | accuracy | +/-10-20 % major/minor |

Failure modes recurring: shot-level leakage, small sample counts (23-138), matrix specificity, instrument change needing calibration transfer, negative transfer and poorer reproducibility.

## Not worth testing
- Differential evolution outer search (storn1997de): 5-10x evaluations of gp_minimize.
- airPLS replacing arPLS (zhang2010airpls): zeroes positive residuals, over-subtracts in noise.
- Humlicek w4 (humlicek1982voigt): alibz uses scipy voigt_profile at 1e-13; only gpu.py could consider it.
- PINN (raissi2019pinn): forward model is already analytic; only a loss term in transformer_plan.
- NMF (lee1999nmf): non-unique, absorbs real elements; only as residual explorer.
- PLS/ICA/ANN calibration (wold2001pls, forni2013ica, sirven2006ann): empirical, matrix-bound, no gain over physics solve.
- Full-text Adam, attention (kingma2015adam, vaswani2017attention): training detail.
- Dense-pixel full SSVS Gibbs per spectrum: too slow; offline prior calibration only.
