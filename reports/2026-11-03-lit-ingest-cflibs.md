# CF-LIBS / quantitative-LIBS literature ingest (2026-11-03)

Notes: `manuscript/lit/notes/<key>.md` (47 written, frontmatter `topic: cflibs`). No source code changed; tests unaffected (counts below).

## Coverage
- Bib keys in `manuscript/bib/cflibs.bib`: 48. Fulltext present for 47. **No fulltext (no note): `sweetapple2015lithium`.**
- Fulltext read in depth (targeted grep + context): hahn2010review1, takahashi2017matrix, urbinamedina2021branching, herrera2009cflibs, cremers2009capabilities, korbel2024lithium, connors2016handheld, eppler1996matrix, wiens2012chemcam, wiens2013preflight, villasboas2023lod/borduchi2022detection (abstract+intro).
- **Abstract + intro only (3-6 kB files; equations/tables absent):** aragon2014csigma, aragon2015generalizedcog, cavalcanti2013onepoint, grifoni2016cfcomparison, praher2010cflibs, aguilera2004spatial, rezaei2020selfabsorption, elsherbini2006halpha, clegg2009pls, clegg2017recalibration, anderson2017submodels, dyar2016univariate, mermet2008loq, harmon2013geochem, senesi2014earth, senesi2021handheld, gornushkin2010radiative. Notes say so; any equation not in the text is marked as background knowledge.
- **Wrong or unusable file (relevance 0, do not cite):** gaudiuso2010review (MDPI cover), gornushkin2022montecarlo (MDPI cover), harmon2023muscovite (MDPI cover), saloman2004neon (AIP user guide), miziolek2006libs (front matter), noll2012libs (TOC only), li2014spectralresponse and palleschi2022selfabsorption (byte-identical copies of korbel2024lithium, md5 d967a2cd...), harmon2019mineral (a 2017 Lithium Australia ASX announcement, not the Minerals review; usable only as a Z300 Li field datum).
- Off topic: palasti2022argon (Au nanoparticle gas breakdown).
- **Fix needed in the retrieval index:** re-retrieve the 7 mis-filed papers above; the self-absorption (Palleschi 2022) and relative-spectral-response (Li 2014) ones matter for ideas 4 and 3.
- Already in alibz (provenance only): H-alpha n_e (`utils/stark.py:80-85`, `pipeline.py:512`); queued items McWhirter, IPD, Tukey IRLS, bootstrap, partition tabulation untouched.

## Top-10 ideas (ranked by expected accuracy gain / risk)
| # | Idea | Source keys | alibz module | Metric + expected direction | Cost | Risk |
|---|------|-------------|--------------|-----------------------------|------|------|
| 1 | Oxide / stoichiometric closure: add unquantified O (default oxidation states) to the closure sum; report oxide wt% and sum-of-oxides QC. Closes assumption A10 | praher2010cflibs, takahashi2017matrix, herrera2009cflibs | `peaky_indexer_v3.py` normalisation; `synthetic.py` (emit O); `bench/synth_cases.py` | Suite S: median rel. error on majors and spurious mass DOWN on oxide cases; real: oxide sum vs 100% | M | M (Fe valence, volatiles, wrong default hides errors) |
| 2 | Stage-specific effective T (and n_e): allow bounded T_II >= T_I offset, report offset as diagnostic | aguilera2004spatial, aragon2014csigma, aragon2015generalizedcog | `peaky_indexer_v3.py` stage-consistency thermometer | Real: stage_disagreement DOWN at equal/higher r2; Fe standard Fe II/Fe I ratio scatter DOWN; suite S T error not worse | M | M (extra degeneracy with Saha ridge; synthetic is single-T so cannot show gain) |
| 3 | One-point-calibration line/segment factors from a reference (Fe standard npz or CRM glass), hold-out validated | cavalcanti2013onepoint, grifoni2016cfcomparison, takahashi2017matrix | new `utils/onepoint.py`; hook before WNNLS | Z300 Fe: Boltzmann T scatter DOWN, resolved-fit count UP, line-matching fraction UP on held-out run means; r2 UP | M | M-H (overfit, absorbs registration/620 nm errors) |
| 4 | Branching-ratio/doublet SA test with discriminance criterion gamma = SA2/SA1 (flag non-discriminating pairs); choose thin pairs with distinct tau for the planned response corpus | urbinamedina2021branching, rezaei2020selfabsorption | `pipeline.py` QC, `utils/absorption.py`, planned corpus response | Suite S SA round trips: false-pass rate of thin check DOWN; K 766/770 check unchanged | S | L |
| 5 | Model-validity exclusion gate for strongest lines (Csigma "model limit") before tau fit | aragon2015generalizedcog, aragon2014csigma | `utils/absorption.py`, `refinement.py` | Suite S SA round trips: error on majors DOWN with strong resonance lines; K doublet closer to thin | M | M (threshold choice) |
| 6 | Per-element LOD (3 sigma) and LOQ (10 sigma) from fitted T and local residual sigma, used for gate and reporting | villasboas2023lod, mermet2008loq | `detections.py`, `synthetic.py` | Suite S: detection probability ~50% at predicted LOD; spurious mass at trace levels DOWN | M | L |
| 7 | Report NRMSE = RMSE/(ymax-ymin) and error-vs-abundance plots next to median relative error | takahashi2017matrix, clegg2017recalibration | `bench/` summaries | Comparability only (no accuracy change) | S | L |
| 8 | Known-element anchor normalisation (controlled calibration) when one element is known (pure standard, stoichiometric host) | cremers2009capabilities | `peaky_indexer_v3.py` | Fe standard composition invariance; suite S majors error DOWN; literature: 2.6x more accurate than F-factor closure (one soil, vs ICP) | S | L (only when anchor is truly known) |
| 9 | Extend queued McWhirter check with relaxation criterion, line-to-continuum and ion-to-neutral consistency flags | hahn2010review1, takahashi2017matrix | `pipeline.py` QC | n_e-at-bound / QC flag rate vs stage_disagreement (diagnostic) | S | L |
| 10 | Gate-width scan on Z300 (wider gate for SNR if fitted T is invariant) | borduchi2022detection, connors2016handheld | acquisition presets (no code) | Fe standard: resolved-fit count UP, Fe I Boltzmann T unchanged | S | M (T drifts quickly at 7-12 kK) |

Notes on interaction: 1 and 8 both change closure; test separately. 3 should be fitted on half of the 26 Fe run means and scored on the rest. 2 cannot be confirmed on suite S as built; needs a two-zone synthetic case.

## Reported accuracy benchmarks (relative error on majors unless stated)
| Paper | Sample | Method | Reported error |
|-------|--------|--------|----------------|
| praher2010cflibs | pressed Fe2O3/MgO/CaO powders, air | CF-LIBS + deconvolution + SA correction + oxide closure | max <10% Fe2O3, <20% MgO, <5% CaO |
| aragon2015generalizedcog | fused glass from slags / pure compounds | Csigma-LIBS (needs a reference) | 3.2% mean relative to Fe (Ca, Mn, Mg, V, Ti, Si, Al); 9.2% mean absolute (Fe, Ca, Mn) |
| herrera2009cflibs | Al, Zn, Cu-Ni, Ti, Ni alloys, soils | classic CF-LIBS, ICCD, 220-700 nm | matrix element ~5% (Al 0.3-6%); other components 30-200% (semi-quantitative) |
| herrera2009cflibs (citing Salle) | rocks, stand-off 3 m, 9 mbar CO2 | CF-LIBS | O 4-63%, other components up to 1700%; external normalisation O <32% |
| herrera2009cflibs (citing) | Mars-condition soil standards | CF-LIBS | Si <5%, others up to 80% |
| cremers2009capabilities | siliceous soils/rocks vs NIST SRMs | CF-LIBS | a few % up to 30% for most elements |
| cremers2009capabilities | one soil vs ICP | CF vs controlled calibration | 6.0% vs 3.2% mean difference (2.6x) |
| wiens2013preflight | general | CF-LIBS (literature) | +/-15-20% average on some matrices |
| takahashi2017matrix | metals; geological | CF-LIBS (Herrera, Salle) | 5-20% metals; 1-190% geological |
| wiens2012chemcam | 69 standards, 1.6-3.0 m | PLS (in-sample median rel. error) | SiO2 0.7-0.8, Al2O3 2.4-2.7, CaO 2.1-2.5, Fe2O3 3.7-3.9, MgO 4.2-7.4, Na2O 2.5-3.2, TiO2 6.1-7.0, K2O 8.8-12.0, P2O5 9.3-13.2 %; requirement +/-10%; cross-validation worse for P, K, Ti, Mn |
| korbel2024lithium | Z300 on Li-mineral pellets, argon | MLR on Li I lines (wt% Li2O) | RMSE 0.15, MAPE 11.75%, R2 0.99; via Rb 0.12 / 11.33%; LOD ~10 ppm Li |
| harmon2019mineral (company release) | Z-300 on pressed RC drill pellets | univariate vs 4-acid/ICP | mean deviation +/-21% (not peer reviewed) |
| connors2016handheld | rock CRMs and NIST soils, argon | analyte/matrix-line ratio curves | % level for major/minor, ~1000 ppm Cr, Ba, Pb, <=100 ppm Be, Li (sensitivity, not error) |
| dyar2016univariate | 452 rocks | univariate vs PLS | best-region univariate +35% (majors); PLS +63% over univariate |
| eppler1996matrix | Pb, Ba in sand/soil | calibration with reference lines | LOD 17-76 ppm; precision <=10% RSD |

Take-away for "good": handheld/mineral CF-LIBS in literature is 5-30% on the matrix and major elements, >30% on minors; <10% requires a reference (Csigma, OPC, controlled calibration) or simple oxides. Treat suite S median relative error on majors of ~10-20% as literature-parity and <10% as strong.

## Ideas not worth testing
- Monte-Carlo / radiative-model CF-LIBS (gornushkin2010radiative, gornushkin2022montecarlo): text unavailable, heavy compute, no handheld evidence.
- Full Csigma parameter fit as a separate engine: duplicates the WNNLS plus tau fixed point; keep only the validity gate (idea 5).
- Width-based SA estimator (SA^-0.54 = dLambda/dLambda0, urbinamedina2021branching/El Sherbini): Stark widths at n_e 1e17 are below the 0.1-0.2 nm instrument LSF; use only for resonant alkali lines.
- Spatially resolved / Abel inversion (aguilera2004spatial): no spatial resolution on Z300; only its physics (idea 2) transfers.
- H-alpha n_e (elsherbini2006halpha): already implemented.
- PLS / ICA / sub-models / univariate (clegg2009pls, clegg2017recalibration, anderson2017submodels, dyar2016univariate): need large labelled corpora and conflict with the physics-based goal; use as benchmark context only.
- Stimulated-emission factor in tau (hahn T4.12): already A/B-refused 2026-07-04.
- Pump-probe diagnostics, nanoparticle breakdown (palasti2022argon), instrument papers (maurice*, wiens2021supercam): off topic.

## Verification / limits
- Numbers were taken from the provided text; where the text lost the micro sign (Connors "1 ms" delay, "125 mm") this is flagged in the note. The sign of the exponent in Delta_lambda/Delta_lambda0 = SA^-0.54 was lost in extraction and is flagged for check; the I/I0 = SA^0.46 relation was confirmed against the SA=0.25 numerical example in the paper.
- Notes for abstract-only papers do not contain equations; adoptable specifics there are limited to the abstract and Takahashi's secondary description.
- No code or test files modified. Test counts: see below.
- Test counts: no source/test files were modified (git status shows only manuscript/lit/notes/ and this report as new). A baseline pytest run (both invocations) was started but had not finished when this report was written; counts not captured. Output would be at the scratchpad t1.txt/t2.txt.
- Test counts (baseline, no code changed): PYTHONPATH=src: 614 passed, 4 skipped, 71 subtests passed. pymatgen-enabled: 620 passed, 4 skipped, 71 subtests passed. (Supersedes the 'not captured' line above.)

## Re-ingest after corpus verification (2026-11-03)
All four full texts were verified as the real papers (title in the first 300 words); notes rewritten in place.
- gaudiuso2010review: relevance 1; no change to the ranked top-10 (qualitative review, line-selection rule already covered by tognoni2010cflibs).
- gornushkin2022montecarlo: relevance 2; does not enter the top-10, but supplies a concrete S-suite recipe (0.5% noise, 84 pix/nm, ~1% ideal ceiling, per-element cost weights) and should replace the "text unavailable" dismissal under "Ideas not worth testing" with "compute-heavy, but useful as an S-suite benchmark".
- harmon2023muscovite: relevance 3; same instrument family (Z-300) with a documented gate/shot protocol and Li/K/Rb ratio data; belongs in the top-10 as a real-spectra check on alkali ratios, displacing the lowest-ranked entry (a relevance-2 item).
- yu2023cnn: relevance 1; no change to the ranked top-10 (needs a labelled corpus; conflicts with the physics-based goal).
