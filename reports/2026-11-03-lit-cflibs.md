# CF-LIBS bibliography report (2026-11-03)

Method: Crossref `query.bibliographic` searches (4 batches, ~100 queries; WebSearch not needed) to find DOIs, then each DOI re-fetched via `api.crossref.org/works/<DOI>` and bib fields (authors, journal, volume, pages, year) taken from that record. Titles were compared by eye against the intended paper.

Queries covered: Hahn/Omenetto I+II, Fortes, Ciucci, Tognoni, Praher, Gornushkin, Aragon/Aguilera (CSigma, curves of growth, Boltzmann/Saha-Boltzmann), Cavalcanti, Herrera, Bulajic, Sun/Yu, Grifoni, Takahashi, Cristoforetti, El Sherbini, Clegg, Anderson, Dyar, Wiens, Maurice, Harmon, Senesi, Sweetapple, Connors, Jochum, Mermet, Rezaei, books (Cremers, Miziolek, Noll), argon/gate/self-absorption/spectral-response/wavelength-calibration topics.

**Entries: 59 in .bib and sources-cflibs.txt; verified title/DOI match: 59; unverified in .bib: 0.** No duplicate keys; bibtex parses the file with no errors. Books are `@book` with DOIs (listed in manifest; PDF retrieval for books will likely fail).

Provider: no provider switch occurred.

Notes: `wiens2021supercam` is dated 2020 by Crossref (online-first; print 2021). `sciaps2021` (10.1177/00037028211006764) is Urbina Medina et al., Applied Spectroscopy 2021, 'Branching Ratio Method for Assessing Optically Thin Conditions in Laser-Induced Plasmas' (Palleschi coauthor), directly relevant to branching-ratio self-absorption checks; it is not a SciAps application note. Titles were brace-protected by a heuristic; proofread acronyms before final build.

| bibkey | DOI | verified-title-match | why it matters to alibz | Crossref record |
|---|---|---|---|---|
| hahn2010review1 | 10.1366/000370210793561691 | yes | LIBS fundamentals review part I: plasma diagnostics, LTE, ablation physics underpinning the forward model | Hahn, David W. 2010: Laser-Induced Breakdown Spectroscopy (LIBS), Part I: Review of Basic D |
| hahn2012review2 | 10.1366/11-06574 | yes | LIBS review part II: instrumentation, calibration and quantification practice | Hahn, David W. 2012: Laser-Induced Breakdown Spectroscopy (LIBS), Part II: Review of Instru |
| fortes2013libs | 10.1021/ac303220r | yes | Concise analytical-chemistry review of LIBS and its quantification state | Fortes, Francisco J. 2012: Laser-Induced Breakdown Spectroscopy |
| cremers2013handbook | 10.1002/9781118567371 | yes | Standard LIBS handbook (book) | Cremers, David A. 2013: Handbook of Laser‐Induced Breakdown Spectroscopy |
| miziolek2006libs | 10.1017/cbo9780511541261 | yes | Fundamentals and applications of LIBS (book) |  2006: Laser-Induced Breakdown Spectroscopy (LIBS) |
| noll2012libs | 10.1007/978-3-642-20668-9 | yes | Noll LIBS fundamentals and applications (book) | Noll, Reinhard 2012: Laser-Induced Breakdown Spectroscopy |
| cremers2009capabilities | 10.1080/05704920903058755 | yes | Capabilities and limitations of LIBS | Cremers, David A. 2009: Laser-Induced Breakdown Spectroscopy—Capabilities and Limitations |
| russo2013ablation | 10.1021/ac4005327 | yes | Laser ablation physics and analytical implications | Russo, Richard E. 2013: Laser Ablation in Analytical Chemistry |
| gaudiuso2010review | 10.3390/s100807434 | yes | Review of LIBS methods incl. CF-LIBS in field/space applications | Gaudiuso, Rosalba 2010: Laser Induced Breakdown Spectroscopy for Elemental Analysis in Environ |
| ciucci1999cflibs | 10.1366/0003702991947612 | yes | Original CF-LIBS procedure | Ciucci, A. 1999: New Procedure for Quantitative Elemental Analysis by Laser-Induced Pla |
| tognoni2010cflibs | 10.1016/j.sab.2009.11.006 | yes | Canonical CF-LIBS state-of-the-art review | Tognoni, E. 2010: Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the ar |
| praher2010cflibs | 10.1016/j.sab.2010.03.010 | yes | CF-LIBS applied to oxide matrices; accuracy and limits | Praher, B. 2010: Calibration free laser-induced breakdown spectroscopy of oxide materia |
| gornushkin2010radiative | 10.1016/j.sab.2010.03.021 | yes | Radiative plasma models relevant to CF-LIBS | Gornushkin, Igor B. 2010: Radiative models of laser-induced plasma and pump-probe diagnostics re |
| gornushkin2022montecarlo | 10.3390/s22197149 | yes | Monte Carlo CF-LIBS intrinsic performance | Gornushkin, Igor B. 2022: Intrinsic Performance of Monte Carlo Calibration-Free Algorithm for La |
| aragon2014csigma | 10.1016/j.jqsrt.2014.07.026 | yes | CSigma graphs, an optically thick-aware plasma characterization | Aragón, C. 2014: CSigma graphs: A new approach for plasma characterization in laser-ind |
| aragon2015generalizedcog | 10.1016/j.sab.2015.06.010 | yes | Quantification via generalized curves of growth | Aragón, C. 2015: Quantitative analysis by laser-induced breakdown spectroscopy based on |
| aragon2008review | 10.1016/j.sab.2008.05.010 | yes | Review of plasma characterization by optical emission | Aragón, C. 2008: Characterization of laser induced plasmas by optical emission spectros |
| cavalcanti2013onepoint | 10.1016/j.sab.2013.05.016 | yes | One-point calibration CF-LIBS | Cavalcanti, G.H. 2013: One-point calibration for calibration-free laser-induced breakdown spe |
| bulajic2002selfabsorption | 10.1016/s0584-8547(01)00398-6 | yes | Self-absorption correction in CF-LIBS | Bulajic, D 2002: A procedure for correcting self-absorption in calibration free-laser i |
| sun2009selfabsorption | 10.1016/j.talanta.2009.03.066 | yes | Self-absorption correction in CF-LIBS | Sun, Lanxiang 2009: Correction of self-absorption effect in calibration-free laser-induced |
| aguilera2004spatial | 10.1016/j.sab.2004.08.003 | yes | Spatially resolved plasma characterization, Boltzmann plot behaviour | Aguilera, J.A 2004: Characterization of a laser-induced plasma by spatially resolved spect |
| aguilera2007sahaboltzmann | 10.1016/j.sab.2007.03.024 | yes | Multi-element Saha-Boltzmann plots | Aguilera, J.A 2007: Multi-element Saha–Boltzmann and Boltzmann plots in laser-induced plas |
| grifoni2016cfcomparison | 10.1016/j.sab.2016.08.022 | yes | Comparison of CF-LIBS with fundamental-parameters analysis | Grifoni, E. 2016: From Calibration-Free to Fundamental Parameters Analysis: A comparison |
| cristoforetti2010lte | 10.1016/j.sab.2009.11.005 | yes | LTE validity beyond the McWhirter criterion | Cristoforetti, G. 2010: Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscop |
| cristoforetti2013columnar | 10.1016/j.sab.2012.11.010 | yes | Columnar density from self-absorbed lines | Cristoforetti, G. 2013: Calculation of elemental columnar density from self-absorbed lines in  |
| rezaei2020selfabsorption | 10.1016/j.sab.2020.105878 | yes | Review of self-absorption evaluation and compensation approaches | Rezaei, Fatemeh 2020: A review of the current analytical approaches for evaluating, compensa |
| bredice2006selfabsorption | 10.1016/j.sab.2006.10.015 | yes | Self-absorption coefficient evaluation for emission lines | Bredice, F. 2006: Evaluation of self-absorption of manganese emission lines in Laser Ind |
| elsherbini2006halpha | 10.1016/j.sab.2006.03.014 | yes | Electron density from Halpha Stark width | El Sherbini, A.M. 2006: Measurement of electron density utilizing the Hα-line from laser produ |
| clegg2009pls | 10.1016/j.sab.2008.10.045 | yes | PLS/PCA multivariate analysis of remote LIBS (ChemCam precursor) | Clegg, Samuel M. 2009: Multivariate analysis of remote laser-induced breakdown spectroscopy s |
| clegg2017recalibration | 10.1016/j.sab.2016.12.003 | yes | ChemCam recalibration with expanded database | Clegg, Samuel M. 2017: Recalibration of the Mars Science Laboratory ChemCam instrument with a |
| anderson2017submodels | 10.1016/j.sab.2016.12.002 | yes | Sub-model multivariate quantification for ChemCam | Anderson, Ryan B. 2017: Improved accuracy in quantitative laser-induced breakdown spectroscopy |
| dyar2016univariate | 10.1016/j.sab.2016.07.010 | yes | Univariate vs multivariate LIBS regression for geologic samples | Dyar, M. Darby 2016: Comparison of univariate and multivariate models for prediction of maj |
| wiens2013preflight | 10.1016/j.sab.2013.02.003 | yes | ChemCam pre-flight calibration and data processing, incl. spectral response | Wiens, R.C. 2013: Pre-flight calibration and initial data processing for the ChemCam las |
| eppler1996matrix | 10.1366/0003702963905123 | yes | Classic matrix-effects study in soils | Eppler, Aaron S. 1996: Matrix Effects in the Detection of Pb and Ba in Soils Using Laser-Indu |
| costa2021calibration | 10.21577/0103-5053.20200175 | yes | Critical review of calibration strategies applied to LIBS | Costa, Vinicius 2021: Calibration Strategies Applied to Laser-Induced Breakdown Spectroscopy |
| mermet2008loq | 10.1016/j.sab.2007.11.029 | yes | Limit of quantitation concepts in atomic spectrometry | Mermet, Jean-Michel 2008: Limit of quantitation in atomic spectrometry: An unambiguous concept? |
| harmon2013geochem | 10.1016/j.sab.2013.05.017 | yes | LIBS review for geochemical and environmental analysis | Harmon, Russell S. 2013: Applications of laser-induced breakdown spectroscopy for geochemical a |
| harmon2019mineral | 10.3390/min9120718 | yes | LIBS for mineral exploration | Harmon, Russell 2019: Laser-Induced Breakdown Spectroscopy—An Emerging Analytical Tool for M |
| harmon2023muscovite | 10.3390/min13050697 | yes | Handheld LIBS of muscovite on lithium pegmatite | Harmon, Russell S. 2023: Rapid Analysis of Muscovites on a Lithium Pegmatite Prospect by Handhe |
| senesi2014earth | 10.1016/j.earscirev.2014.09.008 | yes | LIBS in terrestrial and extraterrestrial geology | Senesi, Giorgio S. 2014: Laser-Induced Breakdown Spectroscopy (LIBS) applied to terrestrial and |
| sweetapple2015lithium | 10.2138/am-2015-5165 | yes | LIBS Li mapping in pegmatite minerals | Sweetapple, Marcus T. 2015: Laser-induced breakdown spectroscopy (LIBS) as a tool for in situ mapp |
| korbel2024lithium | 10.1039/d3ja00423f | yes | Quantification of Li with handheld LIBS and XRF | Korbel, C. 2024: Quantification of lithium using handheld instruments: application of L |
| senesi2021handheld | 10.1016/j.sab.2020.106013 | yes | Field-portable and handheld LIBS historical review | Senesi, Giorgio S. 2021: Field-portable and handheld laser-induced breakdown spectroscopy: Hist |
| connors2016handheld | 10.1177/0003702816638247 | yes | Handheld LIBS for geochemical analysis | Connors, Brendan 2016: Application of Handheld Laser-Induced Breakdown Spectroscopy (LIBS) to |
| jochum2005georem | 10.1111/j.1751-908x.2005.tb00904.x | yes | GeoReM reference material database | Jochum, Klaus Peter 2005: GeoReM: A New Geochemical Database for Reference Materials and Isotopi |
| jochum2011nist | 10.1111/j.1751-908x.2011.00120.x | yes | Reference values for NIST SRM 610-617 glasses | Jochum, Klaus Peter 2011: Determination of Reference Values for NIST SRM 610–617 Glasses Followi |
| palasti2022argon | 10.1016/j.sab.2022.106435 | yes | Argon-induced LIBS signal enhancement | Palásti, Dávid J. 2022: Laser-induced breakdown spectroscopy signal enhancement effect for arg |
| borduchi2022detection | 10.1016/j.sab.2022.106409 | yes | Effect of detection times (delay/gate) on LIBS | Borduchi, Luís Carlos Leva 2022: Study of the effects of detection times in laser-induced breakdown spe |
| li2014spectralresponse | 10.1039/c3ja50371b | yes | Relative spectral response calibration of LIBS spectrometer system | Li, Xiaohui 2014: Relative spectral response calibration of a spectrometer system for la |
| palleschi2022selfabsorption | 10.56530/spectroscopy.xq8876v9 | yes | Self-absorption misunderstandings in LIBS | Palleschi, Vincenzo 2022: Avoiding Misunderstanding Self-Absorption in Laser-Induced Breakdown S |
| saloman2004neon | 10.1063/1.1797771 | yes | Reference wavelengths for spectrometer calibration (neon) | Saloman, E. B. 2004: Wavelengths, Energy Level Classifications, and Energy Levels for the S |
| villasboas2023lod | 10.1016/j.sab.2023.106690 | yes | Statistical LOD definition for CF-LIBS | Villas-Boas, Paulino Ribeiro 2023: A statistical definition of limit of detection for calibration-free la |
| wiens2012chemcam | 10.1007/s11214-012-9902-4 | yes | ? | Wiens, Roger C. 2012: The ChemCam Instrument Suite on the Mars Science Laboratory (MSL) Rove |
| maurice2012chemcam | 10.1007/s11214-012-9912-2 | yes | ? | Maurice, S. 2012: The ChemCam Instrument Suite on the Mars Science Laboratory (MSL) Rove |
| wiens2021supercam | 10.1007/s11214-020-00777-5 | yes | ? | Wiens, Roger C. 2020: The SuperCam Instrument Suite on the NASA Mars 2020 Rover: Body Unit a |
| maurice2021supercam | 10.1007/s11214-021-00807-w | yes | ? | Maurice, S. 2021: The SuperCam Instrument Suite on the Mars 2020 Rover: Science Objectiv |
| sciaps2021 | 10.1177/00037028211006764 | yes | Branching-ratio test for optically thin conditions (Palleschi coauthor); self-absorption diagnostics | Urbina Medina, Ivan A. 2021: Branching Ratio Method for Assessing Optically Thin Conditions in Lase |
| herrera2009cflibs | 10.1039/b820493d | yes | Recursive line-intensity-ratio CF-LIBS on alloys and soils | Herrera, Kathleen K. 2009: Semi-quantitative analysis of metal alloys, brass and soil samples by  |
| takahashi2017matrix | 10.1016/j.sab.2017.09.010 | yes | Review of matrix-effect and self-absorption compensation | Takahashi, Tomoko 2017: Quantitative methods for compensation of matrix effects and self-absor |

## Candidates NOT verified (not in .bib)

- Fabre et al. Li/pegmatite LIBS (Crossref gave only Photoniques 2020 review 10.1051/photon/202010338 and a French ANR project; no verified Fabre pegmatite paper)
- Gornushkin et al. inverse-problem CF-LIBS paper (only the 2022 Sensors MC paper, book chapters, and the 2010 radiative-model paper found)
- Tognoni and Cristoforetti, 'Signal and noise in LIBS' and 'Basic mechanisms' (searches rate-limited, not retried)
- Rare-earth-element LIBS quantification primary paper (only a thesis and a Talanta hair-calibration paper surfaced)
- Lasue et al. ChemCam major-element quantification; Lazic et al. Na/K in soil LIBS (rate-limited)
- Norlen 1973 Ar I/II wavelengths; Whaling 1995 Ar I; Saloman 2010 Ar II (rate-limited). Included instead only the verified neon reference (saloman2004neon); a candidate saloman2006argon (10.1063/1.2204960) turned out to be neutral mercury and was removed
- USGS BHVO-2/AGV-2/GSP-2 certificates: no DOI found; Jochum GeoReM 2005 and NIST 610-617 2011 used instead
- Deuterium-halogen lamp spectral-response calibration paper: none found; li2014spectralresponse is the closest
- Self-absorption of K I 766/770 and Na D in LIBS: no specific primary paper verified; general self-absorption reviews included (rezaei2020, bredice2006, cristoforetti2013, palleschi2022)
- Castro and Pereira-Filho: only a Talanta rare-earth hair paper surfaced; costa2021calibration (JBCS review) used instead
- Dropped for the 60-entry cap (verified, available on request): Rusak 1997 10.1080/10408349708050587; Cremers 2006 handbook 10.1002/0470093013; Bousquet 2007 10.1016/j.sab.2007.10.018; Sirven 2006 10.1007/s00216-006-0322-8; Forni 2013 10.1016/j.sab.2013.05.003; Cousin 2011 10.1016/j.sab.2011.10.004; Afgan 2017 10.1039/c7ja00219j; Safi 2019 10.1016/j.jare.2019.01.008; Gornushkin 2010 JAAS 10.1039/c0ja00016g; Amamou 2002 10.1016/s0022-4073(02)00040-7
