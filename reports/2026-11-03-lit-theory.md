# Theory bibliography report (2026-11-03)

## Method
Candidate DOIs from domain knowledge, then each fetched from Crossref (api.crossref.org/works/DOI; DataCite for 10.18434/T4W30F) and title/author/year compared; all .bib fields come from the Crossref record (Python generator in session scratchpad). Mismatches re-searched with Crossref query.bibliographic. Queries: Irons escape factor; Hermann 1998; Kurucz Including all the lines; Konjevic-Wiese 1976; Dimitrijevic-Konjevic; Cristoforetti-Tognoni columnar density; Hahn-Omenetto Part I; Amamou 2002; Lazic marbles; Gornushkin curve of growth; Saha 1920; Fuhr/Wiese Fe; Kramida NIST ASD; Kelleher-Podobedova; Bredice Mn; Konjevic Balmer; Cremers handbook; Noll; Aragon-Aguilera CoG; Griem books; Kunze; Gaunt factors; Drawin; Gigosos-Cardenoso 1996; Ecker-Kroll 1963; McWhirter 1965.

## Counts
50 entries in theory.bib. 47 DOI-verified (46 Crossref, plus NIST ASD via DataCite, kept in .bib but not in the manifest). 3 without DOI (griem1964plasma, griem1974broadening, mcwhirter1965spectral), from memory, unverified. Manifest sources-theory.txt: 46 lines.
Wrong first-guess DOIs corrected by search: Hermann 1998, Kurucz 2011, Konjevic-Wiese 1976, Gigosos-Cardenoso 1996, Ecker-Kroll, Irons, Cristoforetti 2013, Bredice 2006, Amamou, Dimitrijevic, Gornushkin, Saha, Lazic, Hahn I, Kelleher. Trimmed from 65 to 50: dropped verified marginal items (fuhr1981fe 10.1063/1.555644, kelleher2008namg 10.1063/1.2735328, kurucz2017lines 10.1139/cjp-2016-0794, bredice2010 10.1366/000370210790918454, lazic2004 10.1039/b315606k, janesick2001 10.1117/3.374903, aguilera2004 10.1016/j.sab.2004.08.003, peck1972 10.1364/JOSA.62.000958, humlicek1982 10.1016/0022-4073(82)90078-4, dimitrijevic1980 10.1016/0022-4073(80)90014-x, konjevic2012balmer 10.1016/j.sab.2012.06.026, gornushkin1999 10.1016/s0584-8547(99)00004-x, hahn2010 Part I 10.1366/000370210793561691, drawin1969 10.1007/bf01397532, aragon2015 10.1016/j.sab.2015.06.010).

## Table
| bibkey | DOI | verified-title-match | why it matters |
|---|---|---|---|
| cristoforetti2010lte | 10.1016/j.sab.2009.11.005 | yes (Local Thermodynamic Equilibrium in Laser-Induced B...) | LTE validity beyond McWhirter |
| fujimoto1990validity | 10.1103/PhysRevA.42.6588 | yes (Validity criteria for local thermodynamic equilibr...) | Collisional-radiative LTE validity criteria |
| griem1963validity | 10.1103/PhysRev.131.1170 | yes (Validity of Local Thermal Equilibrium in Plasma Sp...) | Griem LTE criterion |
| aragon2008review | 10.1016/j.sab.2008.05.010 | yes (Characterization of laser induced plasmas by optic...) | Review of plasma characterization by OES |
| aguilera2007multielement | 10.1016/j.sab.2007.03.024 | yes (Multi-element Saha–Boltzmann and Boltzmann plots i...) | Multi-element Saha-Boltzmann plot |
| hermann1998earlyphase | 10.1063/1.366639 | yes (Diagnostics of the early phase of an ultraviolet l...) | Early-phase plasma diagnostics with self-absorption |
| saha1920ionization | 10.1080/14786441008636148 | yes (LIII. <i>Ionization in the solar chromosphere</i>...) | Saha equation |
| irwin1981partition | 10.1086/190730 | yes (Polynomial partition function approximations of 34...) | Partition function polynomials |
| barklem2016partition | 10.1051/0004-6361/201526961 | yes (Partition functions and equilibrium constants for ...) | Modern partition functions |
| kramers1923bremsstrahlung | 10.1080/14786442308565244 | yes (XCIII. <i>On the theory of X-ray absorption and of...) | Bremsstrahlung origin |
| karzas1961gaunt | 10.1086/190063 | yes (Electron Radiative Transitions in a Coulomb Field....) | Gaunt factors for continuum |
| vanhoof2014gaunt | 10.1093/mnras/stu1438 | yes (Accurate determination of the free–free Gaunt fact...) | Accurate free-free Gaunt factors |
| ralchenko2020nist | 10.3390/atoms8030056 | yes (Development of NIST Atomic Databases and Online To...) | NIST ASD overview |
| kurucz2011lines | 10.1139/p10-104 | yes (Including all the lines<sup>1</sup>This article is...) | Kurucz line lists (GFALL) |
| fuhr2006fe | 10.1063/1.2218876 | yes (A Critical Compilation of Atomic Transition Probab...) | Fe I/II transition probabilities |
| obrian1991fe1 | 10.1364/JOSAB.8.001185 | yes (Lifetimes, transition probabilities, and level ene...) | Fe I lab oscillator strengths |
| konjevic1976stark | 10.1063/1.555532 | yes (A critical review of the Stark widths and shifts o...) | Non-hydrogenic Stark widths review |
| konjevic1999nonhydrogenic | 10.1016/S0370-1573(98)00132-X | yes (Plasma broadening and shifting of non-hydrogenic s...) | Non-hydrogenic Stark status |
| konjevic2002review | 10.1063/1.1486456 | yes (Experimental Stark Widths and Shifts for Spectral ...) | Stark widths 1989-2000 review |
| griem1968semiempirical | 10.1103/PhysRev.165.258 | yes (Semiempirical Formulas for the Electron-Impact Wid...) | Semiempirical Stark widths |
| gigosos1996hbeta | 10.1088/0953-4075/29/20/029 | yes (New plasma diagnosis tables of hydrogen Stark broa...) | Hydrogen Stark ne tables |
| gigosos2003halpha | 10.1016/S0584-8547(03)00097-1 | yes (Computer simulated Balmer-alpha, -beta and -gamma ...) | Balmer Stark profiles for ne |
| olivero1977voigt | 10.1016/0022-4073(77)90161-3 | yes (Empirical fits to the Voigt line width: A brief re...) | Voigt FWHM approximation |
| thompson1987voigt | 10.1107/S0021889887087090 | yes (Rietveld refinement of Debye–Scherrer synchrotron ...) | TCH pseudo-Voigt FWHM |
| anderson2000epsf | 10.1086/316632 | yes (Toward High‐Precision Astrometry with WFPC2. I. De...) | ePSF method |
| cowan1948selfabsorption | 10.1103/RevModPhys.20.418 | yes (Self-Absorption of Spectrum Lines...) | Curve of growth |
| irons1979escape | 10.1016/0022-4073(79)90102-x | yes (The escape factor in plasma spectroscopy—I. The es...) | Escape factor |
| bulajic2002selfabsorption | 10.1016/S0584-8547(01)00398-6 | yes (A procedure for correcting self-absorption in cali...) | Self-absorption correction in CF-LIBS |
| elsherbini2005selfabsorption | 10.1016/j.sab.2005.10.011 | yes (Evaluation of self-absorption coefficients of alum...) | Self-absorption coefficients |
| sun2009internalreference | 10.1016/j.talanta.2009.03.066 | yes (Correction of self-absorption effect in calibratio...) | Internal-reference correction |
| bredice2006manganese | 10.1016/j.sab.2006.10.015 | yes (Evaluation of self-absorption of manganese emissio...) | Self-absorption (Mn) |
| amamou2002selfabsorption | 10.1016/s0022-4073(02)00040-7 | yes (Correction of self-absorption spectral line and ra...) | Self-absorption correction, LTE plasma |
| cristoforetti2013columnar | 10.1016/j.sab.2012.11.010 | yes (Calculation of elemental columnar density from sel...) | Columnar density from self-absorbed lines |
| stewart1966lowering | 10.1086/148714 | yes (Lowering of Ionization Potentials in Plasmas...) | Stewart-Pyatt IPD |
| ecker1963lowering | 10.1063/1.1724509 | yes (Lowering of the Ionization Energy for a Plasma in ...) | Ecker-Kroll lowering |
| ciricosta2012ipd | 10.1103/PhysRevLett.109.065002 | yes (Direct Measurements of the Ionization Potential De...) | Experimental IPD |
| edlen1966refractive | 10.1088/0026-1394/2/2/002 | yes (The Refractive Index of Air...) | Air refractive index (Edlen) |
| ciddor1996refractive | 10.1364/AO.35.001566 | yes (Refractive index of air: new equations for the vis...) | Air refractive index (Ciddor) |
| birch1994refractive | 10.1088/0026-1394/31/4/006 | yes (Correction to the Updated Edlén Equation for the R...) | Edlen correction |
| janesick2007photontransfer | 10.1117/3.725073 | yes (Photon Transfer...) | CCD photon-transfer noise |
| ciucci1999cflibs | 10.1366/0003702991947612 | yes (New Procedure for Quantitative Elemental Analysis ...) | Original CF-LIBS |
| tognoni2010cflibs | 10.1016/j.sab.2009.11.006 | yes (Calibration-Free Laser-Induced Breakdown Spectrosc...) | CF-LIBS review |
| hahn2012libs | 10.1366/11-06574 | yes (Laser-Induced Breakdown Spectroscopy (LIBS), Part ...) | LIBS review II (gating, instrumentation) |
| griem1997principles | 10.1017/cbo9780511524578 | yes (Principles of Plasma Spectroscopy...) | Plasma spectroscopy principles |
| kunze2009plasma | 10.1007/978-3-642-02233-3 | yes (Introduction to Plasma Spectroscopy...) | Plasma spectroscopy textbook |
| cremers2013handbook | 10.1002/9781118567371 | yes (Handbook of Laser‐Induced Breakdown Spectroscopy...) | LIBS handbook |
| kramida2023asd | 10.18434/T4W30F | yes (DataCite: NIST Atomic Spectra Database, SRD 78) | Atomic data; version/year in note from memory (5.11, 2023), check at proof |
| griem1964plasma, griem1974broadening, mcwhirter1965spectral | none | n/a | Canonical books/chapter (McWhirter criterion, Stark theory); McWhirter page range from memory |

## Could not verify / not in .bib
- Fuhr, Martin & Wiese 1988 JPCRD 17 Suppl.4: guessed DOI 10.1063/1.555845 resolved to an unrelated erratum; Crossref search found only 1981/2006 compilations (2006 included).
- Hermann et al. 2001 time-evolution/double-pulse paper: no match found.
- Debye-Hueckel 1923, Noll LIBS book, Mihalas Stellar Atmospheres, Lochte-Holtgreven: no usable DOI.
- No arXiv-only items. Titles braces by heuristic plus overrides; not compiled with bibtex.
