# Theory literature ingest, 2026-11-03

## 1. Coverage
50 keys in theory.bib. 40 notes written to manuscript/lit/notes/<key>.md.
Missing (no fulltext file, 10): irwin1981partition, karzas1961gaunt, thompson1987voigt, anderson2000epsf, stewart1966lowering, cremers2013handbook, kramida2023asd, griem1964plasma, griem1974broadening, mcwhirter1965spectral.
Wrong or unusable file (note written, flagged): hermann1998earlyphase (AIP platform guide), janesick2007photontransfer (is a copy of gigosos1996 content), ralchenko2020nist (is Reader 2006 proceedings), konjevic1999nonhydrogenic (title page only), griem1997principles (front matter).
Abstract/first-page only (specifics not verified): cristoforetti2010lte, aragon2008review, tognoni2010cflibs, irons1979escape, elsherbini2005, sun2009, bredice2006, amamou2002, cristoforetti2013, gigosos2003halpha, bulajic2002, olivero1977.
Read in more depth: fujimoto1990, griem1963, barklem2016, ciricosta2012, birch1994, ciddor1996, fuhr2006 (grades). Equations in fujimoto/griem1963 were garbled by pdftotext; coefficients must be re-read from PDF. Several constants (Halpha 0.549/0.67965, El Sherbini exponent 0.54, Ciddor k0) are from memory and marked unverified in the notes.
Not mined in depth: kunze2009plasma, hahn2012libs, konjevic1976/2002 tables, cowan1948 (scan).

## 2. Top 10 candidate improvements
| # | Idea | Source keys | Module | Metric, direction | Cost | Risk |
|---|------|-------------|--------|-------------------|------|------|
| 1 | Air/vacuum conversion audit with Birch-Downs/Ciddor plus p,T scaling (20 C, 85 kPa shifts lines ~25 pm at 500 nm, same size as Ar anchor error) | birch1994, ciddor1996, edlen1966 | line DB loader | Z300 Fe line-matching fraction up; Ar anchor residual down | S | no effect if already standard; Z300 internal p,T unknown |
| 2 | Weight/drop Fe I lines by NIST accuracy grade (A 3%, B 10%, C 25%, D 50%) | fuhr2006fe, obrian1991 | peaky_indexer_v3 WNNLS weights | Fe I Boltzmann T scatter down; resolved-fit count same | S | blends dominate over A error |
| 3 | Per-line McWhirter/Fujimoto gap flag; drop flagged lines (Ar I NIR dE~11.5 eV needs ~2e17 at 8 kK) from the thermometer | griem1963, fujimoto1990, cristoforetti2010 | detections.py, thermometer | stage_disagreement down; T error (S) down | S | necessary-only criterion (already queued as diagnostic) |
| 4 | Tabulated NIST partition functions | barklem2016 | sahaboltzmann.py | S: T error, spurious mass down; stage_disagreement down | M | tables stop at 10 kK; already queued |
| 5 | Two-temperature (neutral vs ion) option in stage thermometer | aguilera2007 | peaky_indexer_v3 | stage_disagreement down; Fe II/Fe I ratio stability up | M | extra parameter, weak identifiability |
| 6 | Neutrality iteration for n_e when no Halpha | ciucci1999 | peaky_indexer_v3 | log n_e error (S) down; n_e-at-bound flag rate down | M | needs all species; H/O/N unseen |
| 7 | Boltzmann-plot self-absorption correction via high-E_k reference line (IRSAC) / generalised shared-upper-level pairs | sun2009, bredice2006, bulajic2002 | refinement.py, absorption.py | Fe I Boltzmann scatter down; K I doublet check | M | needs E_k spread, resolved pairs |
| 8 | Profile-dependent escape factor (Voigt) | cowan1948, irons1979 | absorption.py | SA round-trip majors error down | M | text not read in detail |
| 9 | Debye IPD (Ecker-Kroll low-density limit) | ecker1963, ciricosta2012 | sahaboltzmann.py | stage_disagreement slight down | S | corrected estimate: 0.07 eV per charge at 1e17, 8 kK (not several tenths); Saha factor 1.1-1.25; N_D~3 marginal |
| 10 | Halpha table lookup replacing power-law n_e fit | gigosos1996/2003, konjevic2002 | stark.py | log n_e error on Halpha samples | S | unverified constants |

## 3. Not worth testing
- Stewart-Pyatt vs Ecker-Kroll at solid density (ciricosta2012): irrelevant at 1e17.
- Free-free Gaunt factors (vanhoof2014, kramers1923): synthetic continuum only, outside metrics.
- Non-hydrogenic Stark widths (konjevic1976/1999, griem1968): 0.002 nm unresolvable at 0.1-0.2 nm pixels.
- Width-based SA (elsherbini2005) and profile SA fits (amamou2002): need resolved Stark/profile shape.
- Kurucz predicted lines: wavelength uncertainty about a pixel.
- Olivero Voigt FWHM, Saha 1920: already implemented.
- Textbooks (griem1997, kunze2009, hahn2012): background only.
