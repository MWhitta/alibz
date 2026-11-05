---
key: wiens2013preflight
title: "Pre-flight calibration and initial data processing for the ChemCam laser-induced breakdown spectroscopy instrument on the Mars Science Labor"
year: 2013
topic: cflibs
relevance: 2
---

**Method/claim.** ChemCam pre-flight calibration and data processing: dark subtract, wavelet de-noise, electron-continuum removal, wavelength calibration, instrument-response correction (DN = L A Omega dLambda dt sigma + Off, Eq. 4), distance correction; Level-2 mean-centred variance-normalised spectra per range (Eq. 5). Notes CF-LIBS gives +/-15-20% on some matrices without standards.

**Adoptable specifics.** Eq. 4 instrument response; per-spectrograph MCVN normalisation. CF-LIBS +/-15-20% average (intro). Response measurement uses a calibrated source, which is the lamp calibration alibz has not done.

**Where it would enter alibz.** Template for the planned relative-response (lamp) calibration and the 620 nm step handling in `refinement.py`/`pipeline.py`. Already planned.

**Fit-quality test.** Real spectra: composition tilt across segments before/after response correction (see segment-response notes).

**Caveats.** Stand-off instrument; the distance correction is irrelevant for contact handheld.
