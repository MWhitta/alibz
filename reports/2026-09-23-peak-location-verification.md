# Coordinator verification: peak location, spline vs native-pixel estimators — 2026-09-23

Checks of cal/reports/2026-09-23-peak-location-spline-vs-native.md (specialist, drafts 1–2).

## Draft 2 "decisive Fe accuracy test" is NOT valid
Scored all five estimators on the SAME accepted lines against ONE shared cubic curve (median over
estimators) instead of per-estimator 3xMAD-clipped RMS. Every estimator, including pixel-int (8),
has about half its lines off by more than one pixel (median |r| 93–203 pm; fraction >1 px:
UV 0.54–0.65, VIS 0.47–0.53, NIR 0.44–0.53; tails to ~1 nm). The test is dominated by line
misidentification / non-cubic calibration structure, not estimator error. The claimed
2.5–2.7x accuracy gain is not supported. (Coordinator's own nearest-maximum Fe matching also
failed; Fe I 438.35 sits ~−0.32 nm from db + shift.)

## Identification-free test (cal/scripts/peak_phase_uniformity.py)
Isolated peaks with SNR>30 in the native run means (26 Fe runs + V run2): 141 UV, 196 VIS,
344 NIR. True line positions are uniform in sub-pixel phase, so an unbiased estimator gives a
flat phase histogram (the standard pixel-locking test).
- Spline-root: phases pile up at the pixel centre (VIS histogram 1/3/6/19/43/50/37/24/13/0 over
  −0.5..0.5 px). It pulls centres toward pixel centres.
- Pixel-integrated Gaussian, free width: repels from the centre in UV (bimodal at ±0.3 px); in VIS
  53 % of fits hit σ = 0.25 px (lower bound). The Gaussian shape is wrong for VIS (flat-topped /
  sub-pixel LSF?).
- Spline − pixint and parabola − pixint differences vary periodically with phase: 58–64 pm p-p in
  VIS, 34–39 pm in NIR, ~6 pm in UV.
Verdict: no current estimator is unbiased on real native data. The spline beats argmax and
linear-resample argmax (earlier tables), but it has pixel-locking bias of up to about ±0.25 px in
VIS. The Gaussian pixel fit is not a safe replacement.

## Recommended method
An effective line-spread function (ePSF, as in Anderson & King 2000 for undersampled HST
images) per segment, built self-consistently from many isolated lines at random phases. Iterate
shape and centres until the phase histogram is flat. Then use ePSF fits (amplitude, centre, local
background) for positions and areas. Acceptance test: flat phase histogram; Fe/Ar accuracy only
with validated line matching.

## Correction (later 2026-09-23)
The phase-uniformity test above has little power: the 681 peaks are only 9/18/21 distinct lines
(UV/VIS/NIR) repeated across runs. Counted by distinct line, it accepts all five estimators
(p 0.48–0.78). The "spline pixel-locks" and "Gaussian repels" readings are therefore NOT
established. The estimator differences vs phase (58–64 pm p-p VIS) stand as a disagreement
between methods. The ePSF build replaced this test with a per-run registration ("dither")
consistency test; see cal/reports/2026-09-23-empirical-lsf.md. Separately, all line identities
on native data were wrong by ~18 px: reports/2026-09-23-native-axis-18px-offset.md.
