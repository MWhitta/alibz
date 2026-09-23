# Independent Ar I / O I wavelength calibration engine

Status: in progress.

Scope: implement `alibz/gas_calibration.py` and focused tests only. The estimator
will measure neutral Ar I and O I offsets directly from the incoming instrument
wavelength axis, without consuming an Fe or mixed-element wavelength shift.

Planned acceptance checks:

- known positive and negative synthetic offsets recover independently for Ar I
  and O I;
- a different Fe displacement cannot seed or contaminate the gas result;
- missing lines, blanks, noise, gaps, flat tops, and inconsistent group shifts
  abstain;
- coarse 0.18 nm sampling receives a native-cell uncertainty floor;
- multiplets count as one anchor and unresolved-centroid ambiguity is included
  in uncertainty;
- returned records are JSON safe and report anchor-supported ranges/segments.
