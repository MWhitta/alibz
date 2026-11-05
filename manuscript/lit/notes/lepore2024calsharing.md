---
key: lepore2024calsharing
title: "Sharing calibration information among LIBS instruments"
year: 2024
topic: ml
relevance: 2
---

**Method/claim**: More than 2500 geological targets on three instruments; tests calibration transfer between resolutions and instruments.

**Adoptable specifics**: PDS-PLS (piecewise direct standardization) reduces major-element error when resolutions differ; binning peak areas allows cross-instrument predictions; adding a small target-condition dataset to a large database improves reliability.

**Where it would enter alibz**: Supports peak-area (binned line) features for transfer and aligns with alibz peak list inputs. Z300 vs lab spectrometer transfer set via PDS on peak areas.

**Fit-quality test**: Fe standard fe_aesar_9998_means_20260922.npz/data/remote_samples/*.csv: major-element error across instruments with vs without PDS.

**Caveats**: Empirical, needs paired standards.
