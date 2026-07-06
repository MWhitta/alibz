"""alibz — LIBS spectral analysis toolkit.

Public API
----------
PeakyFinder        Peak detection, background removal, multi-Voigt fitting.
PeakyIndexer       Whole-pattern spectral indexer (alias for PeakyIndexerV3).
PeakyMaker         Forward spectral synthesis via Saha-Boltzmann.
PeakyCorpus        Batch loading, standardisation, and parallel fitting.
PeakyPCA           PCA peak-shape decomposition and broadening classification.
DetectorModel      Three-segment detector artifact removal and background subtraction.

Fit refinement and inspection:
    refine_fit, classify_feature, sa_voigt   Second-iteration refinement,
        blends vs self-absorption asymmetry (``alibz.refinement``).
    seed_minor_lines, match_and_scale   Prior-driven fitting of minor
        lines from established elements (``alibz.minor_lines``).
    peak_table, format_peak_table, estimate_peak_uncertainties,
    plot_spectrum_overview, plot_peak_zoom   Fit visualisation and
        parameter uncertainties (``alibz.inspection``).
    recover_residual_lines   Element-agnostic recovery of significant
        positive residual peaks (``alibz.minor_lines``).

Detection reporting and confounders (``alibz.detections``):
    analyze_detections, classify_detections, element_support,
    contested_support, merge_contests, element_uncertainties,
    confounder_catalog   Per-element detection status with true-negative
        confounder analysis; corpus confounder catalog.

Element metadata (``alibz.elements``):
    element_sort_key, element_periodic_block, element_color

End-to-end directory analysis (``alibz.pipeline``):
    analyze_spectrum, analyze_directory   Full chain to composition +
        detection report; the ``alibz-analyze`` CLI wraps these.

Supporting types exported from ``peaky_indexer_v3``:
    FitResult, LineTable, PeakVector, Species, PhysicsComputationError

Utility re-exports:
    Database, SahaBoltzmann
"""

from alibz.peaky_finder import PeakyFinder
from alibz.peaky_indexer_v3 import (
    FitResult,
    LineTable,
    PeakVector,
    PeakyIndexer,
    PeakyIndexerV3,
    PhysicsComputationError,
    Species,
)
from alibz.peaky_maker import PeakyMaker
from alibz.peaky_corpus import PeakyCorpus
from alibz.peaky_pca import PeakyPCA
from alibz.detector import DetectorModel
from alibz.refinement import classify_feature, refine_fit, sa_voigt
from alibz.minor_lines import (match_and_scale, recover_residual_lines,
                               seed_minor_lines)
from alibz.inspection import (
    estimate_peak_uncertainties,
    format_peak_table,
    peak_table,
    plot_peak_zoom,
    plot_spectrum_overview,
)
from alibz.elements import (
    element_color,
    element_periodic_block,
    element_sort_key,
)
from alibz.detections import (
    analyze_detections,
    classify_detections,
    confounder_catalog,
    contested_support,
    element_support,
    element_uncertainties,
    merge_contests,
    resolve_confounded,
)
from alibz.pipeline import analyze_directory, analyze_spectrum
from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import SahaBoltzmann

__all__ = [
    # Core pipeline
    "PeakyFinder",
    "PeakyIndexer",
    "PeakyIndexerV3",
    "PeakyMaker",
    "PeakyCorpus",
    "PeakyPCA",
    "DetectorModel",
    "plot_spectrum_overview",
    "plot_peak_zoom",
    "peak_table",
    "format_peak_table",
    "estimate_peak_uncertainties",
    "refine_fit",
    "classify_feature",
    "sa_voigt",
    "recover_residual_lines",
    "seed_minor_lines",
    "match_and_scale",
    # Detection reporting + confounders
    "analyze_detections",
    "classify_detections",
    "element_support",
    "contested_support",
    "merge_contests",
    "resolve_confounded",
    "element_uncertainties",
    "confounder_catalog",
    # Element metadata
    "element_sort_key",
    "element_periodic_block",
    "element_color",
    # Directory / spectrum analysis
    "analyze_spectrum",
    "analyze_directory",
    # Indexer data types
    "FitResult",
    "LineTable",
    "PeakVector",
    "Species",
    "PhysicsComputationError",
    # Utilities
    "Database",
    "SahaBoltzmann",
]
