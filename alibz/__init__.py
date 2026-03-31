"""alibz — LIBS spectral analysis toolkit.

Public API
----------
PeakyFinder        Peak detection, background removal, multi-Voigt fitting.
PeakyIndexer       Whole-pattern spectral indexer (alias for PeakyIndexerV3).
PeakyMaker         Forward spectral synthesis via Saha-Boltzmann.
PeakyCorpus        Batch loading, standardisation, and parallel fitting.
PeakyPCA           PCA peak-shape decomposition and broadening classification.
DetectorModel      Three-segment detector artifact removal and background subtraction.

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
