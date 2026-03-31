"""Deprecated compatibility shim for the retired v2 indexer.

``PeakyIndexerV3`` is now the only supported indexer in alibz. This module
keeps the v2 import path alive long enough to migrate callers to the v3
constructor and workflow.
"""

from __future__ import annotations

import warnings

from alibz.peaky_indexer_v3 import (
    FitResult,
    LineTable,
    PeakVector,
    PeakyIndexerV3,
    Species,
)

PeakRecord = PeakVector

__all__ = [
    "FitResult",
    "LineTable",
    "PeakRecord",
    "PeakyIndexerV2",
    "PeakyIndexerV3",
    "Species",
]

_DEPRECATION_MESSAGE = (
    "alibz.peaky_indexer_v2.PeakyIndexerV2 is deprecated. "
    "Use alibz.PeakyIndexer or alibz.peaky_indexer_v3.PeakyIndexerV3. "
    "V3 is the only supported indexer and remains experimental."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


class PeakyIndexerV2(PeakyIndexerV3):
    """Deprecated alias for :class:`alibz.peaky_indexer_v3.PeakyIndexerV3`."""

    def __init__(
        self,
        peak_array,
        pca_scores=None,
        dbpath: str = "db",
        temperature: float = 10_000.0,
        **kwargs,
    ) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        super().__init__(
            peak_array=peak_array,
            pca_scores=pca_scores,
            dbpath=dbpath,
            temperature_init=temperature,
            **kwargs,
        )
