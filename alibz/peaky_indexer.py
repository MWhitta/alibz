"""Deprecated compatibility shim for the retired legacy indexer.

The only supported indexer in alibz is now
``alibz.peaky_indexer_v3.PeakyIndexerV3``. This module is retained only to
provide a clear migration path for older imports.
"""

from __future__ import annotations

import warnings

import numpy as np

from alibz.peaky_indexer_v3 import (
    FitResult,
    LineTable,
    PeakVector,
    PeakyIndexerV3,
    Species,
)

__all__ = [
    "FitResult",
    "LineTable",
    "PeakVector",
    "PeakyIndexer",
    "PeakyIndexerV3",
    "Species",
]

_DEPRECATION_MESSAGE = (
    "alibz.peaky_indexer is deprecated. Use alibz.PeakyIndexer or "
    "alibz.peaky_indexer_v3.PeakyIndexerV3. The old finder-coupled "
    "indexer has been retired."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


def _validate_legacy_signature(args) -> None:
    """Raise a clear error for the retired ``PeakyIndexer(finder)`` API."""
    if not args:
        return

    first_arg = args[0]
    if isinstance(first_arg, (list, tuple, np.ndarray)):
        return
    if hasattr(first_arg, "__array__") or hasattr(first_arg, "shape"):
        return
    if hasattr(first_arg, "fit_spectrum_data") or hasattr(first_arg, "data"):
        raise TypeError(
            "Legacy PeakyIndexer(finder, ...) is retired. "
            "Use PeakyIndexer(peak_array=fit_dict['sorted_parameter_array'], ...)."
        )


class PeakyIndexer(PeakyIndexerV3):
    """Deprecated alias for :class:`alibz.peaky_indexer_v3.PeakyIndexerV3`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        _validate_legacy_signature(args)
        super().__init__(*args, **kwargs)
