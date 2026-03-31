"""Retired experimental fitter.

`PeakyFitter` depended on stale finder/indexer APIs and has been removed from
the supported alibz surface. Use `PeakyFinder` for peak fitting and
`PeakyIndexer` / `PeakyIndexerV3` for the experimental indexing path.
"""

from __future__ import annotations

import warnings

__all__ = ["PeakyFitter"]

_RETIRED_MESSAGE = (
    "PeakyFitter has been removed from alibz because it depended on stale "
    "finder/indexer APIs and produced unreliable results. Use PeakyFinder "
    "for peak fitting. If this fitter is needed again, restore it from "
    "history and rebuild it against the current APIs."
)

warnings.warn(_RETIRED_MESSAGE, DeprecationWarning, stacklevel=2)


class PeakyFitter:
    """Retired placeholder for the removed fitter."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(_RETIRED_MESSAGE)
