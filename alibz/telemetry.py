"""Per-spectrum telemetry: stage wall times, counters, failure attribution.

Every later performance or reliability change is judged against numbers,
not impressions — this module supplies the numbers.  It is deliberately
tiny and dependency-free (safe to import from any alibz module without
cycles), and holds ONE per-process profile: each spectrum is analyzed in
its own worker process (or sequentially in-process), so a process-global
singleton reset at the top of each analysis is race-free by construction.

Usage::

    from alibz import telemetry

    telemetry.reset()
    with telemetry.stage("indexer_pass1"):
        ...
    telemetry.count("outer_objective_evals")
    snap = telemetry.snapshot()

``snapshot()`` returns::

    {"t_total_s": float,          # since reset()
     "stages":   {name: {"s": wall_seconds, "n": call_count}},
     "counters": {name: int},
     "open_stage": str | None,    # innermost stage currently running
     "failure_stage": str | None} # innermost stage an exception escaped

Failure attribution: when an exception (including the pipeline's SIGALRM
``_Timeout``) propagates out of a ``stage`` block, the INNERMOST such
stage is recorded as ``failure_stage`` — the ``with`` blocks unwind
before the driver's ``except`` runs, so the open-stage stack alone would
be empty by then.  This is what turns "it timed out somewhere" into
"it timed out in indexer_pass2".

Overhead is two ``perf_counter()`` calls and a dict update per stage
entry, negligible against stages that run seconds to minutes.
"""

import functools
import sys
import time
from contextlib import contextmanager

__all__ = ["reset", "stage", "count", "snapshot", "timed"]


class _Profile:
    __slots__ = ("stages", "counters", "stack", "failure_stage", "t0")

    def __init__(self):
        self.reset()

    def reset(self):
        self.stages = {}          # name -> [wall_s, n_calls]
        self.counters = {}        # name -> int
        self.stack = []           # open stage names, outermost first
        self.failure_stage = None
        self.t0 = time.perf_counter()


_PROFILE = _Profile()


def reset() -> None:
    """Start a fresh profile (call once per spectrum analysis)."""
    _PROFILE.reset()


@contextmanager
def stage(name: str):
    """Time a named pipeline stage; re-entrant names accumulate."""
    _PROFILE.stack.append(name)
    t = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t
        rec = _PROFILE.stages.setdefault(name, [0.0, 0])
        rec[0] += dt
        rec[1] += 1
        _PROFILE.stack.pop()
        # An exception is unwinding through this stage: the innermost
        # frame's finally runs first, so first-writer-wins records the
        # most specific location.
        if sys.exc_info()[0] is not None and _PROFILE.failure_stage is None:
            _PROFILE.failure_stage = name


def count(name: str, n: int = 1) -> None:
    """Increment a named counter (objective evals, cache hits, ...)."""
    _PROFILE.counters[name] = _PROFILE.counters.get(name, 0) + n


def timed(name: str):
    """Decorator form of :func:`stage`, for whole functions/methods."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def snapshot() -> dict:
    """Current profile as a plain JSON-serializable dict."""
    return dict(
        t_total_s=round(time.perf_counter() - _PROFILE.t0, 3),
        stages={name: dict(s=round(rec[0], 3), n=rec[1])
                for name, rec in _PROFILE.stages.items()},
        counters=dict(_PROFILE.counters),
        open_stage=_PROFILE.stack[-1] if _PROFILE.stack else None,
        failure_stage=_PROFILE.failure_stage,
    )
