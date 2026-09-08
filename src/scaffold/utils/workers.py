"""Worker-count resolution and numba thread control.

Every parallel entry point in the package resolves its worker count here, so
``workers=`` means the same thing everywhere and a single environment variable
can cap a whole batch of jobs.

Resolution order, first hit wins:

1. the explicit ``workers=`` argument,
2. ``SCAFFOLD_NUM_WORKERS``,
3. ``OMP_NUM_THREADS``,
4. ``min(8, len(os.sched_getaffinity(0)))``.

The default is capped at 8 deliberately. Measured on a 2.4M-edge graph, the
tree scorer scales 5.19x at 8 threads and only 6.13x at 16 -- it is
memory-bound past that -- so the extra cores buy little, while grabbing them
all on a shared node makes concurrent jobs fight each other. Callers who really
do want the whole machine pass ``workers=-1``.

Worker count is a *performance* knob only. Every algorithm here returns
bitwise-identical results at any worker count; anything else is a bug.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Optional

try:  # pragma: no cover - depends on the install
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None

__all__ = ["available_cpus", "resolve_workers", "parallel_threads", "split_workers"]

DEFAULT_WORKER_CAP = 8
_ENV_VARS = ("SCAFFOLD_NUM_WORKERS", "OMP_NUM_THREADS")

# Numba's thread mask belongs to the calling Python thread. A shared depth
# counter would leave concurrent pool tasks at their default thread count.
_THREAD_STATE = threading.local()


def available_cpus() -> int:
    """CPUs this process may actually run on, honouring cgroup/taskset limits."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _from_env() -> Optional[int]:
    for name in _ENV_VARS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def resolve_workers(workers=None) -> int:
    """Resolve a worker count to a positive integer.

    ``None``, ``0`` and ``"auto"`` all mean "decide for me". ``-1`` means every
    CPU in the affinity mask. Any other value is taken literally, clamped to at
    least 1 and to the affinity mask.
    """
    cpus = available_cpus()

    if isinstance(workers, str):
        key = workers.strip().lower()
        if key in ("auto", ""):
            workers = None
        elif key == "all":
            workers = -1
        else:
            try:
                workers = int(key)
            except ValueError:
                raise ValueError(
                    f"workers must be an int, 'auto' or 'all', got {workers!r}"
                ) from None

    if workers is not None and int(workers) != 0:
        requested = int(workers)
        if requested < 0:
            return cpus
        return max(1, min(cpus, requested))

    from_env = _from_env()
    if from_env is not None:
        return max(1, min(cpus, from_env))

    return max(1, min(cpus, DEFAULT_WORKER_CAP))


def split_workers(outer_tasks: int, workers: int):
    """Divide ``workers`` between an outer task pool and inner numba threads.

    Used where two parallel axes are available at once -- SCAFFOLD-Sample's
    ``R`` independent backbones each running a parallel tree scorer, for
    instance. Oversubscribing both axes is markedly slower than either alone,
    so the outer pool takes priority and the inner threads get what is left.
    """
    workers = max(1, int(workers))
    outer_tasks = max(1, int(outer_tasks))
    outer = min(outer_tasks, workers)
    inner = max(1, workers // outer)
    return outer, inner


@contextlib.contextmanager
def parallel_threads(workers: int):
    """Run the block with numba's thread count pinned to ``workers``.

    A no-op when numba is missing (the pure-Python kernels are serial anyway).
    ``numba.set_num_threads`` cannot exceed ``NUMBA_NUM_THREADS``, which is
    fixed when the threading layer first launches, so the request is clamped
    rather than allowed to raise.

    Each pool task must enter its own scope. Nested scopes on the same thread
    may lower the budget, but cannot exceed their parent's limit. Both the
    previous mask and the enclosing budget are restored, including on error.
    """
    workers = max(1, int(workers))
    if _numba is None:
        yield workers
        return

    ceiling = int(_numba.config.NUMBA_NUM_THREADS)
    target = max(1, min(workers, ceiling))

    parent = getattr(_THREAD_STATE, "limit", None)
    if parent is not None:
        target = min(target, parent)
    previous = int(_numba.get_num_threads())
    _numba.set_num_threads(target)
    _THREAD_STATE.limit = target
    try:
        yield target
    finally:
        _numba.set_num_threads(previous)
        _THREAD_STATE.limit = parent
