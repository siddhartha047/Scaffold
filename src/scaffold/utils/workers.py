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

# numba.set_num_threads mutates process-global state, so two Python threads
# entering parallel_threads() at once would clobber each other's setting. The
# depth counter makes the outermost scope the only one that touches it.
_THREAD_LOCK = threading.Lock()
_PIN_DEPTH = 0
_PIN_PREVIOUS = None


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

    Only the *outermost* entry changes anything; nested entries -- including
    entries from worker threads of an enclosing pool -- see the pin already in
    place and leave it alone. That nesting rule is what lets an algorithm split
    its budget between an outer task pool and inner numba threads: the outer
    scope pins once, and the per-task calls underneath do not fight over the
    setting.

    The lock is held only across the counter update, never across the body. An
    earlier version wrapped the whole block, which quietly serialized every
    thread of SCAFFOLD-Sample's backbone pool -- correct results, no speedup.
    """
    workers = max(1, int(workers))
    if _numba is None:
        yield workers
        return

    ceiling = int(_numba.config.NUMBA_NUM_THREADS)
    target = max(1, min(workers, ceiling))

    global _PIN_DEPTH, _PIN_PREVIOUS
    with _THREAD_LOCK:
        outermost = _PIN_DEPTH == 0
        if outermost:
            try:
                _PIN_PREVIOUS = int(_numba.get_num_threads())
            except Exception:  # pragma: no cover - very old numba
                _PIN_PREVIOUS = ceiling
            _numba.set_num_threads(target)
        else:
            target = int(_numba.get_num_threads())
        _PIN_DEPTH += 1
    try:
        yield target
    finally:
        with _THREAD_LOCK:
            _PIN_DEPTH -= 1
            if _PIN_DEPTH == 0:
                _numba.set_num_threads(_PIN_PREVIOUS)
