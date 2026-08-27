"""The five SCAFFOLD algorithms.

Each module exposes ``run(graph, ...)``. ``greedy``, ``heap``, ``batch`` and
``fast`` return ``(mask, metadata)``; ``sample`` returns a fitted
:class:`~scaffold.algorithms.sample.ScaffoldSampler`.

Nothing here knows about NetworkX, PyG or SciPy -- everything runs on a
normalized :class:`~scaffold.graph.Graph`.
"""

from . import batch, fast, greedy, heap, sample

__all__ = ["greedy", "heap", "batch", "fast", "sample"]
