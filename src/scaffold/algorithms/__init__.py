"""The four SCAFFOLD algorithms.

Each module exposes ``run(graph, ...)``. ``exact``, ``heap`` and ``fast``
return ``(mask, metadata)``; ``sample`` returns a fitted
:class:`~scaffold.algorithms.sample.ScaffoldSampler`.

Nothing here knows about NetworkX, PyG or SciPy -- everything runs on a
normalized :class:`~scaffold.graph.Graph`.
"""

from . import exact, fast, heap, sample

__all__ = ["exact", "fast", "heap", "sample"]
