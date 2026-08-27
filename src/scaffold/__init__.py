"""SCAFFOLD -- dilation- and congestion-aware graph sparsification.

Five algorithms behind one API::

    import scaffold

    result = scaffold.fast(G, keep_ratio=0.2)    # recommended default
    result = scaffold.greedy(G, keep_ratio=0.2)  # reference greedy
    result = scaffold.heap(G, keep_ratio=0.2)    # lazy greedy
    result = scaffold.batch(G, keep_ratio=0.2)   # sampled-batch growth
    scores = scaffold.sample(G)                  # per-edge weights, not a subgraph

    result = scaffold.sparsify(G, method="fast", keep_ratio=0.2)

``G`` may be a NetworkX graph, a PyG ``Data``, a ``scipy.sparse`` matrix or a
``(2, m)`` edge_index. Results convert back on request::

    result.to_pyg()
    result.to_networkx()
    result.to_scipy()
    result.edge_index, result.edge_weight

See ``docs/`` for the algorithm descriptions and ``examples/`` for runnable
demonstrations, including a visual grid-graph walkthrough.
"""

from ._version import __version__
from .api import METHODS, batch, fast, greedy, heap, sample, sparsify
from .backbone import available_backbones, register_backbone
from .datasets import grid_graph, grid_positions, random_geometric, ring_of_cliques
from .graph import Graph, normalize_graph
from .result import ScaffoldResult, ScaffoldScores
from .scoring import ScoreParams, path_scores, tree_scores

__all__ = [
    # entry points
    "sparsify",
    "greedy",
    "heap",
    "batch",
    "fast",
    "sample",
    "METHODS",
    # data types
    "Graph",
    "ScaffoldResult",
    "ScaffoldScores",
    "ScoreParams",
    "normalize_graph",
    # backbones
    "available_backbones",
    "register_backbone",
    # scoring primitives
    "tree_scores",
    "path_scores",
    # demo graphs
    "grid_graph",
    "grid_positions",
    "ring_of_cliques",
    "random_geometric",
    "__version__",
]


def __getattr__(name):
    """Lazily expose the optional-dependency submodules.

    ``scaffold.viz`` needs matplotlib and ``scaffold.pyg`` needs torch; neither
    should be imported just because someone typed ``import scaffold``.
    """
    if name in ("viz", "pyg"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
