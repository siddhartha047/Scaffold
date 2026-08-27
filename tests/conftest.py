"""Shared fixtures and skip markers."""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from scaffold.graph import from_edge_index


def _importable(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


requires_networkx = pytest.mark.skipif(
    not _importable("networkx"), reason="networkx not installed"
)
requires_scipy = pytest.mark.skipif(
    not _importable("scipy"), reason="scipy not installed"
)
requires_pyg = pytest.mark.skipif(
    not (_importable("torch") and _importable("torch_geometric")),
    reason="torch / torch_geometric not installed",
)
requires_matplotlib = pytest.mark.skipif(
    not _importable("matplotlib"), reason="matplotlib not installed"
)

ALL_METHODS = ("greedy", "heap", "fast", "sample")
GREEDY_METHODS = ("greedy", "heap", "fast")


@pytest.fixture
def grid():
    """8x8 lattice: 64 nodes, 112 edges, delta_min = 63/112 = 0.5625."""
    return scaffold.grid_graph(8, 8)


@pytest.fixture
def small_grid():
    """5x5 lattice, small enough for the greedy variant in a tight loop."""
    return scaffold.grid_graph(5, 5)


@pytest.fixture
def weighted_grid():
    return scaffold.grid_graph(7, 7, weight="random", seed=3)


@pytest.fixture
def disconnected():
    """Two triangles joined by nothing, plus one isolated node.

    Three components and one node of degree zero: the case that breaks
    implementations assuming a single connected component.
    """
    edge_index = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    return from_edge_index(edge_index, num_nodes=7)


@pytest.fixture
def bridged():
    """Two cliques joined by a single bridge; the bridge is mandatory."""
    return scaffold.ring_of_cliques(3, 5)


def sparsify_any(graph, method, keep_ratio, seed=0, **kwargs):
    """Uniform helper: for ``sample``, materialize one draw."""
    result = scaffold.sparsify(
        graph, method=method, keep_ratio=keep_ratio, seed=seed, **kwargs
    )
    if method == "sample":
        result = result.draw(keep_ratio=keep_ratio, seed=seed)
    return result
