"""Small synthetic graphs for demos, tests and documentation figures.

The grid graph is the package's running example, and deliberately so. On a
2-D lattice every edge is interchangeable by symmetry, so a sparsifier's
*preferences* become visible: a method that only minimizes stretch produces a
comb-like skeleton with long detours, while SCAFFOLD's congestion terms push it
towards an evenly loaded mesh. You can see the difference; on Cora you cannot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .graph import Graph, from_edge_index

__all__ = ["grid_graph", "grid_positions", "ring_of_cliques", "random_geometric"]


def grid_graph(
    rows: int,
    cols: Optional[int] = None,
    periodic: bool = False,
    diagonals: bool = False,
    weight: Optional[str] = None,
    seed=None,
) -> Graph:
    """A ``rows x cols`` 2-D lattice, node ``(r, c)`` numbered ``r * cols + c``.

    Parameters
    ----------
    periodic:
        Wrap the boundaries into a torus, which removes the corner/edge
        asymmetry and makes the sparsified pattern easier to read.
    diagonals:
        Also connect the two diagonals of each cell (an 8-neighbour lattice).
    weight:
        ``None`` for an unweighted grid, ``"random"`` for uniform random
        weights in ``[0, 1)``, or ``"distance"`` for weights decreasing with
        Euclidean distance from the grid centre.
    """
    rows = int(rows)
    cols = rows if cols is None else int(cols)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")

    def node(r, c):
        return (r % rows) * cols + (c % cols)

    edges = []
    for r in range(rows):
        for c in range(cols):
            if periodic or c + 1 < cols:
                edges.append((node(r, c), node(r, c + 1)))
            if periodic or r + 1 < rows:
                edges.append((node(r, c), node(r + 1, c)))
            if diagonals:
                if periodic or (r + 1 < rows and c + 1 < cols):
                    edges.append((node(r, c), node(r + 1, c + 1)))
                if periodic or (r + 1 < rows and c > 0):
                    edges.append((node(r, c), node(r + 1, c - 1)))

    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), np.int64)
    num_nodes = rows * cols

    edge_weight = None
    if weight == "random":
        rng = np.random.default_rng(seed)
        edge_weight = rng.random(edge_index.shape[1])
    elif weight == "distance":
        positions = grid_positions(rows, cols)
        centre = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])
        mid = 0.5 * (positions[edge_index[0]] + positions[edge_index[1]])
        radius = np.linalg.norm(mid - centre, axis=1)
        edge_weight = 1.0 / (1.0 + radius)
    elif weight is not None:
        raise ValueError("weight must be None, 'random' or 'distance'")

    graph = from_edge_index(edge_index, num_nodes=num_nodes, edge_weight=edge_weight)
    graph.grid_shape = (rows, cols)  # read by scaffold.viz for the layout
    return graph


def grid_positions(rows: int, cols: Optional[int] = None) -> np.ndarray:
    """``(n, 2)`` xy coordinates for :func:`grid_graph`, row 0 at the top."""
    cols = rows if cols is None else cols
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.stack((c.astype(float), (rows - 1 - r).astype(float)), axis=1)


def ring_of_cliques(num_cliques: int, clique_size: int) -> Graph:
    """Cliques arranged in a ring, joined by single bridge edges.

    The bridges are the only cross-clique paths, so every sparsifier that keeps
    the graph connected must keep them -- a clean test of mandatory-edge and
    backbone handling.
    """
    num_cliques, clique_size = int(num_cliques), int(clique_size)
    if num_cliques < 2 or clique_size < 2:
        raise ValueError("need at least 2 cliques of at least 2 nodes")
    edges = []
    for k in range(num_cliques):
        base = k * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append((base + i, base + j))
        nxt = ((k + 1) % num_cliques) * clique_size
        edges.append((base + clique_size - 1, nxt))
    edge_index = np.asarray(edges, dtype=np.int64).T
    return from_edge_index(edge_index, num_nodes=num_cliques * clique_size)


def random_geometric(num_nodes: int, radius: float, seed=None) -> Tuple[Graph, np.ndarray]:
    """Random geometric graph in the unit square; returns ``(graph, positions)``."""
    rng = np.random.default_rng(seed)
    positions = rng.random((int(num_nodes), 2))
    diff = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    rows, cols = np.nonzero(np.triu(distance <= float(radius), k=1))
    edge_index = np.stack((rows.astype(np.int64), cols.astype(np.int64)))
    return from_edge_index(edge_index, num_nodes=int(num_nodes)), positions
