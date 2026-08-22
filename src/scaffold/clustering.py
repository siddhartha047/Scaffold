"""Node partitioning used by the cluster-parallel growth loops.

SCAFFOLD-Fast's round loop and SCAFFOLD-Heap's lazy heaps can both be run
per-cluster instead of globally. That buys two things: the per-round candidate
sample is spread over the whole graph instead of concentrating wherever scores
happen to be highest, and the clusters are independent, so the work parallelizes.

Only dependency-free partitioners ship with the package. ``metis`` is used when
``pymetis`` is installed and falls back to ``bfs`` otherwise -- with a warning,
never silently, because the partition quality changes the sampling spread.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from .graph import Graph
from .kernels import build_csr

__all__ = ["assign_clusters", "auto_cluster_count", "cluster_edges"]


def auto_cluster_count(graph: Graph) -> int:
    """A reasonable default cluster count for a graph of this size.

    Targets roughly 4k undirected edges per cluster, clamped to ``[1, 256]``,
    so small graphs stay in one piece and large ones get enough spread.
    """
    if graph.num_edges <= 8192:
        return 1
    return int(max(1, min(256, round(graph.num_edges / 4096))))


def assign_clusters(
    graph: Graph,
    clusters=None,
    method: str = "bfs",
    seed=None,
) -> np.ndarray:
    """Return a ``(num_nodes,)`` int64 array of cluster ids.

    ``clusters`` may be ``None`` or ``1`` (single cluster), an integer count, or
    a precomputed per-node assignment array.
    """
    n = graph.num_nodes
    if isinstance(clusters, np.ndarray) or (
        clusters is not None and not isinstance(clusters, (int, np.integer))
    ):
        labels = np.ascontiguousarray(clusters, dtype=np.int64).reshape(-1)
        if labels.shape != (n,):
            raise ValueError(
                f"cluster assignment must have length {n}, got {labels.shape}"
            )
        _, labels = np.unique(labels, return_inverse=True)
        return labels.astype(np.int64, copy=False)

    if clusters is None:
        clusters = auto_cluster_count(graph)
    count = int(max(1, min(int(clusters), n)))
    if count == 1 or n == 0:
        return np.zeros(n, dtype=np.int64)

    method = str(method).strip().lower()
    if method == "metis":
        labels = _metis_clusters(graph, count)
        if labels is not None:
            return labels
        warnings.warn(
            "clustering method 'metis' requires pymetis; falling back to 'bfs'. "
            "Install pymetis for better-balanced partitions.",
            RuntimeWarning,
            stacklevel=2,
        )
        method = "bfs"
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, count, size=n, dtype=np.int64)
    if method == "bfs":
        return _bfs_clusters(graph, count)
    raise ValueError(
        f"Unknown clustering method {method!r}; expected 'bfs', 'metis' or 'random'"
    )


def _bfs_clusters(graph: Graph, count: int) -> np.ndarray:
    """Grow ``count`` balanced BFS balls from the highest-degree seeds.

    Round-robin expansion keeps the parts within one node of each other in
    size, which matters because an unbalanced partition starves some clusters
    of candidates and over-samples others.
    """
    n = graph.num_nodes
    rowptr, col, _ = build_csr(n, graph.src, graph.dst)
    degree = np.diff(rowptr)
    labels = np.full(n, -1, dtype=np.int64)

    seeds = np.lexsort((np.arange(n), -degree))[:count]
    frontiers = []
    for cid, seed_node in enumerate(seeds):
        labels[seed_node] = cid
        frontiers.append([int(seed_node)])

    quota = int(math.ceil(n / count))
    sizes = np.ones(count, dtype=np.int64)
    remaining = n - count
    while remaining > 0:
        progressed = False
        for cid in range(count):
            if not frontiers[cid] or sizes[cid] >= quota:
                continue
            node = frontiers[cid].pop(0)
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = int(col[pos])
                if labels[nb] != -1:
                    continue
                labels[nb] = cid
                sizes[cid] += 1
                remaining -= 1
                frontiers[cid].append(nb)
                progressed = True
                if sizes[cid] >= quota:
                    break
        if not progressed:
            break

    # Isolated nodes and anything the frontiers could not reach.
    stragglers = np.flatnonzero(labels < 0)
    if stragglers.size:
        labels[stragglers] = np.argsort(sizes)[
            np.arange(stragglers.size) % count
        ]
    return labels


def _metis_clusters(graph: Graph, count: int) -> Optional[np.ndarray]:
    try:
        import pymetis
    except Exception:
        return None
    rowptr, col, _ = build_csr(graph.num_nodes, graph.src, graph.dst)
    adjacency = [
        col[rowptr[i] : rowptr[i + 1]].tolist() for i in range(graph.num_nodes)
    ]
    _, membership = pymetis.part_graph(count, adjacency=adjacency)
    return np.asarray(membership, dtype=np.int64)


def cluster_edges(graph: Graph, node_labels: np.ndarray) -> np.ndarray:
    """Assign each edge to a cluster.

    Interior edges take their endpoints' shared cluster. Crossing edges are
    assigned to the lower cluster id, so every edge belongs to exactly one
    cluster and no candidate is scored twice in a round.
    """
    left = node_labels[graph.src]
    right = node_labels[graph.dst]
    return np.minimum(left, right).astype(np.int64, copy=False)
