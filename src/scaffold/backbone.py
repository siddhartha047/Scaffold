"""Support backbones: the spanning forest every SCAFFOLD run starts from.

SCAFFOLD grows a sparsifier by *adding* edges to a connected skeleton rather
than by deleting edges from ``G``. The skeleton -- the *support backbone* -- is
a spanning forest, so as long as the edge budget is at least ``n - c`` the
output has exactly the connected components of the input. Everything above that
floor is spent on the edges that most reduce dilation and congestion.

Available backbones
-------------------

============== =============================================================
``fast-maxst``  **Default.** Bucketed approximate maximum spanning forest.
                Quantizes weights into ``buckets`` priority classes and runs
                one linear-time counting sort instead of an ``O(m log m)``
                comparison sort. On an unweighted graph it degenerates to the
                fastest deterministic forest scan available.
``fast-mst``    Same, minimizing.
``fast-randst`` Seeded coprime-stride randomized Kruskal. Visits every edge
                exactly once without allocating a random permutation. Faster
                and lower-memory than ``randst``, with less random mixing.
``maxst``       Exact maximum spanning forest (stable Kruskal, ``O(m log m)``).
``mst``         Exact minimum spanning forest.
``randst``      Kruskal on a uniform random edge permutation. Better-randomized
                edge order than ``fast-randst``, at the cost of materializing
                that full permutation. Re-drawn per call.
``spt``         Multi-source shortest-path (BFS) forest from the highest-degree
                node of each component. Low diameter, useful when you care more
                about hop distance than about edge weight.
``glst``        Greedy low-stretch tree. Highest quality, ``O(n * m * |cut|)``:
                small graphs only.
``none``        Start from the empty graph. Connectivity is then *not*
                guaranteed -- only use this if you want the pure score ranking.
============== =============================================================

Custom backbones can be registered with :func:`register_backbone` or passed
directly as a boolean mask / edge-id array.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from .graph import Graph
from .kernels import (
    build_csr,
    counting_order,
    spanning_forest_mask,
    strided_spanning_forest_mask,
)

__all__ = [
    "DEFAULT_BACKBONE",
    "available_backbones",
    "build_backbone",
    "canonical_backbone_name",
    "register_backbone",
]

DEFAULT_BACKBONE = "fast-maxst"
DEFAULT_BUCKETS = 256
MAX_BUCKETS = 65_536

_ALIASES = {
    "fast_maxst": "fast-maxst",
    "fastmaxst": "fast-maxst",
    "fast_mst": "fast-mst",
    "fast_minst": "fast-mst",
    "fastmst": "fast-mst",
    "minst": "mst",
    "max_st": "maxst",
    "min_st": "mst",
    "rand_st": "randst",
    "fast_randst": "fast-randst",
    "fastrandst": "fast-randst",
    "random": "randst",
    "bfs": "spt",
    "shortest_path": "spt",
    "low_stretch": "glst",
    "empty": "none",
    "": "none",
}

_REGISTRY: Dict[str, Callable] = {}


def canonical_backbone_name(name) -> str:
    """Normalize user-facing spellings (``fast_maxst``, ``FAST-MAXST``, ...)."""
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower().replace(" ", "")
    normalized = _ALIASES.get(normalized, normalized)
    normalized = _ALIASES.get(normalized.replace("-", "_"), normalized)
    return normalized


def available_backbones():
    """Sorted list of registered backbone names."""
    return sorted(_REGISTRY)


def register_backbone(name: str, builder: Callable):
    """Register a custom backbone.

    ``builder(graph, max_edges, seed, **options) -> boolean mask`` over
    ``graph.edge_index``. The mask must select an acyclic edge set; SCAFFOLD
    does not verify this, and a cyclic "forest" silently corrupts the LCA-based
    scores.
    """
    _REGISTRY[canonical_backbone_name(name)] = builder
    return builder


def build_backbone(
    graph: Graph,
    backbone=DEFAULT_BACKBONE,
    max_edges: Optional[int] = None,
    seed=None,
    **options,
) -> np.ndarray:
    """Return a boolean mask over ``graph.edge_index`` selecting the backbone.

    ``backbone`` may be a registered name, a boolean mask, an array of edge
    ids, or a callable with the signature described in :func:`register_backbone`.
    ``max_edges`` caps the forest size; union-find stops as soon as that many
    acyclic edges have been accepted, so a partial forest is built directly
    rather than built in full and truncated.
    """
    m = graph.num_edges

    if isinstance(backbone, np.ndarray):
        if backbone.dtype == bool:
            if backbone.shape != (m,):
                raise ValueError(
                    f"backbone mask must have length {m}, got {backbone.shape}"
                )
            return np.ascontiguousarray(backbone)
        mask = np.zeros(m, dtype=bool)
        mask[np.asarray(backbone, dtype=np.int64)] = True
        return mask

    if callable(backbone):
        return _validated(backbone(graph, max_edges=max_edges, seed=seed, **options), m)

    name = canonical_backbone_name(backbone)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Available: {available_backbones()}"
        )
    return _validated(
        _REGISTRY[name](graph, max_edges=max_edges, seed=seed, **options), m
    )


def _validated(mask, num_edges) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.shape != (num_edges,):
        raise ValueError(
            f"backbone builder returned a mask of length {mask.shape[0]}, "
            f"expected {num_edges}"
        )
    return mask


def _limit(graph: Graph, max_edges: Optional[int]) -> int:
    """A spanning forest never exceeds ``n - 1`` edges; honour the budget too."""
    cap = max(0, graph.num_nodes - 1)
    if max_edges is None:
        return cap
    return max(0, min(cap, int(max_edges)))


# ----------------------------------------------------------------------
# weighted forests
# ----------------------------------------------------------------------
def _weighted_forest(graph: Graph, max_edges, maximum: bool) -> np.ndarray:
    """Exact stable Kruskal. Ties break on edge id, so runs are reproducible."""
    limit = _limit(graph, max_edges)
    if graph.num_edges == 0 or limit == 0:
        return np.zeros(graph.num_edges, dtype=bool)
    if graph.edge_weight is None:
        order = np.arange(graph.num_edges, dtype=np.int64)
    else:
        keys = -graph.edge_weight if maximum else graph.edge_weight
        order = np.argsort(keys, kind="stable").astype(np.int64, copy=False)
    return spanning_forest_mask(
        graph.num_nodes, graph.src, graph.dst, order, max_edges=limit
    )


def _fast_weighted_forest(
    graph: Graph, max_edges, maximum: bool, buckets: int = DEFAULT_BUCKETS
) -> np.ndarray:
    """Bucketed approximate Kruskal: linear-time ordering, same union-find scan.

    Edge weights are quantized into ``buckets`` priority classes, and edges
    inside a class keep input order. The result is not the exact MaxST, but on
    the graphs SCAFFOLD targets the difference is immaterial while the ordering
    cost drops from ``O(m log m)`` to ``O(m + buckets)``.
    """
    limit = _limit(graph, max_edges)
    if graph.num_edges == 0 or limit == 0:
        return np.zeros(graph.num_edges, dtype=bool)

    buckets = int(buckets)
    if not 2 <= buckets <= MAX_BUCKETS:
        raise ValueError(f"buckets must lie in [2, {MAX_BUCKETS}], got {buckets}")

    if graph.edge_weight is None:
        order = np.arange(graph.num_edges, dtype=np.int64)
    else:
        scores = graph.edge_weight
        low = float(scores.min())
        high = float(scores.max())
        if high - low <= 1e-20:
            order = np.arange(graph.num_edges, dtype=np.int64)
        else:
            keys = np.floor((scores - low) * ((buckets - 1) / (high - low)))
            keys = np.clip(keys, 0, buckets - 1).astype(np.int64)
            if maximum:
                keys = (buckets - 1) - keys
            order = counting_order(keys, buckets)
    return spanning_forest_mask(
        graph.num_nodes, graph.src, graph.dst, order, max_edges=limit
    )


def _random_forest(graph: Graph, max_edges, seed) -> np.ndarray:
    """Kruskal over a uniform random edge permutation."""
    limit = _limit(graph, max_edges)
    if graph.num_edges == 0 or limit == 0:
        return np.zeros(graph.num_edges, dtype=bool)
    rng = np.random.default_rng(seed)
    order = rng.permutation(graph.num_edges).astype(np.int64, copy=False)
    return spanning_forest_mask(
        graph.num_nodes, graph.src, graph.dst, order, max_edges=limit
    )


def _fast_random_forest(graph: Graph, max_edges, seed) -> np.ndarray:
    """Seeded strided random forest without allocating a full permutation."""
    limit = _limit(graph, max_edges)
    m = graph.num_edges
    if m == 0 or limit == 0:
        return np.zeros(m, dtype=bool)
    if m == 1:
        return spanning_forest_mask(
            graph.num_nodes,
            graph.src,
            graph.dst,
            np.zeros(1, dtype=np.int64),
            max_edges=limit,
        )

    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, m))
    stride = int(rng.integers(1, m))
    if stride % 2 == 0:
        stride += 1
    while math.gcd(stride, m) != 1:
        stride += 2
        if stride >= m:
            stride = 1
            break
    return strided_spanning_forest_mask(
        graph.num_nodes,
        graph.src,
        graph.dst,
        offset,
        stride,
        max_edges=limit,
    )


def _shortest_path_forest(graph: Graph, max_edges, seed) -> np.ndarray:
    """BFS forest rooted at the highest-degree node of each component."""
    limit = _limit(graph, max_edges)
    n, m = graph.num_nodes, graph.num_edges
    mask = np.zeros(m, dtype=bool)
    if m == 0 or limit == 0:
        return mask

    rowptr, col, eid = build_csr(n, graph.src, graph.dst)
    degree = np.diff(rowptr)
    # Highest degree first, ties by node id: deterministic root choice.
    roots = np.lexsort((np.arange(n), -degree))
    visited = np.zeros(n, dtype=bool)
    added = 0
    queue = np.empty(n, dtype=np.int64)

    for root in roots:
        if visited[root] or added >= limit:
            continue
        visited[root] = True
        head = tail = 0
        queue[tail] = root
        tail += 1
        while head < tail and added < limit:
            node = queue[head]
            head += 1
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = col[pos]
                if visited[nb]:
                    continue
                visited[nb] = True
                mask[eid[pos]] = True
                added += 1
                queue[tail] = nb
                tail += 1
                if added >= limit:
                    break
    return mask


def _greedy_low_stretch_forest(graph: Graph, max_edges, seed, eta=1.0, alpha=1.0):
    """Greedy low-stretch tree (GLST).

    Repeatedly adds the boundary edge maximizing
    ``(cut / cut_max) ** eta / ((projStretch / stretch_max) ** alpha)``, where
    ``cut[e]`` counts graph edges crossing the two components ``e`` would merge
    and ``projStretch[e]`` is their mean stretch once ``e`` is added.

    Quality comes at ``O(n * m * |cut|)`` cost. Guarded at 5000 edges; raise
    ``glst_max_edges_allowed`` if you really mean it.
    """
    import networkx as nx

    limit = _limit(graph, max_edges)
    m = graph.num_edges
    mask = np.zeros(m, dtype=bool)
    if m == 0 or limit == 0:
        return mask
    guard = int(getattr(graph, "_glst_guard", 5000))
    if m > guard:
        raise ValueError(
            f"backbone='glst' is O(n * m * |cut|) and this graph has {m:,} edges. "
            "Use 'fast-maxst' (the default) for anything beyond a few thousand "
            "edges, or pass a precomputed backbone mask."
        )

    weight = graph.weights_or_ones()
    eps = 1e-8
    comp = np.arange(graph.num_nodes, dtype=np.int64)
    tree = nx.Graph()
    tree.add_nodes_from(range(graph.num_nodes))
    added = 0

    while added < limit:
        boundary = np.flatnonzero(comp[graph.src] != comp[graph.dst])
        if boundary.size == 0:
            break

        pair_key = {}
        for i in boundary:
            key = (
                min(comp[graph.src[i]], comp[graph.dst[i]]),
                max(comp[graph.src[i]], comp[graph.dst[i]]),
            )
            pair_key.setdefault(key, []).append(int(i))

        cut_size = np.zeros(boundary.size, dtype=np.float64)
        stretch = np.zeros(boundary.size, dtype=np.float64)
        for slot, i in enumerate(boundary):
            key = (
                min(comp[graph.src[i]], comp[graph.dst[i]]),
                max(comp[graph.src[i]], comp[graph.dst[i]]),
            )
            cut = pair_key[key]
            cut_size[slot] = len(cut)
            u, v = int(graph.src[i]), int(graph.dst[i])
            tree.add_edge(u, v)
            try:
                total = 0.0
                for j in cut:
                    x, y = int(graph.src[j]), int(graph.dst[j])
                    try:
                        d = nx.shortest_path_length(tree, x, y)
                    except nx.NetworkXNoPath:
                        d = 0.0
                    total += d / max(float(weight[j]), eps)
                stretch[slot] = total / (len(cut) + eps)
            finally:
                tree.remove_edge(u, v)

        cut_max = cut_size.max() if cut_size.size else eps
        stretch_max = stretch.max() if stretch.size else eps
        numerator = ((cut_size + eps) / (cut_max + eps)) ** float(eta)
        denominator = ((stretch + eps) / (stretch_max + eps)) ** float(alpha)
        score = numerator / np.maximum(denominator, eps)

        best = boundary[int(np.lexsort((boundary, -score))[0])]
        u, v = int(graph.src[best]), int(graph.dst[best])
        tree.add_edge(u, v)
        mask[best] = True
        added += 1
        old, new = comp[u], comp[v]
        comp[comp == old] = new
    return mask


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
register_backbone(
    "fast-maxst",
    lambda graph, max_edges=None, seed=None, buckets=DEFAULT_BUCKETS, **_: (
        _fast_weighted_forest(graph, max_edges, maximum=True, buckets=buckets)
    ),
)
register_backbone(
    "fast-mst",
    lambda graph, max_edges=None, seed=None, buckets=DEFAULT_BUCKETS, **_: (
        _fast_weighted_forest(graph, max_edges, maximum=False, buckets=buckets)
    ),
)
register_backbone(
    "maxst",
    lambda graph, max_edges=None, seed=None, **_: _weighted_forest(
        graph, max_edges, maximum=True
    ),
)
register_backbone(
    "mst",
    lambda graph, max_edges=None, seed=None, **_: _weighted_forest(
        graph, max_edges, maximum=False
    ),
)
register_backbone(
    "randst",
    lambda graph, max_edges=None, seed=None, **_: _random_forest(graph, max_edges, seed),
)
register_backbone(
    "fast-randst",
    lambda graph, max_edges=None, seed=None, **_: (
        _fast_random_forest(graph, max_edges, seed)
    ),
)
register_backbone(
    "spt",
    lambda graph, max_edges=None, seed=None, **_: _shortest_path_forest(
        graph, max_edges, seed
    ),
)
register_backbone(
    "glst",
    lambda graph, max_edges=None, seed=None, eta=1.0, alpha=1.0, **_: (
        _greedy_low_stretch_forest(graph, max_edges, seed, eta=eta, alpha=alpha)
    ),
)
register_backbone(
    "none",
    lambda graph, max_edges=None, seed=None, **_: np.zeros(graph.num_edges, dtype=bool),
)
