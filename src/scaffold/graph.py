"""The internal graph representation every SCAFFOLD algorithm runs on.

The algorithms never see NetworkX, PyG or SciPy objects: :func:`normalize_graph`
converts whatever the caller passed into a single :class:`Graph`, and the
adapters in :mod:`scaffold.adapters` convert back. That keeps one copy of each
algorithm and keeps the ecosystem glue out of the hot loops.

A :class:`Graph` is always **canonical**:

* undirected, stored once per edge as ``src < dst``,
* self loops removed,
* duplicate pairs collapsed,
* edges sorted by ``(src, dst)``.

Canonical order matters beyond tidiness: SCAFFOLD-Sample's cached artifacts are
indexed positionally against it, so two runs over the same graph must produce
the same edge sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Graph:
    """A canonical undirected graph on ``0..num_nodes-1``.

    Attributes
    ----------
    num_nodes:
        Node count. Isolated nodes are kept, which is what makes the sparsified
        output line up with a feature matrix.
    edge_index:
        ``(2, m)`` int64 array of canonical undirected edges, ``src < dst``.
    edge_weight:
        Optional ``(m,)`` float64 array. ``None`` means "unweighted", which is
        materially different from "all ones": the unweighted path length is a
        hop count, so the fast hop-based kernels can be used.
    node_labels:
        Original node identifiers when the input used them (NetworkX). Index
        ``i`` of this list is internal node ``i``.
    source:
        The object the user passed in, kept so results can be converted back
        without asking the caller to hold on to it.
    """

    num_nodes: int
    edge_index: np.ndarray
    edge_weight: Optional[np.ndarray] = None
    node_labels: Optional[List[Any]] = None
    source: Any = field(default=None, repr=False)

    def __post_init__(self):
        self.num_nodes = int(self.num_nodes)
        self.edge_index = np.ascontiguousarray(self.edge_index, dtype=np.int64)
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, m), got {self.edge_index.shape}"
            )
        if self.edge_weight is not None:
            self.edge_weight = np.ascontiguousarray(self.edge_weight, dtype=np.float64)
            if self.edge_weight.shape != (self.edge_index.shape[1],):
                raise ValueError(
                    "edge_weight must have one entry per undirected edge, got "
                    f"{self.edge_weight.shape} for {self.edge_index.shape[1]} edges"
                )

    # -- convenience views --------------------------------------------------
    @property
    def num_edges(self) -> int:
        """Number of *undirected* edges."""
        return int(self.edge_index.shape[1])

    @property
    def src(self) -> np.ndarray:
        return self.edge_index[0]

    @property
    def dst(self) -> np.ndarray:
        return self.edge_index[1]

    @property
    def is_weighted(self) -> bool:
        return self.edge_weight is not None

    def weights_or_ones(self) -> np.ndarray:
        """Edge weights, substituting ones when the graph is unweighted."""
        if self.edge_weight is None:
            return np.ones(self.num_edges, dtype=np.float64)
        return self.edge_weight

    def with_weight(self, weight: Optional[np.ndarray]) -> Graph:
        """Copy carrying different edge weights (same topology and labels)."""
        return Graph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index,
            edge_weight=None if weight is None else np.asarray(weight, dtype=np.float64),
            node_labels=self.node_labels,
            source=self.source,
        )

    def subgraph_edge_index(self, mask_or_ids) -> np.ndarray:
        """Canonical ``(2, k)`` edge_index for a boolean mask or an id array."""
        selector = np.asarray(mask_or_ids)
        if selector.dtype == bool:
            return np.ascontiguousarray(self.edge_index[:, selector])
        return np.ascontiguousarray(self.edge_index[:, selector.astype(np.int64)])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "weighted" if self.is_weighted else "unweighted"
        return (
            f"Graph(num_nodes={self.num_nodes}, num_edges={self.num_edges}, {kind})"
        )


def canonicalize(
    num_nodes: int,
    src,
    dst,
    weight=None,
    reduce: str = "first",
):
    """Return the canonical ``(src, dst, weight)`` triple for an edge list.

    Applies, in order: self-loop removal, ``min/max`` orientation, a stable
    sort by ``(src, dst)``, and duplicate collapse. ``reduce`` selects how
    weights of duplicated pairs combine -- ``"first"`` (default), ``"sum"``,
    ``"mean"``, ``"min"`` or ``"max"``.
    """
    num_nodes = int(num_nodes)
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same length")
    if src.size and (src.min() < 0 or dst.min() < 0):
        raise ValueError("node ids must be non-negative")
    if src.size and (src.max() >= num_nodes or dst.max() >= num_nodes):
        raise ValueError(
            f"edge endpoints reference a node id >= num_nodes ({num_nodes})"
        )

    weight = None if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
    if weight is not None and weight.shape != src.shape:
        raise ValueError("weight must have one entry per input edge")

    keep = src != dst
    src, dst = src[keep], dst[keep]
    if weight is not None:
        weight = weight[keep]

    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)

    if lo.size == 0:
        empty_w = None if weight is None else np.zeros(0, dtype=np.float64)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty_w

    # One int64 key per pair: ~50x faster than np.unique(..., axis=1).
    codes = lo * np.int64(num_nodes) + hi
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    lo, hi = lo[order], hi[order]
    if weight is not None:
        weight = weight[order]

    first = np.empty(codes.shape, dtype=bool)
    first[0] = True
    np.not_equal(codes[1:], codes[:-1], out=first[1:])

    out_src = np.ascontiguousarray(lo[first])
    out_dst = np.ascontiguousarray(hi[first])
    if weight is None:
        return out_src, out_dst, None

    group = np.cumsum(first) - 1
    ngroups = int(first.sum())
    if reduce == "first":
        out_w = weight[first]
    elif reduce == "sum":
        out_w = np.bincount(group, weights=weight, minlength=ngroups)
    elif reduce == "mean":
        totals = np.bincount(group, weights=weight, minlength=ngroups)
        counts = np.bincount(group, minlength=ngroups)
        out_w = totals / np.maximum(counts, 1)
    elif reduce == "min":
        out_w = np.full(ngroups, np.inf)
        np.minimum.at(out_w, group, weight)
    elif reduce == "max":
        out_w = np.full(ngroups, -np.inf)
        np.maximum.at(out_w, group, weight)
    else:
        raise ValueError(
            f"reduce must be one of 'first', 'sum', 'mean', 'min', 'max'; got {reduce!r}"
        )
    return out_src, out_dst, np.ascontiguousarray(out_w, dtype=np.float64)


def from_edge_index(
    edge_index,
    num_nodes: Optional[int] = None,
    edge_weight=None,
    reduce: str = "first",
    source: Any = None,
    node_labels: Optional[List[Any]] = None,
) -> Graph:
    """Build a :class:`Graph` from a ``(2, m)`` edge_index (numpy or torch)."""
    edge_index = _to_numpy(edge_index)
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2-D, got shape {edge_index.shape}")
    if edge_index.shape[0] != 2:
        if edge_index.shape[1] == 2:  # tolerate the (m, 2) edge-list convention
            edge_index = edge_index.T
        else:
            raise ValueError(
                f"edge_index must have shape (2, m) or (m, 2), got {edge_index.shape}"
            )
    src, dst = edge_index[0], edge_index[1]
    if num_nodes is None:
        num_nodes = int(max(src.max(), dst.max())) + 1 if src.size else 0
    weight = None if edge_weight is None else _to_numpy(edge_weight).reshape(-1)
    csrc, cdst, cweight = canonicalize(num_nodes, src, dst, weight, reduce=reduce)
    return Graph(
        num_nodes=num_nodes,
        edge_index=np.stack((csrc, cdst)),
        edge_weight=cweight,
        node_labels=node_labels,
        source=source,
    )


def _to_numpy(array):
    """Best-effort conversion of torch tensors / array-likes to numpy."""
    if hasattr(array, "detach"):  # torch.Tensor
        array = array.detach().cpu()
    if hasattr(array, "numpy") and not isinstance(array, np.ndarray):
        array = array.numpy()
    return np.asarray(array)


def normalize_graph(G, num_nodes: Optional[int] = None, reduce: str = "first") -> Graph:
    """Convert any supported graph object into a canonical :class:`Graph`.

    Accepts :class:`Graph`, ``networkx.Graph``/``DiGraph``,
    ``torch_geometric.data.Data``, any ``scipy.sparse`` matrix, a ``(2, m)``
    ``edge_index`` (numpy or torch), or a ``(dense adjacency)`` 2-D array.
    """
    from .adapters import to_graph  # local import keeps optional deps lazy

    return to_graph(G, num_nodes=num_nodes, reduce=reduce)
