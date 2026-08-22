"""The object every SCAFFOLD entry point returns.

One result type for all four algorithms, so swapping ``scaffold.fast`` for
``scaffold.heap`` is a one-word change. Conversions live here rather than
happening implicitly, because on a graph with hundreds of millions of edges a
hidden ``to_networkx()`` is not a detail.

Edge-count conventions -- worth reading once:

* ``edge_index`` / ``edge_weight`` are **symmetric** (both directions present),
  because that is what a GNN consumes. ``edge_index.shape[1] == 2 * sparse_edges``.
* ``original_edges``, ``sparse_edges`` and ``keep_ratio`` count **undirected**
  edges, because that is what a sparsification budget means.
* ``mask`` is a boolean over the input graph's canonical undirected edges and
  is the most useful primitive if you want to do something else entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .graph import Graph


@dataclass
class ScaffoldResult:
    """Result of a sparsification (or, for ``sample``, of a scoring pass)."""

    graph: Graph = field(repr=False)
    method: str
    mask: np.ndarray = field(repr=False)
    undirected_edge_index: np.ndarray = field(repr=False)
    undirected_edge_weight: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.mask = np.ascontiguousarray(self.mask, dtype=bool)
        self.undirected_edge_index = np.ascontiguousarray(
            self.undirected_edge_index, dtype=np.int64
        )
        if self.undirected_edge_weight is not None:
            self.undirected_edge_weight = np.ascontiguousarray(
                self.undirected_edge_weight, dtype=np.float64
            )
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("method", self.method)
        self.metadata.setdefault("num_nodes", self.num_nodes)
        self.metadata.setdefault("original_edges", self.original_edges)
        self.metadata.setdefault("sparse_edges", self.sparse_edges)
        self.metadata.setdefault("keep_ratio", self.keep_ratio)

    # -- basic properties ---------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def original_edges(self) -> int:
        """Undirected edge count of the input graph."""
        return self.graph.num_edges

    @property
    def sparse_edges(self) -> int:
        """Undirected edge count of the result."""
        return int(self.undirected_edge_index.shape[1])

    @property
    def keep_ratio(self) -> float:
        if self.original_edges == 0:
            return 0.0
        return self.sparse_edges / self.original_edges

    @property
    def edge_ids(self) -> np.ndarray:
        """Indices of the kept edges into the input graph's canonical edge list."""
        return np.flatnonzero(self.mask)

    # -- GNN-ready views ----------------------------------------------------
    @property
    def edge_index(self) -> np.ndarray:
        """Symmetric ``(2, 2k)`` edge_index, ready to drop into a GNN."""
        src, dst = self.undirected_edge_index
        return np.ascontiguousarray(
            np.stack((np.concatenate((src, dst)), np.concatenate((dst, src))))
        )

    @property
    def edge_weight(self) -> Optional[np.ndarray]:
        """Symmetric ``(2k,)`` weights aligned with :attr:`edge_index`."""
        if self.undirected_edge_weight is None:
            return None
        return np.concatenate(
            (self.undirected_edge_weight, self.undirected_edge_weight)
        )

    # -- conversions --------------------------------------------------------
    def to_numpy(self):
        """``(edge_index, edge_weight)`` as symmetric numpy arrays."""
        return self.edge_index, self.edge_weight

    def to_scipy(self, fmt: str = "csr"):
        """Symmetric ``scipy.sparse`` adjacency matrix."""
        from .adapters import to_scipy

        return to_scipy(
            self.graph,
            self.undirected_edge_index,
            self.undirected_edge_weight,
            fmt=fmt,
        )

    def to_networkx(self, weight_key: str = "weight"):
        """``networkx.Graph`` with the original node labels restored."""
        from .adapters import to_networkx

        return to_networkx(
            self.graph,
            self.undirected_edge_index,
            self.undirected_edge_weight,
            weight_key=weight_key,
        )

    def to_pyg(self, copy_from: Any = None):
        """``torch_geometric.data.Data`` carrying the sparsified topology.

        Node features, labels and split masks are forwarded from the original
        ``Data`` object (or from ``copy_from``, if given).
        """
        from .adapters import to_pyg

        return to_pyg(
            self.graph,
            self.edge_index,
            self.edge_weight,
            copy_from=copy_from,
        )

    def to_torch(self):
        """``(edge_index, edge_weight)`` as torch tensors."""
        import torch

        edge_index = torch.as_tensor(self.edge_index, dtype=torch.long)
        weight = self.edge_weight
        if weight is None:
            return edge_index, None
        return edge_index, torch.as_tensor(weight, dtype=torch.float)

    # -- diagnostics --------------------------------------------------------
    def num_components(self) -> int:
        """Connected-component count of the sparsified graph."""
        from .kernels import component_count

        return component_count(
            self.num_nodes,
            self.undirected_edge_index[0],
            self.undirected_edge_index[1],
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"scaffold.{self.method}: {self.original_edges:,} -> "
            f"{self.sparse_edges:,} undirected edges "
            f"({self.keep_ratio:.1%} kept, {self.num_nodes:,} nodes)"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScaffoldResult(method={self.method!r}, num_nodes={self.num_nodes}, "
            f"original_edges={self.original_edges}, sparse_edges={self.sparse_edges}, "
            f"keep_ratio={self.keep_ratio:.4f})"
        )


@dataclass
class ScaffoldScores(ScaffoldResult):
    """Result of :func:`scaffold.sample` -- edge *weights*, not a subgraph.

    ``scaffold.sample`` does not choose a subgraph. It scores every edge of the
    input once, and those scores drive whatever sampling you want to do
    afterwards -- typically a fresh draw every training epoch.

    ``edge_index`` / ``edge_weight`` therefore cover **all** edges of the input,
    with ``edge_weight`` holding the SCAFFOLD weight ``pi``. Call :meth:`draw`
    to turn the weights into a concrete sparse graph.
    """

    scores: np.ndarray = field(default=None, repr=False)
    mandatory: np.ndarray = field(default=None, repr=False)
    backbone: np.ndarray = field(default=None, repr=False)
    order: np.ndarray = field(default=None, repr=False)
    sampler: Any = field(default=None, repr=False)

    @property
    def pi(self) -> np.ndarray:
        """Alias for :attr:`scores`, matching the paper's notation."""
        return self.scores

    def inclusion_probabilities(self, keep_ratio=None, num_edges=None) -> np.ndarray:
        """Per-edge inclusion probability for a budget; sums to the budget."""
        return self.sampler.inclusion_probabilities(
            keep_ratio=keep_ratio, num_edges=num_edges
        )

    def draw(self, keep_ratio=None, num_edges=None, seed=None) -> ScaffoldResult:
        """Draw one sparse graph from the weights (exact budget, exact marginals)."""
        return self.sampler.draw(
            keep_ratio=keep_ratio, num_edges=num_edges, seed=seed
        )

    def to_dict(self) -> Dict[Any, float]:
        """``{(u, v): weight}`` using the original node labels where available."""
        labels = self.graph.node_labels
        src, dst = self.undirected_edge_index
        if labels is None:
            keys = zip(src.tolist(), dst.tolist())
        else:
            keys = ((labels[u], labels[v]) for u, v in zip(src.tolist(), dst.tolist()))
        return {key: float(w) for key, w in zip(keys, self.scores)}

    def summary(self) -> str:
        return (
            f"scaffold.sample: {self.original_edges:,} edge weights "
            f"(mandatory={int(self.mandatory.sum()):,}, "
            f"backbone={int(self.backbone.sum()):,}, "
            f"delta_min={self.metadata.get('delta_min', float('nan')):.4f})"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScaffoldScores(num_nodes={self.num_nodes}, "
            f"num_edges={self.original_edges}, "
            f"delta_min={self.metadata.get('delta_min', float('nan')):.4f})"
        )
