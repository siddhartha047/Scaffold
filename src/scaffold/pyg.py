"""PyTorch Geometric integration.

Requires ``torch`` and ``torch-geometric``
(``pip install "scaffold-sparse[pyg]"``). Imported lazily by
``scaffold.pyg``, so plain SciPy/NetworkX users never pay for it.

Two things live here:

* :class:`ScaffoldTransform` -- a ``BaseTransform`` you can hand to a dataset,
  so sparsification happens in the data pipeline like any other transform;
* :class:`ScaffoldResampler` -- per-epoch resparsification for training loops,
  built on ``scaffold.sample``: one precompute, then a fresh view per epoch for
  the cost of a draw.

Both keep node features, labels and split masks intact; only the topology
changes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .api import sample as _sample
from .api import sparsify as _sparsify
from .graph import normalize_graph

__all__ = ["ScaffoldTransform", "ScaffoldResampler", "sparsify_data", "sample_edge_weight"]


def sparsify_data(
    data,
    method: str = "fast",
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    seed=None,
    **kwargs,
):
    """Sparsify a PyG ``Data`` object, returning a new ``Data``.

    Examples
    --------
    >>> from torch_geometric.datasets import Planetoid       # doctest: +SKIP
    >>> data = Planetoid("/tmp/Cora", "Cora")[0]             # doctest: +SKIP
    >>> from scaffold.pyg import sparsify_data               # doctest: +SKIP
    >>> sparse = sparsify_data(data, keep_ratio=0.6)         # doctest: +SKIP
    """
    result = _sparsify(
        data, method=method, keep_ratio=keep_ratio, num_edges=num_edges,
        seed=seed, **kwargs,
    )
    if method == "sample":
        result = result.draw(keep_ratio=keep_ratio, num_edges=num_edges, seed=seed)
    return result.to_pyg()


def sample_edge_weight(data, seed=None, **kwargs):
    """SCAFFOLD-Sample weights for a PyG ``Data``, as a torch tensor.

    Returns ``(edge_index, edge_weight)`` covering **all** edges of ``data``,
    symmetrized and aligned, ready to assign::

        data.edge_index, data.edge_weight = sample_edge_weight(data)

    Note that these are *sampling* weights, not message-passing coefficients:
    they say how much each edge matters, which is what drives the per-epoch
    draw. Feed them to a GNN as edge weights only if that is what you mean.
    """

    scores = _sample(data, seed=seed, **kwargs)
    edge_index, edge_weight = scores.to_torch()
    return edge_index, edge_weight


class ScaffoldTransform:
    """A PyG transform that sparsifies each ``Data`` it sees.

    Parameters
    ----------
    method:
        ``"greedy"``, ``"heap"``, ``"batch"``, ``"fast"`` (default), or
        ``"sample"``.
    keep_ratio, num_edges:
        Edge budget; exactly one.
    seed:
        Base seed. With ``resample=True`` the seed advances on every call, so
        each epoch sees a different view.
    resample:
        Re-run the sparsifier on every call rather than caching one result.
        For per-epoch resparsification prefer :class:`ScaffoldResampler`, which
        does the structural work once.

    Examples
    --------
    >>> from torch_geometric.datasets import Planetoid              # doctest: +SKIP
    >>> from scaffold.pyg import ScaffoldTransform                  # doctest: +SKIP
    >>> dataset = Planetoid("/tmp/Cora", "Cora",                    # doctest: +SKIP
    ...                     transform=ScaffoldTransform(keep_ratio=0.6))
    """

    def __init__(
        self,
        method: str = "fast",
        keep_ratio: Optional[float] = None,
        num_edges: Optional[int] = None,
        seed=None,
        resample: bool = False,
        **kwargs,
    ):
        self.method = method
        self.keep_ratio = keep_ratio
        self.num_edges = num_edges
        self.seed = seed
        self.resample = bool(resample)
        self.kwargs = kwargs
        self._calls = 0

    def __call__(self, data):
        seed = self.seed
        if self.resample and seed is not None:
            seed = int(seed) + self._calls
        self._calls += 1
        return sparsify_data(
            data,
            method=self.method,
            keep_ratio=self.keep_ratio,
            num_edges=self.num_edges,
            seed=seed,
            **self.kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        budget = (
            f"keep_ratio={self.keep_ratio}"
            if self.num_edges is None
            else f"num_edges={self.num_edges}"
        )
        return f"ScaffoldTransform(method={self.method!r}, {budget})"


class ScaffoldResampler:
    """Per-epoch resparsification for a GNN training loop.

    Runs the SCAFFOLD-Sample precompute once, then serves a fresh sparse view
    per epoch for the cost of a uniform draw plus a ``searchsorted``. Every
    view has exactly the requested edge count. When the budget is at or above
    :attr:`delta_min`, it also has exactly the input graph's components; below
    that mathematical floor, preserving them is impossible.

    Examples
    --------
    >>> from scaffold.pyg import ScaffoldResampler                  # doctest: +SKIP
    >>> resampler = ScaffoldResampler(data, keep_ratio=0.6, seed=0) # doctest: +SKIP
    >>> for epoch in range(200):                                    # doctest: +SKIP
    ...     view = resampler.epoch(epoch)
    ...     out = model(view.x, view.edge_index)
    """

    def __init__(
        self,
        data,
        keep_ratio: Optional[float] = None,
        num_edges: Optional[int] = None,
        seed=None,
        backbone: str = "rotate-randst",
        tree_count: int = 8,
        every: int = 1,
        **kwargs,
    ):
        self.data = data
        self.keep_ratio = keep_ratio
        self.num_edges = num_edges
        self.seed = seed
        self.every = max(1, int(every))
        self.graph = normalize_graph(data)
        self.scores = _sample(
            self.graph,
            keep_ratio=keep_ratio,
            num_edges=num_edges,
            seed=seed,
            backbone=backbone,
            tree_count=tree_count,
            **kwargs,
        )
        self._cache = None
        self._cache_epoch = None

    @property
    def edge_weight(self) -> np.ndarray:
        """The precomputed per-edge SCAFFOLD weights (one per undirected edge)."""
        return self.scores.scores

    @property
    def delta_min(self) -> float:
        """Connectivity floor: below this ``keep_ratio`` fragmentation is forced."""
        return float(self.scores.metadata["delta_min"])

    def coverage(self, epochs=(1, 10, 100, 500)):
        """Expected fraction of the graph seen after ``epochs`` draws."""
        return self.scores.sampler.coverage(epochs=epochs)

    def epoch(self, epoch: int):
        """The sparse ``Data`` for ``epoch``, redrawn every ``every`` epochs."""
        slot = int(epoch) // self.every
        if self._cache is None or self._cache_epoch != slot:
            result = self.scores.draw(
                keep_ratio=self.keep_ratio, num_edges=self.num_edges
            )
            self._cache = result.to_pyg()
            self._cache_epoch = slot
        return self._cache

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScaffoldResampler(num_nodes={self.graph.num_nodes}, "
            f"num_edges={self.graph.num_edges}, every={self.every})"
        )
