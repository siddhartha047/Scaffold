"""The public entry points: ``exact``, ``heap``, ``fast``, ``sample``, ``sparsify``.

All five take any supported graph object, all five take the same budget
arguments, and the first four return the same :class:`~scaffold.result.ScaffoldResult`.
Swapping one for another is a one-word change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .algorithms import exact as _exact
from .algorithms import fast as _fast
from .algorithms import heap as _heap
from .algorithms import sample as _sample
from .backbone import DEFAULT_BACKBONE
from .graph import Graph, normalize_graph
from .result import ScaffoldResult, ScaffoldScores
from .scoring import ScoreParams
from .utils.validation import resolve_budget

__all__ = ["exact", "heap", "fast", "sample", "sparsify", "METHODS"]

_SCORE_KEYS = ("alpha", "beta_edge", "beta_node", "edge_norm_p", "node_norm_q")


def _resolve_keep_ratio(keep_ratio, target_ratio):
    """Accept the research code's ``target_ratio`` as a public API alias."""
    if target_ratio is None:
        return keep_ratio
    if keep_ratio is not None:
        raise ValueError("Specify either keep_ratio or target_ratio, not both.")
    return target_ratio


def _split_score_params(kwargs):
    """Pull the objective knobs out of ``**kwargs`` into a ``ScoreParams``."""
    params = kwargs.pop("params", None)
    overrides = {key: kwargs.pop(key) for key in _SCORE_KEYS if key in kwargs}
    if params is None:
        return ScoreParams(**overrides), kwargs
    if overrides:
        merged = params.as_dict()
        merged.update(overrides)
        return ScoreParams(**merged), kwargs
    return params, kwargs


def _prepare(G, kwargs):
    graph = G if isinstance(G, Graph) else normalize_graph(G)
    params, kwargs = _split_score_params(kwargs)
    return graph, params, kwargs


def _build_result(graph: Graph, method: str, mask, metadata) -> ScaffoldResult:
    mask = np.ascontiguousarray(mask, dtype=bool)
    edge_index = graph.subgraph_edge_index(mask)
    weight = None if graph.edge_weight is None else graph.edge_weight[mask]
    return ScaffoldResult(
        graph=graph,
        method=method,
        mask=mask,
        undirected_edge_index=edge_index,
        undirected_edge_weight=weight,
        metadata=metadata,
    )


# ----------------------------------------------------------------------
# the four algorithms
# ----------------------------------------------------------------------
def exact(
    G,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    seed=None,
    target_ratio: Optional[float] = None,
    **kwargs,
) -> ScaffoldResult:
    """SCAFFOLD-Exact -- reference greedy, rescoring everything every round.

    The ground truth the other variants are measured against. ``O(M * m * (n + m))``:
    correct on any graph, practical only on small ones.

    Parameters
    ----------
    G:
        Any supported graph (NetworkX, PyG ``Data``, ``scipy.sparse``,
        ``(2, m)`` edge_index, or a :class:`~scaffold.graph.Graph`).
    keep_ratio:
        Fraction of undirected edges to retain; the budget is
        ``ceil(keep_ratio * m)``. Mutually exclusive with ``target_ratio`` and
        ``num_edges``.
    target_ratio:
        Alias for ``keep_ratio``, matching the research configuration name.
    num_edges:
        Exact undirected edge budget.
    backbone:
        Support backbone; see :mod:`scaffold.backbone`. Default ``"fast-maxst"``.
    seed:
        Seed for any stochastic component (random backbones, tie-breaking
        fallbacks). Never touches the global numpy RNG.
    **kwargs:
        Objective knobs (``alpha``, ``beta_edge``, ``beta_node``,
        ``edge_norm_p``, ``node_norm_q``) and algorithm options
        (``batch_size``, ``max_rounds``, ``verbose``).

    Examples
    --------
    >>> import networkx as nx, scaffold
    >>> result = scaffold.exact(nx.karate_club_graph(), keep_ratio=0.5)
    >>> result.sparse_edges <= result.original_edges
    True
    """
    keep_ratio = _resolve_keep_ratio(keep_ratio, target_ratio)
    graph, params, kwargs = _prepare(G, kwargs)
    mask, metadata = _exact.run(
        graph, keep_ratio=keep_ratio, num_edges=num_edges,
        backbone=backbone, params=params, seed=seed, **kwargs,
    )
    return _build_result(graph, "exact", mask, metadata)


def heap(
    G,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    seed=None,
    target_ratio: Optional[float] = None,
    **kwargs,
) -> ScaffoldResult:
    """SCAFFOLD-Heap -- lazy top-k greedy with local invalidation.

    Tracks Exact closely at a fraction of the cost by keeping stale scores in a
    heap and rescoring only the candidates an insertion actually affected.
    The practical choice for mid-sized graphs.

    Extra options: ``top_k``, ``add_per_round``, ``clusters``,
    ``cluster_method``, ``local_radius``, ``dirty_limit``, ``score_form``.
    """
    keep_ratio = _resolve_keep_ratio(keep_ratio, target_ratio)
    graph, params, kwargs = _prepare(G, kwargs)
    mask, metadata = _heap.run(
        graph, keep_ratio=keep_ratio, num_edges=num_edges,
        backbone=backbone, params=params, seed=seed, **kwargs,
    )
    return _build_result(graph, "heap", mask, metadata)


def fast(
    G,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    seed=None,
    target_ratio: Optional[float] = None,
    **kwargs,
) -> ScaffoldResult:
    """SCAFFOLD-Fast -- exact scores in one ``O(m log n + n)`` tree-prefix pass.

    Scores every candidate against the backbone forest using LCA plus
    root-prefix sums -- no shortest-path search anywhere -- then takes the top
    of the budget. **Use this one** unless you have a specific reason not to.

    Extra options: ``selection`` (``"topk"`` or ``"rounds"``), ``clusters``,
    ``sample_size``, ``add_per_round``, ``weighted_paths``, ``return_scores``.

    Examples
    --------
    >>> import scaffold
    >>> G = scaffold.grid_graph(8, 8)
    >>> result = scaffold.fast(G, keep_ratio=0.5)
    >>> result.num_components() == 1
    True
    """
    keep_ratio = _resolve_keep_ratio(keep_ratio, target_ratio)
    graph, params, kwargs = _prepare(G, kwargs)
    mask, metadata = _fast.run(
        graph, keep_ratio=keep_ratio, num_edges=num_edges,
        backbone=backbone, params=params, seed=seed, **kwargs,
    )
    return _build_result(graph, "fast", mask, metadata)


def sample(
    G,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    seed=None,
    target_ratio: Optional[float] = None,
    **kwargs,
) -> ScaffoldScores:
    """SCAFFOLD-Sample -- per-edge weights for sampling, not a fixed subgraph.

    Unlike the other three, this does **not** choose a subgraph. It scores
    every edge once and returns those weights, so you can draw a fresh sparse
    view every training epoch at the cost of one uniform draw plus one
    ``searchsorted``.

    ``keep_ratio`` / ``num_edges`` are optional here and only set the default
    budget for :meth:`~scaffold.result.ScaffoldScores.draw` and
    :meth:`~scaffold.result.ScaffoldScores.inclusion_probabilities`.

    Extra options: ``tree_count`` (number of random forests to aggregate over),
    ``aggregate_lambda`` (score-vs-frequency mix), ``backbone``
    (``"fixed-maxst"``, ``"fixed-randst"``, or ``"rotate-randst"``),
    ``weighted_paths``.

    Examples
    --------
    >>> import scaffold
    >>> scores = scaffold.sample(scaffold.grid_graph(8, 8), seed=0)
    >>> weights = scores.edge_weight       # one weight per (directed) edge
    >>> view = scores.draw(keep_ratio=0.5)  # a concrete sparse graph
    """
    keep_ratio = _resolve_keep_ratio(keep_ratio, target_ratio)
    graph, params, kwargs = _prepare(G, kwargs)
    # Validated here even though no subgraph is produced: the budget is
    # remembered as the default for draw() / inclusion_probabilities(), and a
    # contradictory pair should fail now, not three calls later.
    resolve_budget(graph.num_edges, keep_ratio, num_edges)
    sampler = _sample.run(graph, params=params, seed=seed, **kwargs)

    weight = sampler.pi
    metadata = {
        "method": "sample",
        "tree_count": int(sampler.tree_count),
        "aggregate_lambda": float(sampler.aggregate_lambda),
        "backbone": sampler.backbone,
        "base_components": int(sampler.base_components),
        "delta_min": float(sampler.delta_min),
        "mandatory_edges": int(sampler.mandatory.sum()),
        "backbone_edges": int(sampler.det_forest.sum()),
        "runtime": float(sampler.build_seconds),
        "default_keep_ratio": keep_ratio,
        "default_num_edges": num_edges,
    }
    if keep_ratio is not None or num_edges is not None:
        sampler._default_budget = (keep_ratio, num_edges)

    return ScaffoldScores(
        graph=graph,
        method="sample",
        mask=np.ones(graph.num_edges, dtype=bool),
        undirected_edge_index=graph.edge_index,
        undirected_edge_weight=weight,
        metadata=metadata,
        scores=weight,
        mandatory=sampler.mandatory,
        backbone=sampler.det_forest,
        order=sampler.order,
        sampler=_BudgetBoundSampler(sampler, keep_ratio, num_edges),
    )


class _BudgetBoundSampler:
    """Remembers the budget passed to ``scaffold.sample`` as the default."""

    def __init__(self, sampler, keep_ratio, num_edges):
        self.sampler = sampler
        self.keep_ratio = keep_ratio
        self.num_edges = num_edges

    def _budget(self, keep_ratio, num_edges):
        if keep_ratio is None and num_edges is None:
            return self.keep_ratio, self.num_edges
        return keep_ratio, num_edges

    def inclusion_probabilities(self, keep_ratio=None, num_edges=None):
        ratio, count = self._budget(keep_ratio, num_edges)
        return self.sampler.inclusion_probabilities(keep_ratio=ratio, num_edges=count)

    def draw(self, keep_ratio=None, num_edges=None, seed=None):
        ratio, count = self._budget(keep_ratio, num_edges)
        return self.sampler.draw(keep_ratio=ratio, num_edges=count, seed=seed)

    def coverage(self, keep_ratio=None, num_edges=None, epochs=(1, 10, 100, 500)):
        ratio, count = self._budget(keep_ratio, num_edges)
        return self.sampler.coverage(keep_ratio=ratio, num_edges=count, epochs=epochs)

    def __getattr__(self, name):
        return getattr(self.sampler, name)


# ----------------------------------------------------------------------
# unified entry point
# ----------------------------------------------------------------------
METHODS = {
    "exact": exact,
    "heap": heap,
    "fast": fast,
    "sample": sample,
}


def sparsify(
    G,
    method: str = "fast",
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    seed=None,
    target_ratio: Optional[float] = None,
    **kwargs,
):
    """Run any SCAFFOLD variant by name.

    ``method`` is one of ``"exact"``, ``"heap"``, ``"fast"`` (default) or
    ``"sample"``. Leading ``"scaffold-"`` / ``"scaffold_"`` prefixes are
    accepted, so config strings from the research code work unchanged.

    Examples
    --------
    >>> import scaffold
    >>> G = scaffold.grid_graph(6, 6)
    >>> for name in ("exact", "heap", "fast"):
    ...     r = scaffold.sparsify(G, method=name, keep_ratio=0.5)
    ...     print(name, r.sparse_edges)
    exact 30
    heap 30
    fast 30
    """
    keep_ratio = _resolve_keep_ratio(keep_ratio, target_ratio)
    key = str(method).strip().lower().replace("_", "-")
    for prefix in ("scaffold-", "scaffold"):
        if key.startswith(prefix):
            key = key[len(prefix) :].lstrip("-")
            break
    if key not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {sorted(METHODS)}."
        )
    return METHODS[key](
        G, keep_ratio=keep_ratio, num_edges=num_edges, seed=seed, **kwargs
    )
