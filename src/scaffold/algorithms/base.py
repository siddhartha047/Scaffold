"""Shared plumbing for the four SCAFFOLD algorithms.

Each algorithm's ``run`` receives a normalized :class:`~scaffold.graph.Graph`
and returns a boolean mask over its canonical edges plus a metadata dict. The
budget arithmetic, backbone construction and mandatory-edge handling are done
once here so the four variants cannot drift apart on the parts that should be
identical.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..backbone import DEFAULT_BACKBONE, build_backbone
from ..graph import Graph
from ..kernels import component_count
from ..scoring import ScoreParams
from ..utils.validation import connectivity_floor, resolve_budget


class GrowthContext:
    """Everything the growth loops share: budget, backbone, bookkeeping."""

    def __init__(
        self,
        graph: Graph,
        keep_ratio: Optional[float] = None,
        num_edges: Optional[int] = None,
        backbone=DEFAULT_BACKBONE,
        params: Optional[ScoreParams] = None,
        seed=None,
        backbone_options=None,
        check_connectivity: bool = True,
    ):
        self.graph = graph
        self.params = params or ScoreParams()
        self.seed = seed
        self.started = time.perf_counter()

        self.target_edges = resolve_budget(graph.num_edges, keep_ratio, num_edges)

        # The backbone is capped at the budget: when the target is below n-1
        # there is no point building a full forest and throwing part of it away.
        self.backbone_name = backbone if isinstance(backbone, str) else "custom"
        self.backbone_mask = build_backbone(
            graph,
            backbone,
            max_edges=self.target_edges,
            seed=seed,
            **(backbone_options or {}),
        )
        self.mask = self.backbone_mask.copy()

        self.base_components = (
            component_count(graph.num_nodes, graph.src, graph.dst)
            if check_connectivity
            else None
        )
        self.delta_min = (
            connectivity_floor(graph.num_nodes, graph.num_edges, self.base_components)
            if self.base_components is not None
            else None
        )

    # -- state --------------------------------------------------------------
    @property
    def selected(self) -> int:
        return int(self.mask.sum())

    @property
    def remaining_budget(self) -> int:
        return max(0, self.target_edges - self.selected)

    def candidate_ids(self) -> np.ndarray:
        return np.flatnonzero(~self.mask).astype(np.int64, copy=False)

    def add(self, edge_ids) -> int:
        ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return 0
        fresh = ids[~self.mask[ids]]
        self.mask[fresh] = True
        return int(fresh.size)

    def trim_to_budget(self) -> int:
        """Drop uniformly at random down to the budget.

        Only reachable when the backbone alone exceeds the budget, i.e. when
        ``keep_ratio < delta_min`` and no sparsifier of any kind can keep the
        graph's components intact.
        """
        excess = self.selected - self.target_edges
        if excess <= 0:
            return 0
        ids = np.flatnonzero(self.mask)
        rng = np.random.default_rng(self.seed)
        drop = rng.permutation(ids)[:excess]
        self.mask[drop] = False
        return int(excess)

    # -- reporting ----------------------------------------------------------
    def finish(self, algorithm: str, **extra):
        elapsed = time.perf_counter() - self.started
        trimmed = self.trim_to_budget()
        metadata = {
            "method": algorithm,
            "backbone": self.backbone_name,
            "backbone_edges": int(self.backbone_mask.sum()),
            "target_edges": int(self.target_edges),
            "selected_edges": self.selected,
            "budget_trimmed": int(trimmed),
            "runtime": float(elapsed),
            "alpha": self.params.alpha,
            "beta_edge": self.params.beta_edge,
            "beta_node": self.params.beta_node,
        }
        if self.base_components is not None:
            spanning_edges = self.graph.num_nodes - self.base_components
            # Tested against the floor directly, not inferred from whether a
            # trim happened: the backbone is capped at the budget, so a run
            # below the floor builds a partial forest and never overshoots.
            metadata["base_components"] = int(self.base_components)
            metadata["delta_min"] = float(self.delta_min)
            metadata["below_connectivity_floor"] = bool(
                trimmed > 0 or self.target_edges < spanning_edges
            )
        metadata.update(extra)
        return self.mask, metadata


def take_top(scores: np.ndarray, candidate_ids: np.ndarray, limit: int) -> np.ndarray:
    """Highest ``limit`` candidates by score, ties broken on ascending edge id.

    Deterministic tie-breaking is not cosmetic: on a symmetric graph large
    blocks of candidates share a score *exactly*, and which of them the budget
    happens to cut through must not depend on array order.

    ``argpartition`` alone is not enough. It finds the k-th largest value in
    ``O(m)``, but among candidates *equal* to that value it keeps an arbitrary
    subset -- so the edges straddling the budget boundary would be chosen by
    memory layout rather than by id. The tie group at the cut is therefore
    resolved explicitly, which costs a sort of that group only.
    """
    limit = int(min(max(0, limit), candidate_ids.size))
    if limit == 0:
        return np.zeros(0, dtype=np.int64)
    values = scores[candidate_ids]
    if limit == candidate_ids.size:
        return candidate_ids[np.lexsort((candidate_ids, -values))]

    cut = float(values[np.argpartition(-values, limit - 1)[limit - 1]])
    above = values > cut
    winners = candidate_ids[above]
    shortfall = limit - int(winners.size)
    if shortfall > 0:
        tied = np.sort(candidate_ids[values == cut])[:shortfall]
        winners = np.concatenate((winners, tied))
    chosen_values = scores[winners]
    return winners[np.lexsort((winners, -chosen_values))]


def degree_weighted_sample(
    candidate_ids: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    degree: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample without replacement with ``P(e) proportional to 1/du + 1/dv``.

    Favouring edges at low-degree endpoints spreads the sample over the whole
    graph instead of piling it onto hubs, which is what keeps the sampled
    round loop from repeatedly proposing edges around the same few nodes.
    """
    if candidate_ids.size == 0:
        return candidate_ids
    size = int(min(size, candidate_ids.size))
    if size >= candidate_ids.size:
        return candidate_ids
    du = np.maximum(degree[src[candidate_ids]], 1)
    dv = np.maximum(degree[dst[candidate_ids]], 1)
    weights = 1.0 / du + 1.0 / dv
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.choice(candidate_ids, size=size, replace=False)
    # Efraimidis-Spirakis: one exponential key per item, take the smallest k.
    keys = rng.exponential(size=candidate_ids.size) / weights
    window = np.argpartition(keys, size - 1)[:size]
    return candidate_ids[np.sort(window)]


def selected_degrees(num_nodes: int, src, dst, mask) -> np.ndarray:
    """Degree of every node in the currently selected subgraph."""
    degree = np.zeros(int(num_nodes), dtype=np.int64)
    ids = np.flatnonzero(mask)
    if ids.size:
        np.add.at(degree, src[ids], 1)
        np.add.at(degree, dst[ids], 1)
    return degree
