"""SCAFFOLD-Exact: the reference greedy loop.

The literal definition of the method. Starting from the support backbone,
every remaining edge is scored against the *current* support graph, the single
best one is added, and the whole candidate set is rescored. Nothing is cached,
sampled or approximated, so this is the ground truth the other three variants
are measured against.

Cost is ``O(M * m * (n + m))`` for a budget of ``M`` edges -- fine for a few
thousand edges, hopeless beyond that. Use it on toy graphs, on the grid demo,
and to validate the faster variants; use :mod:`~scaffold.algorithms.fast` for
real work.

Rescoring after every insertion is the point, not an inefficiency: once an edge
is added, the paths of nearby candidates get shorter and their congestion
shifts, so a stale ranking is a different algorithm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backbone import DEFAULT_BACKBONE
from ..graph import Graph
from ..scoring import PathScorer, ScoreParams
from .base import GrowthContext


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    params: Optional[ScoreParams] = None,
    seed=None,
    batch_size: int = 1,
    backbone_options=None,
    max_rounds: Optional[int] = None,
    verbose: bool = False,
):
    """Grow the support one (or ``batch_size``) best-scoring edge at a time.

    Parameters
    ----------
    batch_size:
        Edges added per rescoring round. ``1`` is the true greedy algorithm;
        larger values trade fidelity for a proportional speed-up, since the
        expensive part is the rescoring, not the insertion.
    max_rounds:
        Optional cap on rescoring rounds, as a safety valve on larger graphs.
    """
    ctx = GrowthContext(
        graph,
        keep_ratio=keep_ratio,
        num_edges=num_edges,
        backbone=backbone,
        params=params,
        seed=seed,
        backbone_options=backbone_options,
    )
    batch_size = max(1, int(batch_size))

    scorer = PathScorer(
        graph.num_nodes,
        graph.src,
        graph.dst,
        weight=graph.edge_weight,
        support_mask=ctx.mask.copy(),
    )

    rounds = 0
    scored_total = 0
    while ctx.remaining_budget > 0:
        if max_rounds is not None and rounds >= max_rounds:
            break
        candidates = ctx.candidate_ids()
        if candidates.size == 0:
            break

        metrics = scorer.evaluate(candidates, ctx.params)
        rounds += 1
        scored_total += int(candidates.size)

        limit = min(batch_size, ctx.remaining_budget, candidates.size)
        # Mandatory (cross-component) candidates carry +inf and sort first,
        # which is exactly the priority they should have.
        order = np.lexsort((candidates, -metrics["score"]))
        chosen = candidates[order[:limit]]

        ctx.add(chosen)
        scorer.add_edges(chosen)

        if verbose:
            best = order[0]
            print(
                f"[scaffold.exact] round={rounds} added={chosen.size} "
                f"score={metrics['score'][best]:.6g} "
                f"dil={metrics['dil'][best]:.4g} "
                f"edges={ctx.selected}/{ctx.target_edges}"
            )

    return ctx.finish(
        "exact",
        rounds=rounds,
        scored_candidates=scored_total,
        batch_size=batch_size,
    )
