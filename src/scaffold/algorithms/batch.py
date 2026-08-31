r"""SCAFFOLD-Batch: sample candidates, score the batch, keep its top edges.

This is the original sampled-growth variant that preceded
:mod:`scaffold.algorithms.fast`.  Unlike Fast, it does not score the complete
candidate set once.  Each round it:

1. draws a degree-weighted candidate batch in every cluster;
2. evaluates dilation, edge congestion, and node congestion against the
   current support graph, using only that batch as the congestion population;
3. commits the best ``add_per_round`` candidates per cluster; and
4. updates the support graph before drawing the next batch.

The repeated shortest-path evaluations make Batch slower than Fast, but its
sampling spreads additions through the graph and its scores can react to edges
added in earlier rounds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backbone import DEFAULT_BACKBONE
from ..clustering import assign_clusters, cluster_edges
from ..graph import Graph
from ..scoring import PathScorer, ScoreParams
from ..utils.workers import resolve_workers
from .base import GrowthContext, degree_weighted_sample, selected_degrees


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    params: Optional[ScoreParams] = None,
    seed=None,
    clusters=None,
    cluster_method: str = "bfs",
    sample_size: int = 64,
    add_per_round: int = 8,
    backbone_options=None,
    workers=None,
    verbose: bool = False,
):
    """Grow a support graph with sampled-batch dilation/congestion scoring.

    ``sample_size`` candidates are drawn per cluster and scored together;
    only the best ``add_per_round`` are added.  Keep
    ``add_per_round < sample_size`` so the score actually affects selection.
    """
    sample_size = int(sample_size)
    add_per_round = int(add_per_round)
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    if add_per_round < 1:
        raise ValueError("add_per_round must be at least 1")
    if add_per_round >= sample_size:
        raise ValueError(
            "add_per_round must be smaller than sample_size; otherwise the "
            "whole sampled batch is committed and scoring has no effect"
        )

    workers = resolve_workers(workers)
    ctx = GrowthContext(
        graph,
        keep_ratio=keep_ratio,
        num_edges=num_edges,
        backbone=backbone,
        params=params,
        seed=seed,
        backbone_options=backbone_options,
    )
    if graph.num_edges == 0 or ctx.remaining_budget <= 0:
        return ctx.finish(
            "batch",
            rounds=0,
            sampled_candidates=0,
            scored_candidates=0,
            sample_size=sample_size,
            add_per_round=add_per_round,
            score_scope="sampled_batch",
            workers=workers,
        )

    node_labels = assign_clusters(graph, clusters, method=cluster_method, seed=seed)
    edge_cluster = cluster_edges(graph, node_labels)
    cluster_ids = np.unique(edge_cluster)
    degree = selected_degrees(graph.num_nodes, graph.src, graph.dst, ctx.mask)
    scorer = PathScorer(
        graph.num_nodes,
        graph.src,
        graph.dst,
        weight=graph.edge_weight,
        support_mask=ctx.mask.copy(),
        workers=workers,
    )

    rounds = 0
    sampled_total = 0
    scored_total = 0
    mandatory_total = 0
    while ctx.remaining_budget > 0:
        rounds += 1
        rng = np.random.default_rng(
            None if seed is None else np.random.SeedSequence([int(seed), rounds])
        )
        selected = []
        for cid in cluster_ids:
            if len(selected) >= ctx.remaining_budget:
                break
            pool = np.flatnonzero((edge_cluster == cid) & ~ctx.mask)
            if pool.size == 0:
                continue
            batch = degree_weighted_sample(
                pool, graph.src, graph.dst, degree, sample_size, rng
            )
            # Only the scores are read here; skip building the path lists.
            metrics = scorer.evaluate(batch, ctx.params, need_paths=False)
            sampled_total += int(batch.size)
            scored_total += int(batch.size)
            mandatory_total += int(metrics["mandatory"].sum())

            # Scores are local to this batch.  Sort by descending score with a
            # stable global-edge-id tie break, matching the other variants.
            order = np.lexsort((batch, -metrics["score"]))
            limit = min(
                add_per_round,
                int(batch.size),
                ctx.remaining_budget - len(selected),
            )
            selected.extend(batch[order[:limit]].tolist())

        if not selected:
            break
        chosen = np.asarray(selected, dtype=np.int64)
        added = ctx.add(chosen)
        if added == 0:
            break
        scorer.add_edges(chosen)
        np.add.at(degree, graph.src[chosen], 1)
        np.add.at(degree, graph.dst[chosen], 1)
        if verbose:
            print(
                f"[scaffold.batch] round={rounds} sampled={sampled_total} "
                f"added={added} edges={ctx.selected}/{ctx.target_edges}"
            )

    return ctx.finish(
        "batch",
        rounds=rounds,
        clusters=int(cluster_ids.size),
        cluster_method=cluster_method,
        sample_size=sample_size,
        add_per_round=add_per_round,
        sampled_candidates=sampled_total,
        scored_candidates=scored_total,
        mandatory_candidates=mandatory_total,
        score_scope="sampled_batch",
        selection="sampled_topk",
        workers=workers,
    )


__all__ = ["run"]
