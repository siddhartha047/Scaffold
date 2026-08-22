r"""SCAFFOLD-Fast: exact scores in one tree-prefix pass, no path search.

This is the variant to use on real graphs, and the one Algorithm 1 of the paper
describes (SCAFFOLD-Fast\ :sup:`+`).

The observation it rests on: while the support graph is still the backbone
forest ``F``, every term of the SCAFFOLD objective is a path aggregate on a
*tree*, and every path aggregate on a tree is a difference of two root-prefix
sums. So instead of running one shortest-path search per candidate, the whole
candidate set is scored together:

1. root ``F``, and build a binary-lifting ancestor table;
2. LCA of every candidate's endpoints, in ``O(log n)`` each;
3. edge and node congestion for **all** ``|I|`` paths at once, via
   "+1 at both endpoints, -2 at the LCA, then fold subtrees upward";
4. root-prefix sums of the congestion terms, making each path aggregate an
   ``O(1)`` difference.

Total: ``O(m log n + n)``, with no path ever enumerated. The scores are
*identical* to what SCAFFOLD-Exact computes in its first round -- the support
is the tree, so ``d_H == d_T``.

Selection
---------

``selection="topk"`` (default) is the "skip the loop" version: because the tree
index is never rebuilt during growth, the score is **static**, so the sampled
round loop was only ever approximating a ranking that can be taken exactly in
one pass. Mandatory (cross-component) edges carry ``+inf`` and are therefore
taken first, which is exactly the priority connectivity demands.

``selection="rounds"`` reproduces Algorithm 1 literally: per-cluster
degree-weighted sampling, top-``r`` per cluster per round, degrees updated as
edges land. It cannot beat ``topk`` on the static score -- both read the same
numbers -- but the selection it produces is spatially spread rather than
concentrated, and that matters more than the ranking does.

Why spreading matters
---------------------

The static score is computed against the backbone forest and never updated. So
when two candidates would repair the *same* region, the score cannot tell: it
rates both highly and ``topk`` takes both, when one would have done. On a graph
with large blocks of exactly-tied scores -- a lattice is the extreme case -- the
budget gets spent in a few neighbourhoods while the rest of the graph keeps its
long detours.

Measured on a 24x24 grid at ``keep_ratio=0.64``, evaluating the *result* with
:func:`~scaffold.scoring.path_scores`:

===================  =========  ==========  ===========
variant              time (ms)  mean dil.   max congest.
===================  =========  ==========  ===========
``exact``              1674         3.83         3.87
``topk``                  0.5      12.06        26.34
``rounds``                2.7      10.97        25.94
===================  =========  ==========  ===========

On less symmetric graphs the gap mostly closes -- on a random geometric graph
``topk`` gets 2.68 against ``exact``'s 2.40, and on Barabasi-Albert 3.82 against
3.66 -- so ``topk`` remains the default. Congestion is the term that suffers
most from concentration; if that is what you care about, use ``rounds``, or use
:mod:`~scaffold.algorithms.sample`, whose tree-locality ordering spreads by
construction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backbone import DEFAULT_BACKBONE
from ..clustering import assign_clusters, cluster_edges
from ..graph import Graph
from ..kernels import build_tree_index
from ..scoring import ScoreParams, tree_scores
from .base import GrowthContext, degree_weighted_sample, selected_degrees, take_top


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    params: Optional[ScoreParams] = None,
    seed=None,
    selection: str = "topk",
    clusters=None,
    cluster_method: str = "bfs",
    sample_size: int = 64,
    add_per_round: int = 8,
    weighted_paths: bool = False,
    backbone_options=None,
    return_scores: bool = False,
    verbose: bool = False,
):
    """Score every candidate once with the tree kernel, then select.

    Parameters
    ----------
    selection:
        ``"topk"`` -- one global top-k over the static scores (default).
        ``"rounds"`` -- the per-cluster sampled round loop of Algorithm 1.
    clusters, cluster_method, sample_size, add_per_round:
        Only used by ``selection="rounds"``. ``sample_size`` (``s``) candidates
        are drawn per cluster per round and the best ``add_per_round`` (``r``)
        of them are committed.

        **Keep ``r < s``.** With ``r >= s`` the top-r step commits the entire
        sample, so the score stops influencing the selection at all and the
        result is degree-weighted random sampling over the backbone. That is a
        legitimate baseline, but it is not this algorithm.
    weighted_paths:
        Measure the numerator of the dilation as a sum of tree edge weights
        rather than as a hop count. Only meaningful on weighted graphs.
    return_scores:
        Also return the per-edge score array in the metadata under
        ``"scores"``. Handy for plotting and for reusing one scoring pass
        across several budgets.
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
    if selection not in ("topk", "rounds"):
        raise ValueError("selection must be 'topk' or 'rounds'")

    if graph.num_edges == 0 or ctx.remaining_budget <= 0:
        return ctx.finish("fast", selection=selection, rounds=0, scored_candidates=0)

    # One scoring pass against the backbone forest. Never recomputed: growth
    # does not rebuild the tree index, so the score is static by construction.
    tree_index = build_tree_index(
        graph.num_nodes, graph.src[ctx.mask], graph.dst[ctx.mask]
    )
    scored = tree_scores(
        graph.num_nodes,
        graph.src,
        graph.dst,
        ctx.mask,
        weight=graph.edge_weight,
        params=ctx.params,
        tree_index=tree_index,
        weighted_paths=weighted_paths,
    )
    scores = scored["score"]
    candidates = ctx.candidate_ids()

    if selection == "topk":
        chosen = take_top(scores, candidates, ctx.remaining_budget)
        ctx.add(chosen)
        rounds = 1
        scored_total = int(candidates.size)
    else:
        rounds, scored_total = _grow_in_rounds(
            ctx,
            graph,
            scores,
            clusters=clusters,
            cluster_method=cluster_method,
            sample_size=sample_size,
            add_per_round=add_per_round,
            seed=seed,
            verbose=verbose,
        )

    extra = {
        "selection": selection,
        "rounds": rounds,
        "scored_candidates": scored_total,
        "mandatory_edges": int(scored["mandatory"].sum()),
        "total_stretch": float(scored["total_stretch"]),
        "weighted_paths": bool(weighted_paths),
    }
    if return_scores:
        extra["scores"] = scores
        extra["dilation"] = scored["dil"]
        extra["edge_congestion_path"] = scored["econ_path"]
        extra["node_congestion_path"] = scored["vcon_path"]
    if verbose:
        print(
            f"[scaffold.fast] selection={selection} scored={scored_total} "
            f"mandatory={extra['mandatory_edges']} "
            f"edges={ctx.selected}/{ctx.target_edges}"
        )
    return ctx.finish("fast", **extra)


def _grow_in_rounds(
    ctx: GrowthContext,
    graph: Graph,
    scores: np.ndarray,
    clusters,
    cluster_method: str,
    sample_size: int,
    add_per_round: int,
    seed,
    verbose: bool,
):
    """Algorithm 1's loop: sample per cluster, take top-r, repeat."""
    node_labels = assign_clusters(graph, clusters, method=cluster_method, seed=seed)
    edge_cluster = cluster_edges(graph, node_labels)
    cluster_ids = np.unique(edge_cluster)
    degree = selected_degrees(graph.num_nodes, graph.src, graph.dst, ctx.mask)
    sample_size = max(1, int(sample_size))
    add_per_round = max(1, int(add_per_round))

    rounds = 0
    scored_total = 0
    while ctx.remaining_budget > 0:
        rounds += 1
        rng = np.random.default_rng(
            None if seed is None else np.random.SeedSequence([int(seed), rounds])
        )
        proposals = []
        for cid in cluster_ids:
            pool = np.flatnonzero((edge_cluster == cid) & ~ctx.mask)
            if pool.size == 0:
                continue
            batch = degree_weighted_sample(
                pool, graph.src, graph.dst, degree, sample_size, rng
            )
            scored_total += int(batch.size)
            proposals.append(take_top(scores, batch, add_per_round))
        if not proposals:
            break

        merged = np.concatenate(proposals)
        chosen = take_top(scores, merged, ctx.remaining_budget)
        added = ctx.add(chosen)
        if added == 0:
            break
        np.add.at(degree, graph.src[chosen], 1)
        np.add.at(degree, graph.dst[chosen], 1)
        if verbose:
            print(
                f"[scaffold.fast] round={rounds} added={added} "
                f"edges={ctx.selected}/{ctx.target_edges}"
            )
    return rounds, scored_total


__all__ = ["run"]
