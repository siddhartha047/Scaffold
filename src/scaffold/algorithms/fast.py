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
*identical* to what SCAFFOLD-Greedy computes in its first round -- the support
is the tree, so ``d_H == d_T``.

Selection is one global top-k. Because the tree index is never rebuilt during
growth, the score is static; the sampled round loop that preceded this version
now lives under :func:`scaffold.batch`, where scores are recomputed within each
sampled batch against the current support.

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
``greedy``             1674         3.83         3.87
``topk``                  0.5      12.06        26.34
===================  =========  ==========  ===========

On less symmetric graphs the gap mostly closes -- on a random geometric graph
``topk`` gets 2.68 against ``greedy``'s 2.40, and on Barabasi-Albert 3.82 against
3.66. Congestion is the term that suffers most from concentration; if that is
what you care about, use
:func:`scaffold.batch` or
:mod:`~scaffold.algorithms.sample`, whose tree-locality ordering spreads by
construction.
"""

from __future__ import annotations

from typing import Optional

from ..backbone import DEFAULT_BACKBONE
from ..graph import Graph
from ..kernels import build_tree_index
from ..scoring import ScoreParams, tree_scores
from ..utils.workers import resolve_workers
from .base import GrowthContext, take_top


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    params: Optional[ScoreParams] = None,
    seed=None,
    selection: str = "topk",
    weighted_paths: bool = False,
    backbone_options=None,
    return_scores: bool = False,
    workers=None,
    verbose: bool = False,
):
    """Score every candidate once with the tree kernel, then select.

    Parameters
    ----------
    selection:
        Compatibility check; must be ``"topk"``. Use :func:`scaffold.batch`
        for sampled-batch top-r growth.
    weighted_paths:
        Measure the numerator of the dilation as a sum of tree edge weights
        rather than as a hop count. Only meaningful on weighted graphs.
    return_scores:
        Also return the per-edge score array in the metadata under
        ``"scores"``. Handy for plotting and for reusing one scoring pass
        across several budgets.
    workers:
        Threads used for the single scoring pass, which is where essentially
        all of Fast's time goes. ``None`` resolves automatically; see
        :func:`scaffold.utils.workers.resolve_workers`. The selected edges are
        identical at any worker count.
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
    if selection != "topk":
        raise ValueError(
            "scaffold.fast supports only selection='topk'; use "
            "scaffold.batch for sampled-batch growth"
        )

    if graph.num_edges == 0 or ctx.remaining_budget <= 0:
        return ctx.finish("fast", selection=selection, rounds=0, scored_candidates=0)

    # One scoring pass against the backbone forest. Never recomputed: growth
    # does not rebuild the tree index, so the score is static by construction.
    tree_index = build_tree_index(
        graph.num_nodes, graph.src[ctx.mask], graph.dst[ctx.mask]
    )
    workers = resolve_workers(workers)
    scored = tree_scores(
        graph.num_nodes,
        graph.src,
        graph.dst,
        ctx.mask,
        weight=graph.edge_weight,
        params=ctx.params,
        tree_index=tree_index,
        weighted_paths=weighted_paths,
        workers=workers,
    )
    scores = scored["score"]
    candidates = ctx.candidate_ids()

    chosen = take_top(scores, candidates, ctx.remaining_budget)
    ctx.add(chosen)
    rounds = 1
    scored_total = int(candidates.size)

    extra = {
        "selection": selection,
        "rounds": rounds,
        "scored_candidates": scored_total,
        "mandatory_edges": int(scored["mandatory"].sum()),
        "total_stretch": float(scored["total_stretch"]),
        "weighted_paths": bool(weighted_paths),
        "workers": int(workers),
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


__all__ = ["run"]
