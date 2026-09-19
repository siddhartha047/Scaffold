"""SCAFFOLD-Heap: lazy top-k greedy with local invalidation.

SCAFFOLD-Greedy rescores *every* candidate after *every* insertion, and almost
all of that work is wasted: adding one edge changes the shortest path of the
candidates routed near it and leaves the rest of the graph alone.

SCAFFOLD-Heap exploits that. Scores live in a max-heap and are allowed to go
stale. Each round pops the top ``k`` entries, rescores only those against the
current support, and adds the best. After an insertion, only candidates whose
recorded path touched the affected edges or nodes are marked dirty and pushed
back with fresh scores; everything else keeps its stale key.

Because a stale score is never *larger* than it should be for the terms that
shrink monotonically, popping the top ``k`` and rescoring is the classic lazy
greedy pattern: usually the same choice as Greedy, at a fraction of the cost.

Scoring note
------------
This variant uses the *max*-congestion form of the objective::

    score(e) = dil(e) * (1 + beta_edge * log1p(maxECon) + beta_node * log1p(maxVCon))

rather than the normalized p-norm product. The max form is cheap to maintain
incrementally and does not depend on a global maximum that shifts every round,
which is what makes stale keys meaningful in the first place. Pass
``score_form="product"`` for the p-norm objective used by the other variants.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from ..backbone import DEFAULT_BACKBONE
from ..clustering import assign_clusters, cluster_edges
from ..graph import Graph
from ..scoring import PathScorer, ScoreParams
from ..utils.workers import resolve_workers
from .base import GrowthContext


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    backbone=DEFAULT_BACKBONE,
    params: Optional[ScoreParams] = None,
    seed=None,
    top_k: int = 16,
    add_per_round: int = 1,
    clusters=None,
    cluster_method: str = "bfs",
    local_radius: int = 1,
    dirty_limit: int = 64,
    score_form: str = "max",
    backbone_options=None,
    workers=None,
    verbose: bool = False,
    return_trace: bool = False,
    weighted_paths: Optional[bool] = None,
):
    """Grow the support with a lazy max-heap over candidate scores.

    Parameters
    ----------
    top_k:
        Candidates rescored per round. Larger values track Greedy more closely
        and cost proportionally more.
    add_per_round:
        Edges committed per round (per cluster, when clustering is on).
    clusters:
        ``None``/``1`` for a single global heap, an integer for that many
        cluster-local heaps, or a per-node assignment array.
    local_radius:
        After an insertion, candidates touching a node within this many hops of
        the affected path are invalidated. ``0`` invalidates only exact
        path-edge/endpoint matches.
    dirty_limit:
        Cap on invalidations per round; ``0`` means unlimited.

        Not a tuning nicety -- it is what keeps the lazy heap lazy. On a
        hub-heavy graph a single insertion near a hub can dirty most of the
        candidate set, at which point "rescore only what changed" degenerates
        into "rescore everything" and Heap becomes *slower* than Greedy. On
        Barabasi-Albert (600 nodes, 1,791 edges, ``keep_ratio=0.45``):
        unlimited 4,757 ms, ``dirty_limit=64`` 591 ms, with mean dilation
        3.630 vs 3.627 -- an 8x speed-up for no measurable quality cost.
        Raise it if you want to track Greedy more tightly.
    score_form:
        ``"max"`` (default, see the module docstring) or ``"product"`` for the
        normalized p-norm objective shared with Greedy and Fast.
    return_trace:
        Record ``added_edge_ids`` and ``addition_round_sizes`` in metadata.
        The sizes group committed edges by insertion round, across clusters.
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
    if score_form not in ("max", "product"):
        raise ValueError("score_form must be 'max' or 'product'")

    top_k = max(1, int(top_k))
    add_per_round = max(1, int(add_per_round))
    local_radius = max(0, int(local_radius))
    dirty_limit = max(0, int(dirty_limit))

    workers = resolve_workers(workers)
    scorer = PathScorer(
        graph.num_nodes,
        graph.src,
        graph.dst,
        weight=graph.edge_weight,
        support_mask=ctx.mask.copy(),
        workers=workers,
        weighted_paths=weighted_paths,
    )

    node_labels = assign_clusters(graph, clusters, method=cluster_method, seed=seed)
    edge_cluster = cluster_edges(graph, node_labels)
    cluster_ids = np.unique(edge_cluster)

    heaps = {int(cid): [] for cid in cluster_ids}
    version = {}          # edge id -> latest push counter (stale entries dropped)
    counter = 0
    # Reverse indices so an insertion can find exactly which candidates it invalidated.
    edge_watchers = {}    # support edge id -> set of candidate edge ids routed over it
    node_watchers = {}    # node id        -> set of candidate edge ids touching it
    recorded = {}         # candidate edge id -> (path edge ids, touched nodes)

    def unindex(edge_id):
        entry = recorded.pop(edge_id, None)
        if entry is None:
            return
        path_edges, touched = entry
        for e in path_edges:
            watchers = edge_watchers.get(e)
            if watchers is not None:
                watchers.discard(edge_id)
        for x in touched:
            watchers = node_watchers.get(x)
            if watchers is not None:
                watchers.discard(edge_id)

    def index(edge_id, path_edges, path_nodes):
        unindex(edge_id)
        touched = set(path_nodes)
        touched.add(int(graph.src[edge_id]))
        touched.add(int(graph.dst[edge_id]))
        recorded[edge_id] = (tuple(path_edges), tuple(touched))
        for e in path_edges:
            edge_watchers.setdefault(e, set()).add(edge_id)
        for x in touched:
            node_watchers.setdefault(x, set()).add(edge_id)

    def push(edge_id, score):
        nonlocal counter
        counter += 1
        version[edge_id] = counter
        key = -np.inf if math.isinf(score) else -float(score)
        heapq.heappush(heaps[int(edge_cluster[edge_id])], (key, counter, int(edge_id)))

    def evaluate(edge_ids, push_back=True):
        edge_ids = np.asarray(sorted(set(int(e) for e in edge_ids)), dtype=np.int64)
        if edge_ids.size == 0:
            return {}
        metrics = scorer.evaluate(edge_ids, ctx.params)
        scores = _scores(metrics, ctx.params, score_form)
        out = {}
        for slot, edge_id in enumerate(edge_ids.tolist()):
            index(edge_id, metrics["path_edges"][slot] or (), metrics["path_nodes"][slot] or ())
            out[edge_id] = float(scores[slot])
            if push_back:
                push(edge_id, scores[slot])
        return out

    # Seed every cluster heap.
    trace = {"added_edge_ids": [], "addition_round_sizes": []} if return_trace else {}
    initial = ctx.candidate_ids()
    if initial.size == 0:
        return ctx.finish("heap", rounds=0, rescored_candidates=0, heap_rebuilds=0,
                          weighted_paths=scorer.weighted_paths,
                          **trace)
    evaluate(initial)

    rounds = 0
    rescored = 0
    rebuilds = 0
    remaining = set(initial.tolist())

    while ctx.remaining_budget > 0 and remaining:
        rounds += 1
        active = []
        for cid in cluster_ids:
            heap = heaps[int(cid)]
            picked = 0
            while heap and picked < top_k:
                _, ver, edge_id = heapq.heappop(heap)
                if edge_id not in remaining or version.get(edge_id) != ver:
                    continue
                active.append(edge_id)
                picked += 1

        if not active:
            # Everything popped was stale: rebuild from the live candidate set.
            live = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
            if live.size == 0:
                break
            evaluate(live)
            rebuilds += 1
            rescored += int(live.size)
            continue

        fresh = evaluate(active, push_back=False)
        rescored += len(fresh)

        # Commit the best `add_per_round` per cluster, respecting the budget.
        by_cluster = {}
        for edge_id, score in fresh.items():
            by_cluster.setdefault(int(edge_cluster[edge_id]), []).append((score, edge_id))
        committed = []
        for cid in sorted(by_cluster):
            ranked = sorted(by_cluster[cid], key=lambda item: (-item[0], item[1]))
            for _score, edge_id in ranked[:add_per_round]:
                committed.append(edge_id)
                if len(committed) >= ctx.remaining_budget:
                    break
            if len(committed) >= ctx.remaining_budget:
                break
        if not committed:
            break

        affected_edges = set()
        affected_nodes = set()
        for edge_id in committed:
            path_edges, touched = recorded.get(edge_id, ((), ()))
            affected_edges.update(path_edges)
            affected_nodes.update(touched)
            ctx.add([edge_id])
            scorer.add_edge(edge_id)
            remaining.discard(edge_id)
            unindex(edge_id)
            version.pop(edge_id, None)

        if return_trace:
            trace["added_edge_ids"].extend(committed)
            trace["addition_round_sizes"].append(len(committed))
        if verbose:
            print(
                f"[scaffold.heap] round={rounds} active={len(active)} "
                f"added={len(committed)} edges={ctx.selected}/{ctx.target_edges}"
            )
        if ctx.remaining_budget <= 0:
            break

        dirty = set()
        for e in affected_edges:
            dirty.update(edge_watchers.get(e, ()))
        for x in _expand(scorer, affected_nodes, local_radius, graph.num_nodes):
            dirty.update(node_watchers.get(x, ()))
        dirty &= remaining
        if dirty_limit and len(dirty) > dirty_limit:
            rng = np.random.default_rng(seed if seed is None else seed + rounds)
            dirty = set(rng.choice(sorted(dirty), size=dirty_limit, replace=False).tolist())

        # Untouched actives keep their fresh scores; dirty ones get rescored.
        for edge_id, score in fresh.items():
            if edge_id in remaining and edge_id not in dirty:
                push(edge_id, score)
        if dirty:
            evaluate(dirty)
            rescored += len(dirty)

    return ctx.finish(
        "heap",
        rounds=rounds,
        rescored_candidates=rescored,
        heap_rebuilds=rebuilds,
        top_k=top_k,
        add_per_round=add_per_round,
        clusters=int(cluster_ids.size),
        score_form=score_form,
        weighted_paths=scorer.weighted_paths,
        workers=workers,
        **trace,
    )


def _scores(metrics, params: ScoreParams, score_form: str) -> np.ndarray:
    if score_form == "product":
        return metrics["score"]
    dil = metrics["dil"]
    econ = metrics["econ_path"]
    vcon = metrics["vcon_path"]
    out = dil * (
        1.0
        + params.beta_edge * np.log1p(np.maximum(econ, 0.0))
        + params.beta_node * np.log1p(np.maximum(vcon, 0.0))
    )
    out[~np.isfinite(dil)] = np.inf
    return out


def _expand(scorer: PathScorer, nodes, radius: int, num_nodes: int):
    """Nodes within ``radius`` hops of ``nodes`` in the current support graph."""
    if radius <= 0 or not nodes:
        return set(nodes)
    scorer._ensure_csr()
    rowptr, col = scorer._rowptr, scorer._col
    frontier = set(int(x) for x in nodes)
    seen = set(frontier)
    for _ in range(radius):
        nxt = set()
        for node in frontier:
            if node >= num_nodes:
                continue
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = int(col[pos])
                if nb not in seen:
                    seen.add(nb)
                    nxt.add(nb)
        if not nxt:
            break
        frontier = nxt
    return seen


# Referenced by the docstring's score formula; kept importable for tests.
__all__ = ["run"]
