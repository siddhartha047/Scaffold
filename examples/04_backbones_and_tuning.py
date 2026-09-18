"""Choosing a support backbone and tuning the objective.

Run::

    python examples/04_backbones_and_tuning.py

Two knobs matter most, and they do different jobs:

* the **backbone** decides what is guaranteed -- connectivity, and the skeleton
  the rest of the budget is spent relative to;
* the **objective weights** (``alpha``, ``beta_edge``, ``beta_node``) decide
  what "important" means for the edges bought with the remaining budget.
"""

from __future__ import annotations

import numpy as np

import scaffold
from scaffold.backbone import build_backbone, register_backbone
from scaffold.scoring import ScoreParams, tree_scores


def section(title):
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


G = scaffold.grid_graph(14, 14)
KEEP = 0.7
print(f"graph: {G}  (delta_min = {(G.num_nodes - 1) / G.num_edges:.3f})")


# ----------------------------------------------------------------------
section("1. The available backbones")
# ----------------------------------------------------------------------
print(f"  {scaffold.available_backbones()}\n")
print(f"  {'backbone':12s} {'forest':>7s} {'stretch':>10s} {'edges':>7s} {'comp':>5s}")
print(f"  {'-' * 12} {'-' * 7} {'-' * 10} {'-' * 7} {'-' * 5}")
for name in ("fast-maxst", "maxst", "mst", "fast-randst", "randst", "spt", "glst", "llst"):
    mask = build_backbone(G, name, seed=0)
    # total_stretch: sum over non-tree edges of dist_T(u,v)/w(e). Lower is a
    # better starting point -- fewer long detours for the growth phase to fix.
    stretch = tree_scores(G.num_nodes, G.src, G.dst, mask)["total_stretch"]
    # Reuse exactly the forest measured above, including LLST's refinement.
    result = scaffold.fast(G, keep_ratio=KEEP, backbone=mask, seed=0)
    print(
        f"  {name:12s} {int(mask.sum()):7d} {stretch:10.0f} "
        f"{result.sparse_edges:7d} {result.num_components():5d}"
    )
print("\n  On an *unweighted* grid, fast-maxst and maxst coincide: with no weights")
print("  to sort by, both fall back to the same deterministic scan. Give the grid")
print("  weights and they diverge.")
print("  fast-randst avoids constructing randst's full random permutation; it is")
print("  normally faster, while randst provides a more thoroughly shuffled order.")
print("  llst refines glst with up to 10 exhaustive improving cycle swaps;")
print("  its forest has the same edge budget, with shorter total detours.")


# ----------------------------------------------------------------------
section("2. Weighted graphs: the backbone follows the weights")
# ----------------------------------------------------------------------
Gw = scaffold.grid_graph(14, 14, weight="distance")  # heavier near the centre
print(f"  {Gw}, weights in [{Gw.edge_weight.min():.3f}, {Gw.edge_weight.max():.3f}]\n")
for name in ("fast-maxst", "maxst", "fast-mst", "mst"):
    mask = build_backbone(Gw, name, seed=0)
    mean_weight = float(Gw.edge_weight[mask].mean())
    print(f"  {name:12s} mean weight of backbone edges = {mean_weight:.4f}")
print("\n  maxst keeps the heavy (central) edges, mst the light (peripheral) ones.")
print("  'fast-' variants bucket the weights into 256 priority classes and sort in")
print("  linear time; the ordering is approximate, the union-find scan identical.")


# ----------------------------------------------------------------------
section("3. Objective weights: what gets bought with the rest of the budget")
# ----------------------------------------------------------------------
settings = [
    ("pure dilation", dict(alpha=1.0, beta_edge=0.0, beta_node=0.0)),
    ("dilation + edge cong.", dict(alpha=1.0, beta_edge=1.0, beta_node=0.0)),
    ("dilation + node cong.", dict(alpha=1.0, beta_edge=0.0, beta_node=1.0)),
    ("all three (default)", dict(alpha=1.0, beta_edge=1.0, beta_node=1.0)),
    ("congestion-dominant", dict(alpha=0.5, beta_edge=2.0, beta_node=2.0)),
]


def quality(mask):
    """Measure the *result*: how well does the kept subgraph serve what it dropped?

    ``path_scores`` evaluates the objective against an arbitrary support graph,
    so it works on the sparsified output, not just on a forest.
    """
    from scaffold.scoring import path_scores

    metrics = path_scores(G.num_nodes, G.src, G.dst, mask, params=ScoreParams())
    dil = metrics["dil"]
    finite = dil[np.isfinite(dil)]
    return {
        "mean_dilation": float(finite.mean()) if finite.size else 0.0,
        "max_dilation": float(finite.max()) if finite.size else 0.0,
        "max_edge_congestion": float(np.max(metrics["econ_path"]))
        if metrics["econ_path"].size
        else 0.0,
    }


baseline = scaffold.fast(G, keep_ratio=KEEP, seed=0).mask
print("  Measured on the *dropped* edges: how long a detour do they get, and how")
print("  concentrated is the traffic on the edges that survived?\n")
print(
    f"  {'setting':24s} {'mean dil':>9s} {'max dil':>8s} {'max cong':>9s} {'vs default':>11s}"
)
print(f"  {'-' * 24} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 11}")
for label, knobs in settings:
    result = scaffold.fast(G, keep_ratio=KEEP, seed=0, **knobs)
    stats = quality(result.mask)
    overlap = (result.mask & baseline).sum() / max(1, result.mask.sum())
    print(
        f"  {label:24s} {stats['mean_dilation']:9.3f} {stats['max_dilation']:8.1f} "
        f"{stats['max_edge_congestion']:9.2f} {overlap:10.1%}"
    )
print("""
  How to read this. alpha rewards candidates with a long detour; the betas
  reward candidates whose detour runs through an already-overloaded part of the
  support, on the grounds that adding them relieves a bottleneck. Both terms
  are evaluated against the *backbone forest*, once -- that is what makes
  scaffold.fast fast.

  The columns above measure something different: the final subgraph, after all
  the additions. On this uniform grid the two disagree, and pure dilation ends
  up with the lower post-hoc numbers. That is not a bug and not a
  recommendation; a lattice is the least structured input there is, so the
  congestion signal has little to latch onto. Measure this on *your* graph
  before picking exponents -- which is precisely why the knobs are exposed.""")


# ----------------------------------------------------------------------
section("4. Reusing one scoring pass across many budgets")
# ----------------------------------------------------------------------
# scaffold.fast scores every candidate once; if you want several budgets, ask
# for the scores and threshold them yourself instead of re-running.
result = scaffold.fast(G, keep_ratio=0.99, seed=0, return_scores=True)
scores = result.metadata["scores"]
backbone = build_backbone(G, "fast-maxst", seed=0)
print(
    f"  one scoring pass over {G.num_edges} edges "
    f"({result.metadata['runtime'] * 1000:.1f} ms)\n"
)
for ratio in (0.6, 0.7, 0.8, 0.9):
    budget = int(np.ceil(ratio * G.num_edges)) - int(backbone.sum())
    candidates = np.flatnonzero(~backbone)
    order = np.lexsort((candidates, -scores[candidates]))
    mask = backbone.copy()
    mask[candidates[order[:budget]]] = True
    direct = scaffold.fast(G, keep_ratio=ratio, seed=0).mask
    print(
        f"  keep_ratio={ratio:.2f}: {int(mask.sum()):4d} edges, "
        f"identical to a fresh run: {np.array_equal(mask, direct)}"
    )


# ----------------------------------------------------------------------
section("5. Registering a custom backbone")


# ----------------------------------------------------------------------
def star_forest(graph, max_edges=None, seed=None, **_):
    """A silly backbone: greedily attach each node to its lowest-id neighbour."""
    from scaffold.kernels import spanning_forest_mask

    order = np.lexsort((graph.dst, graph.src))
    return spanning_forest_mask(
        graph.num_nodes, graph.src, graph.dst, order, max_edges=max_edges
    )


register_backbone("lowest-id", star_forest)
result = scaffold.fast(G, keep_ratio=KEEP, backbone="lowest-id", seed=0)
print(f"  registered 'lowest-id' -> {result.summary()}")
print(f"  components: {result.num_components()}")
print(f"  available now: {scaffold.available_backbones()}")

# A precomputed mask works without registering anything.
custom = build_backbone(G, "randst", seed=99)
result = scaffold.fast(G, keep_ratio=KEEP, backbone=custom, seed=0)
print(f"\n  passing a mask directly -> {result.summary()}")
