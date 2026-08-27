"""Standalone usage: NetworkX, SciPy and raw numpy, no PyTorch anywhere.

Run::

    python examples/02_standalone.py

SCAFFOLD's core needs only numpy. NetworkX and SciPy are optional adapters, so
this whole file runs in an environment with no deep-learning stack at all.
"""

from __future__ import annotations

import numpy as np

import scaffold


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
section("1. The simplest thing that works")
# ----------------------------------------------------------------------
G = scaffold.grid_graph(20, 20)
result = scaffold.fast(G, keep_ratio=0.6, seed=0)

print(result.summary())
print(f"  connected components: {result.num_components()}")
print(f"  edge_index shape:     {result.edge_index.shape}  (symmetric)")
print(f"  undirected shape:     {result.undirected_edge_index.shape}")
print(f"  runtime:              {result.metadata['runtime'] * 1000:.1f} ms"
      f"  (first call includes numba JIT warm-up; see section 6)")


# ----------------------------------------------------------------------
section("2. Budgets: keep_ratio or num_edges, never both")
# ----------------------------------------------------------------------
by_ratio = scaffold.fast(G, keep_ratio=0.25, seed=0)
by_count = scaffold.fast(G, num_edges=190, seed=0)
print(f"  keep_ratio=0.25 -> {by_ratio.sparse_edges} edges")
print(f"  num_edges=190   -> {by_count.sparse_edges} edges")
try:
    scaffold.fast(G, keep_ratio=0.25, num_edges=190)
except ValueError as exc:
    print(f"  both at once    -> ValueError: {exc}")


# ----------------------------------------------------------------------
section("3. NetworkX in, NetworkX out (node labels survive)")
# ----------------------------------------------------------------------
try:
    import networkx as nx

    Gnx = nx.karate_club_graph()
    Gnx = nx.relabel_nodes(Gnx, {i: f"member-{i:02d}" for i in Gnx.nodes()})

    result = scaffold.heap(Gnx, keep_ratio=0.5, seed=0)
    H = result.to_networkx()

    print(f"  {result.summary()}")
    print(f"  labels preserved: {sorted(H.nodes())[:3]} ...")
    print(f"  H is a subgraph of G: "
          f"{set(map(frozenset, H.edges())) <= set(map(frozenset, Gnx.edges()))}")
    print(f"  connected: {nx.is_connected(H)}")
except ImportError:
    print("  networkx not installed; skipping")


# ----------------------------------------------------------------------
section("4. SciPy sparse in, SciPy sparse out")
# ----------------------------------------------------------------------
try:
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = rng.random((300, 300)) < 0.03
    dense = np.triu(dense, 1)
    A = sp.csr_matrix(dense + dense.T)

    result = scaffold.fast(A, keep_ratio=0.3, seed=0)
    A_sparse = result.to_scipy()

    print(f"  {result.summary()}")
    print(f"  nnz {A.nnz} -> {A_sparse.nnz}, symmetric: {(A_sparse != A_sparse.T).nnz == 0}")
except ImportError:
    print("  scipy not installed; skipping")


# ----------------------------------------------------------------------
section("5. Raw edge_index in, numpy out")
# ----------------------------------------------------------------------
edge_index = np.array([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
result = scaffold.fast(edge_index, keep_ratio=0.7, seed=0)
print(f"  {result.summary()}")
print(f"  kept edge ids: {result.edge_ids.tolist()}")
print(f"  mask:          {result.mask.astype(int).tolist()}")


# ----------------------------------------------------------------------
section("6. Comparing the five methods")
# ----------------------------------------------------------------------
G = scaffold.grid_graph(16, 16)
print(f"  {G}\n")
print(f"  {'method':8s} {'edges':>7s} {'comp':>5s} {'ms':>8s}   notes")
print(f"  {'-' * 8} {'-' * 7} {'-' * 5} {'-' * 8}   {'-' * 30}")
for method in ("greedy", "heap", "batch", "fast", "sample"):
    result = scaffold.sparsify(G, method=method, keep_ratio=0.7, seed=0)
    note = ""
    if method == "sample":
        note = "weights only; drawing one view"
        result = result.draw(keep_ratio=0.7, seed=0)
    print(
        f"  {method:8s} {result.sparse_edges:7d} {result.num_components():5d} "
        f"{result.metadata['runtime'] * 1000:8.1f}   {note}"
    )


# ----------------------------------------------------------------------
section("7. The connectivity floor is a real constraint")
# ----------------------------------------------------------------------
floor = (G.num_nodes - 1) / G.num_edges
print(f"  A spanning tree of this graph needs {G.num_nodes - 1} of {G.num_edges} edges,")
print(f"  so no sparsifier can stay connected below keep_ratio={floor:.3f}.\n")
for ratio in (0.60, floor, 0.40):
    result = scaffold.fast(G, keep_ratio=ratio, seed=0)
    flag = "" if result.num_components() == 1 else "  <- fragmented, as it must be"
    print(
        f"  keep_ratio={ratio:.3f} -> {result.sparse_edges:4d} edges, "
        f"{result.num_components():3d} component(s){flag}"
    )
print("\n  scaffold reports this in metadata as 'delta_min' and "
      "'below_connectivity_floor'.")
