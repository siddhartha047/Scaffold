"""Low-level array kernels shared by every SCAFFOLD variant.

Everything here operates on plain ``numpy`` arrays describing a canonical
undirected edge list (``src < dst``, no self loops, no duplicates). The bodies
are written in explicit-loop style so that a single source works both as pure
Python/NumPy and as a ``numba`` JIT kernel; :func:`jit` picks whichever is
available at import time.

Contents:

* union-find spanning-forest scans (:func:`spanning_forest_mask`),
* the rooted-forest index with binary lifting (:func:`build_tree_index`),
* batched lowest-common-ancestor queries (:func:`tree_lca`),
* the tree congestion counters and root-prefix sums that turn every path
  aggregate into an ``O(1)`` difference.

The last three are what make SCAFFOLD-Fast and SCAFFOLD-Sample run in
``O(m log n + n)`` instead of one shortest-path search per candidate edge.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised implicitly by whichever branch is installed
    from numba import njit as _njit
except Exception:  # pragma: no cover
    _njit = None

HAVE_NUMBA = _njit is not None


def jit(func):
    """Compile ``func`` with numba when available, else return it unchanged.

    The pure-Python fallback stays correct but is only fast enough for the
    small graphs used in the tests and the tutorial notebooks.
    """
    if _njit is None:
        return func
    return _njit(cache=True, nogil=True)(func)


# ----------------------------------------------------------------------
# union-find
# ----------------------------------------------------------------------
@jit
def _dsu_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@jit
def _forest_mask_kernel(num_nodes, src, dst, visit, max_edges):
    """Kruskal over ``visit`` (an edge-id order); boolean mask over ``src``."""
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=np.int8)
    mask = np.zeros(src.shape[0], dtype=np.bool_)
    added = 0
    for j in range(visit.shape[0]):
        if added >= max_edges:
            break
        i = visit[j]
        u = src[i]
        v = dst[i]
        if u == v:
            continue
        ru = _dsu_find(parent, u)
        rv = _dsu_find(parent, v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        mask[i] = True
        added += 1
    return mask


@jit
def _component_count_kernel(num_nodes, src, dst):
    parent = np.arange(num_nodes)
    components = num_nodes
    for i in range(src.shape[0]):
        ru = _dsu_find(parent, src[i])
        rv = _dsu_find(parent, dst[i])
        if ru != rv:
            parent[ru] = rv
            components -= 1
    return components


@jit
def _component_labels_kernel(num_nodes, src, dst):
    parent = np.arange(num_nodes)
    for i in range(src.shape[0]):
        ru = _dsu_find(parent, src[i])
        rv = _dsu_find(parent, dst[i])
        if ru != rv:
            parent[ru] = rv
    out = np.empty(num_nodes, dtype=np.int64)
    for x in range(num_nodes):
        out[x] = _dsu_find(parent, x)
    return out


# ----------------------------------------------------------------------
# priority bucketing (the "fast" in fast-maxst / fast-mst)
# ----------------------------------------------------------------------
@jit
def _counting_order_kernel(buckets, bucket_count):
    """Stable counting sort of edge ids by bucket, in O(m + bucket_count)."""
    counts = np.zeros(bucket_count, dtype=np.int64)
    for i in range(buckets.shape[0]):
        counts[buckets[i]] += 1
    cursors = np.empty(bucket_count, dtype=np.int64)
    position = 0
    for b in range(bucket_count):
        cursors[b] = position
        position += counts[b]
    order = np.empty(buckets.shape[0], dtype=np.int64)
    for i in range(buckets.shape[0]):
        b = buckets[i]
        order[cursors[b]] = i
        cursors[b] += 1
    return order


# ----------------------------------------------------------------------
# rooted forest index (parent / depth / root / binary lifting / preorder)
# ----------------------------------------------------------------------
@jit
def _tree_index_kernel(num_nodes, tree_src, tree_dst, levels):
    m = tree_src.shape[0]
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i in range(m):
        rowptr[tree_src[i] + 1] += 1
        rowptr[tree_dst[i] + 1] += 1
    for i in range(1, num_nodes + 1):
        rowptr[i] += rowptr[i - 1]
    nbr = np.empty(2 * m, dtype=np.int64)
    cursor = np.empty(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        cursor[i] = rowptr[i]
    for i in range(m):
        u = tree_src[i]
        v = tree_dst[i]
        nbr[cursor[u]] = v
        cursor[u] += 1
        nbr[cursor[v]] = u
        cursor[v] += 1

    parent = np.arange(num_nodes)
    depth = np.zeros(num_nodes, dtype=np.int32)
    root = np.arange(num_nodes)
    tin = np.zeros(num_nodes, dtype=np.int64)
    seen = np.zeros(num_nodes, dtype=np.bool_)
    stack = np.empty(num_nodes, dtype=np.int64)
    clock = 0

    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        parent[start] = start
        depth[start] = 0
        root[start] = start
        top = 0
        stack[top] = start
        top += 1
        while top > 0:
            top -= 1
            node = stack[top]
            tin[node] = clock  # DFS preorder: the tree-locality key
            clock += 1
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = nbr[pos]
                if seen[nb]:
                    continue
                seen[nb] = True
                parent[nb] = node
                depth[nb] = depth[node] + 1
                root[nb] = start
                stack[top] = nb
                top += 1

    up = np.empty((levels, num_nodes), dtype=np.int64)
    for i in range(num_nodes):
        up[0, i] = parent[i]
    for lvl in range(1, levels):
        for i in range(num_nodes):
            up[lvl, i] = up[lvl - 1, up[lvl - 1, i]]
    return depth, root, up, parent, tin


@jit
def _lca_kernel(src, dst, depth, root, up):
    n = src.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    levels = up.shape[0]
    for i in range(n):
        a = src[i]
        b = dst[i]
        if root[a] != root[b]:
            continue  # different components: no tree path at all
        if depth[a] < depth[b]:
            tmp = a
            a = b
            b = tmp
        diff = depth[a] - depth[b]
        bit = 0
        while diff > 0:
            if diff & 1:
                a = up[bit, a]
            diff >>= 1
            bit += 1
        if a != b:
            for lvl in range(levels - 1, -1, -1):
                if up[lvl, a] != up[lvl, b]:
                    a = up[lvl, a]
                    b = up[lvl, b]
            a = up[0, a]
        out[i] = a
    return out


# ----------------------------------------------------------------------
# congestion counters
# ----------------------------------------------------------------------
@jit
def _edge_congestion_kernel(src, dst, lca, parent, order, num_nodes):
    """``econ[x]`` = number of candidate paths using the tree edge (x, parent[x]).

    Computed with the standard "+1 at both endpoints, -2 at the LCA, then fold
    subtrees upward" trick, so all ``|I|`` paths are counted in one ``O(m + n)``
    pass rather than enumerated one at a time.
    """
    econ = np.zeros(num_nodes, dtype=np.int64)
    for i in range(src.shape[0]):
        anc = lca[i]
        if anc < 0:
            continue
        econ[src[i]] += 1
        econ[dst[i]] += 1
        econ[anc] -= 2
    for idx in range(order.shape[0] - 1, -1, -1):  # deepest first
        x = order[idx]
        p = parent[x]
        if p != x:  # never fold a root into itself
            econ[p] += econ[x]
    return econ


@jit
def _node_congestion_kernel(src, dst, lca, econ):
    """``vcon[x] = econ[x] + lcaCount[x] - endpointDegree[x]``.

    A candidate path touches ``x`` in exactly one of two disjoint ways: it uses
    the tree edge ``(x, parent[x])`` -- counted by ``econ``, which implies the
    LCA sits strictly above ``x`` -- or ``x`` *is* the LCA. Only interior path
    nodes count, so paths having ``x`` as an endpoint are subtracted back out.
    """
    vcon = econ.copy()
    for i in range(src.shape[0]):
        anc = lca[i]
        if anc < 0:
            continue
        vcon[anc] += 1
        vcon[src[i]] -= 1
        vcon[dst[i]] -= 1
    return vcon


@jit
def _root_prefix_kernel(values, parent, order, power, eps, include_root):
    """``out[x] = sum over the root->x path of (values[.] + eps) ** power``.

    ``include_root`` distinguishes *edge* prefixes (a root has no parent edge,
    so the sum starts at 0) from *node* prefixes (a root is a node, so it
    contributes its own term).
    """
    n = order.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for idx in range(n):  # ascending depth: parents precede children
        x = order[idx]
        term = (values[x] + eps) ** power
        p = parent[x]
        if p == x:
            out[x] = term if include_root else 0.0
        else:
            out[x] = out[p] + term
    return out


# ----------------------------------------------------------------------
# CSR adjacency + shortest paths (used by the exact / heap variants)
# ----------------------------------------------------------------------
@jit
def _csr_kernel(num_nodes, src, dst):
    m = src.shape[0]
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i in range(m):
        rowptr[src[i] + 1] += 1
        rowptr[dst[i] + 1] += 1
    for i in range(1, num_nodes + 1):
        rowptr[i] += rowptr[i - 1]
    col = np.empty(2 * m, dtype=np.int64)
    eid = np.empty(2 * m, dtype=np.int64)
    cursor = rowptr[:num_nodes].copy()
    for i in range(m):
        u = src[i]
        v = dst[i]
        col[cursor[u]] = v
        eid[cursor[u]] = i
        cursor[u] += 1
        col[cursor[v]] = u
        eid[cursor[v]] = i
        cursor[v] += 1
    return rowptr, col, eid


@jit
def _bfs_parents_kernel(source, rowptr, col, eid, parent, parent_edge, dist, stamp, token):
    """Single-source BFS writing parent pointers into pre-allocated buffers.

    ``stamp``/``token`` implement O(visited) reuse of the buffers across calls:
    a node counts as visited only when ``stamp[node] == token``, so nothing has
    to be cleared between sources.
    """
    queue = np.empty(rowptr.shape[0] - 1, dtype=np.int64)
    head = 0
    tail = 0
    queue[tail] = source
    tail += 1
    stamp[source] = token
    parent[source] = -1
    parent_edge[source] = -1
    dist[source] = 0
    while head < tail:
        node = queue[head]
        head += 1
        for pos in range(rowptr[node], rowptr[node + 1]):
            nb = col[pos]
            if stamp[nb] == token:
                continue
            stamp[nb] = token
            parent[nb] = node
            parent_edge[nb] = eid[pos]
            dist[nb] = dist[node] + 1
            queue[tail] = nb
            tail += 1
    return tail


# ----------------------------------------------------------------------
# public wrappers
# ----------------------------------------------------------------------
def spanning_forest_mask(num_nodes, src, dst, visit_order, max_edges=None):
    """Kruskal over ``visit_order``; returns a boolean mask over ``src``."""
    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    visit_order = np.ascontiguousarray(visit_order, dtype=np.int64)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    limit = num_nodes if max_edges is None else int(max_edges)
    limit = max(0, min(limit, num_nodes))
    if limit == 0:
        return np.zeros(src.shape[0], dtype=bool)
    return np.asarray(_forest_mask_kernel(num_nodes, src, dst, visit_order, limit))


def component_count(num_nodes, src, dst):
    """Number of connected components of ``(num_nodes, src, dst)``."""
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return int(num_nodes)
    return int(_component_count_kernel(int(num_nodes), src, dst))


def component_labels(num_nodes, src, dst):
    """Per-node component representative (not necessarily 0..k-1)."""
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return np.arange(int(num_nodes), dtype=np.int64)
    return np.asarray(_component_labels_kernel(int(num_nodes), src, dst))


def counting_order(buckets, bucket_count):
    """Stable ``argsort`` of small non-negative integer keys, in linear time."""
    buckets = np.ascontiguousarray(buckets, dtype=np.int64)
    if buckets.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(_counting_order_kernel(buckets, int(bucket_count)))


def build_tree_index(num_nodes, tree_src, tree_dst):
    """Return ``(depth, root, up, parent, tin)`` for a rooted spanning forest.

    ``tin`` is the DFS preorder index, used as the tree-locality sort key by
    SCAFFOLD-Sample.
    """
    num_nodes = int(num_nodes)
    tree_src = np.ascontiguousarray(tree_src, dtype=np.int64)
    tree_dst = np.ascontiguousarray(tree_dst, dtype=np.int64)
    levels = max(1, int(math.ceil(math.log2(max(2, num_nodes)))) + 1)
    return _tree_index_kernel(num_nodes, tree_src, tree_dst, levels)


def depth_order(depth):
    """Node ids sorted by ascending depth (a valid topological order)."""
    depth = np.asarray(depth)
    return np.argsort(depth, kind="stable").astype(np.int64, copy=False)


def tree_lca(src, dst, depth, root, up):
    """LCA per pair; ``-1`` when the endpoints lie in different components."""
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(_lca_kernel(src, dst, depth, root, up))


def build_csr(num_nodes, src, dst):
    """Symmetric CSR adjacency ``(rowptr, col, edge_id)`` over an edge list."""
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return (
            np.zeros(int(num_nodes) + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return _csr_kernel(int(num_nodes), src, dst)


def parent_edge_weights(depth, tree_src, tree_dst, tree_weight, num_nodes):
    """Weight of the tree edge joining each node to its parent (roots get 0).

    On a tree the deeper endpoint of an edge is always the child, so the
    assignment is unambiguous.
    """
    tree_src = np.asarray(tree_src, dtype=np.int64)
    tree_dst = np.asarray(tree_dst, dtype=np.int64)
    tree_weight = np.asarray(tree_weight, dtype=np.float64)
    out = np.zeros(int(num_nodes), dtype=np.float64)
    if tree_src.size:
        deeper = np.where(depth[tree_src] > depth[tree_dst], tree_src, tree_dst)
        out[deeper] = tree_weight
    return out
