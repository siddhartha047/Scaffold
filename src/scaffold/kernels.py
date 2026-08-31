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
    from numba import prange as _prange
except Exception:  # pragma: no cover
    _njit = None
    _prange = range

HAVE_NUMBA = _njit is not None

# Written as ``prange`` in the kernel bodies below; it degrades to the builtin
# ``range`` when numba is absent, so one source serves both backends.
prange = _prange

# Below these problem sizes, spawning threads costs more than the work saved,
# so the kernels are run serially however many workers were asked for. The
# numbers come from measured crossovers on this hardware and are deliberately
# conservative -- set past the break-even, not at it.
PARALLEL_PATH_MIN_CANDIDATES = 256
PARALLEL_TREE_MIN_CANDIDATES = 8192


def jit(func):
    """Compile ``func`` with numba when available, else return it unchanged.

    The pure-Python fallback stays correct but is only fast enough for the
    small graphs used in the tests and the tutorial notebooks.
    """
    if _njit is None:
        return func
    return _njit(cache=True, nogil=True)(func)


def jit_parallel(func):
    """Like :func:`jit`, but with numba's auto-parallelizer enabled.

    Reserved for kernels whose loop iterations are genuinely independent and
    write to disjoint output slots. The thread count is controlled by
    :func:`scaffold.utils.workers.parallel_threads` at the call site, not here,
    and one thread is a perfectly good setting -- a ``prange`` kernel pinned to
    a single thread measured *slightly faster* than the same body compiled
    serially, so there is no separate serial build of any of these.

    That is not merely a tidiness decision. **Never decorate one function object
    with both :func:`jit` and :func:`jit_parallel`.** numba keys its on-disk
    cache on the function's code location, so the two compilations collide:
    whichever runs first wins the slot and the other silently loads it. The
    failure is invisible -- correct results, no speedup, no warning. It cost a
    full benchmark round to find (parallel LCA 1.22 s from the poisoned cache
    versus 0.23 s when compiled properly).

    Every kernel decorated with this must produce bitwise-identical output at
    any thread count: no floating-point reduction may cross iterations, since
    the summation order would then depend on the scheduler.
    """
    if _njit is None:
        return func
    return _njit(cache=True, nogil=True, parallel=True)(func)


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
def _strided_forest_mask_kernel(
    num_nodes, src, dst, offset, stride, max_edges
):
    """Kruskal over a cyclic arithmetic permutation without materializing it."""
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=np.int8)
    mask = np.zeros(src.shape[0], dtype=np.bool_)
    added = 0
    m = src.shape[0]
    for step in range(m):
        if added >= max_edges:
            break
        i = (offset + step * stride) % m
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


def _lca_body(src, dst, depth, root, up):
    """One LCA per candidate pair; ``-1`` for cross-component pairs.

    Iterations touch only ``out[i]`` and read the shared ancestor table, so this
    parallelizes exactly. The body is compiled twice -- once serial, once with
    the auto-parallelizer -- from this single source; numba treats ``prange`` as
    ``range`` when ``parallel=False``, and so does the no-numba fallback.
    """
    n = src.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    levels = up.shape[0]
    for i in prange(n):
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


_lca_kernel = jit_parallel(_lca_body)


# ----------------------------------------------------------------------
# fused per-candidate term assembly (the tree scorer's inner loop)
# ----------------------------------------------------------------------
def _tree_terms_body(
    cu, cv, cand, lca, depth, sp, nq, vcon, weight, edge_p, node_q, eps
):
    """Dilation and the two path-congestion norms, one candidate per iteration.

    Every quantity is a difference of root-prefix sums through the candidate's
    LCA, so each iteration is a handful of gathers and writes to its own slot.
    Fusing them here rather than expressing them as chained NumPy operations
    removes eight length-``m`` temporaries and lets the loop parallelize.

    Two traps preserved from the array formulation:

    * ``eConPath`` averages over the path's ``path_len`` *edges*, while
      ``vConPath`` averages over its ``path_len - 1`` *interior nodes*; the LCA
      and the two endpoints are added back / subtracted out explicitly.
    * cross-component candidates (``lca < 0``) get infinite dilation and take no
      part in the congestion statistics.
    """
    n = cu.shape[0]
    dil = np.empty(n, dtype=np.float64)
    econ_path = np.zeros(n, dtype=np.float64)
    vcon_path = np.zeros(n, dtype=np.float64)
    path_len = np.zeros(n, dtype=np.int64)
    inv_p = 1.0 / edge_p
    inv_q = 1.0 / node_q
    for i in prange(n):
        anc = lca[i]
        if anc < 0:
            dil[i] = np.inf
            continue
        u = cu[i]
        v = cv[i]
        hops = depth[u] + depth[v] - 2 * depth[anc]
        path_len[i] = hops

        w = weight[cand[i]]
        if w < eps:
            w = eps
        dil[i] = hops / w

        sigma_e = sp[u] + sp[v] - 2.0 * sp[anc]
        if sigma_e < 0.0:
            sigma_e = 0.0
        econ_path[i] = (sigma_e / (hops + eps)) ** inv_p

        sigma_v = (
            nq[u]
            + nq[v]
            - 2.0 * nq[anc]
            + (vcon[anc] + eps) ** node_q
            - (vcon[u] + eps) ** node_q
            - (vcon[v] + eps) ** node_q
        )
        if sigma_v < 0.0:
            sigma_v = 0.0
        vcon_path[i] = (sigma_v / (hops - 1 + eps)) ** inv_q
    return dil, econ_path, vcon_path, path_len


_tree_terms_kernel = jit_parallel(_tree_terms_body)


def _tree_score_body(
    dil, econ_path, vcon_path, d_max, e_max, v_max, alpha, beta_edge, beta_node, eps
):
    """Normalize the three terms by their maxima and combine them.

    The maxima arrive as scalars computed by the caller, so no reduction
    crosses iterations here and the result cannot depend on the thread count.
    """
    n = dil.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        if not np.isfinite(dil[i]):
            out[i] = np.inf
        else:
            out[i] = (
                ((dil[i] + eps) / (d_max + eps)) ** alpha
                * ((econ_path[i] + eps) / (e_max + eps)) ** beta_edge
                * ((vcon_path[i] + eps) / (v_max + eps)) ** beta_node
            )
    return out


_tree_score_kernel = jit_parallel(_tree_score_body)


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
# CSR adjacency + shortest paths (used by the greedy / heap variants)
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
# multi-source path evaluation (the greedy / heap / batch inner loop)
# ----------------------------------------------------------------------
# Candidates are grouped by source so one search serves every candidate leaving
# it, and the groups are then dealt into contiguous *blocks* -- one per worker.
# Blocks, rather than groups, are the unit of parallelism: each block owns a row
# of the scratch arrays for the whole time it runs, which avoids both a
# per-group O(n) allocation and any dependence on numba's thread ids.


@jit
def _heap_push(heap_dist, heap_node, size, dist, node):
    """Binary min-heap ordered by ``(dist, node)``.

    The node id is part of the key, not a tiebreak afterthought: Python's
    ``heapq`` compares the ``(dist, node)`` tuples the reference implementation
    pushes, so matching that ordering is what makes the compiled Dijkstra
    settle vertices in the same sequence and therefore produce the same
    parent pointers -- and the same paths, and the same congestion.
    """
    i = size
    heap_dist[i] = dist
    heap_node[i] = node
    while i > 0:
        parent = (i - 1) >> 1
        if heap_dist[parent] < heap_dist[i] or (
            heap_dist[parent] == heap_dist[i] and heap_node[parent] <= heap_node[i]
        ):
            break
        td = heap_dist[parent]; heap_dist[parent] = heap_dist[i]; heap_dist[i] = td
        tn = heap_node[parent]; heap_node[parent] = heap_node[i]; heap_node[i] = tn
        i = parent
    return size + 1


@jit
def _heap_pop(heap_dist, heap_node, size):
    """Remove and return the ``(dist, node)`` minimum; returns the new size too."""
    top_dist = heap_dist[0]
    top_node = heap_node[0]
    size -= 1
    heap_dist[0] = heap_dist[size]
    heap_node[0] = heap_node[size]
    i = 0
    while True:
        left = 2 * i + 1
        if left >= size:
            break
        smallest = left
        right = left + 1
        if right < size and (
            heap_dist[right] < heap_dist[left]
            or (
                heap_dist[right] == heap_dist[left]
                and heap_node[right] < heap_node[left]
            )
        ):
            smallest = right
        if heap_dist[i] < heap_dist[smallest] or (
            heap_dist[i] == heap_dist[smallest]
            and heap_node[i] <= heap_node[smallest]
        ):
            break
        td = heap_dist[smallest]; heap_dist[smallest] = heap_dist[i]; heap_dist[i] = td
        tn = heap_node[smallest]; heap_node[smallest] = heap_node[i]; heap_node[i] = tn
        i = smallest
    return top_dist, top_node, size


@jit
def _search_from(
    source, rowptr, col, eid, weight, weighted,
    parent, parent_edge, dist, hops, stamp, token,
    heap_dist, heap_node, settled,
):
    """One single-source search, writing parent pointers into caller scratch.

    BFS when ``weighted`` is false, Dijkstra otherwise. ``stamp``/``token`` mark
    reached nodes so the O(n) scratch never has to be cleared between sources.
    Both branches mirror the reference implementation's traversal order exactly.
    """
    stamp[source] = token
    parent[source] = -1
    parent_edge[source] = -1
    dist[source] = 0.0
    hops[source] = 0

    if not weighted:
        # A plain FIFO queue reusing heap_node as its backing store.
        head = 0
        tail = 0
        heap_node[tail] = source
        tail += 1
        while head < tail:
            node = heap_node[head]
            head += 1
            for pos in range(rowptr[node], rowptr[node + 1]):
                nb = col[pos]
                if stamp[nb] == token:
                    continue
                stamp[nb] = token
                parent[nb] = node
                parent_edge[nb] = eid[pos]
                hops[nb] = hops[node] + 1
                dist[nb] = hops[nb]
                heap_node[tail] = nb
                tail += 1
        return

    size = _heap_push(heap_dist, heap_node, 0, 0.0, source)
    while size > 0:
        d, node, size = _heap_pop(heap_dist, heap_node, size)
        if settled[node] == token:
            continue
        settled[node] = token
        for pos in range(rowptr[node], rowptr[node + 1]):
            nb = col[pos]
            if settled[nb] == token:
                continue
            cand = d + weight[eid[pos]]
            if stamp[nb] != token or cand < dist[nb]:
                stamp[nb] = token
                dist[nb] = cand
                hops[nb] = hops[node] + 1
                parent[nb] = node
                parent_edge[nb] = eid[pos]
                size = _heap_push(heap_dist, heap_node, size, cand, nb)


def _path_lengths_body(
    block_start, group_start, group_source, cand_dst,
    rowptr, col, eid, weight, weighted,
    parent, parent_edge, dist, hops, stamp, heap_dist, heap_node, settled,
):
    """Pass 1: reachability, distance and hop count for every candidate.

    No path is walked here. Knowing each path's length up front turns the flat
    path buffer into an exact prefix-sum layout, so pass 2 can fill it from
    several threads without any of them needing to know what the others found.
    """
    n_cand = cand_dst.shape[0]
    out_dist = np.zeros(n_cand, dtype=np.float64)
    out_hops = np.zeros(n_cand, dtype=np.int64)
    reached = np.zeros(n_cand, dtype=np.bool_)
    n_blocks = block_start.shape[0] - 1

    for b in prange(n_blocks):
        token = 0
        for g in range(block_start[b], block_start[b + 1]):
            token += 1
            _search_from(
                group_source[g], rowptr, col, eid, weight, weighted,
                parent[b], parent_edge[b], dist[b], hops[b], stamp[b], token,
                heap_dist[b], heap_node[b], settled[b],
            )
            for i in range(group_start[g], group_start[g + 1]):
                target = cand_dst[i]
                if stamp[b, target] != token:
                    continue
                reached[i] = True
                out_dist[i] = dist[b, target]
                out_hops[i] = hops[b, target]
    return out_dist, out_hops, reached


_path_lengths_kernel = jit_parallel(_path_lengths_body)


def _path_fill_body(
    block_start, group_start, group_source, cand_dst,
    rowptr, col, eid, weight, weighted,
    reached, edge_offset, node_offset,
    edge_flat, node_flat,
    parent, parent_edge, dist, hops, stamp, heap_dist, heap_node, settled,
):
    """Pass 2: walk each candidate's path into its preallocated slot.

    Re-running the search costs one extra traversal, but the search was only
    ~9% of this routine's cost even before compilation, and paying it buys a
    layout that is fixed before any thread starts writing -- which is what makes
    the output independent of the scheduler.
    """
    n_blocks = block_start.shape[0] - 1
    for b in prange(n_blocks):
        token = 0
        for g in range(block_start[b], block_start[b + 1]):
            token += 1
            _search_from(
                group_source[g], rowptr, col, eid, weight, weighted,
                parent[b], parent_edge[b], dist[b], hops[b], stamp[b], token,
                heap_dist[b], heap_node[b], settled[b],
            )
            for i in range(group_start[g], group_start[g + 1]):
                if not reached[i]:
                    continue
                # Walk target -> source. The reference reverses the path before
                # use; order within a slot is irrelevant here because every
                # consumer either counts or sums over the whole slot.
                node = cand_dst[i]
                e_at = edge_offset[i]
                n_at = node_offset[i]
                while parent[b, node] != -1:
                    edge_flat[e_at] = parent_edge[b, node]
                    e_at += 1
                    node = parent[b, node]
                    if parent[b, node] != -1:
                        # Interior nodes only: the source itself is excluded,
                        # as is the target, which was never appended.
                        node_flat[n_at] = node
                        n_at += 1


_path_fill_kernel = jit_parallel(_path_fill_body)


def _path_norms_body(flat, offset, count, load, order, eps):
    """Pass 3: the normalized p-norm of ``load`` over each candidate's slot.

    Mirrors the reference ``_norm``: ``(sum((v + eps) ** p) / (len + eps)) **
    (1 / p)``, and 0 for an empty slot.
    """
    n = offset.shape[0]
    out = np.zeros(n, dtype=np.float64)
    inv = 1.0 / order
    for i in prange(n):
        k = count[i]
        if k <= 0:
            continue
        total = 0.0
        base = offset[i]
        for j in range(k):
            total += (load[flat[base + j]] + eps) ** order
        out[i] = (total / (k + eps)) ** inv
    return out


_path_norms_kernel = jit_parallel(_path_norms_body)


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


def strided_spanning_forest_mask(
    num_nodes, src, dst, offset, stride, max_edges=None
):
    """Forest from a coprime strided edge scan, without an ``O(m)`` order array.

    When ``gcd(stride, m) == 1``, ``(offset + step * stride) % m`` visits every
    edge exactly once. This is the allocation-free traversal used by the
    ``fast-randst`` backbone.
    """
    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    m = int(src.shape[0])
    if m == 0:
        return np.zeros(0, dtype=bool)
    offset = int(offset) % m
    stride = int(stride) % m
    if stride == 0 or math.gcd(stride, m) != 1:
        raise ValueError(f"stride must be coprime with the edge count ({m})")
    limit = num_nodes if max_edges is None else int(max_edges)
    limit = max(0, min(limit, num_nodes))
    if limit == 0:
        return np.zeros(m, dtype=bool)
    return np.asarray(
        _strided_forest_mask_kernel(
            num_nodes, src, dst, offset, stride, limit
        )
    )


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
    """LCA per pair; ``-1`` when the endpoints lie in different components.

    Thread count comes from whatever
    :func:`~scaffold.utils.workers.parallel_threads` scope encloses the call;
    the result is identical at any of them, since each pair is resolved
    independently.
    """
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(_lca_kernel(src, dst, depth, root, up))


def tree_terms(cu, cv, cand, lca, depth, sp, nq, vcon, weight, edge_p, node_q, eps):
    """Per-candidate dilation and path-congestion norms; see ``_tree_terms_body``."""
    dil, econ_path, vcon_path, path_len = _tree_terms_kernel(
        np.ascontiguousarray(cu, dtype=np.int64),
        np.ascontiguousarray(cv, dtype=np.int64),
        np.ascontiguousarray(cand, dtype=np.int64),
        np.ascontiguousarray(lca, dtype=np.int64),
        np.ascontiguousarray(depth, dtype=np.int64),
        np.ascontiguousarray(sp, dtype=np.float64),
        np.ascontiguousarray(nq, dtype=np.float64),
        np.ascontiguousarray(vcon, dtype=np.float64),
        np.ascontiguousarray(weight, dtype=np.float64),
        float(edge_p),
        float(node_q),
        float(eps),
    )
    return np.asarray(dil), np.asarray(econ_path), np.asarray(vcon_path), np.asarray(path_len)


def tree_score(dil, econ_path, vcon_path, maxima, alpha, beta_edge, beta_node, eps):
    """Combine the three normalized terms into the SCAFFOLD score."""
    d_max, e_max, v_max = maxima
    return np.asarray(
        _tree_score_kernel(
            np.ascontiguousarray(dil, dtype=np.float64),
            np.ascontiguousarray(econ_path, dtype=np.float64),
            np.ascontiguousarray(vcon_path, dtype=np.float64),
            float(d_max),
            float(e_max),
            float(v_max),
            float(alpha),
            float(beta_edge),
            float(beta_node),
            float(eps),
        )
    )


def multi_source_paths(
    num_nodes, rowptr, col, eid, cand_src, cand_dst, weight=None, workers: int = 1
):
    """Shortest path from each candidate's source to its target, all at once.

    Candidates must arrive sorted by source. Returns
    ``(reached, dist, hops, edge_flat, edge_offset, node_flat, node_offset)``,
    where the two ``*_flat`` arrays hold every path's edge ids and interior node
    ids concatenated, and the ``*_offset`` arrays say where each candidate's
    slice begins. A candidate's slices have length ``hops[i]`` and
    ``hops[i] - 1`` respectively.

    Returning a flat buffer rather than a list of lists is the point: the
    congestion counters become one ``bincount`` over the whole buffer instead of
    a Python dict updated a few million times, and the norms become a strided
    scan. Both are what made this loop slow.
    """
    n_cand = int(cand_dst.shape[0])
    empty_i = np.zeros(0, dtype=np.int64)
    if n_cand == 0:
        return (
            np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64), empty_i,
            empty_i, empty_i, empty_i, empty_i,
        )

    weighted = weight is not None
    weight_arr = (
        np.ascontiguousarray(weight, dtype=np.float64)
        if weighted
        else np.zeros(1, dtype=np.float64)
    )

    # Group boundaries: one search serves every candidate leaving a source.
    boundary = np.flatnonzero(np.diff(cand_src)) + 1
    group_start = np.concatenate(
        (np.zeros(1, dtype=np.int64), boundary.astype(np.int64),
         np.array([n_cand], dtype=np.int64))
    )
    group_source = np.ascontiguousarray(cand_src[group_start[:-1]], dtype=np.int64)
    n_groups = int(group_source.shape[0])

    # Deal the groups into contiguous blocks of roughly equal candidate count,
    # one per worker. Balancing on candidates rather than on groups matters:
    # a single high-degree source can own a large share of the work.
    #
    # Below a few hundred candidates the thread dispatch costs more than the
    # searches do, and SCAFFOLD-Heap calls this with a few dozen candidates
    # thousands of times over, so the threshold is load-bearing rather than a
    # micro-optimization. Measured crossover is well under 1,000 candidates
    # (1,336 candidates already ran 1.6x faster on 8 threads).
    if n_cand < PARALLEL_PATH_MIN_CANDIDATES:
        workers = 1
    n_blocks = max(1, min(int(workers), n_groups))
    if n_blocks == 1:
        block_start = np.array([0, n_groups], dtype=np.int64)
    else:
        cuts = np.linspace(0, n_cand, n_blocks + 1)[1:-1]
        block_start = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.unique(np.searchsorted(group_start[1:-1], cuts) + 1).astype(np.int64),
                np.array([n_groups], dtype=np.int64),
            )
        )
        block_start = np.unique(block_start)
        n_blocks = int(block_start.shape[0]) - 1

    # Per-block scratch: each block owns a row for as long as it runs, which is
    # why no thread ids are needed and no two blocks can ever collide.
    heap_capacity = max(int(num_nodes), int(eid.shape[0])) + 1
    parent = np.empty((n_blocks, num_nodes), dtype=np.int64)
    parent_edge = np.empty((n_blocks, num_nodes), dtype=np.int64)
    dist = np.empty((n_blocks, num_nodes), dtype=np.float64)
    hops = np.empty((n_blocks, num_nodes), dtype=np.int64)
    stamp = np.zeros((n_blocks, num_nodes), dtype=np.int64)
    settled = np.zeros((n_blocks, num_nodes), dtype=np.int64)
    heap_dist = np.empty((n_blocks, heap_capacity), dtype=np.float64)
    heap_node = np.empty((n_blocks, heap_capacity), dtype=np.int64)

    args = (
        block_start, group_start, group_source, cand_dst,
        rowptr, col, eid, weight_arr, weighted,
    )
    scratch = (parent, parent_edge, dist, hops, stamp, heap_dist, heap_node, settled)

    out_dist, out_hops, reached = _path_lengths_kernel(*args, *scratch)

    edge_count = np.where(reached, out_hops, 0)
    node_count = np.maximum(edge_count - 1, 0)
    edge_offset = np.zeros(n_cand, dtype=np.int64)
    node_offset = np.zeros(n_cand, dtype=np.int64)
    np.cumsum(edge_count[:-1], out=edge_offset[1:])
    np.cumsum(node_count[:-1], out=node_offset[1:])

    edge_flat = np.zeros(int(edge_count.sum()), dtype=np.int64)
    node_flat = np.zeros(int(node_count.sum()), dtype=np.int64)

    # Reset the visit stamps: pass 2 restarts its token count from zero.
    stamp.fill(0)
    settled.fill(0)
    _path_fill_kernel(
        *args, reached, edge_offset, node_offset, edge_flat, node_flat, *scratch
    )
    return reached, out_dist, edge_count, edge_flat, edge_offset, node_flat, node_offset


def path_norms(flat, offset, count, load, order, eps):
    """Normalized ``order``-norm of ``load`` over each candidate's path slice."""
    if offset.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        _path_norms_kernel(
            np.ascontiguousarray(flat, dtype=np.int64),
            np.ascontiguousarray(offset, dtype=np.int64),
            np.ascontiguousarray(count, dtype=np.int64),
            np.ascontiguousarray(load, dtype=np.float64),
            float(order),
            float(eps),
        )
    )


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
