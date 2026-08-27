r"""The SCAFFOLD objective, and two ways to evaluate it.

For a support graph ``H`` and a candidate edge ``e = (u, v)`` not in ``H``, let
``P_H(e)`` be the shortest path between ``u`` and ``v`` in ``H``. SCAFFOLD
scores ``e`` by how badly ``H`` currently serves it:

* **dilation** -- how much longer the detour is than the edge itself::

      dil[e] = dist_H(u, v) / w(e)

* **edge congestion** -- how many other candidates are routed over the same
  support edges, aggregated as a normalized p-norm over ``P_H(e)``'s edges;
* **node congestion** -- the same over ``P_H(e)``'s *interior* nodes, q-norm.

Each term is divided by its maximum over the candidate set and raised to a
tunable exponent, so the three are commensurable and the score is::

    score(e) = ((dil + eps)      / (D_max + eps)) ** alpha
             * ((eConPath + eps) / (E_max + eps)) ** beta_edge
             * ((vConPath + eps) / (V_max + eps)) ** beta_node

Candidates whose endpoints sit in different components of ``H`` have no path;
they get ``dil = score = inf`` and are reported as **mandatory** -- adding them
is the only way to preserve connectivity, so they are always taken first.

Two evaluators
--------------

:func:`tree_scores` is the fast one and the reason SCAFFOLD scales. When ``H``
is a *forest*, every path aggregate is a root-prefix difference, so all ``|I|``
candidates are scored together in ``O(m log n + n)`` -- no path is ever
enumerated. This is ``ExactTreeScores`` in the paper.

:func:`path_scores` is the literal definition: one shortest-path search per
distinct source, valid for any ``H`` (forest or not). It is what
SCAFFOLD-Greedy and SCAFFOLD-Heap use once they have started adding cycles, and
it is the reference the tree kernel is tested against.

Three traps in the tree formulation, each of which produces plausible-looking
wrong numbers rather than a crash:

1. Cross-component candidates must be excluded from the congestion counters,
   not merely flagged with an infinite dilation.
2. Edge prefixes index tree *edges* on ``root -> x`` (so a root contributes 0)
   while node prefixes index *nodes* inclusively (so a root contributes its own
   term). A single recurrence self-references at a root, where ``parent == x``.
3. The subtree fold must skip roots for the same reason, or ``eCon[root]``
   silently doubles.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .kernels import (
    _bfs_parents_kernel,
    _edge_congestion_kernel,
    _node_congestion_kernel,
    _root_prefix_kernel,
    build_csr,
    build_tree_index,
    depth_order,
    parent_edge_weights,
    tree_lca,
)
from .utils.validation import validate_norm_order

__all__ = ["ScoreParams", "tree_scores", "path_scores", "combine_terms"]

EPS = 1e-8


class ScoreParams:
    """The five knobs of the SCAFFOLD objective.

    Parameters
    ----------
    alpha:
        Dilation exponent. ``alpha=1, beta_*=0`` gives pure stretch.
    beta_edge, beta_node:
        Edge- and node-congestion exponents. Setting both to 0 disables the
        congestion terms entirely and makes scoring markedly cheaper.
    edge_norm_p, node_norm_q:
        Norm orders for aggregating congestion along a path. ``2.0`` (the
        default) is a soft maximum; larger values approach a hard max.
    """

    __slots__ = ("alpha", "beta_edge", "beta_node", "edge_norm_p", "node_norm_q", "eps")

    def __init__(
        self,
        alpha: float = 1.0,
        beta_edge: float = 1.0,
        beta_node: float = 1.0,
        edge_norm_p: float = 2.0,
        node_norm_q: float = 2.0,
        eps: float = EPS,
    ):
        self.alpha = float(alpha)
        self.beta_edge = float(beta_edge)
        self.beta_node = float(beta_node)
        self.edge_norm_p = validate_norm_order(edge_norm_p, "edge_norm_p")
        self.node_norm_q = validate_norm_order(node_norm_q, "node_norm_q")
        self.eps = float(eps)

    @property
    def congestion_free(self) -> bool:
        return self.beta_edge == 0.0 and self.beta_node == 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta_edge": self.beta_edge,
            "beta_node": self.beta_node,
            "edge_norm_p": self.edge_norm_p,
            "node_norm_q": self.node_norm_q,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ScoreParams({self.as_dict()})"


def combine_terms(dil, econ_path, vcon_path, params: ScoreParams, finite_mask=None):
    """Normalize the three terms by their maxima and combine them."""
    eps = params.eps
    dil = np.asarray(dil, dtype=np.float64)
    econ_path = np.asarray(econ_path, dtype=np.float64)
    vcon_path = np.asarray(vcon_path, dtype=np.float64)
    if finite_mask is None:
        finite_mask = np.isfinite(dil)

    finite = dil[finite_mask]
    d_max = float(finite.max()) if finite.size else eps
    e_max = float(econ_path.max()) if econ_path.size else eps
    v_max = float(vcon_path.max()) if vcon_path.size else eps

    score = np.full(dil.shape, np.inf, dtype=np.float64)
    if finite_mask.any():
        score[finite_mask] = (
            ((dil[finite_mask] + eps) / (d_max + eps)) ** params.alpha
            * ((econ_path[finite_mask] + eps) / (e_max + eps)) ** params.beta_edge
            * ((vcon_path[finite_mask] + eps) / (v_max + eps)) ** params.beta_node
        )
    return score, (d_max, e_max, v_max)


# ----------------------------------------------------------------------
# the tree-prefix scorer  (paper: ExactTreeScores)
# ----------------------------------------------------------------------
def tree_scores(
    num_nodes: int,
    src,
    dst,
    tree_mask,
    weight=None,
    params: Optional[ScoreParams] = None,
    tree_index=None,
    weighted_paths: bool = False,
):
    """Score every non-tree edge exactly, in ``O(m log n + n)``.

    ``src`` / ``dst`` are the canonical undirected edge list and ``tree_mask``
    selects a spanning forest. All returned arrays have length ``m``; entries
    for tree edges are zero and are excluded from every statistic, matching the
    greedy variants, which only ever score ``E \\ E(H)``.

    ``weight`` is always the *denominator* of the dilation, i.e. ``w_G(e)``.
    ``weighted_paths`` selects the numerator: ``False`` (default) counts hops,
    ``True`` sums tree edge weights.

    Returns a dict with ``dil``, ``length``, ``econ_path``, ``vcon_path``,
    ``score``, ``mandatory``, ``candidate_mask``, ``edge_congestion``,
    ``node_congestion`` and ``total_stretch``.
    """
    params = params or ScoreParams()
    num_nodes = int(num_nodes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    tree_mask = np.ascontiguousarray(tree_mask, dtype=bool)
    m = int(src.shape[0])
    eps = params.eps
    weight = (
        np.ones(m, dtype=np.float64)
        if weight is None
        else np.ascontiguousarray(weight, dtype=np.float64)
    )

    if tree_index is None:
        tree_index = build_tree_index(num_nodes, src[tree_mask], dst[tree_mask])
    depth, root, up, parent = tree_index[0], tree_index[1], tree_index[2], tree_index[3]
    order = depth_order(depth)

    candidate_mask = ~tree_mask
    cand_idx = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)

    dil = np.zeros(m, dtype=np.float64)
    length = np.zeros(m, dtype=np.int64)
    econ_path = np.zeros(m, dtype=np.float64)
    vcon_path = np.zeros(m, dtype=np.float64)
    score = np.zeros(m, dtype=np.float64)
    mandatory = np.zeros(m, dtype=bool)

    if cand_idx.size == 0:
        return {
            "dil": dil,
            "length": length,
            "econ_path": econ_path,
            "vcon_path": vcon_path,
            "score": score,
            "mandatory": mandatory,
            "candidate_mask": candidate_mask,
            "edge_congestion": np.zeros(num_nodes, dtype=np.int64),
            "node_congestion": np.zeros(num_nodes, dtype=np.int64),
            "total_stretch": 0.0,
            "maxima": (eps, eps, eps),
        }

    cu = src[cand_idx]
    cv = dst[cand_idx]
    lca = tree_lca(cu, cv, depth, root, up)
    connected = lca >= 0

    # TRAP 1: congestion counters see connected candidates only.
    lca_c = np.ascontiguousarray(lca[connected], dtype=np.int64)
    cu_c = np.ascontiguousarray(cu[connected], dtype=np.int64)
    cv_c = np.ascontiguousarray(cv[connected], dtype=np.int64)

    econ = np.asarray(
        _edge_congestion_kernel(cu_c, cv_c, lca_c, parent, order, num_nodes)
    )
    vcon = np.asarray(_node_congestion_kernel(cu_c, cv_c, lca_c, econ))

    # TRAP 2: edge prefixes exclude the root's (nonexistent) parent edge;
    # node prefixes include the root itself.
    sp = np.asarray(
        _root_prefix_kernel(econ, parent, order, float(params.edge_norm_p), eps, False)
    )
    nq = np.asarray(
        _root_prefix_kernel(vcon, parent, order, float(params.node_norm_q), eps, True)
    )

    conn_idx = cand_idx[connected]
    disc_idx = cand_idx[~connected]

    # Hop counts drive the norm denominators: |path_edges| and |path_nodes|.
    path_len = (
        depth[cu_c].astype(np.int64)
        + depth[cv_c].astype(np.int64)
        - 2 * depth[lca_c].astype(np.int64)
    )
    length[conn_idx] = path_len

    if weighted_paths:
        pw = parent_edge_weights(
            depth, src[tree_mask], dst[tree_mask], weight[tree_mask], num_nodes
        )
        wdepth = np.asarray(_root_prefix_kernel(pw, parent, order, 1.0, 0.0, False))
        path_dist = wdepth[cu_c] + wdepth[cv_c] - 2.0 * wdepth[lca_c]
    else:
        path_dist = path_len.astype(np.float64)

    dil[conn_idx] = path_dist / np.maximum(weight[conn_idx], eps)
    dil[disc_idx] = np.inf
    mandatory[disc_idx] = True

    # eConPath: p-norm over the path's tree edges, count = path_len.
    sigma_e = sp[cu_c] + sp[cv_c] - 2.0 * sp[lca_c]
    econ_path[conn_idx] = np.maximum(sigma_e, 0.0) / (path_len + eps)
    econ_path[conn_idx] **= 1.0 / float(params.edge_norm_p)

    # vConPath: q-norm over the path's *interior* nodes, count = path_len - 1.
    q = float(params.node_norm_q)
    sigma_v = (
        nq[cu_c]
        + nq[cv_c]
        - 2.0 * nq[lca_c]
        + (vcon[lca_c] + eps) ** q
        - (vcon[cu_c] + eps) ** q
        - (vcon[cv_c] + eps) ** q
    )
    vcon_path[conn_idx] = np.maximum(sigma_v, 0.0) / (path_len - 1 + eps)
    vcon_path[conn_idx] **= 1.0 / q

    finite = dil[conn_idx]
    d_max = float(finite.max()) if finite.size else eps
    e_max = float(econ_path[cand_idx].max()) if cand_idx.size else eps
    v_max = float(vcon_path[cand_idx].max()) if cand_idx.size else eps

    score[conn_idx] = (
        ((dil[conn_idx] + eps) / (d_max + eps)) ** params.alpha
        * ((econ_path[conn_idx] + eps) / (e_max + eps)) ** params.beta_edge
        * ((vcon_path[conn_idx] + eps) / (v_max + eps)) ** params.beta_node
    )
    score[disc_idx] = np.inf

    return {
        "dil": dil,
        "length": length,
        "econ_path": econ_path,
        "vcon_path": vcon_path,
        "score": score,
        "mandatory": mandatory,
        "candidate_mask": candidate_mask,
        "edge_congestion": econ,
        "node_congestion": vcon,
        "total_stretch": float(dil[conn_idx].sum()),
        "maxima": (d_max, e_max, v_max),
    }


# ----------------------------------------------------------------------
# the literal shortest-path scorer  (reference; works on any support graph)
# ----------------------------------------------------------------------
class PathScorer:
    """Scores candidates against an arbitrary support graph ``H``.

    Rebuilding CSR after every edge insertion would dominate the runtime, so
    the support adjacency is kept as growable per-node lists and rebuilt only
    when :meth:`add_edge` has been called. Unweighted graphs use BFS; weighted
    graphs use Dijkstra.
    """

    def __init__(self, num_nodes, src, dst, weight=None, support_mask=None):
        self.num_nodes = int(num_nodes)
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.dst = np.ascontiguousarray(dst, dtype=np.int64)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        self.num_edges = int(self.src.shape[0])
        mask = (
            np.zeros(self.num_edges, dtype=bool)
            if support_mask is None
            else np.ascontiguousarray(support_mask, dtype=bool)
        )
        self.support_mask = mask
        self._dirty = True
        # Scratch buffers reused across BFS calls (see _bfs_parents_kernel).
        self._parent = np.empty(self.num_nodes, dtype=np.int64)
        self._parent_edge = np.empty(self.num_nodes, dtype=np.int64)
        self._dist = np.empty(self.num_nodes, dtype=np.int64)
        self._stamp = np.full(self.num_nodes, -1, dtype=np.int64)
        self._token = 0

    def add_edge(self, edge_id: int):
        self.support_mask[int(edge_id)] = True
        self._dirty = True

    def add_edges(self, edge_ids):
        self.support_mask[np.asarray(edge_ids, dtype=np.int64)] = True
        self._dirty = True

    def _ensure_csr(self):
        if not self._dirty:
            return
        ids = np.flatnonzero(self.support_mask).astype(np.int64, copy=False)
        rowptr, col, local = build_csr(self.num_nodes, self.src[ids], self.dst[ids])
        self._rowptr = np.ascontiguousarray(rowptr)
        self._col = np.ascontiguousarray(col)
        # Map CSR slots back to *global* edge ids so paths report real edges.
        self._eid = (
            np.ascontiguousarray(ids[local]) if local.size else np.zeros(0, dtype=np.int64)
        )
        self._dirty = False

    # -- path queries -------------------------------------------------------
    def _bfs(self, source):
        self._ensure_csr()
        self._token += 1
        _bfs_parents_kernel(
            int(source),
            self._rowptr,
            self._col,
            self._eid,
            self._parent,
            self._parent_edge,
            self._dist,
            self._stamp,
            self._token,
        )
        return self._stamp == self._token

    def _dijkstra(self, source):
        import heapq

        self._ensure_csr()
        self._token += 1
        weight = self.weight
        dist = np.full(self.num_nodes, np.inf, dtype=np.float64)
        dist[source] = 0.0
        self._parent[source] = -1
        self._parent_edge[source] = -1
        self._stamp[source] = self._token
        heap = [(0.0, int(source))]
        settled = np.zeros(self.num_nodes, dtype=bool)
        while heap:
            d, node = heapq.heappop(heap)
            if settled[node]:
                continue
            settled[node] = True
            for pos in range(self._rowptr[node], self._rowptr[node + 1]):
                nb = int(self._col[pos])
                if settled[nb]:
                    continue
                cand = d + float(weight[self._eid[pos]])
                if cand < dist[nb]:
                    dist[nb] = cand
                    self._parent[nb] = node
                    self._parent_edge[nb] = self._eid[pos]
                    self._stamp[nb] = self._token
                    heapq.heappush(heap, (cand, nb))
        self._dist_float = dist
        return settled

    def _walk(self, target):
        """Path from a scored source to ``target`` as (node list, edge id list)."""
        nodes = [int(target)]
        edges = []
        node = int(target)
        while self._parent[node] != -1:
            edges.append(int(self._parent_edge[node]))
            node = int(self._parent[node])
            nodes.append(node)
        nodes.reverse()
        edges.reverse()
        return nodes, edges

    def evaluate(self, candidate_ids, params: ScoreParams):
        """Score ``candidate_ids`` against the current support graph.

        Congestion is measured over exactly the candidate set passed in, which
        is what makes cluster-local and top-k evaluation meaningful: a
        candidate is penalised for competing with the batch it is ranked
        against.
        """
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        n_cand = candidate_ids.size
        dil = np.full(n_cand, np.inf, dtype=np.float64)
        econ_path = np.zeros(n_cand, dtype=np.float64)
        vcon_path = np.zeros(n_cand, dtype=np.float64)
        connected = np.zeros(n_cand, dtype=bool)
        if n_cand == 0:
            return {
                "dil": dil,
                "econ_path": econ_path,
                "vcon_path": vcon_path,
                "score": np.zeros(0, dtype=np.float64),
                "mandatory": np.zeros(0, dtype=bool),
                "path_edges": [],
                "path_nodes": [],
            }

        weighted = self.weight is not None
        graph_weight = (
            self.weight if weighted else np.ones(self.num_edges, dtype=np.float64)
        )

        # Group by source so one search serves every candidate leaving it.
        sources = self.src[candidate_ids]
        order = np.argsort(sources, kind="stable")
        path_edges = [None] * n_cand
        path_nodes = [None] * n_cand
        edge_load: Dict[int, int] = {}
        node_load: Dict[int, int] = {}

        pos = 0
        while pos < order.size:
            slot = order[pos]
            source = int(sources[slot])
            end = pos
            while end < order.size and int(sources[order[end]]) == source:
                end += 1

            reachable = self._dijkstra(source) if weighted else self._bfs(source)
            for k in range(pos, end):
                s = order[k]
                target = int(self.dst[candidate_ids[s]])
                if not reachable[target]:
                    continue
                nodes, edges = self._walk(target)
                if weighted:
                    distance = float(self._dist_float[target])
                else:
                    distance = float(len(edges))
                connected[s] = True
                dil[s] = distance / max(
                    float(graph_weight[candidate_ids[s]]), params.eps
                )
                path_edges[s] = edges
                interior = nodes[1:-1]
                path_nodes[s] = interior
                for e in edges:
                    edge_load[e] = edge_load.get(e, 0) + 1
                for x in interior:
                    node_load[x] = node_load.get(x, 0) + 1
            pos = end

        if not params.congestion_free:
            for s in range(n_cand):
                if not connected[s]:
                    continue
                econ_path[s] = _norm(
                    [edge_load.get(e, 0) for e in path_edges[s]],
                    params.edge_norm_p,
                    params.eps,
                )
                vcon_path[s] = _norm(
                    [node_load.get(x, 0) for x in path_nodes[s]],
                    params.node_norm_q,
                    params.eps,
                )

        score, _ = combine_terms(dil, econ_path, vcon_path, params, finite_mask=connected)
        return {
            "dil": dil,
            "econ_path": econ_path,
            "vcon_path": vcon_path,
            "score": score,
            "mandatory": ~connected,
            "path_edges": path_edges,
            "path_nodes": path_nodes,
        }


def _norm(values, order, eps):
    """Normalized p-norm: ``(mean of (v + eps) ** p) ** (1/p)``."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (float(v) + eps) ** order
    return (total / (len(values) + eps)) ** (1.0 / order)


def path_scores(
    num_nodes,
    src,
    dst,
    support_mask,
    weight=None,
    params: Optional[ScoreParams] = None,
    candidate_ids=None,
):
    """Convenience wrapper: score candidates against ``support_mask`` directly.

    This is the reference implementation of the objective. It works for any
    support graph, forest or not, and is what :func:`tree_scores` is validated
    against.
    """
    params = params or ScoreParams()
    support_mask = np.ascontiguousarray(support_mask, dtype=bool)
    if candidate_ids is None:
        candidate_ids = np.flatnonzero(~support_mask)
    scorer = PathScorer(num_nodes, src, dst, weight=weight, support_mask=support_mask.copy())
    return scorer.evaluate(candidate_ids, params)
