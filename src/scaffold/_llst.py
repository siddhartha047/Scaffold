"""Local-search low-stretch backbones, ported from the research LLST.

The edge-swap loop, sampled objective, LCA distances, and cycle sampling
follow ICML_SPARSIFICATION/sparsifiers/local_search_low_stretch_tree.py.
This module is imported only when the LLST backbone is requested. It needs
NetworkX, but never imports PyTorch, PyG, or the research checkout.
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Optional

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The LLSF (legacy LLST) backbone requires NetworkX; install scaffold-sparse[networkx]."
    ) from exc

from .backbone import DEFAULT_BUCKETS, build_backbone, canonical_backbone_name
from .graph import Graph, from_edge_index
from .kernels import spanning_forest_mask


def local_search_low_stretch_forest(
    graph: Graph,
    max_edges: Optional[int] = None,
    seed=None,
    **options,
) -> np.ndarray:
    """Return a local-search forest mask in the canonical input edge order.

    Start from GLST (or ``init_support``) in each connected component. At
    each pass, add a non-tree edge and remove one edge from its fundamental
    cycle, accepting the best strictly improving swap. The objective is
    ``sum(dist_T(u,v) / w(u,v) for (u,v) in E)`` with weighted tree distances.
    All edge weights must be finite and strictly positive.

    ``max_passes`` defaults to 10 per component. Candidate, evaluation and
    cycle sample sizes default to zero, meaning exhaustive evaluation. With
    evaluation sampling, only the sampled objective is guaranteed to improve.
    A partial ``max_edges`` budget returns the budgeted initializer for any
    component it cannot span; swaps require a complete component tree.

    This is a small-graph reference: one exhaustive pass can evaluate
    O(m*n) swaps, each rebuilding an index and measuring O(m) stretches.
    The public ``build_backbone`` entry point rejects inputs above 1,000
    undirected edges by default, before importing this module. Its
    ``max_input_edges`` option explicitly raises or lowers that runtime guard.
    """
    search = _LocalSearch(seed=seed, **options)
    if graph.edge_weight is not None and (
        not np.all(np.isfinite(graph.edge_weight)) or np.any(graph.edge_weight <= 0)
    ):
        raise ValueError("LLSF (legacy LLST) requires finite, strictly positive edge weights")
    mask = np.zeros(graph.num_edges, dtype=bool)
    remaining = max(0, graph.num_nodes - 1) if max_edges is None else max(0, int(max_edges))
    if remaining == 0 or graph.num_edges == 0:
        return mask
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    ids = {}
    for i, (u, v) in enumerate(zip(graph.src, graph.dst)):
        u, v = int(u), int(v)
        ids[(u, v)] = i
        attrs = {} if graph.edge_weight is None else {"weight": float(graph.edge_weight[i])}
        G.add_edge(u, v, **attrs)
    for nodes in nx.connected_components(G):
        if remaining == 0:
            break
        if len(nodes) <= 1:
            continue
        component = G.subgraph(nodes).copy()
        # A full component must finish its initializer, especially Dijkstra:
        # discovering the last node is earlier than settling its shortest path.
        component_budget = None if remaining >= len(nodes) - 1 else remaining
        tree = search._build_component_tree(component, max_edges=component_budget)
        for u, v in tree.edges():
            mask[ids[(min(u, v), max(u, v))]] = True
        remaining -= tree.number_of_edges()
    return mask


class _LocalSearch:
    """Component-local implementation of the research edge-swap search."""

    def __init__(
        self,
        max_passes=10,
        init_support="glst",
        fast_tree_buckets=DEFAULT_BUCKETS,
        glst_alpha=1.0,
        glst_eta=1.0,
        candidate_strategy="random",
        candidate_sample_size=0,
        eval_sample_size=0,
        cycle_sample_size=0,
        resample_eval_each_pass=False,
        verbose=False,
        seed=None,
    ):
        self.max_passes = max(1, int(max_passes))
        self.init_support = canonical_backbone_name(init_support)
        self.fast_tree_buckets = int(fast_tree_buckets)
        self.glst_alpha = float(glst_alpha)
        self.glst_eta = float(glst_eta)
        self.candidate_strategy = str(candidate_strategy)
        self.candidate_sample_size = max(0, int(candidate_sample_size))
        self.eval_sample_size = max(0, int(eval_sample_size))
        self.cycle_sample_size = max(0, int(cycle_sample_size))
        self.resample_eval_each_pass = bool(resample_eval_each_pass)
        self.verbose = verbose
        self.seed = None if seed is None else int(seed)
        self._eps = 1e-12
        self._rng = random.Random(self.seed)
        if self.candidate_strategy not in ("random", "tree_distance"):
            raise ValueError("LLSF (legacy LLST) candidate_strategy must be 'random' or 'tree_distance'")
        if self.init_support not in (
            "glst",
            "maxst",
            "mst",
            "fast-maxst",
            "fast-mst",
            "randst",
            "fast-randst",
            "spt",
            "slst",
            "randspt",
        ):
            raise ValueError(f"Unknown LLSF (legacy LLST) init_support: {self.init_support!r}")

    def _build_component_tree(self, G, max_edges=None):
        full_tree_edges = max(0, G.number_of_nodes() - 1)
        edge_limit = (
            full_tree_edges
            if max_edges is None
            else min(full_tree_edges, max(0, int(max_edges)))
        )
        if edge_limit <= 0:
            T = nx.Graph()
            T.add_nodes_from(G.nodes(data=True))
            return T
        if G.number_of_edges() <= max(0, G.number_of_nodes() - 1) and nx.is_tree(G):
            if edge_limit >= G.number_of_edges():
                return G.copy()

        initial = self._build_initial_tree(G, max_edges=max_edges)

        # The LLST swap objective requires a connected tree. A lower target is
        # intentionally a partial forest, so return the budgeted initializer
        # directly instead of constructing a full tree and trimming it.
        if edge_limit < full_tree_edges:
            return initial

        best_tree = initial.copy()
        eval_edges = self._sample_eval_edges(G)
        best_stretch = self._total_stretch(G, best_tree, eval_edges=eval_edges)

        if self.verbose:
            sampled_eval_edges = (
                G.number_of_edges() if eval_edges is None else len(eval_edges)
            )
            print(
                f"[LLSF] init total_stretch={best_stretch:.4f} "
                f"mean_stretch={best_stretch / max(G.number_of_edges(), 1):.4f} "
                f"sampled_eval_edges={sampled_eval_edges}"
            )

        passes = 0
        while passes < self.max_passes:
            if passes > 0 and self.resample_eval_each_pass:
                eval_edges = self._sample_eval_edges(G)
                best_stretch = self._total_stretch(G, best_tree, eval_edges=eval_edges)
            move = self._best_improving_swap(
                G,
                best_tree,
                best_stretch,
                eval_edges=eval_edges,
            )
            if move is None:
                break
            best_tree, best_stretch, added_edge, removed_edge = move
            passes += 1
            if self.verbose:
                print(
                    f"[LLSF] pass={passes} add={added_edge} remove={removed_edge} "
                    f"total_stretch={best_stretch:.4f} "
                    f"mean_stretch={best_stretch / max(G.number_of_edges(), 1):.4f}"
                )

        return best_tree

    def _build_initial_tree(self, G, max_edges=None):
        if self.init_support == "randspt":
            builder = _RandomShortestPathTree(self.seed)
            root = builder._rng.choice(list(G.nodes()))
            build = (
                builder._random_dijkstra_tree
                if nx.is_weighted(G)
                else builder._random_bfs_tree
            )
            parent = build(G, root, max_edges=max_edges)
            tree = nx.Graph()
            tree.add_nodes_from(G.nodes())
            for node, pred in parent.items():
                if pred is not None:
                    tree.add_edge(node, pred, **G[node][pred])
            return tree

        nodes = list(G.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        rows = list(G.edges(data=True))
        data = np.asarray([(index[u], index[v]) for u, v, _ in rows], dtype=np.int64).T
        weights = [d.get("weight", 1.0) for _, _, d in rows] if nx.is_weighted(G) else None
        graph = from_edge_index(data, num_nodes=len(nodes), edge_weight=weights)
        if self.init_support == "randst":
            # Preserve the research LLST initializer's seeded random priorities.
            # Public randst uses a permutation with the same distribution, but
            # that would give a different tree for the same integer seed.
            order = np.argsort(
                np.random.default_rng(self.seed).random(graph.num_edges), kind="stable"
            ).astype(np.int64)
            mask = spanning_forest_mask(
                graph.num_nodes, graph.src, graph.dst, order, max_edges=max_edges
            )
        else:
            options = (
                {} if self.init_support == "slst" else {
                    "buckets": self.fast_tree_buckets,
                    "alpha": self.glst_alpha,
                    "eta": self.glst_eta,
                }
            )
            mask = build_backbone(
                graph,
                self.init_support,
                max_edges=max_edges,
                seed=self.seed,
                **options,
            )
        tree = nx.Graph()
        tree.add_nodes_from(nodes)
        for i in np.flatnonzero(mask):
            u, v = nodes[int(graph.src[i])], nodes[int(graph.dst[i])]
            tree.add_edge(u, v, **G[u][v])
        return tree

    def _sample_edges(self, edges, sample_size):
        edge_list = list(edges)
        if sample_size <= 0 or len(edge_list) <= sample_size:
            return edge_list
        return self._rng.sample(edge_list, sample_size)

    def _sample_eval_edges(self, G):
        if self.eval_sample_size <= 0:
            return None
        eval_edges = [self._canon_edge(u, v) for u, v in G.edges()]
        return self._sample_edges(eval_edges, self.eval_sample_size)

    def _best_improving_swap(self, G, T, current_stretch, eval_edges=None):
        best_trial = None
        best_stretch = current_stretch
        tree_index = self._build_tree_index(T)
        tree_edge_set = {self._canon_edge(u, v) for u, v in T.edges()}
        candidate_edges = [
            self._canon_edge(u, v)
            for u, v in G.edges()
            if self._canon_edge(u, v) not in tree_edge_set
        ]

        for candidate_edge in self._choose_candidate_edges(candidate_edges, tree_index):
            u, v = candidate_edge
            path_edges = self._path_edges(u, v, tree_index)
            path_edges = self._sample_edges(path_edges, self.cycle_sample_size)

            edge_data = G.get_edge_data(u, v) or {}
            for remove_u, remove_v in path_edges:
                trial = T.copy()
                trial.add_edge(u, v, **edge_data)
                trial.remove_edge(remove_u, remove_v)
                if not nx.is_tree(trial):
                    continue

                trial_index = self._build_tree_index(trial)
                trial_stretch = self._total_stretch(
                    G,
                    trial,
                    eval_edges=eval_edges,
                    tree_index=trial_index,
                )
                if trial_stretch + self._eps < best_stretch:
                    best_trial = (
                        trial,
                        trial_stretch,
                        candidate_edge,
                        self._canon_edge(remove_u, remove_v),
                    )
                    best_stretch = trial_stretch

        return best_trial

    def _choose_candidate_edges(self, candidate_edges, tree_index):
        if self.candidate_strategy == "random":
            return self._sample_edges(candidate_edges, self.candidate_sample_size)

        if self.candidate_strategy != "tree_distance":
            raise ValueError(
                f"Unknown LLSF (legacy LLST) candidate_strategy: {self.candidate_strategy!r}"
            )

        if self.candidate_sample_size <= 0 or self.candidate_sample_size >= len(
            candidate_edges
        ):
            k = len(candidate_edges)
        else:
            k = self.candidate_sample_size

        scored = (
            (self._tree_distance(u, v, tree_index), (u, v)) for u, v in candidate_edges
        )
        return [edge for _, edge in heapq.nlargest(k, scored, key=lambda item: item[0])]

    def _total_stretch(self, G, T, eval_edges=None, tree_index=None):
        tree_index = self._build_tree_index(T) if tree_index is None else tree_index
        total = 0.0
        sampled_edges = (
            list(eval_edges)
            if eval_edges is not None
            else [self._canon_edge(u, v) for u, v in G.edges()]
        )
        if not sampled_edges:
            return 0.0
        for u, v in sampled_edges:
            data = G.get_edge_data(u, v) or {}
            edge_weight = float(data.get("weight", 1.0))
            edge_weight = max(edge_weight, self._eps)
            dist = self._tree_distance(u, v, tree_index)
            total += dist / edge_weight
        scale = (
            1.0
            if eval_edges is None
            else float(G.number_of_edges()) / float(len(sampled_edges))
        )
        return total * scale

    def _build_tree_index(self, T):
        nodes = list(T.nodes())
        if not nodes:
            return None

        index_of = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        root = nodes[0]
        root_idx = index_of[root]

        parent = [-1] * n
        depth = [0] * n
        dist_to_root = [0.0] * n
        edge_to_parent = [None] * n
        seen = {root}
        stack = [root]

        while stack:
            u = stack.pop()
            iu = index_of[u]
            for v, data in T[u].items():
                if v in seen:
                    continue
                seen.add(v)
                iv = index_of[v]
                parent[iv] = iu
                depth[iv] = depth[iu] + 1
                dist_to_root[iv] = dist_to_root[iu] + float((data or {}).get("weight", 1.0))
                edge_to_parent[iv] = self._canon_edge(u, v)
                stack.append(v)

        log_n = max(1, n.bit_length())
        up = [[root_idx] * n for _ in range(log_n)]
        for idx in range(n):
            up[0][idx] = root_idx if parent[idx] < 0 else parent[idx]
        for level in range(1, log_n):
            prev = up[level - 1]
            curr = up[level]
            for idx in range(n):
                curr[idx] = prev[prev[idx]]

        return {
            "index_of": index_of,
            "parent": parent,
            "depth": depth,
            "dist_to_root": dist_to_root,
            "edge_to_parent": edge_to_parent,
            "up": up,
        }

    def _lca_index(self, idx_u, idx_v, tree_index):
        depth = tree_index["depth"]
        up = tree_index["up"]

        if depth[idx_u] < depth[idx_v]:
            idx_u, idx_v = idx_v, idx_u

        diff = depth[idx_u] - depth[idx_v]
        bit = 0
        while diff:
            if diff & 1:
                idx_u = up[bit][idx_u]
            diff >>= 1
            bit += 1

        if idx_u == idx_v:
            return idx_u

        for level in range(len(up) - 1, -1, -1):
            if up[level][idx_u] != up[level][idx_v]:
                idx_u = up[level][idx_u]
                idx_v = up[level][idx_v]
        return up[0][idx_u]

    def _tree_distance(self, u, v, tree_index):
        if u == v:
            return 0.0
        index_of = tree_index["index_of"]
        dist_to_root = tree_index["dist_to_root"]
        idx_u = index_of[u]
        idx_v = index_of[v]
        idx_lca = self._lca_index(idx_u, idx_v, tree_index)
        return dist_to_root[idx_u] + dist_to_root[idx_v] - 2.0 * dist_to_root[idx_lca]

    def _path_edges(self, u, v, tree_index):
        index_of = tree_index["index_of"]
        parent = tree_index["parent"]
        depth = tree_index["depth"]
        edge_to_parent = tree_index["edge_to_parent"]

        idx_u = index_of[u]
        idx_v = index_of[v]
        path_from_u = []
        path_from_v = []

        while depth[idx_u] > depth[idx_v]:
            path_from_u.append(edge_to_parent[idx_u])
            idx_u = parent[idx_u]
        while depth[idx_v] > depth[idx_u]:
            path_from_v.append(edge_to_parent[idx_v])
            idx_v = parent[idx_v]
        while idx_u != idx_v:
            path_from_u.append(edge_to_parent[idx_u])
            path_from_v.append(edge_to_parent[idx_v])
            idx_u = parent[idx_u]
            idx_v = parent[idx_v]

        path_from_v.reverse()
        return path_from_u + path_from_v

    def _canon_edge(self, u, v):
        return (u, v) if repr(u) <= repr(v) else (v, u)


class _RandomShortestPathTree:
    """Research randomized BFS/Dijkstra traversal shared by LLST and RandSPF."""

    def __init__(self, seed):
        self._rng = random.Random(seed)
        self._eps = 1e-12

    def _random_bfs_tree(self, G, root, max_edges=None):
        adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        for nbrs in adj.values():
            self._rng.shuffle(nbrs)

        parent = {root: None}
        if max_edges is not None and max_edges <= 0:
            return parent
        queue = deque([root])
        added = 0
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                added += 1
                if max_edges is not None and added >= max_edges:
                    return parent
                queue.append(v)
        return parent

    def _random_dijkstra_tree(self, G, root, max_edges=None):
        adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        tie_break = {}
        for u, nbrs in adj.items():
            self._rng.shuffle(nbrs)
            for v in nbrs:
                tie_break[(u, v)] = self._rng.random()

        dist = {root: 0.0}
        parent = {root: None}
        if max_edges is not None and max_edges <= 0:
            return parent
        best_tie = {root: -1.0}
        heap = [(0.0, -1.0, repr(root), root)]

        while heap:
            d, in_tie, _, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")) + self._eps:
                continue
            if in_tie > best_tie.get(u, float("inf")) + self._eps:
                continue

            for v in adj[u]:
                w = float(G[u][v].get("weight", 1.0))
                nd = d + max(w, self._eps)
                edge_tie = tie_break[(u, v)]
                cur = dist.get(v)
                cur_tie = best_tie.get(v, float("inf"))
                if (
                    cur is None
                    or nd < cur - self._eps
                    or (abs(nd - cur) <= self._eps and edge_tie < cur_tie - self._eps)
                ):
                    discovered = cur is None
                    dist[v] = nd
                    parent[v] = u
                    best_tie[v] = edge_tie
                    heapq.heappush(heap, (nd, edge_tie, repr(v), v))
                    if (
                        discovered
                        and max_edges is not None
                        and max(0, len(parent) - 1) >= max_edges
                    ):
                        return parent

        return parent
