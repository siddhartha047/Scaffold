"""Optional shortest-path backbones adapted from the research implementation.

SLSF compares multiple rooted BFS/Dijkstra trees by estimated mean stretch;
this is a heuristic, without the theoretical low-stretch guarantees of
recursive literature constructions. RandSPF uses the randomized traversal
already used by the package's local-search initializer. NetworkX stays lazy;
no research checkout, PyTorch, or PyG is required.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "SLSF and RandSPF require NetworkX; install scaffold-sparse[networkx]."
    ) from exc

from ._llst import _RandomShortestPathTree


def _component_forest(graph, build_component, max_edges):
    """Adapt a component builder to the canonical graph/mask package API."""
    if graph.edge_weight is not None and (
        not np.all(np.isfinite(graph.edge_weight)) or np.any(graph.edge_weight <= 0)
    ):
        raise ValueError("Shortest-path backbones require finite, strictly positive edge weights")
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
        # Finish Dijkstra when spanning a component: discovery of its last
        # vertex does not mean that every shortest path has been settled.
        component_budget = None if remaining >= len(nodes) - 1 else remaining
        tree = build_component(component, max_edges=component_budget)
        for u, v in tree.edges():
            mask[ids[(min(u, v), max(u, v))]] = True
        remaining -= tree.number_of_edges()
    return mask


def scalable_low_stretch_forest(graph, max_edges=None, seed=None, **options):
    """SLSF mask; options match the research multi-root stretch heuristic."""
    search = _ScalableLowStretch(seed=seed, **options)
    return _component_forest(graph, search._build_component_tree, max_edges)


def random_shortest_path_forest(graph, max_edges=None, seed=None):
    """Seeded random-root BFS/Dijkstra forest with randomized tie breaking."""
    search = _RandomShortestPathTree(None if seed is None else int(seed))

    def build_component(G, max_edges=None):
        root = search._rng.choice(list(G.nodes()))
        traversal = search._random_dijkstra_tree if nx.is_weighted(G) else search._random_bfs_tree
        parent = traversal(G, root, max_edges=max_edges)
        tree = nx.Graph()
        tree.add_nodes_from(G.nodes())
        tree.add_edges_from((u, v) for u, v in parent.items() if v is not None)
        return tree

    return _component_forest(graph, build_component, max_edges)


class _ScalableLowStretch:
    """Multi-root search from sparsifiers/scalable_low_stretch_tree.py."""

    def __init__(
        self,
        num_roots=8,
        eval_sample_size=8192,
        exact_eval_threshold=20000,
        seed=None,
        verbose=False,
        fast_mode=False,
    ):
        self.num_roots = max(1, int(num_roots))
        self.eval_sample_size = int(eval_sample_size) if eval_sample_size is not None else None
        self.exact_eval_threshold = max(0, int(exact_eval_threshold))
        self.seed = None if seed is None else int(seed)
        self.verbose = verbose
        self._eps = 1e-12
        self._rng = random.Random(self.seed)
        self.fast_mode = bool(fast_mode)

    def _build_component_tree(self, G, max_edges=None):
        edge_limit = max(0, G.number_of_nodes() - 1)
        if max_edges is not None:
            edge_limit = min(edge_limit, max(0, int(max_edges)))
        if edge_limit <= 0:
            T = nx.Graph()
            T.add_nodes_from(G.nodes(data=True))
            return T
        if G.number_of_edges() <= max(0, G.number_of_nodes() - 1) and nx.is_tree(G):
            if edge_limit >= G.number_of_edges():
                return G.copy()

        adj, degrees, is_weighted = self._build_adjacency(G)

        if edge_limit < max(0, G.number_of_nodes() - 1):
            # Stretch scoring requires a connected tree. For a deliberately
            # sub-tree budget, use the best degree root and stop the SPT
            # traversal as soon as the requested number of edges is found.
            root = min(G.nodes(), key=lambda node: (-degrees[node], self._node_key(node)))
            parent, _, _ = self._shortest_path_tree(
                adj,
                degrees,
                root,
                is_weighted,
                max_edges=edge_limit,
            )
            T = nx.Graph()
            T.add_nodes_from(G.nodes(data=True))
            for node, pred in parent.items():
                if pred is not None:
                    T.add_edge(node, pred, **(G.get_edge_data(node, pred) or {}))
            return T

        if self.fast_mode:
            # Use the first candidate root without the Tarjan-LCA comparison.
            roots = self._candidate_roots(G, adj, degrees, is_weighted)
            root = roots[0] if roots else next(iter(G.nodes()))
            parent, _, _ = self._shortest_path_tree(adj, degrees, root, is_weighted)
            T = nx.Graph()
            T.add_nodes_from(G.nodes(data=True))
            for node, pred in parent.items():
                if pred is None:
                    continue
                T.add_edge(node, pred, **(G.get_edge_data(node, pred) or {}))
            if self.verbose:
                print(
                    f"[SLSF-fast] chosen_root={root} candidates={len(roots)} "
                    f"(stretch scoring skipped)"
                )
            return T

        edge_list = list(G.edges(data=True))
        eval_edges = self._sample_eval_edges(edge_list)
        roots = self._candidate_roots(G, adj, degrees, is_weighted)

        best = None
        for root in roots:
            parent, _, dist = self._shortest_path_tree(adj, degrees, root, is_weighted)
            if len(parent) != G.number_of_nodes():
                continue
            stretch = self._estimate_average_stretch(eval_edges, parent, dist, root)
            if best is None or stretch < best[0]:
                best = (stretch, root, parent)

        if best is None:
            fallback_root = next(iter(G.nodes()))
            parent, _, _ = self._shortest_path_tree(adj, degrees, fallback_root, is_weighted)
            best = (float("inf"), fallback_root, parent)

        _, root, parent = best
        T = nx.Graph()
        T.add_nodes_from(G.nodes(data=True))
        for node, pred in parent.items():
            if pred is None:
                continue
            T.add_edge(node, pred, **(G.get_edge_data(node, pred) or {}))

        if self.verbose:
            score_str = "inf" if math.isinf(best[0]) else f"{best[0]:.4f}"
            print(
                f"[SLSF] chosen_root={root} candidates={len(roots)} "
                f"estimated_mean_stretch={score_str}"
            )
        return T

    def _build_adjacency(self, G):
        degrees = {u: G.degree(u) for u in G.nodes()}
        weighted = nx.is_weighted(G)
        adj = {u: [] for u in G.nodes()}

        for u, v, data in G.edges(data=True):
            w = self._edge_weight_from_data(data) if weighted else 1.0
            adj[u].append((v, w))
            adj[v].append((u, w))

        for nbrs in adj.values():
            nbrs.sort(key=lambda item: (-degrees[item[0]], self._node_key(item[0])))
        return adj, degrees, weighted

    def _sample_eval_edges(self, edge_list):
        m = len(edge_list)
        if m == 0:
            return []
        if m <= self.exact_eval_threshold:
            return edge_list
        if self.eval_sample_size is None or self.eval_sample_size <= 0:
            return edge_list
        if self.eval_sample_size >= m:
            return edge_list
        sample_idx = self._rng.sample(range(m), self.eval_sample_size)
        return [edge_list[i] for i in sample_idx]

    def _candidate_roots(self, G, adj, degrees, is_weighted):
        nodes = list(G.nodes())
        if len(nodes) <= self.num_roots:
            return nodes

        ordered_by_degree = sorted(nodes, key=lambda u: (-degrees[u], self._node_key(u)))
        roots = []
        roots.extend(ordered_by_degree[: min(2, self.num_roots)])

        seed = ordered_by_degree[0]
        a, _, _ = self._farthest_from(adj, degrees, seed, is_weighted)
        b, parent_a, dist_a = self._farthest_from(adj, degrees, a, is_weighted)
        roots.extend([a, b])

        path_ab = self._path_from_parent(parent_a, a, b)
        roots.extend(self._path_landmarks(path_ab, dist_a, a))

        seen = set()
        deduped = []
        for root in roots:
            if root not in seen:
                seen.add(root)
                deduped.append(root)
            if len(deduped) >= self.num_roots:
                return deduped

        remaining = [u for u in nodes if u not in seen]
        if remaining:
            take = min(self.num_roots - len(deduped), len(remaining))
            for root in self._rng.sample(remaining, take):
                deduped.append(root)
        return deduped

    def _farthest_from(self, adj, degrees, source, is_weighted):
        parent, depth, dist = self._shortest_path_tree(adj, degrees, source, is_weighted)
        del depth
        farthest = max(
            dist,
            key=lambda node: (dist[node], degrees[node], self._node_key(node)),
        )
        return farthest, parent, dist

    def _path_from_parent(self, parent, root, target):
        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            if cur == root:
                break
            cur = parent[cur]
        path.reverse()
        return path

    def _path_landmarks(self, path, dist_from_root, root):
        if not path:
            return []
        targets = []
        total = dist_from_root[path[-1]] - dist_from_root[root]
        for frac in (0.25, 0.5, 0.75):
            target = frac * total
            best = min(
                path,
                key=lambda node: (
                    abs((dist_from_root[node] - dist_from_root[root]) - target),
                    self._node_key(node),
                ),
            )
            targets.append(best)
        return targets

    def _shortest_path_tree(self, adj, degrees, root, is_weighted, max_edges=None):
        if is_weighted:
            return self._dijkstra_tree(adj, degrees, root, max_edges=max_edges)
        return self._bfs_tree(adj, root, max_edges=max_edges)

    def _bfs_tree(self, adj, root, max_edges=None):
        parent = {root: None}
        depth = {root: 0}
        dist = {root: 0.0}
        if max_edges is not None and max_edges <= 0:
            return parent, depth, dist
        queue = deque([root])

        while queue:
            u = queue.popleft()
            for v, _ in adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                dist[v] = dist[u] + 1.0
                if max_edges is not None and len(parent) - 1 >= max_edges:
                    return parent, depth, dist
                queue.append(v)
        return parent, depth, dist

    def _dijkstra_tree(self, adj, degrees, root, max_edges=None):
        parent = {root: None}
        dist = {root: 0.0}
        if max_edges is not None and max_edges <= 0:
            return parent, {root: 0}, dist
        heap = [(0.0, -degrees[root], self._node_key(root), root)]

        while heap:
            d, _, _, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")) + self._eps:
                continue

            for v, w in adj[u]:
                nd = d + max(w, self._eps)
                current = dist.get(v)
                if current is None or nd < current - self._eps:
                    discovered = current is None
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, -degrees[v], self._node_key(v), v))
                    if discovered and max_edges is not None and len(parent) - 1 >= max_edges:
                        depth = {}
                        for node in sorted(dist, key=lambda item: (dist[item], self._node_key(item))):
                            pred = parent[node]
                            depth[node] = 0 if pred is None else depth[pred] + 1
                        return parent, depth, dist
                elif abs(nd - current) <= self._eps and self._prefer_parent(u, parent[v], degrees):
                    parent[v] = u

        depth = {}
        for node in sorted(dist, key=lambda item: (dist[item], self._node_key(item))):
            pred = parent[node]
            depth[node] = 0 if pred is None else depth[pred] + 1
        return parent, depth, dist

    def _estimate_average_stretch(self, edge_list, parent, dist_root, root):
        if not edge_list:
            return 0.0

        nodes = list(parent.keys())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)

        parent_idx = [-1] * n
        children = [[] for _ in range(n)]
        dist_idx = [0.0] * n
        root_idx = node_to_idx[root]

        for node, pred in parent.items():
            i = node_to_idx[node]
            dist_idx[i] = dist_root[node]
            if pred is None:
                continue
            p = node_to_idx[pred]
            parent_idx[i] = p
            children[p].append(i)

        queries = [[] for _ in range(n)]
        lca_idx = [-1] * len(edge_list)
        weights = [1.0] * len(edge_list)
        endpoints = [None] * len(edge_list)

        for qid, (u, v, data) in enumerate(edge_list):
            ui = node_to_idx[u]
            vi = node_to_idx[v]
            endpoints[qid] = (ui, vi)
            weights[qid] = max(self._edge_weight_from_data(data), self._eps)
            if ui == vi:
                lca_idx[qid] = ui
                continue
            queries[ui].append((vi, qid))
            queries[vi].append((ui, qid))

        uf_parent = list(range(n))
        uf_rank = [0] * n
        ancestor = list(range(n))
        black = [False] * n

        def find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return ra
            if uf_rank[ra] < uf_rank[rb]:
                ra, rb = rb, ra
            uf_parent[rb] = ra
            if uf_rank[ra] == uf_rank[rb]:
                uf_rank[ra] += 1
            return ra

        stack = [(root_idx, 0)]
        while stack:
            u, state = stack.pop()
            if state == 0:
                ancestor[find(u)] = u
                stack.append((u, 1))
                for child in reversed(children[u]):
                    stack.append((child, 0))
                continue

            black[u] = True
            for v, qid in queries[u]:
                if black[v] and lca_idx[qid] == -1:
                    lca_idx[qid] = ancestor[find(v)]

            p = parent_idx[u]
            if p != -1:
                rep = union(p, u)
                ancestor[rep] = p

        total = 0.0
        for qid, (ui, vi) in enumerate(endpoints):
            lca = lca_idx[qid]
            if lca == -1:
                raise RuntimeError("Failed to resolve LCA while estimating stretch")
            tree_dist = dist_idx[ui] + dist_idx[vi] - 2.0 * dist_idx[lca]
            total += tree_dist / weights[qid]
        return total / len(edge_list)

    def _prefer_parent(self, candidate, current, degrees):
        cand_rank = (-degrees[candidate], self._node_key(candidate))
        curr_rank = (-degrees[current], self._node_key(current))
        return cand_rank < curr_rank

    @staticmethod
    def _edge_weight_from_data(data):
        if not data:
            return 1.0
        return float(data.get("weight", 1.0))

    @staticmethod
    def _node_key(node):
        return repr(node)
