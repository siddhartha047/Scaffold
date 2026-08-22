# API reference

```python
import scaffold
```

The distribution is `scaffold-sparse`; the canonical import name is `scaffold`.
`import scaffold_sparse` works too and exposes the same API.

---

## Entry points

### `scaffold.sparsify(G, method="fast", keep_ratio=None, num_edges=None, seed=None, target_ratio=None, **kwargs)`

Run any variant by name. `method` is `"exact"`, `"heap"`, `"fast"` or
`"sample"`; a leading `"scaffold-"` / `"scaffold_"` is accepted, so config
strings work unchanged.

### `scaffold.exact(G, keep_ratio=None, num_edges=None, backbone="fast-maxst", seed=None, target_ratio=None, **kwargs)`
### `scaffold.heap(G, ...)`
### `scaffold.fast(G, ...)`

Return a [`ScaffoldResult`](#scaffoldresult).

### `scaffold.sample(G, keep_ratio=None, num_edges=None, seed=None, **kwargs)`

Returns a [`ScaffoldScores`](#scaffoldscores) — per-edge weights, **not** a
subgraph. `keep_ratio` / `num_edges` are optional and only set the default
budget for later `draw()` calls.

---

## Shared arguments

| argument | type | meaning |
|---|---|---|
| `G` | graph | `Graph`, `networkx.Graph`, PyG `Data`, `scipy.sparse`, `(2, m)` `edge_index`, or a dense square adjacency |
| `keep_ratio` | `float` in `[0, 1]` | budget as a fraction: `ceil(keep_ratio · m)` undirected edges |
| `target_ratio` | `float` in `[0, 1]` | alias for `keep_ratio`, matching the research configuration name |
| `num_edges` | `int` | exact undirected edge budget |
| `backbone` | `str`, mask, or callable | support forest — see [backbones.md](backbones.md) |
| `seed` | `int` or `None` | seeds every stochastic component; never touches the global NumPy RNG |

`keep_ratio` and `target_ratio` mean the same thing. Specify at most one of
them, and do not combine either one with `num_edges`; contradictory budgets
raise `ValueError`.

### Objective knobs (all variants)

| argument | default | meaning |
|---|---|---|
| `alpha` | `1.0` | dilation exponent |
| `beta_edge` | `1.0` | edge-congestion exponent; `0` disables the term |
| `beta_node` | `1.0` | node-congestion exponent; `0` disables the term |
| `edge_norm_p` | `2.0` | `p`-norm order along the detour's edges |
| `node_norm_q` | `2.0` | `q`-norm order along the detour's interior nodes |

`inf` norm orders raise `NotImplementedError` rather than being silently
approximated — the root-prefix formulation computes sums, and a max-on-path
query needs a different data structure.

You may also pass a prebuilt `params=ScoreParams(...)`; individual keyword
overrides win over it.

### `scaffold.exact` extras

| argument | default | meaning |
|---|---|---|
| `batch_size` | `1` | edges committed per rescoring round |
| `max_rounds` | `None` | safety cap on rescoring rounds |
| `verbose` | `False` | per-round progress |

### `scaffold.heap` extras

| argument | default | meaning |
|---|---|---|
| `top_k` | `16` | candidates rescored per round |
| `add_per_round` | `1` | edges committed per round, per cluster |
| `clusters` | `None` | `None`/`1` for one global heap, an `int`, or a per-node label array |
| `cluster_method` | `"bfs"` | `"bfs"`, `"metis"` (needs `pymetis`), `"random"` |
| `local_radius` | `1` | invalidation radius, in hops, after an insertion |
| `dirty_limit` | `64` | cap on invalidations per round; `0` = unlimited |
| `score_form` | `"max"` | `"max"` or `"product"` — see [algorithms.md](algorithms.md) |

### `scaffold.fast` extras

| argument | default | meaning |
|---|---|---|
| `selection` | `"topk"` | `"topk"` (one global top-k) or `"rounds"` (paper Algorithm 1) |
| `clusters` | `None` | `"rounds"` only |
| `cluster_method` | `"bfs"` | `"rounds"` only |
| `sample_size` | `64` | `"rounds"` only: candidates drawn per cluster per round |
| `add_per_round` | `8` | `"rounds"` only: edges committed per cluster per round |
| `weighted_paths` | `False` | measure detour length as a sum of tree edge weights instead of hops |
| `return_scores` | `False` | put the per-edge score array in `metadata["scores"]` |

### `scaffold.sample` extras

| argument | default | meaning |
|---|---|---|
| `tree_count` | `8` | random spanning forests to aggregate over (`R`) |
| `aggregate_lambda` | `1.0` | mix of score vs. backbone frequency; `0` = frequency only |
| `backbone` | `"fixed-maxst"` | `"fixed-maxst"` or `"rotate-randst"` |
| `scheme` | `"systematic"` | sampling scheme (only systematic π-ps is implemented) |
| `weighted_paths` | `False` | as above |
| `verbose` | `False` | per-forest progress |

---

## `ScaffoldResult`

Returned by `exact`, `heap`, `fast`, and by `ScaffoldScores.draw()`.

### Counting convention

- `edge_index` / `edge_weight` are **symmetric** — both directions present,
  ready for a GNN. `edge_index.shape[1] == 2 * sparse_edges`.
- `original_edges`, `sparse_edges` and `keep_ratio` count **undirected** edges,
  because that is what a budget means.

### Properties

| | |
|---|---|
| `.mask` | `(m,)` bool array over the input's canonical undirected edges |
| `.edge_ids` | indices of the kept edges |
| `.undirected_edge_index` | `(2, k)`, `src < dst` |
| `.undirected_edge_weight` | `(k,)` or `None` |
| `.edge_index` | `(2, 2k)` symmetric |
| `.edge_weight` | `(2k,)` or `None` |
| `.num_nodes` | node count (unchanged by sparsification) |
| `.original_edges`, `.sparse_edges`, `.keep_ratio` | undirected counts |
| `.method` | `"exact"` / `"heap"` / `"fast"` / `"sample"` |
| `.metadata` | dict; see below |

### Methods

| | |
|---|---|
| `.to_numpy()` | `(edge_index, edge_weight)` |
| `.to_torch()` | the same as torch tensors |
| `.to_pyg(copy_from=None)` | a `Data` with features, labels and masks forwarded |
| `.to_networkx(weight_key="weight")` | a `networkx.Graph`, original labels restored |
| `.to_scipy(fmt="csr")` | a symmetric sparse adjacency |
| `.num_components()` | connected components of the result |
| `.summary()` | one-line human-readable summary |

### `metadata`

Always present: `method`, `num_nodes`, `original_edges`, `sparse_edges`,
`keep_ratio`, `backbone`, `support_budget_mode`, `backbone_edges`,
`target_edges`, `selected_edges`, `budget_trimmed`, `runtime`, `alpha`,
`beta_edge`, `beta_node`,
`base_components`, `delta_min`, `below_connectivity_floor`.

For `exact`, `heap`, and `fast`, `budget_trimmed` is the number of edges
randomly removed from the complete support forest when the target lies below
the connectivity floor. The trim is reproducible with `seed=`. A concrete
`sample` draw reports the same situation through `forced_edges`,
`sampled_edges=0`, `budget_trimmed`, and `below_connectivity_floor=True`;
successive draws re-trim the complete forest.

Per method: `exact` adds `rounds`, `scored_candidates`, `batch_size`; `heap`
adds `rescored_candidates`, `heap_rebuilds`, `top_k`, `clusters`, `score_form`;
`fast` adds `selection`, `mandatory_edges`, `total_stretch` (and `scores`,
`dilation`, `edge_congestion_path`, `node_congestion_path` with
`return_scores=True`).

---

## `ScaffoldScores`

Returned by `scaffold.sample`. Subclasses `ScaffoldResult`, so every property
above still works — but note that it covers **all** edges of the input, since
nothing has been sparsified.

| | |
|---|---|
| `.scores` / `.pi` | `(m,)` SCAFFOLD weight per undirected edge |
| `.edge_weight` | the same, symmetrized to `(2m,)` |
| `.mandatory` | `(m,)` bool: cross-component edges |
| `.backbone` | `(m,)` bool: the deterministic MaxST forest |
| `.order` | tree-locality permutation of the edges |
| `.sampler` | the underlying `ScaffoldSampler` |
| `.inclusion_probabilities(keep_ratio=None, num_edges=None)` | `(m,)`, sums to the budget, all `≤ 1` |
| `.draw(keep_ratio=None, num_edges=None, seed=None)` | one `ScaffoldResult` |
| `.to_dict()` | `{(u, v): weight}` with the original node labels |

### `ScaffoldSampler`

```python
from scaffold.algorithms.sample import ScaffoldSampler

sampler = scores.sampler
sampler.coverage(keep_ratio=0.2, epochs=(1, 10, 100))
sampler.save("weights.npz")
sampler = ScaffoldSampler.load("weights.npz", graph)
```

`load` verifies that the artifact describes exactly this graph — same node
count, same canonical edge list — and raises `ValueError` otherwise. Weights
are indexed positionally, so a silent mismatch would corrupt every draw.

---

## `Graph`

The internal representation. Always canonical: undirected, `src < dst`, no self
loops, no duplicates, sorted by `(src, dst)`.

```python
graph = scaffold.normalize_graph(G)

graph.num_nodes, graph.num_edges
graph.edge_index          # (2, m) int64
graph.edge_weight         # (m,) float64 or None
graph.src, graph.dst
graph.is_weighted
graph.node_labels         # original labels, when the input had them
graph.subgraph_edge_index(mask)
```

`edge_weight is None` means *unweighted*, which is materially different from
"all ones": unweighted detour length is a hop count, which the fast integer
kernels can compute directly.

Duplicate input pairs collapse to one edge; `reduce="first"` (default),
`"sum"`, `"mean"`, `"min"` or `"max"` selects how their weights combine.

---

## Scoring primitives

```python
from scaffold.scoring import ScoreParams, tree_scores, path_scores
```

### `tree_scores(num_nodes, src, dst, tree_mask, weight=None, params=None, tree_index=None, weighted_paths=False)`

The `O(m log n + n)` tree-prefix kernel. `tree_mask` **must** select an acyclic
edge set. Returns a dict: `dil`, `length`, `econ_path`, `vcon_path`, `score`,
`mandatory`, `candidate_mask`, `edge_congestion`, `node_congestion`,
`total_stretch`, `maxima`. Entries for tree edges are zero.

### `path_scores(num_nodes, src, dst, support_mask, weight=None, params=None, candidate_ids=None)`

The literal shortest-path definition. Works on **any** support graph, forest or
not. Slower; used as the reference the tree kernel is tested against, and the
only option once cycles exist.

---

## Backbones

```python
from scaffold.backbone import build_backbone, register_backbone, available_backbones

available_backbones()
mask = build_backbone(graph, "fast-maxst", max_edges=None, seed=0)
register_backbone("mine", my_builder)
```

See [backbones.md](backbones.md).

---

## Demo graphs

```python
scaffold.grid_graph(rows, cols=None, periodic=False, diagonals=False, weight=None, seed=None)
scaffold.grid_positions(rows, cols=None)
scaffold.ring_of_cliques(num_cliques, clique_size)
scaffold.random_geometric(num_nodes, radius, seed=None)   # -> (graph, positions)
```

`weight` is `None`, `"random"` or `"distance"`.

---

## Plotting

Needs matplotlib.

```python
from scaffold import viz

viz.draw_graph(graph, mask=result.mask, highlight=backbone_mask)
viz.draw_result(result)
viz.draw_edge_scores(graph, scores.scores)
fig, results = viz.compare_methods(graph, keep_ratio=0.5)
```

---

## PyTorch Geometric

Needs torch and torch-geometric.

```python
from scaffold.pyg import (
    sparsify_data, sample_edge_weight, ScaffoldTransform, ScaffoldResampler
)
```

See [pytorch-geometric.md](pytorch-geometric.md).
