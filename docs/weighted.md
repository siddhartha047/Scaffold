# Weighted graphs and feature-derived weights

All five Scaffold variants accept weighted graphs and retain the selected
edges' original weights. Use `weighted_paths=True` to measure supporting paths
by the sum of their edge weights, including in Fast and Sample.

## Existing edge weights

Weights come from NetworkX's `weight` attribute, SciPy/dense adjacency values,
PyG's `edge_weight` (or scalar `edge_attr`), or an explicit edge list:

```python
import scaffold
from scaffold.adapters import from_edge_index

G = from_edge_index(edge_index, num_nodes=n, edge_weight=edge_weight)
result = scaffold.fast(
    G, keep_ratio=0.6, backbone="fast-randsf", weighted_paths=True,
    seed=0, workers=8,
)
A_sparse = result.to_scipy()
```

`from_edge_index` aligns weights while canonicalizing undirected edges and
removing loops/duplicates. Its default duplicate reduction is `first`; pass
`reduce="mean"` when needed. For an already normalized `Graph`,
`G.with_weight(weights)` expects one value per **canonical undirected edge**.
Never assign weights in the original edge-list order after normalization.

Scoring requires finite, nonnegative weights. Zero-weight edges are allowed;
the dilation denominator is clamped by `ScoreParams.eps` (default `1e-8`).
RandSPF, SLSF, and LLSF require strictly positive weights for weighted stretch.

## Derive weights from node features

```python
G = scaffold.with_feature_weights(graph, features=X, metric="cosine")
result = scaffold.batch(
    G, keep_ratio=0.6, backbone="randsf", weighted_paths=True, seed=0,
    sample_size=512, add_per_round=64, clusters=10, workers=8,
)

# Reuse weighted Sample preprocessing for repeated sparse draws.
scores = scaffold.sample(
    G, backbone="fixed-randsf", weighted_paths=True, tree_count=8, seed=0,
)
view = scores.draw(keep_ratio=0.6, seed=1)
```

The helpers accept dense NumPy arrays, CPU/GPU Torch tensors (copied to CPU),
and SciPy sparse feature matrices. With PyG, omitted `features` uses `data.x`.
Feature rows must follow the normalized node order; for NetworkX this is
`scaffold.normalize_graph(graph).node_labels` (node insertion order).
Isolated nodes and original labels are preserved, and the input is not mutated.
Existing edge weights are replaced only in the returned graph.

| `metric` | Default `kind="similarity"` | `kind="distance"` |
|---|---|---|
| `cosine` | `(1 + cos(x_u, x_v)) / 2` | `1 - cos(x_u, x_v)` |
| `euclidean` / `l2` | `1 / (1 + ‖x_u − x_v‖₂)` | Euclidean distance |
| `sqeuclidean` | `1 / (1 + ‖x_u − x_v‖₂²)` | Squared Euclidean distance |
| `manhattan` / `cityblock` / `l1` | `1 / (1 + ‖x_u − x_v‖₁)` | Manhattan distance |
| `chebyshev` | `1 / (1 + max(abs(x_u − x_v)))` | Chebyshev distance |
| `dot` | Dot products, min–max scaled over existing edges | Unsupported |
| `uniform` | All ones; no features needed | All ones |

Cosine, Euclidean affinity, dot, and uniform follow the research formulas.
A zero feature vector has cosine zero, so its shifted similarity is `0.5`.
Constant dot products become ones. Unlike the research's literal zeros, the
helper floors outputs at `min_weight=1e-12` so they also work with strictly
positive low-stretch backbones. Non-finite features or computed distances
raise an error. Set a different positive `min_weight` if needed.

`scaffold.feature_edge_weights(G, X, ...)` returns only the float64 weight
array, in canonical edge order. `scaffold.with_feature_weights(G, X, ...)`
returns a `Graph` carrying that array.

### Additional scikit-learn distances

Install the `metrics` extra alongside any existing extras (e.g.
`scaffold-sparse[speed,metrics]`; use the repository install URL until the PyPI
release). Named metrics use scikit-learn's
[DistanceMetric API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html):

```python
G = scaffold.with_feature_weights(
    graph, X, metric="minkowski", metric_kwargs={"p": 3}, kind="distance",
)
G = scaffold.with_feature_weights(graph, X, metric="canberra")
# Supply required metric parameters explicitly, e.g. VI for Mahalanobis.
G = scaffold.with_feature_weights(
    graph, X, metric="mahalanobis", metric_kwargs={"VI": inverse_covariance},
    kind="distance",
)
```

These metric names follow the installed `DistanceMetric` backend; not every
`pairwise_distances`/SciPy distance string is accepted. A callable is also
supported: `metric=lambda a, b: float(...)`, with optional `metric_kwargs`.
It must return a symmetric, finite, nonnegative distance. For any such distance
`d`, the default similarity transform is `1 / (1 + d)`.

Only **existing edge pairs** are evaluated. Common metrics are vectorized in
batches (`batch_size=8192`), with a feature-dimension cap on temporary arrays.
No all-pairs node matrix is built. Additional backend metrics/callables evaluate
one pair at a time to bound memory; they may be much slower on large graphs.
Built-in metrics do not import or require scikit-learn. Feature weighting is
separate preprocessing and is not included in the README's unweighted timings.

## What the weights mean

With `weighted_paths=True`, dilation for an omitted edge `e` is
`sum(w[a] for a in P_e) / max(w[e], eps)`. Greedy, Heap, and Batch use
Dijkstra on the current support. Fast and Sample use weighted root-prefix
distances along their scoring forests. Congestion still counts supporting-path
usage; it does not treat edge weights as traffic demands.

**The supplied scalar is used directly as a path length; Scaffold does not
invert similarities automatically.** The research affinity formulas are
available for reproducing its settings. When smaller values should mean
closer/cheaper connections, use `kind="distance"` explicitly.

| Method | Default `weighted_paths` | Explicit `True` |
|---|---|---|
| Greedy, Heap, Batch | `None`: Dijkstra if weights exist, BFS otherwise | Weighted shortest paths |
| Fast, Sample | `False`: hop-count numerator | Sum of forest-edge weights |

These defaults preserve previous results. In the explicit `False` mode, the
numerator counts hops but dilation still divides by the candidate edge weight.
To ignore weights entirely, pass `G.with_weight(None)`. Unweighted graphs
produce the same scoring under either mode.

Backbone construction is independent of the scoring flag. MaxSF/MinSF order
by descending/ascending weight; their fast versions use buckets. RandSF and
Fast-RandSF remain randomized. SPF uses BFS for unweighted inputs and Dijkstra
for weighted inputs, with degree-selected roots. RandSPF and the low-stretch
backbones also account for weighted path lengths.

Sample's `scores.scores` are **sampling priorities**, not the input graph's
weights. `scores.draw(...)` preserves the selected input weights. Newly saved
artifacts record graph weights and the path mode; loading rejects weight/mode
mismatches. Legacy unweighted artifacts remain supported. Legacy weighted
artifacts lack weight metadata and must be rebuilt.

## Weighted PyG usage

```python
weighted = scaffold.with_feature_weights(data, metric="euclidean")  # uses data.x
view = scaffold.fast(
    weighted, keep_ratio=0.6, backbone="fast-randsf", weighted_paths=True,
).to_pyg()
out = model(view.x, view.edge_index, edge_weight=view.edge_weight)
```

Use the edge-weight argument supported by your GNN layer. Preserving weights in
`Data` does not make a layer consume them automatically. Features, labels, and
split masks survive conversion. A weighted graph from this helper also works
with `ScaffoldResampler(..., weighted_paths=True)`.

Runnable example: [`07_weighted_graphs.py`](../examples/07_weighted_graphs.py).
Offline research comparison:
[`compare_weighted_research.py`](../validation/compare_weighted_research.py).
