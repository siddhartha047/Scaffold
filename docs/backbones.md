# Support backbones

The backbone is the spanning forest SCAFFOLD grows from. It does two jobs:

1. **It is the connectivity guarantee.** A spanning forest of a graph with `n`
   nodes and `c` components has `n − c` edges and, by construction, exactly
   those `c` components. Every edge added afterwards can only merge things
   further, so the output's component count is fixed the moment the backbone is
   built.
2. **It is the reference the objective is measured against.** Dilation is
   `dist_F(u, v) / w(e)` and congestion counts detours *through the forest*, so
   a different backbone means a different ranking of the same candidates.

The default is `fast-maxst`.

![backbones on a grid](images/grid_backbones.png)

---

## The built-in backbones

### `fast-maxst` (default) and `fast-mst`

Bucketed approximate weighted Kruskal. Edge weights are quantized into
`buckets` (default 256) priority classes and ordered with a linear-time counting
sort instead of an `O(m log m)` comparison sort; edges within a class keep input
order. The union-find scan is identical to the exact version.

The ordering is approximate. On the graphs SCAFFOLD targets the difference is
immaterial, and the ordering cost drops from `O(m log m)` to `O(m + buckets)`.

On an **unweighted** graph there is nothing to sort, so `fast-maxst`,
`fast-mst`, `maxst` and `mst` all reduce to the same deterministic scan and
produce identical forests.

```python
scaffold.fast(G, keep_ratio=0.2)                                     # fast-maxst
scaffold.fast(G, keep_ratio=0.2, backbone="fast-mst")
scaffold.fast(G, keep_ratio=0.2, backbone_options={"buckets": 1024})
```

### `maxst` and `mst`

Exact stable Kruskal, `O(m log m)`. Ties break on edge id, so runs are
reproducible. Use these when the weight ordering genuinely matters and you can
afford the sort.

### `fast-randst` and `randst`

Both are randomized Kruskal forests and are redrawn on every call:

- `fast-randst` draws only a seeded offset and coprime stride. The resulting
  arithmetic traversal visits every edge once without allocating an `m`-entry
  permutation. It is the faster, lower-memory choice and matches the research
  tensor backend's fast randomized support.
- `randst` constructs a full uniform random edge permutation before the same
  union-find scan. Its edge order is more thoroughly randomized, but creating
  and storing the permutation costs additional time and `O(m)` memory.

Neither is an exact uniform spanning-tree sampler. The distinction is the
randomness and cost of the *edge order*, not the union-find acceptance rule.

Warmed median construction times with numba on the development machine were:

| undirected edges | `fast-randst` | `randst` | speedup |
|---:|---:|---:|---:|
| 130,560 | 2.4 ms | 4.5 ms | 1.9× |
| 523,264 | 9.9 ms | 22.6 ms | 2.3× |
| 2,095,104 | 134.3 ms | 177.6 ms | 1.3× |

The exact ratio is graph-, seed-, cache-, and machine-dependent. The stable
distinction is that `fast-randst` generates two random integers and no order
array, whereas `randst` generates and stores all `m` permutation entries.

```python
a = scaffold.fast(G, keep_ratio=0.2, backbone="fast-randst", seed=1)
b = scaffold.fast(G, keep_ratio=0.2, backbone="fast-randst", seed=2)
# a.mask != b.mask
```

### `spt`

Multi-source BFS forest rooted at the highest-degree node of each component.
Low diameter, so dilations start out small. Useful when hop distance matters
more than edge weight.

### `glst`

Greedy low-stretch tree. Repeatedly adds the boundary edge maximizing

```
(cut / cut_max)^eta / ((projStretch / stretch_max)^alpha)
```

where `cut[e]` counts graph edges crossing the two components `e` would merge,
and `projStretch[e]` is their mean stretch once `e` is added.

The best-quality backbone here — on the grid it gives the lowest total stretch
of any deterministic option — at `O(n · m · |cut|)`. Guarded at 5,000 edges;
beyond that it will raise rather than hang.

```python
scaffold.greedy(G, keep_ratio=0.5, backbone="glst",
               backbone_options={"eta": 1.0, "alpha": 1.0})
```

### `none`

Start from the empty graph. **Connectivity is then not guaranteed** — you get
the pure score ranking with no structural floor. Occasionally what you want for
an ablation; almost never what you want in production.

---

## Choosing one

| | |
|---|---|
| Large graph, want it fast | `fast-maxst` (default) |
| Weight ordering matters exactly | `maxst` / `mst` |
| Fast randomized support | `fast-randst` |
| More thoroughly shuffled support | `randst` |
| Per-epoch resparsification | `fast-randst`, or `scaffold.sample(backbone="rotate-randst")` |
| Hop distance matters more than weight | `spt` |
| Small graph, want the best skeleton | `glst` |
| Ablation with no structural guarantee | `none` |

Measured on a 14×14 grid (196 nodes, 364 edges) — `total stretch` is the sum
over non-tree edges of `dist_F(u,v)/w(e)`, so lower is a better starting point:

| backbone | forest edges | total stretch |
|---|---|---|
| `fast-maxst` | 195 | 2535 |
| `maxst` | 195 | 2535 |
| `mst` | 195 | 2535 |
| `fast-randst` | 195 | 1719 |
| `randst` | 195 | 1649 |
| `spt` | 195 | 2355 |
| `glst` | 195 | 1961 |

(On this unweighted grid the three weighted variants coincide, as noted above.
Both randomized variants improve stretch here because a random maze has no
long thin comb structure; `randst` happens to win for seed 0. This is a quality
sample, not a claim that one randomized distribution always dominates.)

Reproduce with [`examples/04_backbones_and_tuning.py`](../examples/04_backbones_and_tuning.py).

---

## Passing your own

### A precomputed mask

Any boolean array of length `m` over the graph's canonical edges, or an array of
edge ids:

```python
graph = scaffold.normalize_graph(G)
mask = my_spanning_forest(graph)        # (m,) bool
result = scaffold.fast(graph, keep_ratio=0.2, backbone=mask)
```

### A registered builder

```python
import numpy as np
from scaffold.backbone import register_backbone
from scaffold.kernels import spanning_forest_mask

def degree_ordered_forest(graph, max_edges=None, seed=None, **kwargs):
    """Kruskal over edges sorted by the sum of their endpoint degrees."""
    degree = np.bincount(
        np.concatenate([graph.src, graph.dst]), minlength=graph.num_nodes
    )
    priority = -(degree[graph.src] + degree[graph.dst])
    order = np.lexsort((np.arange(graph.num_edges), priority))
    return spanning_forest_mask(
        graph.num_nodes, graph.src, graph.dst, order, max_edges=max_edges
    )

register_backbone("degree", degree_ordered_forest)
result = scaffold.fast(G, keep_ratio=0.2, backbone="degree")
```

A builder receives `(graph, max_edges=..., seed=..., **options)` and must return
a boolean mask of length `graph.num_edges`.

> **The mask must be acyclic.** SCAFFOLD does not verify this. `scaffold.fast`
> and `scaffold.sample` root the mask as a forest and compute LCAs on it; a mask
> containing a cycle produces wrong scores silently rather than raising.
> `spanning_forest_mask` guarantees acyclicity for any edge ordering, so build
> on top of it.

---

## `max_edges` and partial forests

`build_backbone` itself can construct a deterministic partial forest when
called with `max_edges`:

```python
mask = build_backbone(graph, "fast-maxst", max_edges=50)
```

The result is a genuine partial forest — still acyclic, just not spanning.
The sparsification algorithms intentionally use a different policy when their
budget is smaller than `n − c`: they build the complete support forest and
uniformly randomly drop edges until the exact budget is reached. This matches
the research implementation and avoids favoring the prefix of the backbone's
edge order. The trim is reproducible with `seed=` and is reported in metadata:

```python
result.metadata["below_connectivity_floor"]   # True
result.metadata["delta_min"]                  # (n - c) / m
result.metadata["budget_trimmed"]             # forest edges dropped (greedy)
```

---

## Feature-derived edge weights

If your graph has node features but no edge weights, deriving weights from
feature similarity gives the backbone something meaningful to sort by:

```python
import numpy as np
import scaffold

graph = scaffold.normalize_graph(data)          # PyG Data
x = data.x.numpy()

left, right = x[graph.src], x[graph.dst]
cosine = (left * right).sum(1) / (
    np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
)
weighted = graph.with_weight((cosine + 1) / 2)  # map to [0, 1]

result = scaffold.fast(weighted, keep_ratio=0.2, backbone="fast-maxst")
```

With `fast-maxst` the backbone then prefers edges joining similar nodes, which
on a homophilous graph is usually the right structural prior.
