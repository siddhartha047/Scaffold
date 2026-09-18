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

The 12×12 grid comparison includes LLST beside its GLST initializer. Every
backbone has 143 edges and one connected component; each label reports the
sum of supporting-path lengths for the omitted edges. LLST uses up to 10
exhaustive improving swaps with seed 0. The two randomized backbones also use
seed 0; minimum-weight variants coincide with maximum-weight variants on this
unweighted graph and are omitted from the image.
LLST reduces omitted-edge stretch from 1,161 to 795 (31.5%) in this example.
The [animated comparison](images/grid_backbones.gif) cycles through the same
seven completed forests, ending with GLST and its LLST refinement.

Regenerate the PNG, GIF, and [measurements](images/grid_backbones.json) with:

```bash
python examples/01_grid_demo.py --only backbones --out docs/images
```

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

Construction costs `O(n · m · |cut|)`. GLST also serves as the default
initializer for LLST, which can further reduce its stretch through cycle swaps.
Guarded at 5,000 edges; beyond that it will raise rather than hang.

```python
scaffold.greedy(G, keep_ratio=0.5, backbone="glst",
               backbone_options={"eta": 1.0, "alpha": 1.0})
```

### `llst`

Local-search low-stretch tree, ported from the research implementation. It
builds an initial tree in each component, then repeatedly adds a non-tree
edge and removes an edge on the resulting cycle. Each accepted swap strictly
reduces the evaluated total stretch:

```
sum(dist_T(u, v) / w(u, v) for (u, v) in E(G))
```

Tree distances sum edge weights (or count hops on unweighted inputs). Weighted
LLST inputs must have finite, strictly positive weights. This is a stretch
objective; the subsequent Scaffold growth stage also accounts for congestion.

Install `scaffold-sparse[networkx]` to use LLST. It works with Greedy, Heap,
Batch and Fast and with every graph format accepted by those methods:

```python
result = scaffold.fast(
    G, keep_ratio=0.7, backbone="llst", seed=0,
    backbone_options={"init_support": "maxst", "max_passes": 5},
)

# Or build just the forest, as a mask over canonical edges:
from scaffold.backbone import build_backbone
graph = scaffold.normalize_graph(G)
forest = build_backbone(graph, "llst", seed=0, init_support="maxst", max_passes=5)
```

Options go directly to `build_backbone`, or inside `backbone_options` for the
Scaffold methods:

| option | default | meaning |
|---|---|---|
| `init_support` | `"glst"` | Starting tree; choices below |
| `max_passes` | `10` | Maximum accepted improving swaps per component; minimum 1 |
| `candidate_strategy` | `"random"` | `"random"` or longest current `"tree_distance"` |
| `candidate_sample_size` | `0` | Add-edge candidates per pass; 0 evaluates all |
| `eval_sample_size` | `0` | Original edges used to estimate stretch; 0 evaluates all |
| `cycle_sample_size` | `0` | Removable edges tested per cycle; 0 tests all |
| `resample_eval_each_pass` | `False` | Redraw the objective sample each pass |
| `glst_alpha`, `glst_eta` | `1.0`, `1.0` | GLST initializer exponents |
| `fast_tree_buckets` | `256` | Buckets for fast weighted initializers |
| `verbose` | `False` | Report accepted swaps and evaluated stretch |

Initializers are `glst`, `maxst`, `mst`, `fast-maxst`, `fast-mst`, `randst`,
`randspt`, `fast-randst`, and `spt`; underscore aliases also work for the fast
weighted names. `randspt` is available here as an LLST initializer, not as a
separate registered backbone. LLST preserves the research RandST initializer's
seeded random-priority ordering; the package's standalone `randst` uses a
permutation and can produce a different initial tree for the same seed.

Exhaustive LLST is for small graphs: each pass may test many cycles and build
a tree index for every trial swap. Sampling reduces the work; when
`eval_sample_size` is nonzero, improvements concern the sampled objective,
not necessarily the full-graph objective. The default GLST initializer also
retains its existing 5,000-edge guard.

Disconnected components and isolated nodes are preserved. A direct
`max_edges` cap that cannot span a component returns its budgeted initializer
without swaps. Scaffold methods instead construct the complete LLST forest
and use the package's usual random trim if their final budget is below the
connectivity floor. Sample's specialized fixed/rotating-backbone modes are
unchanged; `backbone="llst"` is not a Sample mode.

See [`examples/05_local_search_backbone.py`](../examples/05_local_search_backbone.py)
and the [research parity check](validation.md#llst-backbone).

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
| Small graph, grow a tree using projected stretch | `glst` |
| Small graph, refine a tree by improving stretch | `llst` |
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
| `llst` (GLST initializer, 10 passes) | 195 | 1257 |

(On this unweighted grid the three weighted variants coincide, as noted above.
Both randomized variants improve stretch here because a random maze has no
long thin comb structure; `randst` happens to win for seed 0. This is a quality
sample, not a claim that one randomized distribution always dominates.)

LLST uses exhaustive candidate, cycle, and objective evaluation with seed 0
and reduces GLST's omitted-edge stretch by 35.9% here. All forests have one
component; growing each with `scaffold.fast` at `keep_ratio=0.7` retains
exactly 255 edges and preserves connectivity. These are illustrative grid
results, not a guarantee that one backbone always gives the best final support.

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
