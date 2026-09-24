# Support backbones

The backbone is the spanning forest Scaffold grows from.

**Backbone names use forest terminology:** MaxSF, MinSF, RandSF, SPF, GLSF,
and LLSF. Each construction produces one tree per input connected component;
isolated vertices remain as one-node trees. A connected input is the special
case with a single spanning tree. Python accepts the forest spellings and the
historical tree spellings as exact synonyms. Metadata and saved artifacts
retain the historical keys for compatibility.

A full backbone has `n − c` edges for `n` nodes and `c` input components.
This applies to the built-in constructions other than `none`, without a
restrictive `max_edges` cap. A smaller cap or a budget below `n − c` can only
produce a partial forest; custom backbones must satisfy the same spanning
condition to provide the connectivity guarantee.

The backbone does two jobs:

1. **It is the connectivity guarantee.** A spanning forest of a graph with `n`
   nodes and `c` components has `n − c` edges and, by construction, exactly
   those `c` components. Added edges come from the input graph, so they cannot
   join distinct input components. The final support preserves each component
   when its budget can hold the full backbone; it can contain cycles after
   additional edges are retained.
2. **It is the reference the objective is measured against.** Dilation is
   `dist_F(u, v) / w(e)` and congestion counts detours *through the forest*, so
   a different backbone means a different ranking of the same candidates.

The default is Fast-MaxSF (`fast-maxsf`; historical key `fast-maxst`).

## Forest aliases

These aliases match the research `configs/BACKBONE_ALIASES.md`. Names are
case-insensitive, surrounding whitespace is ignored, and hyphens and
underscores are interchangeable. For the same graph, seed, and options,
aliases produce identical forests, scores, and sparse draws. Timing fields
can naturally differ between runs.

| Forest spelling | Historical spelling / canonical package key |
|---|---|
| `maxsf` | `maxst` |
| `msf`, `minsf` | `mst` (`minst` is also accepted) |
| `fast-maxsf` | `fast-maxst` |
| `fast-msf`, `fast-minsf` | `fast-mst` (`fast-minst` is also accepted) |
| `randsf` | `randst` |
| `fast-randsf` | `fast-randst` |
| `slsf` | `slst` |
| `glsf` | `glst` |
| `llsf` | `llst` |
| `randspf` | `randspt` |
| `sf`, `st` | `maxst` |
| `spf` (package extension) | `spt` |

The package keeps its existing **hyphenated** canonical `fast-*` keys;
research configs use underscores internally. Both input spellings work.
`sf` means `maxst`, regardless of the package default `fast-maxst`. On weighted
graphs, `minsf`/`msf` minimize weights and `maxsf` maximizes them.

Sample has its own mode aliases, normalized before draw-plan keys and artifact
storage are formed:

| Sample forest spelling | Canonical mode |
|---|---|
| `fixed-maxsf`, `fixed-msf`, `fixed-sf`, `fixed-st` | `fixed-maxst` |
| `fixed-slsf` | `fixed-slst` |
| `rotate-randsf` | `rotate-randst` |
| `fixed-randsf` (package extension) | `fixed-randst` |

```python
scaffold.fast(G, keep_ratio=0.2, backbone="Fast_RandSF", seed=0)
scaffold.sample(G, backbone="rotate-randsf", seed=0)
scaffold.fast(G, keep_ratio=0.7, backbone="llsf", seed=0,
              backbone_options={"init_support": "maxsf", "max_passes": 5})
```

The same aliases work in `build_backbone` and LLSF's supported `init_support`
choices. `available_backbones(notation="forest")` lists the preferred SF names;
`available_backbones()` retains the historical list for existing callers.
`SF`, `MaxSF`, `LLSF`, and compact `FastMaxSF`/`FastRandSF` spellings also work.
SLSF, RandSPF,
GLSF, and LLSF require NetworkX: run `python -m pip install ".[networkx]"`
from the repository root. All other
built-in backbones and the default Sample modes work without NetworkX.

## Visual comparison

The figure spells out each name: **MaxSF** is a maximum-weight spanning forest,
**RandSF** a random spanning forest, **SPF** a shortest-path forest, **GLSF** a
greedy low-stretch forest, and **LLSF** a local-search low-stretch forest.
The `fast-` variants use cheaper edge ordering. Minimum-weight spanning forests
use `minsf` (MinSF) or `fast-minsf` in the API. The connected grid below makes
each spanning forest a single tree.

![Seven support backbones with full method names and measured stretch](images/grid_backbones.png)

The unweighted 12×12 grid comparison includes LLSF beside its GLSF initializer. Every
backbone has 143 edges and one connected component; each label reports the
sum of supporting-path lengths for the omitted edges. LLSF uses up to 10
exhaustive improving swaps with seed 0. The two randomized backbones also use
seed 0; minimum-weight variants coincide with maximum-weight variants on this
unweighted graph and are omitted from the image.
LLSF reduces omitted-edge stretch from 1,161 to 795 (31.5%) in this example.
The [animated comparison](images/grid_backbones.gif) cycles through the same
seven completed forests, ending with GLSF and its LLSF refinement.

Regenerate the PNG, GIF, and [measurements](images/grid_backbones.json) with:

```bash
python examples/01_grid_demo.py --only backbones --out docs/images
```

---

## The built-in backbones

### `fast-maxsf` (default) and `fast-minsf`

Bucketed approximate weighted Kruskal. Edge weights are quantized into
`buckets` (default 256) priority classes and ordered with a linear-time counting
sort instead of an `O(m log m)` comparison sort; edges within a class keep input
order. The union-find scan is identical to the exact version.

The ordering is approximate. On the graphs SCAFFOLD targets the difference is
immaterial, and the ordering cost drops from `O(m log m)` to `O(m + buckets)`.

On an **unweighted** graph there is nothing to sort, so `fast-maxsf`,
`fast-minsf`, `maxsf` and `minsf` all reduce to the same deterministic scan and
produce identical forests.

```python
scaffold.fast(G, keep_ratio=0.2)                                     # fast-maxsf
scaffold.fast(G, keep_ratio=0.2, backbone="fast-minsf")
scaffold.fast(G, keep_ratio=0.2, backbone_options={"buckets": 1024})
```

### `maxsf` and `minsf`

Exact stable Kruskal, `O(m log m)`. Ties break on edge id, so runs are
reproducible. Use these when the weight ordering genuinely matters and you can
afford the sort.

### `fast-randsf` and `randsf`

Both are randomized Kruskal forests and are redrawn on every call:

- `fast-randsf` draws only a seeded offset and coprime stride. The resulting
  arithmetic traversal visits every edge once without allocating an `m`-entry
  permutation. It is the faster, lower-memory choice and matches the research
  tensor backend's fast randomized support.
- `randsf` constructs a full uniform random edge permutation before the same
  union-find scan. Its edge order is more thoroughly randomized, but creating
  and storing the permutation costs additional time and `O(m)` memory.

Neither is an exact uniform spanning-tree sampler. The distinction is the
randomness and cost of the *edge order*, not the union-find acceptance rule.

Warmed median construction times with numba on the development machine were:

| undirected edges | `fast-randsf` | `randsf` | speedup |
|---:|---:|---:|---:|
| 130,560 | 2.4 ms | 4.5 ms | 1.9× |
| 523,264 | 9.9 ms | 22.6 ms | 2.3× |
| 2,095,104 | 134.3 ms | 177.6 ms | 1.3× |

The exact ratio is graph-, seed-, cache-, and machine-dependent. The stable
distinction is that `fast-randsf` generates two random integers and no order
array, whereas `randsf` generates and stores all `m` permutation entries.

```python
a = scaffold.fast(G, keep_ratio=0.2, backbone="fast-randsf", seed=1)
b = scaffold.fast(G, keep_ratio=0.2, backbone="fast-randsf", seed=2)
# a.mask != b.mask
```

<a id="spt"></a>

### `spf`

BFS forest on unweighted inputs; Dijkstra forest on weighted inputs. Each
component is rooted at its highest-degree node. Weights must be finite and
nonnegative. Use `graph.with_weight(None)` to request hop distances explicitly.

### `randspf`

Random shortest-path forest (RandSPF), using a seeded random root in each
component and randomized traversal/tie ordering. Uses BFS on unweighted inputs
and Dijkstra on weighted inputs; weights must be finite and strictly positive.
Requires NetworkX. This is also available as an LLSF initializer.

### `slsf`

Scalable low-stretch forest (SLSF), ported from the research multi-root
shortest-path heuristic. For each component it constructs several BFS/Dijkstra
trees, estimates mean stretch, and keeps the best. It does not implement the
theoretical recursive low-stretch constructions or their guarantees. Requires
NetworkX; weighted inputs use finite, strictly positive edge lengths.

Options passed through `backbone_options` are `num_roots` (default 8),
`eval_sample_size` (8,192), `exact_eval_threshold` (20,000), `fast_mode`
(False), and `verbose` (False). `fast_mode=True` uses the first candidate root
without comparing stretch. Sample's `fixed-slsf` mode uses the default SLSF
settings for its fixed forest; `tree_count` still controls the separate random
forests used for scoring.

```python
scaffold.fast(G, keep_ratio=0.2, backbone="slsf", seed=0,
              backbone_options={"num_roots": 4})
scaffold.sample(G, backbone="fixed-slsf", seed=0)
```

<a id="glst"></a>

### `glsf`

Greedy low-stretch forest (GLSF). Repeatedly adds the boundary edge maximizing

```
(cut / cut_max)^eta / ((projStretch / stretch_max)^alpha)
```

where `cut[e]` counts graph edges crossing the two components `e` would merge,
and `projStretch[e]` is their mean stretch once `e` is added.

Construction costs `O(n · m · |cut|)`. GLSF also serves as the default
initializer for LLSF, which can further reduce its stretch through cycle swaps.
Guarded at 5,000 edges; beyond that it will raise rather than hang.

```python
scaffold.greedy(G, keep_ratio=0.5, backbone="glsf",
               backbone_options={"eta": 1.0, "alpha": 1.0})
```

<a id="llst"></a>

### `llsf`

Local-search low-stretch forest (LLSF), ported from the research implementation. It
builds an initial tree in each component, then repeatedly adds a non-tree
edge and removes an edge on the resulting cycle. Each accepted swap strictly
reduces the evaluated total stretch:

```
sum(dist_T(u, v) / w(u, v) for (u, v) in E(G))
```

Tree distances sum edge weights (or count hops on unweighted inputs). Weighted
LLSF inputs must have finite, strictly positive weights. This is a stretch
objective; the subsequent Scaffold growth stage also accounts for congestion.

**Runtime limit: 1,000 undirected input edges by default.** LLSF is a
small-graph reference; exhaustive local search can take many minutes or
longer. Larger inputs raise `ValueError` before LLSF initialization or search,
with a suggestion to use `randsf`, `fast-randsf`, or `fast-maxsf`. The limit
counts unique undirected edges after graph normalization, not the requested
output budget or the largest component. It applies to sampled LLSF too.

Install the `networkx` extra from the repository root to use LLSF:
`python -m pip install ".[networkx]"`. It works with Greedy, Heap,
Batch and Fast and with every graph format accepted by those methods:

```python
result = scaffold.fast(
    G, keep_ratio=0.7, backbone="llsf", seed=0,
    backbone_options={"init_support": "maxsf", "max_passes": 5},
)

# Or build just the forest, as a mask over canonical edges:
from scaffold.backbone import build_backbone
graph = scaffold.normalize_graph(G)
forest = build_backbone(graph, "llsf", seed=0, init_support="maxsf", max_passes=5)
```

Options go directly to `build_backbone`, or inside `backbone_options` for the
Scaffold methods:

| option | default | meaning |
|---|---|---|
| `max_input_edges` | `1000` | Positive integer limiting the full undirected input; explicitly increase it to opt into larger, potentially slow experiments |
| `init_support` | `"glsf"` | Starting spanning forest (one tree per component); choices below |
| `max_passes` | `10` | Maximum accepted improving swaps per component; minimum 1 |
| `candidate_strategy` | `"random"` | `"random"` or longest current `"tree_distance"` |
| `candidate_sample_size` | `0` | Add-edge candidates per pass; 0 evaluates all |
| `eval_sample_size` | `0` | Original edges used to estimate stretch; 0 evaluates all |
| `cycle_sample_size` | `0` | Removable edges tested per cycle; 0 tests all |
| `resample_eval_each_pass` | `False` | Redraw the objective sample each pass |
| `glst_alpha`, `glst_eta` | `1.0`, `1.0` | GLSF initializer exponents |
| `fast_tree_buckets` | `256` | Buckets for fast weighted initializers |
| `verbose` | `False` | Report accepted swaps and evaluated stretch |

Initializers are `glsf`, `maxsf`, `minsf`, `fast-maxsf`, `fast-minsf`, `randsf`,
`randspf`, `fast-randsf`, `spf`, and `slsf`; the corresponding forest aliases
also work. LLSF preserves the research RandSF initializer's
seeded random-priority ordering; the package's standalone `randsf` uses a
permutation and can produce a different initial tree for the same seed.

An exhaustive pass may test `O(m n)` cycle swaps, rebuilding a tree index and
evaluating stretch across all input edges for each trial. A more detailed
bound is `C_init + O(P m ell (n + m) log n)`, where `P` is the pass limit,
`ell` is the mean number of tree edges tested per candidate cycle, and
`C_init` is the initializer cost. Even graphs below the guard can be slow.

For a deliberate larger experiment, raise the limit explicitly and consider
a lightweight initializer with sampled search:

```python
result = scaffold.fast(
    G, keep_ratio=0.7, backbone="llsf", seed=0,
    backbone_options={
        "max_input_edges": 2000,
        "init_support": "randsf",
        "max_passes": 2,
        "candidate_sample_size": 16,
        "cycle_sample_size": 2,
        "eval_sample_size": 64,
    },
)
# For build_backbone(..., "llsf"), pass these options directly.
```

Sampling reduces the work; when
`eval_sample_size` is nonzero, improvements concern the sampled objective,
not necessarily the full-graph objective. Raising `max_input_edges` does not
remove the default GLSF initializer's separate 5,000-edge guard.

Disconnected components and isolated nodes are preserved. A direct
`max_edges` cap that cannot span a component returns its budgeted initializer
without swaps. Scaffold methods instead construct the complete LLSF forest
and use the package's usual random trim if their final budget is below the
connectivity floor. Sample's specialized fixed/rotating-backbone modes are
unchanged; `backbone="llsf"` is not a Sample mode.

See [`examples/05_local_search_backbone.py`](../examples/05_local_search_backbone.py)
and the [research parity check](validation.md#llsf-backbone).

### `none`

Start from the empty graph. **Connectivity is then not guaranteed** — you get
the pure score ranking with no structural floor. Occasionally what you want for
an ablation; almost never what you want in production.

---

## Choosing one

| | |
|---|---|
| Large graph, want it fast | `fast-maxsf` (default) |
| Weight ordering matters exactly | `maxsf` / `minsf` |
| Fast randomized support | `fast-randsf` |
| More thoroughly shuffled support | `randsf` |
| Per-epoch resparsification | `fast-randsf`, or `scaffold.sample(backbone="rotate-randsf")` |
| Rooted shortest paths (hops or weighted distance) | `spf` |
| Small graph, grow a tree using projected stretch | `glsf` |
| Small graph, refine a tree by improving stretch | `llsf` |
| Ablation with no structural guarantee | `none` |

Measured on a 14×14 grid (196 nodes, 364 edges) — `total stretch` is the sum
over non-tree edges of `dist_F(u,v)/w(e)`, so lower is a better starting point:

| backbone | forest edges | total stretch |
|---|---|---|
| `fast-maxsf` | 195 | 2535 |
| `maxsf` | 195 | 2535 |
| `minsf` | 195 | 2535 |
| `fast-randsf` | 195 | 1719 |
| `randsf` | 195 | 1649 |
| `spf` | 195 | 2355 |
| `glsf` | 195 | 1961 |
| `llsf` (GLSF initializer, 10 passes) | 195 | 1257 |

(On this unweighted grid the three weighted variants coincide, as noted above.
Both randomized variants improve stretch here because a random maze has no
long thin comb structure; `randsf` happens to win for seed 0. This is a quality
sample, not a claim that one randomized distribution always dominates.)

LLSF uses exhaustive candidate, cycle, and objective evaluation with seed 0
and reduces GLSF's omitted-edge stretch by 35.9% here. All forests have one
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
mask = build_backbone(graph, "fast-maxsf", max_edges=50)
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
import scaffold

weighted = scaffold.with_feature_weights(data, metric="cosine")  # uses PyG data.x
result = scaffold.fast(
    weighted, keep_ratio=0.6, backbone="fast-maxsf", weighted_paths=True,
)
```

With `fast-maxsf`, the backbone prefers larger similarities. For smaller-is-closer
distances, use `kind="distance"` and `minsf`/`fast-minsf`. Path scoring uses the
supplied scalar directly; it does not invert similarities. The helper evaluates
existing edges in batches. See [weighted graphs](weighted.md) for Euclidean,
scikit-learn metrics, and the distinction between backbone and scoring weights.
