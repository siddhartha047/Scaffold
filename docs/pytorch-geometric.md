# PyTorch Geometric integration

```bash
pip install "scaffold-sparse[pyg]"
```

Everything here lives in `scaffold.pyg`, which is imported lazily — plain
NumPy/SciPy users never pay for torch.

---

## Data in, Data out

```python
from torch_geometric.datasets import Planetoid
import scaffold

data = Planetoid(root="/tmp/Cora", name="Cora")[0]

result = scaffold.fast(data, keep_ratio=0.6, seed=0)
sparse = result.to_pyg()

print(data.num_edges, "->", sparse.num_edges)
print(result.summary())
```

`to_pyg()` forwards **every** non-edge attribute of the source object — `x`,
`y`, `train_mask`, `val_mask`, `test_mask`, anything else you attached — and
replaces only `edge_index` / `edge_weight` / `edge_attr`. Downstream training
code keeps working unchanged.

One call, if you don't need the intermediate result:

```python
from scaffold.pyg import sparsify_data

sparse = sparsify_data(data, method="fast", keep_ratio=0.6, seed=0)
```

### Conventions

- Input `edge_index` is canonicalized: self loops dropped, `(u,v)` and `(v,u)`
  collapsed to one undirected edge, sorted.
- Output `edge_index` is **symmetric** — both directions present, `(2, 2k)`.
- `keep_ratio` counts **undirected** edges. `keep_ratio=0.6` on Cora's
  `edge_index=[2, 10556]` gives `[2, 6334]`, i.e. 3,167 undirected edges kept
  of 5,278. This is above Cora's 2,630-edge connectivity floor, so it exercises
  both the support forest and the LCA-ranked additions.
- `edge_attr` is read as an edge weight only when it is 1-D or single-column;
  multi-dimensional attributes are ignored rather than guessed at.

---

## As a dataset transform

```python
from scaffold.pyg import ScaffoldTransform

dataset = Planetoid(
    root="/tmp/Cora",
    name="Cora",
    transform=ScaffoldTransform(method="fast", keep_ratio=0.6, seed=0),
)
data = dataset[0]      # already sparsified
```

`transform` runs on every access, so with `resample=True` and a `seed` the view
changes each time:

```python
ScaffoldTransform(method="fast", keep_ratio=0.6, seed=0, resample=True)
```

That re-runs the whole sparsifier per access. For per-epoch resparsification
`ScaffoldResampler` is far cheaper — it does the structural work once.

Use `pre_transform=` instead if you want the sparsification baked into the
processed dataset on disk. Remember PyG caches `processed/`, so delete it when
you change the ratio.

---

## Per-epoch resparsification

The interesting use case. A single fixed sparsifier shows the model
`keep_ratio` of the graph and discards the rest permanently. Redrawing every
epoch shows it nearly all of the graph while every individual view stays small.

```python
from scaffold.pyg import ScaffoldResampler

resampler = ScaffoldResampler(
    data,
    keep_ratio=0.6,
    seed=0,
    backbone="rotate-randst",   # the guaranteed forest rotates too
    tree_count=8,               # forests aggregated during precompute
    every=1,                    # redraw every N epochs
)

for epoch in range(500):
    view = resampler.epoch(epoch)
    model.train()
    optimizer.zero_grad()
    out = model(view.x, view.edge_index)
    loss = F.cross_entropy(out[view.train_mask], view.y[view.train_mask])
    loss.backward()
    optimizer.step()
```

Each `view` has **exactly** the requested edge count. At or above
`resampler.delta_min`, it also has **exactly** the input graph's connected
components — with probability 1, not in expectation. Below that floor, no
sparsifier can preserve all components.

### Is it buying you anything?

Check before you commit to it:

```python
resampler.coverage(epochs=(1, 10, 100, 500))
# {'always_included': 2630,
#  'median_epochs_to_first_inclusion': 2.04,
#  'coverage_curve': {1: 0.60, 10: 0.91, 100: 0.99, 500: 0.99}}
```

`always_included` is the *deterministic core* — edges present in every draw.
It can never be smaller than the backbone. If the core is most of your budget,
resampling has little room to vary and you may as well use `scaffold.fast`.
`backbone="rotate-randst"` shrinks the core.

```python
resampler.delta_min      # the connectivity floor for this graph
```

### Cost

The precompute is `O(R · (m log n + n))` once. Each `epoch()` call is one
uniform draw plus one `searchsorted` — microseconds on Cora. With `every=k` the
view is cached and reused for `k` epochs.

---

## Edge weights instead of a subgraph

```python
scores = scaffold.sample(data, seed=0)
data.edge_index, data.edge_weight = scores.to_torch()
```

This covers **all** edges of the input; `edge_weight` holds the SCAFFOLD
importance `π` for each.

> These are *sampling* weights — how much each edge matters structurally — not
> message-passing coefficients. Feeding them to `GCNConv(..., edge_weight=...)`
> is a modelling choice, not the intended use. The intended use is driving the
> per-epoch draw.

The convenience wrapper:

```python
from scaffold.pyg import sample_edge_weight

edge_index, edge_weight = sample_edge_weight(data, seed=0)
```

---

## Large graphs

**Precompute the sample artifact once and cache it.** For `ogbn-products`-scale
graphs the precompute dominates; `save`/`load` makes it a one-time cost.

```python
from pathlib import Path
import scaffold
from scaffold.algorithms.sample import ScaffoldSampler

graph = scaffold.normalize_graph(data)
cache = Path("cache/products_weights.npz")

if cache.exists():
    sampler = ScaffoldSampler.load(cache, graph)
else:
    sampler = scaffold.sample(graph, seed=0, tree_count=4).sampler
    cache.parent.mkdir(parents=True, exist_ok=True)
    sampler.save(cache)

view = sampler.draw(keep_ratio=0.1)
```

`load` verifies the artifact against the graph's canonical edge list and raises
on any mismatch — weights are indexed positionally, so a silent mismatch would
corrupt every draw.

**Install numba.** Without it the union-find, LCA and prefix-sum kernels run as
pure Python loops. `pip install "scaffold-sparse[speed]"`.

**Lower `tree_count`.** The precompute is linear in it. `tree_count=4` is a
reasonable large-graph setting; `1` gives frequency-free weights driven purely
by the score.

**Turn off the congestion terms** if you only need stretch:

```python
scaffold.fast(data, keep_ratio=0.1, beta_edge=0.0, beta_node=0.0)
```

**Keep everything on CPU.** The sparsifier is NumPy end to end; only the final
`to_pyg()` produces tensors. Move the result to the GPU as usual.

---

## Mini-batch loaders

Sparsify **before** the loader, so every batch is sampled from the sparse graph:

```python
from torch_geometric.loader import NeighborLoader

sparse = sparsify_data(data, keep_ratio=0.2, seed=0)
loader = NeighborLoader(sparse, num_neighbors=[10, 10], batch_size=1024)
```

With per-epoch resparsification, rebuild the loader when the view changes:

```python
for epoch in range(epochs):
    view = resampler.epoch(epoch)
    loader = NeighborLoader(view, num_neighbors=[10, 10], batch_size=1024)
    for batch in loader:
        ...
```

Building a loader per epoch is not free; `every=5` or `every=10` amortizes it
while still varying the topology.

---

## A complete example

[`examples/03_pytorch_geometric.py`](../examples/03_pytorch_geometric.py) runs
end to end — with `--dataset Cora` for the real thing, or with no arguments on
a synthetic graph so it works offline.

```bash
python examples/03_pytorch_geometric.py
python examples/03_pytorch_geometric.py --dataset Cora --keep-ratio 0.6
```
