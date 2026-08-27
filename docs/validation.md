# Research implementation parity

The package includes a real-data comparison against the implementation used by
`ICML_SPARSIFICATION/main.py`:

```bash
pip install -e ".[dev,pyg]"
python validation/compare_research.py
```

By default this loads Cora through the research repository's own dataset loader
and sends the same PyG `Data` object through both implementations. The research
side runs the tensor backend with one cluster, `fast_maxst`, and
`fast_score="tree_exact"`. Those settings are the exact counterpart of:

```python
scaffold.fast(G, backbone="fast-maxst", selection="topk")
```

The script fails unless the two implementations return the identical canonical
undirected edge set. It defaults to `keep_ratio=0.6`, above Cora's connectivity
floor, so the comparison exercises the LCA scores and top-k selection rather
than merely comparing partial support forests.

The separate research `scaffold_batch` config uses METIS partitions and
`tree_exact_loop`. That output is intentionally not expected to match the
package Fast default: partition-local budgeting and sampled-batch selection
define a different procedure. Use `scaffold.batch(...)` in the package for the
sample-score-top-r loop; use `scaffold.fast(...)` for LCA score-once/top-k.

Paths can be overridden when the repositories are not siblings:

```bash
python validation/compare_research.py \
  --research-root /path/to/ICML_SPARSIFICATION \
  --data-dir /path/to/data \
  --dataset cora \
  --keep-ratio 0.6 \
  --seed 123
```
