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

## Backbone aliases

`tests/test_backbone_aliases.py` checks the forest/tree synonyms listed in
[the backbone guide](backbones.md#forest-aliases). The tests cover seeded
forest masks, weighted min/max distinctions, all four growth variants,
canonical metadata, Sample scores and draw-plan keys, artifact save/reload,
LLSF initializer aliases, custom backbones, and disconnected/partial forests.

The two alias maps were compared directly with the research
`configs/BACKBONE_ALIASES.md` implementation. The SLSF component builder also
matched the research implementation in 72 comparisons: weighted/unweighted
and connected/disconnected graphs, three seeds, full/partial budgets, and
default, sampled-evaluation, and fast-mode settings. Inputs used the same
canonical node/edge order to preserve tie breaking.

`validation/check_distributions.py` exercises forest aliases through all five
methods after installing built artifacts outside the checkout, with only core
dependencies. The NetworkX wheel installation was additionally checked with
`slsf`, `randspf`, and Sample's `fixed-slsf` mode.

## LLSF backbone

The LLSF comparison uses small synthetic graphs and does not download datasets:

```bash
python validation/compare_llst_research.py
```

It checks 84 forest masks against the original
`LocalSearchLowStretchTreeSparsifier`: seven research initializers, four
exact/sampled configurations, and unweighted, weighted and disconnected
fixtures. `--research-root` overrides the sibling checkout path. The research
side requires its PyG dependencies; packaged LLSF itself only adds NetworkX
to the NumPy/SciPy core. Inputs are placed in the same canonical order before
comparison, so node/edge insertion order does not alter tie breaking.

`tests/test_llst.py` independently verifies best improving swaps and local
optimality using NetworkX distances, as well as forest validity, budgets,
seeded sampling, original labels/weights, and optional-framework isolation.
It also covers LLSF's 1,000-edge runtime guard, normalized edge counting,
explicit overrides through each supported Scaffold method, and early rejection
of oversized backbone demos.
