# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-21

First release. Private install from the GitHub repository; see
[RELEASING.md](RELEASING.md).

### Added

**Algorithms**
- `scaffold.exact` — reference greedy; rescores every candidate after every
  insertion.
- `scaffold.heap` — lazy greedy with a stale-score heap and local invalidation.
  Supports cluster-local heaps and both the max and p-norm score forms.
- `scaffold.fast` — the `O(m log n + n)` tree-prefix scorer (LCA + root-prefix
  sums), with `selection="topk"` (default) and `selection="rounds"`
  (paper Algorithm 1).
- `scaffold.sample` — ratio-independent per-edge weights plus systematic π-ps
  drawing, with exact budget and exact component preservation per draw.
- `scaffold.sparsify` — dispatch by method name.

**Support backbones**
- `fast-maxst` (default), `fast-mst`, `maxst`, `mst`, `randst`, `spt`, `glst`,
  `none`.
- `register_backbone` for custom builders; precomputed masks and edge-id arrays
  accepted directly.

**Graph I/O**
- Adapters for NetworkX, PyTorch Geometric `Data`, `scipy.sparse`, `(2, m)`
  edge_index arrays (NumPy or torch), and dense adjacency matrices.
- Canonical internal representation with stable edge ordering, so cached
  `scaffold.sample` artifacts stay valid across runs.
- NetworkX node labels are preserved through a round trip.

**Results**
- `ScaffoldResult` with `.mask`, `.edge_index`, `.edge_weight`, `.metadata`,
  and `.to_pyg()` / `.to_networkx()` / `.to_scipy()` / `.to_torch()`.
- `ScaffoldScores` for `sample`, adding `.inclusion_probabilities()`,
  `.draw()`, `.to_dict()` and coverage reporting.
- `delta_min` and `below_connectivity_floor` reported on every result.
- Below the connectivity floor, every variant builds its complete support
  forest and applies a seeded uniform random trim to the exact edge budget;
  `sample.draw()` re-trims it for every view.

**PyTorch Geometric**
- `ScaffoldTransform` for dataset pipelines.
- `ScaffoldResampler` for per-epoch resparsification.
- `sparsify_data`, `sample_edge_weight`.

**Other**
- `scaffold-sparse` distribution with canonical `import scaffold` and optional
  `import scaffold_sparse` alias.
- Optional numba acceleration for the union-find, LCA and prefix-sum kernels,
  with a correct pure-Python fallback.
- Demo graphs (`grid_graph`, `ring_of_cliques`, `random_geometric`) and
  plotting helpers in `scaffold.viz`.
- Four runnable examples and a visual grid-graph walkthrough.
- 247 tests, including exact equivalence between the tree-prefix kernel and the
  shortest-path reference across five backbones and four parameter sets.

### Notes

- `edge_index` / `edge_weight` on a result are **symmetric** (both directions);
  `keep_ratio`, `original_edges` and `sparse_edges` count **undirected** edges.
- `inf` norm orders (`edge_norm_p`, `node_norm_q`) raise `NotImplementedError`
  rather than being silently approximated.
- The low-stretch-tree research backbones `slst`, `randspt` and `llst` are not
  included in this release. `glst` is, guarded at 5,000 edges.

[Unreleased]: https://github.com/siddhartha047/Scaffold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/siddhartha047/Scaffold/releases/tag/v0.1.0
