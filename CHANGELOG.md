# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `backbone="llst"`: the research local-search low-stretch forest, available
  to Greedy, Heap, Batch and Fast through `backbone_options`. Includes exact
  improving cycle swaps, random/tree-distance candidate selection, sampled
  objectives and cycles, and the research initializers (including private
  RandSPT initialization). Requires the optional NetworkX extra, not PyTorch.
- LLST property tests, a standalone example, and an 84-case comparison with
  the research implementation on exact, sampled, weighted and disconnected cases.
- LLST in the README grid-backbone visualization and backbone-tuning example,
  with larger panel labels, an animated GIF comparison, measured stretch,
  reproducible JSON results, and `--only backbones` for regenerating the
  comparison independently of the other demo figures.
- README demo gallery with visible backbone and resampling GIFs, larger labels,
  2×3 method/budget comparisons, and 2×2 sampling/coverage figures. The
  resampling simulation records every draw in `grid_coverage.json`.

- **`workers` on all five algorithms.** Resolved from the argument, then
  `SCAFFOLD_NUM_WORKERS`, then `OMP_NUM_THREADS`, then `min(8, cpus)`. The cap
  is deliberate: several unthrottled jobs on one node slow each other down far
  more than the extra threads gain. `workers="all"` opts out.
  Results are bit-for-bit identical at every worker count — the new
  `tests/test_parallel.py` asserts exact equality, not tolerance, for all five
  variants across grids, weighted, disconnected and truncated-forest graphs.
- `scaffold.utils.workers` with `resolve_workers`, `parallel_threads` and
  `split_workers`.
- `--workers` on `benchmarks/bench_methods.py`.

### Changed

- `viz.compare_methods` now defaults to a three-column grid; `ncols` selects
  a different layout, including a single row.
- GLST now uses weighted path lengths on weighted inputs, matching the research
  initializer used by LLST; previously its projected stretch used hop counts.

- Large systematic Sample draws now use compiled parallel tick blocks. The
  cumulative sum, random offset, and boundary correction remain unchanged;
  selected edges are identical across worker counts. Draw metadata records
  the sampling kernel's worker budget separately from precompute time.
- Greedy, Heap, and Batch path scoring now reduce the actual Numba thread
  mask for small candidate sets and bound it by the number of source groups.
- Numba masks are now set and restored per calling thread. Concurrent
  backbone scorers each honor their inner budget, nested scopes can reduce
  that budget, and Sample's locality-ordering LCA pass honors `workers` too.

- **Tree scorer (`fast`, `sample`) parallelized over candidates.** The LCA
  queries and per-candidate term assembly moved into fused `prange` kernels,
  replacing a chain of NumPy temporaries. ~3x on 8 threads, and ~1.4x even
  serially because the fusion drops eight length-`m` intermediates.
- **`sample` precompute runs its `tree_count` backbones through a thread pool**,
  its widest parallel axis — it covers the serial union-find forest build too.
  Contributions are accumulated in backbone order, never completion order, so
  `pi` cannot drift with the schedule.
- **Path scorer (`greedy`, `heap`, `batch`) compiled.** `PathScorer.evaluate`
  was ~90% Python interpreter overhead — path reconstruction and dict-based
  congestion counters — which no amount of threading could fix. It is now a
  three-pass compiled pipeline (lengths → prefix-sum layout → parallel fill),
  with congestion via `bincount` over a flat buffer and a numba binary-heap
  Dijkstra replacing `heapq`. Scoring is ~8x faster unweighted and ~40x
  weighted at 8 threads; 2.5–3.4x end to end.
  `PathScorer._evaluate_python` is kept as the executable specification, and is
  asserted against on random weighted and unweighted graphs.
- `PathScorer.evaluate` takes `need_paths`; `greedy` and `batch` skip building
  per-candidate path lists they never read.
- Small inputs fall back to serial automatically, below measured crossovers.

## [0.1.0] — 2026-08-21

First release. Private install from the GitHub repository; see
[RELEASING.md](RELEASING.md).

### Added

**Algorithms**
- `scaffold.greedy` — reference greedy; rescores every candidate after every
  insertion.
- `scaffold.heap` — lazy greedy with a stale-score heap and local invalidation.
  Supports cluster-local heaps and both the max and p-norm score forms.
- `scaffold.batch` — sampled per-cluster growth that computes dilation and
  congestion within each candidate batch, then commits only the batch's top
  edges against the current support graph.
- `scaffold.fast` — the `O(m log n + n)` tree-prefix scorer (LCA + root-prefix
  sums), followed by one global top-k. The sampled loop has the distinct
  `scaffold.batch` identity.
- `scaffold.sample` — ratio-independent per-edge weights plus systematic π-ps
  drawing, with exact budget and exact component preservation per draw;
  `fixed-randst` lets visual comparisons share a seeded random backbone.
- `scaffold.sparsify` — dispatch by method name.

**Support backbones**
- `fast-maxst` (default), `fast-mst`, `fast-randst`, `maxst`, `mst`, `randst`,
  `spt`, `glst`, `none`.
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
- 263 tests, including exact equivalence between the tree-prefix kernel and the
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
