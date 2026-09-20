# Runtime and memory expectations

These are the **2026-09-20 audited measurements**, using the package's public API.
Use Fast for one sparse support and Sample for repeated draws; dynamic Batch
repeatedly searches the evolving support and can be much more expensive.

## Complete construction at 20% retention

Eight CPU workers, unweighted synthetic graphs, three-call medians:

| Method | Nodes | Edges | Complete time (s) |
|---|---:|---:|---:|
| Greedy | 400 | 3,131 | **0.78** |
| Heap | 4,000 | 39,893 | **25.16** |
| Batch | 10,000 | 250,000 | **31.01** |
| Fast | 200,000 | 2,399,834 | **2.30** |
| Sample, 8 random scoring forests + first draw | 200,000 | 2,399,834 | **9.02** |

The timer starts with a bidirectional NumPy edge list already in memory and
ends after a bidirectional sparse output has been materialized. It includes
normalization, **fresh forest construction**, scoring, selection, and output
assembly. Sample includes **fresh preprocessing plus the first draw**. No fitted
sampler or forest is reused for complete-call timings. The kernel warm-up,
generation/loading of the input, output validation, GNN training, and GPU
transfer are excluded. An initial call on a clean installation can take longer
because Numba compiles its kernels.

All runs use seed 0, objective exponents `(1,1,1)`, norm orders `(2,2)`,
Fast-MaxSF for Greedy/Heap/Batch/Fast, and fixed MaxSF plus eight randomized
scoring forests for Sample. Heap uses product scores, `top_k=16`, and
`dirty_limit=64`. Batch uses `sample_size=512`, `add_per_round=64`, ten BFS
candidate clusters. The same settings and graphs are used at every worker
count. These are not timings of the separate `(1,1,0)` training configuration.

The 200K-node graph retains **479,967 edges**: a **199,999-edge forest plus
279,968 additions**. The other addition counts are 228 (Greedy), 3,980 (Heap),
and 40,001 (Batch). Every budget therefore exceeds the forest floor by more
than two edges; the requested 20% ratio did not need adjustment. Every output
passes exact-budget, connectivity, and reproducibility checks. Graph hashes
match the earlier experiments.

Hardware: shared dual AMD EPYC 7282 (32 physical cores), with affinity
`0,4,8,12,16,20,24,28`; Python 3.12.9, NumPy 2.2.6,
Numba 0.66.0. `OMP_PROC_BIND=false`; BLAS is limited to one thread.
The machine was not exclusively reserved. No observations were discarded;
all repetitions and their ranges are available in the raw JSON.

## Worker scaling

Complete construction, including a fresh forest and (for Sample) a first draw.
Graph sizes and all other settings match the table above:

| Method | 1 worker (s) | 4 workers (s) | 8 workers (s) | Speedup |
|---|---:|---:|---:|---:|
| Greedy | 1.63 | 1.01 | 0.78 | 2.08× |
| Heap | 25.09 | 25.02 | 25.16 | 1.00× |
| Batch | 206.12 | 57.54 | 31.01 | 6.65× |
| Fast | 3.53 | 2.78 | 2.30 | 1.54× |
| Sample | 33.77 | 11.81 | 9.02 | 3.74× |

The selected support is identical across worker counts under fixed randomness.
More workers do not guarantee lower time: Python bookkeeping, forest scans,
selection, and small candidate refreshes include sequential work. In particular,
Heap shows little speedup in this measured configuration. The earlier appendix
mixed different ratios, Batch settings, and repetition statistics; the current
main and appendix tables are generated from the same observations.

## Why Fast can finish quickly

Fast builds one forest, computes all initial dilation/congestion scores using
LCA and root-prefix operations, then selects the top edges. It does not run a
separate shortest-path search for each omitted edge. On unweighted inputs,
Fast-MaxSF also bypasses weight ordering. Every measured Fast call scored
**2,199,835 candidates** and added **279,968 edges**; no forest-only early exit
was used.

Independent instrumented calls measured these stages:

| Fast stage | Median seconds |
|---|---:|
| normalization | 0.4381 |
| forest | 0.0208 |
| component check | 0.0467 |
| tree index | 0.0136 |
| candidate scoring | 0.7040 |
| selection | 0.0676 |
| result assembly | 0.0163 |
| bidirectional export | 0.0047 |
| other | 0.0210 |

Stage medians need not sum to the median total. The complete time reported above
is measured by an outer wall-clock timer, not reconstructed from stage medians.
The separately timed **normalized-input API** median is
**1.654 s**. The previous **0.73 s** measurement
used that narrower warmed scope in an older environment; its timer already
included forest construction. It is retained only in the historical record,
not used as the new complete input-to-output time.

A fresh check in the historical Python 3.9.21 / NumPy 2.0.2 / Numba 0.60.0 environment measured **0.789 s** for the same normalized-input API (median of three), still including a new forest. This supports the plausibility of the older 0.73 s observation; it is not used in the current-environment input-to-output tables.

## Sample preprocessing versus repeated draws

At eight workers, the preprocessing API (including input normalization) takes
**8.900 s**, and the first draw takes
**0.126 s** (separate component medians).
The complete-call median, including output assembly, is
**9.02 s**. The old main-table Sample entry timed preprocessing
alone and did not yet produce a sparse graph.

Later cached draws, including bidirectional output assembly, take
**0.055 s** (median of nine). These deliberately reuse preprocessing and
the budget-specific draw plan; they must not be described as fresh complete
support construction.

## Reproduce the audit

The [audit record](benchmarks/runtime_audit_20260920/README.md) gives commands,
settings, graph hashes, source hashes, every observation, and validation rules.
`benchmarks/bench_runtime_audit.py` measures the public API;
`benchmarks/render_runtime_audit.py` renders both paper snippets from the same
completed JSON records. Separate files also report unweighted backbone costs
and BFS/Dijkstra scoring-only diagnostics. Those component costs are not mixed
with complete variant runtimes.

## Measure your input-to-output time

```python
from time import perf_counter
import scaffold

# Input edge_index is already loaded; n preserves isolated nodes.
# Warm representative kernels first for a steady-state measurement.
start = perf_counter()
G = scaffold.normalize_graph(edge_index, num_nodes=n)
result = scaffold.fast(
    G, keep_ratio=0.2, backbone="fast-randsf", workers=8, seed=0,
)
sparse_edge_index = result.edge_index
seconds = perf_counter() - start
```

The example uses the recommended Fast-RandSF; the recorded table uses
Fast-MaxSF. Graph topology, weights, backbone, retention, software versions,
and memory pressure can change runtime. Count unique **undirected** edges;
a bidirectional edge list normally contains two entries per edge.

## Larger graphs and memory

No 1M-node/100M-edge graph was run in this audit. Earlier planning estimates
and the completed 200K-node dynamic Batch run are retained in the
[historical record](performance_history_20260919.md); those estimates are not
new measurements or guarantees.

At 100M unique undirected edges, two int64 endpoint arrays alone occupy
1.6 GB, and each float64 per-edge array occupies another 0.8 GB. Normalization,
forest indices, scoring buffers, output arrays, and concurrent Sample forests
add to this. These are arithmetic storage costs, not peak-memory measurements;
swapping can substantially change runtime.

## Worker configuration

All variants accept `workers`. With Numba installed (`[speed]`), use an integer
to set a CPU thread budget, `1` for serial execution, or `"all"` for all available
CPUs. Omitting it checks `SCAFFOLD_NUM_WORKERS`, then `OMP_NUM_THREADS`, then
defaults to at most eight available CPUs. The selected edges do not change
with the worker count.

```python
result = scaffold.fast(
    G, keep_ratio=0.2, backbone="fast-randsf", seed=0, workers=8,
)
```

Cap workers per job when running several jobs on one machine. More workers
need not be faster: forest construction, reductions, and selection include
sequential work, and small problems use serial paths automatically. Without
Numba, kernels fall back to Python and `workers` has no effect. The first call
may also include JIT compilation or cache loading.
