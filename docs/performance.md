# Runtime and memory expectations

Start with **Fast for one sparse support** and **Sample for repeated draws**
on large graphs. Batch adapts scores as edges are inserted, but repeated
shortest-path searches can make it much more expensive. Graph size, retention
ratio, topology, backbone, batching and available memory all affect runtime.

## Current measurements at 20% retention

All variants now have a common `keep_ratio=0.2` benchmark with eight CPU
workers. On the original 200,000-node / 2,399,834-edge graph:

| Method / phase | Seconds | Measurement |
|---|---:|---|
| Fast, complete construction | **0.73** | Median of three |
| Sample preprocessing, R=8 | **5.01** | Median of three |
| Sample cached draw | **0.048** | Median of six |
| Batch, complete construction | **3444.83** | One complete run |

The connected target has **479,967 edges**, with **279,968 insertions** after
the 199,999-edge backbone. Large Batch used `sample_size=512`,
`add_per_round=256`, ten BFS candidate clusters and `workers=8`. Its full
runtime was **57.4 minutes**, over 110 rounds;
it was allowed to finish beyond the initial 300-second target. This is the
dynamic full-support algorithm, with no frozen-tree approximation. The
earlier stopped trial is excluded from the completed runtime.

The main-paper Batch case uses **10,000 nodes / 250,000 edges**, with
**512 candidates / 64 insertions per cluster**, ten BFS candidate clusters,
and eight workers. Three complete calls took **32.64, 32.34, and 32.94 s**
(median **32.64 s**). Each retained the same connected **50,000-edge** support
after **65 rounds**. These are measurements at 20% retention with fresh
backbone construction, not estimates. The
[raw record](benchmarks/batch_10k_250k_b512_r64_20260919.json) includes the
exact graph generator, settings, source hashes, and all three observations.

Other smaller cases at the same ratio/workers took **0.75 s** for Greedy on
400 nodes / 3,131 edges and **28.65 s** for Heap on 4,000 / 39,893.
The earlier Batch case took **1.58 s** on 10,000 nodes / 100,000 edges,
using 512/256 batches and ten clusters, and retained exactly 20,000 edges.
These are three-call medians at different graph sizes and settings.

```python
result = scaffold.batch(
    G, keep_ratio=0.2, backbone="fast-maxst", seed=0, workers=8,
    sample_size=512, add_per_round=64, clusters=10, cluster_method="bfs",
    alpha=1.0, beta_edge=1.0, beta_node=1.0,
    edge_norm_p=2.0, node_norm_q=2.0,
)
```

All runs use seed 0, exponents `(1,1,1)`, and path norm orders `(2,2)`,
matching the historical timing configuration. These are not measurements
of the paper's separate `(1,1,0)` training setting. Heap explicitly uses
`score_form="product"`; its default `"max"` form is a different objective.
Greedy/Heap/Batch/Fast use `fast-maxst`; Sample uses its fixed maximum-weight
backbone and eight randomized scoring backbones.

Each completed output passes exact-budget and connectivity checks. Fast's
three 20% calls took **0.722--0.876 s** and scored every non-backbone
candidate. Including bidirectional array normalization took **0.915--0.963 s**
and produced identical masks. At the old 50% ratio, Fast was rechecked at
**0.904--0.968 s**, supporting the historical 0.92 s measurement. Its first
call in that process took **6.711 s** with startup/cache loading; this was
not a clean-install JIT benchmark.

Sample's first draw in the initial run included kernel startup (0.695 s).
The two warm first-draw observations took 0.121 and 0.125 s; six later
draws reused their budget-specific plans. All raw observations, including
Heap's 26.40--68.45 s range on the shared host, are retained in the
[new measurement record](benchmarks/variant_delta20_20260918.json).
The benchmark script now warms Sample.draw explicitly.

Hardware/software: shared dual AMD EPYC 7282 host; eight physical CPUs,
affinity `0,4,8,12,16,20,24,28`; Python 3.9.21, NumPy 2.0.2, Numba 0.60.0.
`OMP_PROC_BIND=false` avoids narrowing the caller's affinity and accidentally
reducing the package's effective worker count. CPU indices are host-specific.
The full API timer includes fresh backbone construction and result assembly
but excludes graph generation, normalization, warm-up and final validation.
Large Batch also includes its lightweight round-progress diagnostics.
No package algorithm changed during these measurements.

## Historical worker scaling

All edge counts below are unique undirected edges. Timings use the standalone
package with Numba, normalized input already in memory, and warmed kernels.
Fast, Batch, Heap and Greedy include backbone construction, selection and
result assembly. Sample measures preprocessing only, not a draw.

| Method | Nodes | Edges | Retention | 1 worker (s) | 4 workers (s) | 8 workers (s) |
|---|---:|---:|---:|---:|---:|---:|
| Fast | 200,000 | 2,399,834 | 50% | 2.79 | 1.11 | 0.92 |
| Sample, 8 backbones | 200,000 | 2,399,834 | Preprocessing | 29.58 | 9.32 | 10.49 |
| Batch | 4,000 | 39,893 | 50% | 14.73 | 4.55 | 2.78 |
| Heap | 4,000 | 39,893 | 50% | 130.92 | 51.08 | 38.89 |
| Greedy | 400 | 3,131 | 72% | 18.04 | 9.28 | 6.99 |

**Do not compare all rows as if they used one graph.** The dynamic methods
were measured on smaller inputs. Fast, Sample, Heap and Greedy are historical
single-run measurements from 2026-08-31 on a 32-core node, preserved in
[the timing record](benchmarks/variant_scaling_20260831.json). Those cells
preserve the historical runs; the common-retention measurements above use
the current package. Batch was measured on 2026-09-18 and reports
the minimum of three calls per worker count, with
[every repetition and its settings recorded](benchmarks/batch_scaling_20260918.json).
The earlier description of all variant rows as "best of three" was incorrect;
only the Batch row in this historical scaling table uses that statistic.

### Batch settings and validation

The graph matches the historical Heap case: 4,000 nodes, 39,893 edges,
unweighted, graph seed 0. It retains exactly **19,947 edges** with:

```python
result = scaffold.batch(
    G, keep_ratio=0.5, backbone="fast-maxst", seed=0,
    clusters=10, cluster_method="bfs",
    sample_size=256, add_per_round=64, workers=8,
    alpha=1.0, beta_edge=1.0, beta_node=1.0,
    edge_norm_p=2.0, node_norm_q=2.0,
)
```

Measurements used a shared 32-core machine with two AMD EPYC 7282 processors,
Python 3.9.21, NumPy 2.0.2 and Numba 0.60.0. Graph creation, input
normalization, JIT warm-up and output validation are outside the timer.
Each run includes clustering and backbone construction; it does not reuse
an already constructed support. There was no exclusive machine reservation.

All twelve runs at 1, 4, 8 and 16 workers returned connected supports with
the same edge-mask hash and exact budget. The 16-worker minimum was **3.34 s**,
slower than **2.78 s** at eight workers. The eight-worker speedup was **5.3×**
over the **14.73 s** serial result. Worker scaling is not necessarily monotonic.

The 256/64 batch sizes are the current defaults for inputs with at least
1,024 edges. The older default was 64/8; these measurements do not describe
that older configuration. This run used all three objective exponents equal
to one, including node congestion.

The synthetic graph generator draws `n * degree_argument` pairs of uniform
random endpoints using `numpy.random.default_rng(0)`, removes self-loops,
canonicalizes each pair as `(min(u,v), max(u,v))`, removes duplicates, and
passes both directions to `scaffold.graph.from_edge_index`. The generator arguments
are `(4000,10)` for Batch/Heap, `(200000,12)` for Fast/Sample, and `(400,8)`
for Greedy. The degree argument is not the realized mean degree. The JSON
records retain the graph hashes and the Batch source-file hashes.

## Estimated cost at 1M nodes and 100M edges

**No graph of this size was benchmarked.** These are rough planning estimates
for eight CPU workers, an unweighted connected graph, compiled Numba kernels,
normalized input already in RAM, and enough memory to avoid swapping. They
exclude loading, graph conversion, JIT compilation and GNN training. Different
hardware, weighted paths and graph structure can change costs substantially.

| Method / setting | Planning estimate | What is included |
|---|---|---|
| Fast, 50% retention | **1–3 minutes** | One support construction |
| Sample, 8 backbones | **10–30 minutes** | Preprocessing, reusable across draws |
| Batch, 10% retention | **Days to weeks** | Exact dynamic construction, 256/64 batches |
| Batch, 50% retention | **Potentially months** | Exact dynamic construction, 256/64 batches |

These are not confidence intervals, guarantees, or experimental results.
No matched package measurement supports a precise per-draw time at this
scale; Sample's preprocessing estimate must not be charged to every epoch.

### How the estimates were obtained

For Fast and Sample, scaling the leading `m log2(n)` work from the historical
200K-node / 2,399,834-edge eight-worker measurements gives:

```text
factor = (100,000,000 / 2,399,834) * log2(1,000,000) / log2(200,000)
       ≈ 47.16
Fast:   0.92 s * factor ≈ 43 s
Sample: 10.49 s * factor ≈ 495 s (8.25 minutes)
```

The stated 1–3 and 10–30 minute ranges add judgment-based allowances for
larger working sets, cache behavior and allocation; they have not been
validated at 100M edges.

Batch behaves differently. Its clusters partition **candidate pools**, while
each shortest-path search traverses the **entire current support graph**.
The compiled scorer performs two search passes per distinct source to size
and fill its path buffers. At 256 candidates and 64 insertions, approximately
four candidate evaluations are needed per added edge, before source sharing.

An illustrative workload scaling of the measured 2.776 s Batch run is:

```text
T = q - (n - 1)                    # additions beyond a connected backbone
h_mean = (q + n - 1) / 2           # approximate mean support size
time_target ≈ time_base * (T_target / T_base)
                         * (n_target + 2*h_mean_target)
                         / (n_base + 2*h_mean_base)
```

With the same batch sizes, this gives roughly **8 days at 10% retention**
and **180 days at 50%**. Holding search throughput and source sharing fixed
across such different sizes is a strong assumption; these figures explain
the broad days-to-months guidance, not precise completion dates. They ignore
several changes in cache behavior, sampling overhead and cluster balance.
Raising the insertion size or changing the path approximation requires a new
measurement and may change the resulting support.

The separate research tensor implementation has much shorter reported Batch
times on large datasets because it reuses an initial tree index for scoring.
Those times do not measure this package's dynamic shortest-path Batch.

## Memory and input representation

At 100M unique undirected edges, two int64 endpoint arrays require **1.6 GB**
in decimal units, and each dense float64 per-edge array requires another
**0.8 GB**. The algorithms hold several arrays at once; graph normalization,
sorting, output conversion and concurrent Sample backbone scoring add more.
These arithmetic examples are not peak-RAM measurements or a recommended
RAM capacity. Swapping can invalidate the runtime estimates entirely.

Count unique undirected edges when choosing a budget. If a PyG edge index
contains both directions, 100M entries normally represent about 50M unique
undirected edges before removing duplicates and self-loops. Loading a graph
through Python objects can also cost much more memory than its array form.

## Measure on your hardware

Install the `speed` extra, start with a manageable graph and the intended
retention ratio, and time preprocessing separately from repeated draws:

```python
from time import perf_counter
import scaffold

# G should already be loaded and normalized. Warm compiled kernels on a
# small representative input before measuring if you want steady-state time.
start = perf_counter()
scores = scaffold.sample(G, tree_count=8, workers=8, seed=0)
preprocessing_seconds = perf_counter() - start

start = perf_counter()
result = scores.draw(keep_ratio=0.1, seed=0)
first_draw_seconds = perf_counter() - start
# The first draw may prepare a budget-specific sampling plan; later draws
# can reuse it. Report first and subsequent draws separately.
```

For a small grid comparison from a repository checkout:

```bash
python benchmarks/bench_methods.py --sizes 16 32 --workers 8 --skip greedy heap
```

That script includes Sample preprocessing **and** a draw in its Sample timing,
so its timing scope differs from the preprocessing-only table above. It is a
convenient local check, not a reproduction of the synthetic graph experiment.
See [algorithm choices](algorithms.md) and [worker configuration](../README.md#parallelism)
for the relevant trade-offs and controls.
