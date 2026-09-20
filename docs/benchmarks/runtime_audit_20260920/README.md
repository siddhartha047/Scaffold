# Runtime audit, 2026-09-20

For a separate breakdown of **forest construction versus Scaffold work**,
see [split.md](split.md) and the raw `split.json` records. That diagnostic
synchronizes Sample's forest/scoring phases; the original normal-API records
and tables below remain unchanged.

This audit replaces the paper's mixed-scope construction times with one
definition: **a bidirectional NumPy edge list in memory to a materialized,
bidirectional sparse edge list**. The wall-clock timer surrounds the public API
and output conversion. It includes normalization, fresh backbone construction,
scoring, selection, and output assembly. Sample includes fresh preprocessing
and its **first draw**; a fitted sampler is never reused for a complete-call
measurement. Cached draws have separate records.

The benchmark does not change algorithm code. Graph generation/loading, JIT
warm-up, validation, training, and GPU transfer are outside the timer. These
are warmed construction measurements, not clean-install startup estimates.
The first public Fast call in the process is additionally recorded, with disk
JIT caches allowed; it is not presented as an empty-cache compilation test.

## Files and reproducibility

- `variants.json`: all repetitions at 1, 4, and 8 workers; Sample phase times
  and cached draws; graph hashes, exact budgets, settings, and source hashes.
- `profiles.json`: independent instrumented Fast calls plus normalized-input
  API checks. Each instrumented call verifies that a new forest is built once
  and that every omitted edge is scored. The phase times are not used to
  synthesize the main table: the table uses the outer complete-call timer.
- `profiles_python39.json`: the same diagnostic in the historical Python
  3.9.21 / NumPy 2.0.2 / Numba 0.60.0 environment. Its normalized-input Fast
  median is 0.789 s, including a fresh forest. This is a scope/environment
  comparison, not a replacement for the primary current-environment totals.
- `backbones.json`: complete backbone builds from normalized **unweighted**
  graphs. These are component costs, distinct from complete variant costs.
- `scorers.json`: BFS/Dijkstra scoring-only diagnostics on a fixed support.
  These do not produce a new sparse graph and are not mixed into variant totals.

Only reports with `status="complete"` may be rendered. Any timed-out backbone
call remains explicitly censored rather than being used as a completed time.
LLSF is deliberately measured once with GLSF initialization, an up-to-ten-pass
exhaustive search, and a 30-minute cap; other cells use three-call medians.
Its normal 1,000-edge guard is explicitly raised to 1,542 for this benchmark.
No large GLSF/LLSF run is attempted.

Run each section **sequentially**, with no competing benchmark process:

```bash
export OMP_PROC_BIND=false NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for section in variants profiles scorers backbones; do
    taskset -c 0,4,8,12,16,20,24,28 python benchmarks/bench_runtime_audit.py \
        --section "$section" \
        --output "docs/benchmarks/runtime_audit_20260920/$section.json"
done
python benchmarks/render_runtime_audit.py \
    --records docs/benchmarks/runtime_audit_20260920 --output /tmp/scaffold-runtime-latex
```

The affinity mask is specific to the recorded 32-core dual AMD EPYC 7282 host.
Use available physical CPUs on another host and retain their actual mask in
the report. The current environment is Python 3.12.9, NumPy 2.2.6, Numba 0.66.0.
The machine is shared, not exclusively reserved. Every repetition is retained;
worker order is rotated to reduce systematic ordering effects.

All sections completed successfully, with unchanged algorithm source hashes.
At eight workers, complete-call medians are **0.78 s** (Greedy), **25.16 s**
(Heap), **31.01 s** (Batch), **2.30 s** (Fast), and **9.02 s** (Sample), on
the different graph sizes listed below. LLSF completed its one small-graph
construction in **392.81 s**, including initialization; no run timed out.

For the historical-environment diagnostic, repeat the `profiles` command with
that environment's Python and save to `profiles_python39.json`. Do not replace
`profiles.json` or mix those observations into the primary runtime table.

## Budgets and checks

Every variant uses `q=max(ceil(0.2*m), n-components+2)`. On these connected
inputs the maximum always chooses `ceil(0.2*m)`, so no ratio increase is needed.

| Method | Nodes | Unique undirected edges | Forest edges | Output edges | Added edges |
|---|---:|---:|---:|---:|---:|
| Greedy | 400 | 3,131 | 399 | 627 | 228 |
| Heap | 4,000 | 39,893 | 3,999 | 7,979 | 3,980 |
| Batch | 10,000 | 250,000 | 9,999 | 50,000 | 40,001 |
| Fast / Sample | 200,000 | 2,399,834 | 199,999 | 479,967 | 279,968 |

The graph hashes match the earlier measurements exactly. Every timed output
is subsequently checked for exact budget, preserved connectivity, and identical
selection across worker counts/repetitions under fixed randomness. Fast must
report one scoring pass over all **2,199,835** non-backbone candidates; its
forest-only early-return branch cannot satisfy the checks.

Use the same settings in both paper tables: `(alpha,beta_edge,beta_node)=(1,1,1)`,
path norm orders `(2,2)`, seed 0, Fast-MaxSF for the four growth methods, fixed
MaxSF and eight randomized scoring forests for Sample. Heap uses product scores,
`top_k=16`, `dirty_limit=64`; Batch uses `sample_size=512`, `add_per_round=64`,
ten BFS candidate clusters. These are the historical timing settings, distinct
from the separate `(1,1,0)` training setting.

## Problems found in the earlier presentation

1. The main Sample cell was preprocessing only. It did not yet return the
   sparse support described by the table. The new total includes the first draw.
2. Main and appendix tables used different retention ratios, Batch graph sizes,
   batch sizes, and repetition statistics. They now share the exact same
   complete-call records at each worker count.
3. The old backbone table used random positive edge weights, although the
   supplied appendix described unweighted benchmarks. Its older SPF also used
   BFS on weighted input. The new table measures unit-weight inputs explicitly;
   the old weighted values remain historical records, not fresh measurements.
4. The old Fast 0.73-second measurement surrounded the normalized-input API
   and **did include a fresh forest**. Its budget was above the connectivity
   floor. A short runtime alone was not evidence of a missing forest. The new
   audit records that narrower scope separately and uses the broader common
   input-to-output scope for both replacement tables.

Component-wise medians do not necessarily sum to the median total. Use the
measured complete-call median for a reported total, not a sum of medians or
the fastest observation from each stage.
