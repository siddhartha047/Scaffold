# Forest versus Scaffold runtime

These are separate instrumented calls at eight workers and 20% retention.
Each row uses the components of the median-total run among three fresh calls.

| Method | Nodes | Edges | Forest (ms) | Scaffold (s) | I/O (s) | Total (s) |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 400 | 3,131 | 0.055 | 0.760 | 0.0004 | 0.760 |
| Heap | 4,000 | 39,893 | 0.297 | 25.758 | 0.0024 | 25.761 |
| Batch | 10,000 | 250,000 | 0.838 | 31.541 | 0.0149 | 31.557 |
| Fast | 200,000 | 2,399,834 | 19.825 | 1.076 | 0.6496 | 1.745 |
| Sample | 200,000 | 2,399,834 | 416.134 | 9.366 | 1.5978 | 11.380 |

Forest time is **milliseconds**, other columns are **seconds**.
Scaffold includes score computation, selection, indexing, bookkeeping,
result assembly, and (for Sample) its first draw. I/O is in-memory input
normalization and bidirectional output export, not disk loading.

The first four methods each build one Fast-MaxSF. Sample includes one
fixed MaxSF plus eight randomized scoring forests. A benchmark-only
barrier separates its concurrent forest builds from their scoring so
wall-clock phases do not overlap. Forest times are not summed over workers.
This changes scheduling, so these totals do not replace the normal API
measurements in the main audit. Differences between calls also reflect
the shared host and allocation/cache variation. Every selected edge mask
matches the original audit; algorithm source hashes are unchanged.

Raw data: `split.json`. Reproduce from the package root:

```bash
OMP_PROC_BIND=false NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0,4,8,12,16,20,24,28 python benchmarks/bench_runtime_split.py \
  --output /tmp/scaffold-runtime-split.json
python benchmarks/render_runtime_split.py \
  --records /tmp/scaffold-runtime-split.json --output /tmp/scaffold-split-latex
```
