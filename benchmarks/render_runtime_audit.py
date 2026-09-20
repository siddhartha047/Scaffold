"""Render paper snippets and a readable audit from completed raw observations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METHODS = ("greedy", "heap", "batch", "fast", "sample")
LABELS = dict(greedy="Greedy", heap="Heap", batch="Batch", fast="Fast", sample="Sample")


def median(rows, key="seconds"):
    return statistics.median(row[key] for row in rows)


def format_ms(value):
    if value < 1:
        return f"{value:.3f}"
    return f"{value:,.2f}" if value < 100 else f"{value:,.0f}"


def update_package_docs(root, variant, times, summary):
    """Publish the same complete-call medians without disturbing other README sections."""
    graphs = variant["graphs"]
    rows = ["| Method | Nodes | Edges | Complete time (s) |", "|---|---:|---:|---:|"]
    for method in METHODS:
        g = graphs[method]
        name = "Sample, 8 random scoring forests + first draw" if method == "sample" else LABELS[method]
        rows.append(f"| {name} | {g['n']:,} | {g['m']:,} | **{times[method, 8]:.2f}** |")
    table = "\n".join(rows)
    cached = summary["sample_cached_draw_median"]
    section = f"""## Runtime expectations

**20% edge retention · 8 CPU workers · unweighted synthetic graphs.**
Counts are unique undirected edges; graph sizes differ across rows.

{table}

Medians of three fresh constructions on a shared dual AMD EPYC 7282 host.
The timer includes **input normalization, a new spanning forest, scoring,
selection, and bidirectional sparse-output assembly**. Sample includes
preprocessing and its **first draw**. Kernels are warm; graph loading, JIT
startup, GNN training, and GPU transfer are excluded.

Batch uses **512 candidates / 64 insertions per cluster**, ten BFS clusters.
The benchmarks use Fast-MaxSF; Sample uses fixed MaxSF plus eight randomized
scoring forests. These settings differ from the recommended Fast-RandSF
quick start. Every result satisfies the exact budget and connectivity checks.

After Sample preprocessing, later **cached draws take {cached:.3f} s** (median of
nine), including output assembly. Use Fast for one support and Sample for
repeated draws. Dynamic Batch can cost much more on larger inputs.
[Full audited settings, worker scaling, stage timings, and raw records](docs/performance.md).

"""
    readme = root / "README.md"
    content = readme.read_text(encoding="utf-8")
    content, count = re.subn(r"## Runtime expectations\n.*?(?=## Choose an algorithm\n)",
                            lambda _: section, content, flags=re.S)
    assert count == 1
    readme.write_text(content, encoding="utf-8")
    performance = root / "docs/performance.md"
    archive = root / "docs/performance_history_20260919.md"
    old = performance.read_text(encoding="utf-8")
    if not archive.exists():
        archive.write_text(
            "> Archived measurements and planning estimates. The current audited\n"
            "> input-to-output timings are in [performance.md](performance.md).\n\n" + old,
            encoding="utf-8")
    historical = archive.read_text(encoding="utf-8")
    worker_section = historical[historical.index("## Worker configuration"):]
    worker_rows = ["| Method | 1 worker (s) | 4 workers (s) | 8 workers (s) | Speedup |",
                   "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        worker_rows.append(f"| {LABELS[method]} | {times[method,1]:.2f} | {times[method,4]:.2f} | {times[method,8]:.2f} | {times[method,1]/times[method,8]:.2f}× |")
    stages = summary["fast_stage_medians"]
    stage_rows = ["| Fast stage | Median seconds |", "|---|---:|"]
    for key, value in stages.items():
        stage_rows.append(f"| {key.replace('_', ' ')} | {value:.4f} |")
    text = f"""# Runtime and memory expectations

These are the **2026-09-20 audited measurements**, using the package's public API.
Use Fast for one sparse support and Sample for repeated draws; dynamic Batch
repeatedly searches the evolving support and can be much more expensive.

## Complete construction at 20% retention

Eight CPU workers, unweighted synthetic graphs, three-call medians:

{table}

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
`0,4,8,12,16,20,24,28`; Python {variant['python'].split()[0]}, NumPy {variant['numpy']},
Numba {variant['numba']}. `OMP_PROC_BIND=false`; BLAS is limited to one thread.
The machine was not exclusively reserved. No observations were discarded;
all repetitions and their ranges are available in the raw JSON.

## Worker scaling

Complete construction, including a fresh forest and (for Sample) a first draw.
Graph sizes and all other settings match the table above:

{chr(10).join(worker_rows)}

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

{chr(10).join(stage_rows)}

Stage medians need not sum to the median total. The complete time reported above
is measured by an outer wall-clock timer, not reconstructed from stage medians.
The separately timed **normalized-input API** median is
**{summary['fast_normalized_api_median']:.3f} s**. The previous **0.73 s** measurement
used that narrower warmed scope in an older environment; its timer already
included forest construction. It is retained only in the historical record,
not used as the new complete input-to-output time.

## Sample preprocessing versus repeated draws

At eight workers, the preprocessing API (including input normalization) takes
**{summary['sample_preprocessing_median']:.3f} s**, and the first draw takes
**{summary['sample_first_draw_median']:.3f} s** (separate component medians).
The complete-call median, including output assembly, is
**{times['sample',8]:.2f} s**. The old main-table Sample entry timed preprocessing
alone and did not yet produce a sparse graph.

Later cached draws, including bidirectional output assembly, take
**{cached:.3f} s** (median of nine). These deliberately reuse preprocessing and
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

{worker_section}
"""
    if "fast_python39_normalized_api_median" in summary:
        legacy_note = ("A fresh check in the historical Python 3.9.21 / NumPy 2.0.2 / "
                       "Numba 0.60.0 environment measured **"
                       f"{summary['fast_python39_normalized_api_median']:.3f} s** for the same normalized-input "
                       "API (median of three), still including a new forest. This supports the "
                       "plausibility of the older 0.73 s observation; it is not used in the "
                       "current-environment input-to-output tables.\n\n")
        text = text.replace("## Sample preprocessing versus repeated draws", legacy_note + "## Sample preprocessing versus repeated draws")
    performance.write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-docs", action="store_true", help="also refresh package README/performance from these records")
    parser.add_argument("--main-only", action="store_true", help="render the main snippet once variant timings finish")
    args = parser.parse_args()
    sections = ("variants",) if args.main_only else ("variants", "profiles", "backbones", "scorers")
    reports = {name: json.loads((args.records / f"{name}.json").read_text(encoding="utf-8"))
               for name in sections}
    assert all(value["status"] == "complete" for value in reports.values())
    variant = reports["variants"]
    assert all(value["source_sha256"] == variant["source_sha256"] for value in reports.values())
    groups = {(method, p): [r for r in variant["records"]
                            if r["kind"] == "variant" and r["method"] == method and r["workers"] == p]
              for method in METHODS for p in (1, 4, 8)}
    assert all(len(rows) == variant["repeats"] for rows in groups.values())
    times = {key: median(rows) for key, rows in groups.items()}
    graphs = variant["graphs"]
    legacy_path = args.records / "profiles_python39.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else None
    legacy_normalized = None
    if legacy is not None:
        assert legacy["status"] == "complete"
        legacy_normalized = median([r for r in legacy["records"] if r["kind"] == "fast_normalized_api"])
    graph_tex = {"greedy": r"$(400,\,3.1\mathrm{K})$", "heap": r"$(4\mathrm{K},\,39.9\mathrm{K})$",
                 "batch": r"$(10\mathrm{K},\,250\mathrm{K})$", "fast": r"$(200\mathrm{K},\,2.4\mathrm{M})$",
                 "sample": r"$(200\mathrm{K},\,2.4\mathrm{M})$"}
    policies = dict(greedy="dynamic; best one", heap="dynamic; heap max.", batch="sampled; top-$r$",
                    fast="static; top-$T$", sample="$R$ forests; draw")
    complexities = {
        "greedy": r"$\mathcal{O}(TnC_{\rm sp})$",
        "heap": r"\begin{tabular}[c]{@{}l@{}}$\mathcal{O}((n+T(\kappa+c))C_{\rm sp}$\\$\qquad{}+m\log m)$\end{tabular}",
        "batch": r"$\mathcal{O}(\lceil T/r\rceil bC_{\rm sp})$",
        "fast": r"$\mathcal{O}((m+n)\log n)$",
        "sample": r"\begin{tabular}[c]{@{}l@{}}$\mathcal{O}(R(m+n)\log n)$ pre.\\$\mathcal{O}(m)$ draw\end{tabular}",
    }
    uses = dict(greedy="Reference", heap="Small", batch="Medium/large", fast="Large", sample="Large")
    main_tex = r"""% Requires amsmath, booktabs, graphicx. Replaces the main-paper variants paragraph/table.
\paragraph{Scalable variants.}
Full recomputation makes \texttt{Greedy} expensive. \texttt{Heap} caches paths
and selectively refreshes candidates; \texttt{Batch} scores sampled batches
and inserts their top-$r$ edges; \texttt{Fast} scores the initial forest once
using LCA and prefix sums; and \texttt{Sample} amortizes scores from multiple
forests over repeated budgeted draws. Table~\ref{tab:scaffold-variants} reports
complete support-construction times; Appendix~\ref{app:scaffold-runtime}
gives the measurement protocol and worker scaling.

\begin{table}[!htbp]
\centering
\caption{\sgt{} variants: selection complexity and complete construction time
($\delta=0.2$, eight CPU workers).}
\label{tab:scaffold-variants}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lllrrl@{}}
\toprule
\textbf{Variant} & \textbf{Scoring / selection} & \textbf{Selection cost}
& \textbf{Graph $(n,m)$} & \textbf{Total (s)} & \textbf{Use} \\
\midrule
"""
    for method in METHODS:
        main_tex += f"{LABELS[method]} & {policies[method]} & {complexities[method]} & {graph_tex[method]} & {times[method, 8]:.2f} & {uses[method]} \\\\\n"
    main_tex += r"""\bottomrule
\end{tabular}}
\vspace{2pt}
\parbox{\linewidth}{\scriptsize
$T=q-|\gE_{\gF}|$; $C_{\rm sp}$ is one BFS/Dijkstra search.
Unweighted synthetic graphs; medians of three complete calls, including input normalization, fresh forest
construction, scoring/selection, and sparse-output assembly. Sample includes
$R=8$ preprocessing \emph{and the first draw}. Batch uses ten clusters with
$b=512$, $r=64$ per cluster. Compiled kernels are warm; loading and GNN
training are excluded. Graph sizes differ across rows.}
\end{table}
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "main_variants_runtime.tex").write_text(main_tex, encoding="utf-8")
    if args.main_only:
        print(args.output / "main_variants_runtime.tex")
        return
    appendix = r"""% Requires amsmath, amssymb, booktabs, array, graphicx.
% Replaces only the Runtime and Scalability subsection; other appendix text stays unchanged.
\subsection{Runtime and Scalability}
\label{app:scaffold-runtime}

For $q=\lceil\delta m\rceil$, construction comprises a spanning-forest
backbone $\gF$ and selection of $T=q-|\gE_{\gF}|$ further edges.
The complexities in Table~\ref{tab:scaffold-variants} describe selection;
total algorithmic cost also includes the chosen backbone construction.

\paragraph{Measurement protocol.}
All measurements below use the current standalone package on a shared
dual AMD EPYC 7282 machine (32 physical cores; Python 3.12.9, NumPy 2.2.6,
Numba 0.66.0), with an affinity mask of eight physical cores.
We report the median of three warm calls at each worker count, without
discarding outliers. The complete-construction timer starts with an in-memory
bidirectional NumPy edge list and includes normalization, a newly constructed
forest, candidate scoring, edge selection, and bidirectional sparse-output
assembly. For Sample it includes fresh preprocessing \emph{and its first draw};
cached draws are reported separately. Graph generation/loading, JIT warm-up,
correctness checks, GNN training, and GPU transfer are excluded.
No fitted sampler or constructed forest is reused between timed complete calls.

The variant inputs are seeded unweighted synthetic graphs.
The objective exponents are $(1,1,1)$ and the path
norm orders are $(2,2)$; Heap uses product scores with $\kappa=16,c=64$.
Greedy, Heap, Batch, and Fast use Fast-MaxSF; Sample uses a fixed MaxSF and
$R=8$ randomized scoring forests. Batch uses ten BFS candidate clusters with
$b=512$ and $r=64$ per cluster. These settings are held fixed across worker
counts. All three score terms are enabled for these construction benchmarks.

\paragraph{Budgets and validation.}
For these benchmarks, we require $q\ge n-c_{\gG}+2$, where $c_{\gG}$ is the number of input
components. All measured inputs are connected, and $\delta=0.2$ already
satisfies this condition, so no ratio increase is needed:
\begin{center}
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Variant} & $n$ & $m$ & $q$ & $T$ \\
\midrule
"""
    for method in ("greedy", "heap", "batch", "fast"):
        g = graphs[method]
        name = "Fast / Sample" if method == "fast" else LABELS[method]
        appendix += f"{name} & {g['n']:,} & {g['m']:,} & {g['q']:,} & {g['added_edges']:,} \\\\\n"
    appendix += r"""\bottomrule
\end{tabular}
\end{center}
Every result is checked for its exact edge budget, preserved connectivity,
and identical selected edges across repetitions and worker counts under fixed
randomness. Fast scores all $2{,}199{,}835$ non-forest candidates.

\paragraph{Parallel scaling.}
Independent source searches and forest-scoring operations can use multiple
CPU workers, while dynamic insertion rounds remain sequential.
Table~\ref{tab:scaffold-scaling} uses the same graphs, budgets, and settings
as the main table; its eight-worker column therefore matches that table.

\begin{table}[!htbp]
\centering
\caption{Complete support-construction time at $\delta=0.2$, including fresh
forest construction. Sample includes preprocessing and its first draw.}
\label{tab:scaffold-scaling}
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}llrrrr@{}}
\toprule
\textbf{Variant} & \textbf{Graph $(n,m)$} & \multicolumn{3}{c}{\textbf{Total time (s)}} & \textbf{Speedup} \\
\cmidrule(lr){3-5}
& & $P=1$ & $P=4$ & $P=8$ & $t_1/t_8$ \\
\midrule
"""
    for method in METHODS:
        appendix += f"{LABELS[method]} & {graph_tex[method]} & {times[method,1]:.2f} & {times[method,4]:.2f} & {times[method,8]:.2f} & ${times[method,1]/times[method,8]:.2f}\\times$ \\\\\n"
    appendix += r"""\bottomrule
\end{tabular}
\end{table}

"""
    samples = groups["sample", 8]
    cached = [r for r in variant["records"] if r["kind"] == "cached_draw" and r["workers"] == 8]
    appendix += (f"At eight workers, Sample preprocessing takes {median(samples, 'api_seconds'):.2f}~s "
                 f"and its first draw takes {median(samples, 'first_draw_seconds'):.3f}~s "
                 f"(component-wise medians). Later cached draws, including sparse-output "
                 f"assembly, take {median(cached):.3f}~s (median of nine draws). "
                 "The complete-call median, rather than a sum of separately rounded medians, "
                 "is used in both variant tables.\n\n")
    profile = reports["profiles"]
    traces = [r for r in profile["records"] if r["kind"] == "fast_profile"]
    normalized = [r for r in profile["records"] if r["kind"] == "fast_normalized_api"]
    stages = {key: statistics.median(r["phases"][key] for r in traces) for key in traces[0]["phases"]}
    appendix += r"""\paragraph{Backbone construction.}
The following table times only a complete backbone build from normalized
input, separately from the variant totals above. The inputs are unweighted.
Exact MaxSF/MinSF use sorted Kruskal for
general weights; their fast versions use $B=256$ buckets. The unit-weight
input bypasses weight ordering, so those constructions select the same forest.
RandSF shuffles edge order, Fast-RandSF uses a seeded strided order, and SPF
uses a degree-rooted BFS forest. GLSF and LLSF are measured only on the small
input; LLSF includes GLSF initialization and up to ten exhaustive local-search passes.

\begin{table}[!htbp]
\centering
\caption{Support-backbone construction times on unweighted inputs (ms).
Complexities also describe general weighted input where applicable.}
\label{tab:scaffold-backbones}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}llrr@{}}
\toprule
\textbf{Backbone} & \textbf{Complexity}
& \shortstack{$n=200\mathrm{K}$\\$m=3{,}119{,}758$}
& \shortstack{$n=400$\\$m=1{,}542$} \\
\midrule
"""
    bc = reports["backbones"]
    for graph in ("small", "large"):
        unit_forests = {r["mask_sha256"] for r in bc["records"] if r["graph"] == graph
                        and r["method"] in ("fast-maxsf", "fast-minsf", "maxsf", "minsf")}
        assert len(unit_forests) == 1, "unit-weight MinSF/MaxSF forests differ"
    backbone_specs = [
        (("fast-maxsf", "fast-minsf"), "Fast-MaxSF/MinSF", r"$\mathcal{O}(m\alpha_{\rm UF}(n)+B)$"),
        (("maxsf", "minsf"), "MaxSF/MinSF", r"$\mathcal{O}(m\log m)$"),
        (("fast-randsf",), "Fast-RandSF", r"$\mathcal{O}(m\alpha_{\rm UF}(n))$"),
        (("randsf",), "RandSF", r"$\mathcal{O}(m\alpha_{\rm UF}(n))$"),
        (("spf",), "SPF", r"$\mathcal{O}(m+n\log n)$"),
        (("glsf",), "GLSF", r"$\mathcal{O}(nm\bar c)$"),
        (("llsf",), "LLSF", r"$C_{\rm init}+\mathcal{O}(L m\bar\ell(n+m)\log n)$"),
    ]
    backbone_summary = {}
    for names, name, complexity in backbone_specs:
        cells = []
        for graph in ("large", "small"):
            all_rows = [r for r in bc["records"] if r["method"] in names and r["graph"] == graph]
            if not all_rows:
                cells.append("---")
                continue
            if any(r["status"] == "timeout" for r in all_rows):
                lower = min(r["lower_bound_seconds"] for r in all_rows if r["status"] == "timeout")
                cells.append(f"$>{lower*1000:,.0f}$")
                continue
            ms = [1000 * median([r for r in all_rows if r["method"] == key]) for key in names]
            endpoints = list(dict.fromkeys(format_ms(value) for value in (min(ms), max(ms))))
            cell = "--".join(endpoints)
            if names == ("llsf",):
                cell += "$^\\dagger$"
            cells.append(cell)
            backbone_summary[f"{name}/{graph}"] = ms
        appendix += f"{name} & {complexity} & {' & '.join(cells)} \\\\\n"
    appendix += r"""\bottomrule
\end{tabular}
\vspace{2pt}
\parbox{\linewidth}{\scriptsize
$\alpha_{\rm UF}$ is the inverse Ackermann function; $\bar c$ is the mean
component count during GLSF construction, $L=10$ is the LLSF pass cap, and $\bar\ell$ is the
mean number of tested cycle edges. Ranges span the separate MaxSF/MinSF
medians. $^\dagger$One completed LLSF call; all other entries are three-call
medians. LLSF's input guard was explicitly raised to 1,542 edges for this
benchmark. Dashes mean not measured.}
\end{table}

\paragraph{Selection complexity.}
Candidates sharing a source reuse a search. Writing
$C_{\rm sp}=\mathcal{O}(n+q)$ for BFS and
$C_{\rm sp}=\mathcal{O}((n+q)\log n)$ for Dijkstra gives
\[
\begin{aligned}
\text{Greedy}:&\ \mathcal{O}(TnC_{\rm sp}),\\
\text{Heap}:&\ \mathcal{O}((n+T(\kappa+c))C_{\rm sp}+m\log m),\\
\text{Batch}:&\ \mathcal{O}(\lceil T/r\rceil bC_{\rm sp}),\\
\text{Fast}:&\ \mathcal{O}((m+n)\log n),\\
\text{Sample}:&\ \mathcal{O}(R(m+n)\log n)\text{ preprocessing},\quad
\mathcal{O}(m)\text{ per draw}.
\end{aligned}
\]
Here $b,r$ denote Batch candidate/insertion sizes; in the clustered
implementation they apply per cluster. Forest construction must be added
to these selection bounds; the measured variant totals already include it.

\paragraph{Scoring-only diagnostic.}
For completeness, an independent scoring pass on a fixed support is measured
below. These are component timings, not complete sparsification times, and
are excluded from the complete-construction comparison.
\begin{center}
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Scorer} & $P=1$ (s) & $P=4$ (s) & $P=8$ (s) \\
\midrule
"""
    sc = reports["scorers"]
    for method in ("bfs", "dijkstra"):
        vals = [median([r for r in sc["records"] if r["method"] == method and r["workers"] == p]) for p in (1, 4, 8)]
        appendix += f"{method.upper() if method == 'bfs' else 'Dijkstra'} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\\n"
    appendix += r"""\bottomrule
\end{tabular}
\end{center}
"""
    sg = sc["graphs"]["scoring"]
    appendix += (f"The diagnostic uses $n={sg['n']:,}$, $m={sg['m']:,}$, and "
                 f"{sg['m']-sg['forest_edges']:,} omitted candidates. BFS uses unit weights; "
                 "Dijkstra uses seeded positive weights on the same topology. "
                 "The already-built forest is held fixed.\n\n"
                 "Detailed GNN training and GPU-memory measurements are separate from these "
                 "CPU construction measurements (Appendix~\\ref{app:runtime-details}).\n")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "main_variants_runtime.tex").write_text(main_tex, encoding="utf-8")
    (args.output / "appendix_runtime_scalability.tex").write_text(appendix, encoding="utf-8")
    summary = dict(complete_median_seconds={method: {str(p): times[method, p] for p in (1, 4, 8)} for method in METHODS},
                   graphs=graphs, fast_stage_medians=stages,
                   fast_normalized_api_median=median(normalized),
                   sample_preprocessing_median=median(samples, "api_seconds"),
                   sample_first_draw_median=median(samples, "first_draw_seconds"),
                   sample_cached_draw_median=median(cached), backbone_median_ms=backbone_summary)
    if legacy_normalized is not None:
        summary["fast_python39_normalized_api_median"] = legacy_normalized
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if args.update_docs:
        update_package_docs(Path(__file__).resolve().parents[1], variant, times, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
