"""Render the forest/Scaffold split without overwriting ordinary API timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_runtime_audit import LABELS, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.records.read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["source_unchanged"]
    chosen = {}
    for method in METHODS:
        rows = [r for r in report["records"] if r["method"] == method]
        assert len(rows) == 3 and len({r["mask_sha256"] for r in rows}) == 1
        for r in rows:
            p = r["phases"]
            assert abs(p["forest"] + p["scaffold"] + p["io"] - r["seconds"]) < 1e-8
            assert r["added_edges"] >= 2 and r["workers"] == 8
            assert r["forest_calls"] == (9 if method == "sample" else 1)
        chosen[method] = sorted(rows, key=lambda r: r["seconds"])[1]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(statistic="components of the median-total run among three fresh calls",
                   graphs=report["graphs"], rows=chosen, protocol=report["split_protocol"])
    (args.output / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sizes = {"greedy": r"$(400,\,3.1\mathrm{K})$", "heap": r"$(4\mathrm{K},\,39.9\mathrm{K})$",
             "batch": r"$(10\mathrm{K},\,250\mathrm{K})$",
             "fast": r"$(200\mathrm{K},\,2.4\mathrm{M})$",
             "sample": r"$(200\mathrm{K},\,2.4\mathrm{M})$"}
    costs = {
        "greedy": r"$\mathcal{O}(TnC_{\rm sp})$",
        "heap": r"\begin{tabular}[c]{@{}l@{}}$\mathcal{O}((n+T(\kappa+c))C_{\rm sp}$\\$\qquad{}+m\log m)$\end{tabular}",
        "batch": r"$\mathcal{O}(\lceil T/r\rceil bC_{\rm sp})$",
        "fast": r"$\mathcal{O}((m+n)\log n)$",
        "sample": r"\begin{tabular}[c]{@{}l@{}}$\mathcal{O}(R(m+n)\log n)$ pre.\\$\mathcal{O}(m)$ draw\end{tabular}",
    }
    uses = dict(greedy="Reference", heap="Small", batch="Medium/large", fast="Large", sample="Large")
    tex = r"""% Alternative main table; use INSTEAD OF the earlier main table (same label).
% Requires amsmath, booktabs, graphicx and the paper's graph macros.
\begin{table}[!htbp]
\centering
\caption{\sgt{} variants: selection cost and construction-time breakdown
($\delta=0.2$, eight CPU workers).}
\label{tab:scaffold-variants}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}llrrrrl@{}}
\toprule
\textbf{Variant} & \textbf{Selection cost} & \textbf{Graph $(n,m)$}
& \shortstack{\textbf{Forest}\\\textbf{(ms)}}
& \shortstack{\textbf{Scaffold + I/O}\\\textbf{(s)}}
& \shortstack{\textbf{Total}\\\textbf{(s)}} & \textbf{Use} \\
\midrule
"""
    for method, r in chosen.items():
        p = r["phases"]
        tex += (f"{LABELS[method]} & {costs[method]} & {sizes[method]} & "
                f"{p['forest']*1000:.3f} & {p['scaffold']+p['io']:.3f} & "
                f"{r['seconds']:.3f} & {uses[method]} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\vspace{2pt}
\parbox{\linewidth}{\scriptsize
$T=q-|\gE_{\gF}|$; $C_{\rm sp}$ is one BFS/Dijkstra search.
Unweighted synthetic graphs; kernels warm. Each row uses all components
from the median-total run of three fresh constructions. Forest time is in
\emph{milliseconds}; the remaining work includes normalization, scoring,
selection, and output assembly. Sample includes its fixed forest, $R=8$
random forests, and first draw; forest/scoring phases are synchronized for
this diagnostic. Batch uses ten clusters, $b=512,r=64$ per cluster.
Loading, compilation, and training are excluded.}
\end{table}
"""
    (args.output / "main_variants_runtime_split.tex").write_text(tex, encoding="utf-8")

    appendix = r"""% Append to the runtime subsection; keeps the normal API scaling table unchanged.
% Requires booktabs. This is a separately instrumented diagnostic.
\paragraph{Separating forest construction from sparsification.}
We instrument fresh constructions using the same unweighted graphs, budgets,
backbones, and algorithm parameters as the runtime audit, with eight workers
and $\delta=0.2$. The phases are disjoint wall-clock intervals:
\emph{forest} builds the backbone(s), \emph{Scaffold} covers the remaining
scoring, selection, indexing, bookkeeping, and result assembly, and
\emph{I/O} covers input normalization and bidirectional output export.
For Sample, Scaffold includes score aggregation and the first sparse draw.

\begin{table}[!htbp]
\centering
\caption{Stage-separated runtime. Each row reports components from the
median-total run of three fresh eight-worker calls.}
\label{tab:scaffold-runtime-split}
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}llrrrr@{}}
\toprule
\textbf{Variant} & \textbf{Graph $(n,m)$} & \textbf{Forest (ms)}
& \textbf{Scaffold (s)} & \textbf{I/O (s)} & \textbf{Total (s)} \\
\midrule
"""
    for method, r in chosen.items():
        p = r["phases"]
        appendix += (f"{LABELS[method]} & {sizes[method]} & {p['forest']*1000:.3f} & "
                     f"{p['scaffold']:.3f} & {p['io']:.4f} & {r['seconds']:.3f} \\\\\n")
    appendix += r"""\bottomrule
\end{tabular}
\end{table}

Greedy, Heap, Batch, and Fast each build one Fast-MaxSF. Sample's forest
column includes its fixed MaxSF and all eight randomized scoring forests,
including randomized edge ordering. The eight random builds run concurrently;
their wall time is measured as one phase, not summed across workers.
A barrier holds their scoring until all eight forests are ready, preventing
double-counting of overlapping work. This synchronization is used only in the
diagnostic: the normal Sample implementation overlaps work across forests.
Consequently, these totals are separate measurements and do not replace the
normal API worker-scaling results. All outputs match the original audit exactly.
Choosing the components of the median-total run, rather than taking separate
component medians, ensures that forest plus Scaffold plus I/O equals the
measured total before rounding.
"""
    (args.output / "appendix_runtime_split.tex").write_text(appendix, encoding="utf-8")

    lines = ["# Forest versus Scaffold runtime", "",
             "These are separate instrumented calls at eight workers and 20% retention.",
             "Each row uses the components of the median-total run among three fresh calls.",
             "", "| Method | Nodes | Edges | Forest (ms) | Scaffold (s) | I/O (s) | Total (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, r in chosen.items():
        p, g = r["phases"], report["graphs"][method]
        lines.append(f"| {LABELS[method]} | {g['n']:,} | {g['m']:,} | {p['forest']*1000:.3f} | "
                     f"{p['scaffold']:.3f} | {p['io']:.4f} | {r['seconds']:.3f} |")
    lines.extend(["", "Forest time is **milliseconds**, other columns are **seconds**.",
                  "Scaffold includes score computation, selection, indexing, bookkeeping,",
                  "result assembly, and (for Sample) its first draw. I/O is in-memory input",
                  "normalization and bidirectional output export, not disk loading.", "",
                  "The first four methods each build one Fast-MaxSF. Sample includes one",
                  "fixed MaxSF plus eight randomized scoring forests. A benchmark-only",
                  "barrier separates its concurrent forest builds from their scoring so",
                  "wall-clock phases do not overlap. Forest times are not summed over workers.",
                  "This changes scheduling, so these totals do not replace the normal API",
                  "measurements in the main audit. Differences between calls also reflect",
                  "the shared host and allocation/cache variation. Every selected edge mask",
                  "matches the original audit; algorithm source hashes are unchanged.", "",
                  "Raw data: `split.json`. Reproduce from the package root:", "", "```bash",
                  "OMP_PROC_BIND=false NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 \\",
                  "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
                  "taskset -c 0,4,8,12,16,20,24,28 python benchmarks/bench_runtime_split.py \\",
                  "  --output /tmp/scaffold-runtime-split.json",
                  "python benchmarks/render_runtime_split.py \\",
                  "  --records /tmp/scaffold-runtime-split.json --output /tmp/scaffold-split-latex",
                  "```", ""])
    (args.records.parent / "split.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({m: {"forest_ms": r["phases"]["forest"]*1000,
                          "scaffold_seconds": r["phases"]["scaffold"],
                          "io_seconds": r["phases"]["io"], "total_seconds": r["seconds"]}
                      for m, r in chosen.items()}, indent=2))


if __name__ == "__main__":
    main()
