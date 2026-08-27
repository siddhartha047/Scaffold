"""Runtime and quality comparison of the four SCAFFOLD variants.

Run::

    python benchmarks/bench_methods.py                     # default sweep
    python benchmarks/bench_methods.py --sizes 20 40 80    # grid side lengths
    python benchmarks/bench_methods.py --skip greedy heap   # large graphs only

Reports, per graph size and method: wall time, resulting component count, and
two quality measures taken on the *result* rather than on the backbone forest:

* **mean dilation** -- over the dropped edges, how long a detour each one gets.
* **max congestion** -- how concentrated the traffic is on the surviving edges.

Quality is measured with ``path_scores``, which evaluates the objective against
an arbitrary support graph. It costs a full shortest-path pass, so it is capped
by ``--quality-max-edges`` and skipped above that.

Numbers here are single-run and single-threaded. They are for spotting
order-of-magnitude differences between the variants, not for a paper table.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import scaffold
from scaffold.kernels import HAVE_NUMBA
from scaffold.scoring import ScoreParams, path_scores

METHODS = ("greedy", "heap", "fast", "sample")


def measure_quality(graph, mask):
    metrics = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask, params=ScoreParams()
    )
    dil = metrics["dil"]
    finite = dil[np.isfinite(dil)]
    econ = metrics["econ_path"]
    return (
        float(finite.mean()) if finite.size else 0.0,
        float(econ.max()) if econ.size else 0.0,
    )


def run_one(graph, method, keep_ratio, seed, repeats):
    """Time ``repeats`` runs after one warm-up, and return the best."""
    call = lambda: scaffold.sparsify(  # noqa: E731
        graph, method=method, keep_ratio=keep_ratio, seed=seed
    )
    result = call()          # warm-up: absorbs the one-off numba JIT cost
    if method == "sample":
        result.draw(keep_ratio=keep_ratio, seed=seed)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = call()
        if method == "sample":
            out = out.draw(keep_ratio=keep_ratio, seed=seed)
        best = min(best, time.perf_counter() - start)
        result = out
    return result, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--keep-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip", nargs="*", default=[], choices=METHODS)
    parser.add_argument("--quality-max-edges", type=int, default=5000)
    args = parser.parse_args()

    methods = [m for m in METHODS if m not in args.skip]
    print(f"numba: {HAVE_NUMBA}   keep_ratio: {args.keep_ratio}   "
          f"best of {args.repeats} after one warm-up\n")

    for side in args.sizes:
        graph = scaffold.grid_graph(side, side)
        floor = (graph.num_nodes - 1) / graph.num_edges
        note = "" if args.keep_ratio >= floor else "   (below the connectivity floor)"
        print(f"grid {side}x{side}: {graph.num_nodes:,} nodes, "
              f"{graph.num_edges:,} edges, delta_min={floor:.3f}{note}")

        quality_ok = graph.num_edges <= args.quality_max_edges
        header = f"  {'method':8s} {'edges':>8s} {'comp':>5s} {'ms':>10s} {'x fast':>8s}"
        if quality_ok:
            header += f" {'mean dil':>9s} {'max cong':>9s}"
        print(header)
        print(f"  {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 10} {'-' * 8}"
              + (f" {'-' * 9} {'-' * 9}" if quality_ok else ""))

        timings = {}
        for method in methods:
            try:
                result, seconds = run_one(
                    graph, method, args.keep_ratio, args.seed, args.repeats
                )
            except Exception as exc:  # a variant may simply be too slow to finish
                print(f"  {method:8s} failed: {type(exc).__name__}: {exc}")
                continue
            timings[method] = seconds
            row = (
                f"  {method:8s} {result.sparse_edges:8,d} "
                f"{result.num_components():5d} {seconds * 1000:10.2f}"
            )
            reference = timings.get("fast")
            row += f" {seconds / reference:8.1f}" if reference else f" {'-':>8s}"
            if quality_ok:
                mean_dil, max_cong = measure_quality(graph, result.mask)
                row += f" {mean_dil:9.3f} {max_cong:9.2f}"
            print(row)
        print()

    if not HAVE_NUMBA:
        print("Note: numba is not installed, so the kernels are running as pure")
        print("Python loops. Install it (pip install \"scaffold-sparse[speed]\")")
        print("for representative timings on anything but toy graphs.")


if __name__ == "__main__":
    main()
