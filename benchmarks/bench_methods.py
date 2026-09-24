"""Runtime and quality comparison of the five SCAFFOLD variants.

Run::

    python benchmarks/bench_methods.py                     # default sweep
    python benchmarks/bench_methods.py --sizes 20 40 80    # grid side lengths
    python benchmarks/bench_methods.py --skip greedy heap   # large graphs only
    python benchmarks/bench_methods.py --reference-max-edges 0  # allow slow references

Reports, per graph size and method: wall time, resulting component count, and
two quality measures taken on the *result* rather than on the backbone forest:

* **mean dilation** -- over the dropped edges, how long a detour each one gets.
* **max congestion** -- how concentrated the traffic is on the surviving edges.

Quality is measured with ``path_scores``, which evaluates the objective against
an arbitrary support graph. It costs a full shortest-path pass, so it is capped
by ``--quality-max-edges`` and skipped above that.

Numbers here are single-run. Pass ``--workers`` to compare thread counts; the
quality columns are expected to be byte-identical across them, so a change
there is a bug, not a tuning result. These are for spotting order-of-magnitude
differences between the variants, not for a paper table.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import scaffold
from scaffold.kernels import HAVE_NUMBA
from scaffold.scoring import ScoreParams, path_scores
from scaffold.utils.workers import resolve_workers

METHODS = ("greedy", "heap", "batch", "fast", "sample")


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


def run_one(graph, method, keep_ratio, seed, repeats, workers=None, **options):
    """Time ``repeats`` runs after one warm-up, and return the best."""
    call = lambda: scaffold.sparsify(  # noqa: E731
        graph, method=method, keep_ratio=keep_ratio, seed=seed, workers=workers,
        **options,
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
    parser.add_argument(
        "--reference-max-edges", type=int, default=1000,
        help="Skip Greedy/Heap above this input edge count (0 disables the limit).",
    )
    parser.add_argument("--batch-sample-size", type=int, default=None)
    parser.add_argument("--batch-add-per-round", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker threads per run (default: resolved from the environment, "
             "capped at 8). Pass 1 for serial timings.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(side < 2 for side in args.sizes):
        parser.error("--repeats must be positive and grid side lengths must be at least 2")
    if args.reference_max_edges < 0 or args.quality_max_edges < 0:
        parser.error("edge limits must be nonnegative")
    if args.batch_sample_size is not None or args.batch_add_per_round is not None:
        sample = 64 if args.batch_sample_size is None else args.batch_sample_size
        add = 8 if args.batch_add_per_round is None else args.batch_add_per_round
        if not 1 <= add < sample:
            parser.error("Batch requires 1 <= --batch-add-per-round < --batch-sample-size")

    methods = [m for m in METHODS if m not in args.skip]
    workers = resolve_workers(args.workers)
    print(f"numba: {HAVE_NUMBA}   keep_ratio: {args.keep_ratio}   "
          f"workers: {workers}   best of {args.repeats} after one warm-up")
    print("Quality columns must not move with --workers; only the timings may.\n")

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
        rows = []
        for method in methods:
            if (method in ("greedy", "heap") and args.reference_max_edges
                    and graph.num_edges > args.reference_max_edges):
                print(f"  {method:8s} skipped: above {args.reference_max_edges:,} reference edges; "
                      "use --reference-max-edges 0 to run")
                continue
            options = {}
            if method == "batch":
                options = {"sample_size": args.batch_sample_size,
                           "add_per_round": args.batch_add_per_round}
            print(f"  timing {method}...", flush=True)
            try:
                result, seconds = run_one(
                    graph, method, args.keep_ratio, args.seed, args.repeats,
                    workers=workers,
                    **options,
                )
            except Exception as exc:
                print(f"  {method:8s} failed: {type(exc).__name__}: {exc}")
                continue
            timings[method] = seconds
            quality = measure_quality(graph, result.mask) if quality_ok else None
            rows.append((method, result, seconds, quality))
        # The Fast reference must be measured before any ratios are printed.
        for method, result, seconds, quality in rows:
            row = (
                f"  {method:8s} {result.sparse_edges:8,d} "
                f"{result.num_components():5d} {seconds * 1000:10.2f}"
            )
            reference = timings.get("fast")
            row += f" {seconds / reference:8.1f}" if reference else f" {'-':>8s}"
            if quality_ok:
                mean_dil, max_cong = quality
                row += f" {mean_dil:9.3f} {max_cong:9.2f}"
            print(row)
            if method == "batch":
                print(f"    batch: sample={result.metadata['sample_size']}, "
                      f"add={result.metadata['add_per_round']}, "
                      f"rounds={result.metadata['rounds']}")
        print()

    if not HAVE_NUMBA:
        print("Note: numba is not installed, so the kernels are running as pure")
        print("Python loops. Install it (python -m pip install numba)")
        print("for representative timings on anything but toy graphs.")


if __name__ == "__main__":
    main()
