"""Reproducible complete-construction runtime audit for the paper.

Run each section alone, with OMP_PROC_BIND=false and eight available CPUs.
Variant timers start at the public API with a bidirectional NumPy edge list
and stop after a bidirectional sparse output is materialized. Sample includes
fresh preprocessing AND its first draw. No fitted sampler or backbone is reused.
JIT warm-up, graph generation, and correctness checks are outside those timers.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scaffold  # noqa: E402
from scaffold.backbone import build_backbone  # noqa: E402
from scaffold.graph import from_edge_index  # noqa: E402
from scaffold.kernels import component_count  # noqa: E402
from scaffold.scoring import PathScorer, ScoreParams  # noqa: E402
from scaffold.utils.workers import parallel_threads  # noqa: E402


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def original_graph(n, degree, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n, size=n * degree)
    d = rng.integers(0, n, size=n * degree)
    valid = s != d
    lo, hi = np.minimum(s[valid], d[valid]), np.maximum(s[valid], d[valid])
    _, first = np.unique(lo * n + hi, return_index=True)
    return from_edge_index(np.stack((lo[first], hi[first])), num_nodes=n)


def exact_graph(n=10000, m=250000):
    rng = np.random.default_rng(0)
    s = rng.integers(0, n, size=int(m * 1.2))
    d = rng.integers(0, n, size=int(m * 1.2))
    lo, hi = np.minimum(s, d), np.maximum(s, d)
    valid = lo != hi
    lo, hi = lo[valid], hi[valid]
    _, first = np.unique(lo * n + hi, return_index=True)
    first = np.sort(first)[:m]
    assert first.size == m
    return from_edge_index(np.stack((lo[first], hi[first])), num_nodes=n)


def backbone_graph(n, nominal_m, seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n, size=int(nominal_m * 1.3))
    d = rng.integers(0, n, size=int(nominal_m * 1.3))
    # Same topology as the historical backbone table, now explicitly unweighted.
    return from_edge_index(np.stack((s, d)), num_nodes=n)


def budget(graph, delta):
    components = component_count(graph.num_nodes, graph.src, graph.dst)
    forest = graph.num_nodes - components
    q = max(int(np.ceil(delta * graph.num_edges)), forest + 2)
    if q > graph.num_edges:
        raise ValueError("Input needs at least two non-forest edges")
    return int(q), int(forest), int(components)


def options(method, workers):
    opts = dict(seed=0, workers=workers, **ScoreParams().as_dict())
    if method == "sample":
        opts.update(backbone="fixed-maxsf", tree_count=8)
    else:
        opts["backbone"] = "fast-maxsf"
    if method == "heap":
        opts.update(score_form="product", top_k=16, dirty_limit=64)
    if method == "batch":
        opts.update(sample_size=512, add_per_round=64, clusters=10, cluster_method="bfs")
    return opts


class Audit:
    def __init__(self, args):
        self.args = args
        self.hashes = {}
        self.report = dict(
            status="running", started_utc=datetime.now(timezone.utc).isoformat(),
            section=args.section, package_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            python=sys.version, numpy=np.__version__, numba=numba.__version__,
            hostname=platform.node(), cpu_info=subprocess.check_output(["lscpu"], text=True),
            affinity=sorted(os.sched_getaffinity(0)), load_average_start=os.getloadavg(),
            environment={key: os.environ.get(key) for key in (
                "OMP_PROC_BIND", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            repeats=args.repeats, delta_requested=args.delta,
            statistic="median of every completed warm repetition; no outliers removed",
            variant_scope="raw bidirectional NumPy input -> normalization -> fresh forest -> scoring/selection (Sample precompute + first draw) -> bidirectional sparse output",
            exclusions=["graph generation/loading", "JIT warm-up", "validation", "GNN training", "GPU transfer"],
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / "src/scaffold").rglob("*.py"))},
            graphs={}, records=[],
        )
        self.save()

    def save(self):
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.args.output)

    def record(self, **data):
        self.report["records"].append(data)
        self.save()
        print(json.dumps({k: v for k, v in data.items() if k in
                          ("kind", "method", "workers", "repetition", "seconds", "q", "added_edges", "status")}), flush=True)

    def describe(self, label, graph):
        q, forest, components = budget(graph, self.args.delta)
        self.report["graphs"][label] = dict(n=graph.num_nodes, m=graph.num_edges,
            weighted=graph.is_weighted, edge_index_sha256=digest(graph.edge_index),
            components=components, forest_edges=forest, q=q, added_edges=q - forest,
            effective_ratio=q / graph.num_edges, requested_ratio=self.args.delta)
        self.save()
        return q

    def validate(self, result, graph, q, group):
        count = component_count(graph.num_nodes, graph.src, graph.dst)
        assert result.num_nodes == graph.num_nodes
        assert result.original_edges == graph.num_edges
        assert result.sparse_edges == q
        assert result.num_components() == count
        assert result.metadata["backbone_edges"] == graph.num_nodes - count
        assert q - result.metadata["backbone_edges"] >= 2
        sha = digest(result.mask)
        if group in self.hashes:
            assert self.hashes[group] == sha, f"selection changed: {group}"
        self.hashes[group] = sha
        if result.method == "fast":
            assert result.metadata["scored_candidates"] == graph.num_edges - (graph.num_nodes - count)
            assert result.metadata["rounds"] == 1
        return dict(q=q, added_edges=q - (graph.num_nodes - count), components=count,
                    mask_sha256=sha, metadata=result.metadata)

    def construction(self, graph, raw, method, workers, repetition, kind="variant"):
        q, _, _ = budget(graph, self.args.delta)
        kwargs = options(method, workers)
        gc.collect()
        start = time.perf_counter()
        result = scaffold.sparsify(raw, method=method, num_edges=q, **kwargs)
        after_precompute = time.perf_counter()
        sampler = None
        if method == "sample":
            sampler = result
            result = sampler.draw(num_edges=q, seed=0)
        after_draw = time.perf_counter()
        output = result.edge_index
        stop = time.perf_counter()
        assert output.shape == (2, 2 * q)
        info = self.validate(result, graph, q, method)
        phases = dict(api_seconds=after_precompute - start,
                      first_draw_seconds=after_draw - after_precompute if sampler is not None else 0.,
                      export_seconds=stop - after_draw)
        self.record(kind=kind, method=method, workers=workers, repetition=repetition,
                    seconds=stop-start, settings=kwargs, **phases, **info)
        if sampler is not None and kind == "variant":
            for draw in range(3):
                start = time.perf_counter()
                fresh = sampler.draw(num_edges=q, seed=draw + 1)
                output = fresh.edge_index
                elapsed = time.perf_counter() - start
                assert fresh.sparse_edges == q and fresh.num_components() == info["components"]
                self.record(kind="cached_draw", method=method, workers=workers, repetition=repetition,
                            draw=draw+1, seconds=elapsed, q=q)

    def variants(self):
        cases = {"fast": original_graph(200000, 12), "sample": original_graph(200000, 12),
                 "greedy": original_graph(400, 8), "heap": original_graph(4000, 10),
                 "batch": exact_graph()}
        for method, graph in cases.items():
            self.describe(method, graph)
        raw_large = np.concatenate((cases["fast"].edge_index, cases["fast"].edge_index[::-1]), axis=1)
        # Report process-first-call cost separately; disk JIT caches may exist.
        self.construction(cases["fast"], raw_large, "fast", 8, 0, kind="process_first_call")
        del raw_large
        for method in ("fast", "sample", "greedy", "heap", "batch"):
            graph = cases[method]
            raw = np.concatenate((graph.edge_index, graph.edge_index[::-1]), axis=1)
            warm = original_graph(300, 8)
            for workers in (1, 4, 8):
                kwargs = options(method, workers)
                if method == "batch":
                    kwargs["clusters"] = 1
                fitted = scaffold.sparsify(warm, method=method, keep_ratio=0.3, **kwargs)
                if method == "sample":
                    fitted.draw(keep_ratio=0.3, seed=0)
                del fitted
            for rep in range(1, self.args.repeats + 1):
                # Rotate the order so one worker count is not always measured last.
                for workers in ((8, 1, 4), (4, 8, 1), (1, 4, 8))[(rep - 1) % 3]:
                    print(f"START {method} P={workers} repetition={rep}", flush=True)
                    self.construction(graph, raw, method, workers, rep)
            del raw

    def profiles(self):
        import scaffold.algorithms.base as base
        import scaffold.algorithms.fast as fast
        import scaffold.api as api

        graph = original_graph(200000, 12)
        q = self.describe("fast", graph)
        raw = np.concatenate((graph.edge_index, graph.edge_index[::-1]), axis=1)
        scaffold.fast(original_graph(10000, 12), keep_ratio=0.2, **options("fast", 8))
        for rep in range(1, self.args.repeats + 1):
            phases = {}
            calls = {}

            def wrap(fn, label, phases=phases, calls=calls):
                def measured(*a, **kw):
                    t = time.perf_counter()
                    value = fn(*a, **kw)
                    phases[label] = phases.get(label, 0.) + time.perf_counter() - t
                    calls[label] = calls.get(label, 0) + 1
                    return value
                return measured

            with contextlib.ExitStack() as stack:
                for module, name, label in (
                    (api, "normalize_graph", "normalization"),
                    (base, "build_backbone", "forest"),
                    (base, "component_count", "component_check"),
                    (fast, "build_tree_index", "tree_index"),
                    (fast, "tree_scores", "candidate_scoring"),
                    (fast, "take_top", "selection"),
                    (api, "_build_result", "result_assembly"),
                ):
                    stack.enter_context(patch.object(module, name, wrap(getattr(module, name), label)))
                start = time.perf_counter()
                result = scaffold.fast(raw, num_edges=q, **options("fast", 8))
                t = time.perf_counter()
                output = result.edge_index
                stop = time.perf_counter()
                phases["bidirectional_export"] = stop - t
            assert output.shape[1] == 2 * q
            assert all(value == 1 for value in calls.values())
            phases["other"] = stop - start - sum(phases.values())
            self.record(kind="fast_profile", method="fast", workers=8, repetition=rep,
                        seconds=stop-start, phases=phases, calls=calls,
                        **self.validate(result, graph, q, "fast"))
        # Independent baseline: the historical normalized-input API scope.
        for rep in range(1, self.args.repeats + 1):
            start = time.perf_counter()
            result = scaffold.fast(graph, num_edges=q, **options("fast", 8))
            seconds = time.perf_counter()-start
            self.record(kind="fast_normalized_api", method="fast", workers=8, repetition=rep,
                        seconds=seconds, **self.validate(result, graph, q, "fast"))

    def backbones(self):
        for label, graph in (("large", backbone_graph(200000, 2400000, 0)),
                             ("small", backbone_graph(400, 1200, 1))):
            self.describe(label, graph)
            names = ["fast-maxsf", "fast-minsf", "maxsf", "minsf", "fast-randsf", "randsf", "spf"]
            if label == "small":
                names += ["glsf", "llsf"]
            for name in names:
                kwargs = dict(seed=0)
                if name == "llsf":
                    kwargs.update(max_input_edges=graph.num_edges, max_passes=10, init_support="glsf")
                # Only a tiny graph is used to warm the builder; no measured forest reused.
                build_backbone(scaffold.grid_graph(3, 3), name, **kwargs)
                repeats = 1 if name == "llsf" else self.args.repeats
                for rep in range(1, repeats+1):
                    print(f"START backbone {name} {label} repetition={rep}", flush=True)

                    def timeout(*_):
                        raise TimeoutError("backbone construction reached 1800 seconds")

                    previous = signal.signal(signal.SIGALRM, timeout)
                    signal.setitimer(signal.ITIMER_REAL, 1800)
                    started = time.perf_counter()
                    try:
                        mask = build_backbone(graph, name, **kwargs)
                        seconds = time.perf_counter() - started
                    except TimeoutError:
                        self.record(kind="backbone", method=name, graph=label, repetition=rep,
                                    status="timeout", lower_bound_seconds=time.perf_counter()-started, settings=kwargs)
                        break
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous)
                    nc = component_count(graph.num_nodes, graph.src, graph.dst)
                    assert int(mask.sum()) == graph.num_nodes - nc
                    assert component_count(graph.num_nodes, graph.src[mask], graph.dst[mask]) == nc
                    key = f"{label}/{name}"
                    sha = digest(mask)
                    if key in self.hashes:
                        assert self.hashes[key] == sha
                    self.hashes[key] = sha
                    self.record(kind="backbone", method=name, graph=label, repetition=rep,
                                status="complete", seconds=seconds, forest_edges=int(mask.sum()),
                                mask_sha256=sha, settings=kwargs)

    def scorers(self):
        graph = original_graph(5000, 10)
        self.describe("scoring", graph)
        mask = build_backbone(graph, "fast-maxsf", seed=0)
        candidates = np.flatnonzero(~mask)
        for weighted in (False, True):
            weights = np.random.default_rng(0).uniform(0.1, 1., graph.num_edges) if weighted else None
            scorer = PathScorer(graph.num_nodes, graph.src, graph.dst, weights, mask.copy(), workers=8)
            scorer.evaluate(candidates[:300], ScoreParams(), need_paths=False)
            expected = None
            for workers in (1, 4, 8):
                scorer.workers = workers
                for rep in range(1, self.args.repeats+1):
                    start = time.perf_counter()
                    output = scorer.evaluate(candidates, ScoreParams(), need_paths=False)
                    seconds = time.perf_counter()-start
                    sha = digest(output["score"])
                    if expected is not None:
                        assert expected == sha
                    expected = sha
                    self.record(kind="scoring_only", method="dijkstra" if weighted else "bfs",
                                workers=workers, repetition=rep, seconds=seconds,
                                candidates=int(candidates.size), score_sha256=sha)

    def run(self):
        with parallel_threads(8):
            getattr(self, self.args.section)()
        for name, sha in self.report["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
        self.report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                           load_average_end=os.getloadavg(), source_unchanged=True)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("variants", "profiles", "backbones", "scorers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.2)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.delta <= 1:
        parser.error("repeats must be positive and 0 < delta <= 1")
    Audit(args).run()


if __name__ == "__main__":
    main()
