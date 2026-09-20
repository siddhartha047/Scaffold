"""Split forest construction from Scaffold work in fresh eight-worker calls.

For Sample only, synchronize its random-forest tasks before they enter scoring.
This prevents forest/scoring overlap from being counted twice. The result is a
stage-separated diagnostic, not a replacement for the normal API runtime audit.
No algorithm source files are modified and every output must match that audit.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from bench_runtime_audit import Audit, exact_graph, options, original_graph

import scaffold
import scaffold.algorithms.base as base
import scaffold.algorithms.sample as sample
import scaffold.api as api


class SplitAudit(Audit):
    def split(self):
        reference = json.loads(self.args.reference.read_text(encoding="utf-8"))
        assert reference["status"] == "complete"
        assert reference["source_sha256"] == self.report["source_sha256"]
        self.report["reference"] = str(self.args.reference)
        self.report["split_protocol"] = (
            "8 workers; forest + other Scaffold work + input normalization/export = total; "
            "Sample uses a barrier before scoring its 8 random forests, with all 9 "
            "forest builds included. Stage components are disjoint wall-clock intervals."
        )
        cases = {"fast": original_graph(200000, 12), "sample": original_graph(200000, 12),
                 "greedy": original_graph(400, 8), "heap": original_graph(4000, 10),
                 "batch": exact_graph()}
        for method, graph in cases.items():
            q = self.describe(method, graph)
            assert self.report["graphs"][method] == reference["graphs"][method]
            expected = {r["mask_sha256"] for r in reference["records"]
                        if r["kind"] == "variant" and r["method"] == method}
            assert len(expected) == 1
            self.hashes[method] = expected.pop()
            raw = np.concatenate((graph.edge_index, graph.edge_index[::-1]), axis=1)
            warm_options = options(method, 8)
            if method == "batch":
                warm_options["clusters"] = 1
            warm = scaffold.sparsify(original_graph(300, 8), method=method,
                                     keep_ratio=0.3, **warm_options)
            if method == "sample":
                warm.draw(keep_ratio=0.3, seed=0)
            del warm
            for rep in range(1, self.args.repeats + 1):
                print(f"START split {method} repetition={rep}", flush=True)
                self.split_call(graph, raw, q, method, rep)

    def split_call(self, graph, raw, q, method, repetition):
        phases, calls = {}, {}
        kwargs = options(method, 8)

        def timed(fn, label):
            def measured(*a, **kw):
                start = time.perf_counter()
                value = fn(*a, **kw)
                phases[label] = phases.get(label, 0.) + time.perf_counter() - start
                calls[label] = calls.get(label, 0) + 1
                return value
            return measured

        gc.collect()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(api, "normalize_graph",
                                              timed(api.normalize_graph, "normalization")))
            if method != "sample":
                stack.enter_context(patch.object(base, "build_backbone",
                                                  timed(base.build_backbone, "forest_fixed")))
            else:
                # The fixed forest is constructed on the main thread. Capture
                # ordering preparation as well as its union-find kernel.
                main_thread = threading.get_ident()
                component_count = sample.component_count
                forest_mask = sample.spanning_forest_mask
                tree_scores = sample.tree_scores
                stamps = {}
                mask_threads = []

                def components(*a, **kw):
                    result = component_count(*a, **kw)
                    if "fixed_start" not in stamps:
                        stamps["fixed_start"] = time.perf_counter()
                    return result

                def build(*a, **kw):
                    mask = forest_mask(*a, **kw)
                    thread = threading.get_ident()
                    mask_threads.append(thread)
                    if thread == main_thread:
                        phases["forest_fixed"] = time.perf_counter() - stamps["fixed_start"]
                    return mask

                def all_forests_ready():
                    phases["forest_random_pool"] = time.perf_counter() - stamps["pool_start"]

                gate = threading.Barrier(8, action=all_forests_ready, timeout=300)

                def scores(*a, **kw):
                    if threading.get_ident() != main_thread:
                        gate.wait()
                    return tree_scores(*a, **kw)

                class ForestPool(ThreadPoolExecutor):
                    def map(self, *a, **kw):
                        stamps["pool_start"] = time.perf_counter()
                        return super().map(*a, **kw)

                for name, replacement in (("component_count", components),
                                           ("spanning_forest_mask", build),
                                           ("tree_scores", scores),
                                           ("ThreadPoolExecutor", ForestPool)):
                    stack.enter_context(patch.object(sample, name, replacement))

            start = time.perf_counter()
            result = scaffold.sparsify(raw, method=method, num_edges=q, **kwargs)
            if method == "sample":
                draw_start = time.perf_counter()
                result = result.draw(num_edges=q, seed=0)
                phases["first_draw"] = time.perf_counter() - draw_start
            export_start = time.perf_counter()
            output = result.edge_index
            stop = time.perf_counter()

        phases["export"] = stop - export_start
        phases["forest"] = phases["forest_fixed"] + phases.get("forest_random_pool", 0.)
        phases["io"] = phases["normalization"] + phases["export"]
        phases["scaffold"] = stop - start - phases["forest"] - phases["io"]
        assert all(value >= 0 for value in phases.values())
        assert calls["normalization"] == 1
        if method == "sample":
            assert mask_threads.count(main_thread) == 1 and len(mask_threads) == 9
            assert "forest_random_pool" in phases
            forest_calls = len(mask_threads)
        else:
            forest_calls = calls["forest_fixed"]
            assert forest_calls == 1
        assert output.shape == (2, 2*q)
        info = self.validate(result, graph, q, method)
        self.record(kind="stage_split", method=method, workers=8, repetition=repetition,
                    seconds=stop-start, phases=phases, settings=kwargs,
                    forest_calls=forest_calls, sample_barrier=method == "sample", **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path("docs/benchmarks/runtime_audit_20260920/variants.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.section, args.delta = "split", 0.2
    SplitAudit(args).run()


if __name__ == "__main__":
    main()
