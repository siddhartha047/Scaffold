"""Compare the packaged LLST backbone with the original research implementation.

Uses small canonical fixtures, including weighted and disconnected graphs,
all seven research initializers, and exact/sampled swap settings. Requires the
research checkout and its PyG dependencies only for this comparison script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-root", type=Path, default=ROOT.parent / "ICML_SPARSIFICATION"
    )
    args = parser.parse_args()
    research = args.research_root.resolve()
    if not (research / "sparsifiers/local_search_low_stretch_tree.py").is_file():
        raise SystemExit(f"Research LLST not found in {research}")
    sys.path.insert(0, str(research))
    sys.path.insert(0, str(ROOT / "src"))
    from sparsifiers.local_search_low_stretch_tree import (
        LocalSearchLowStretchTreeSparsifier,
    )

    import scaffold
    from scaffold.backbone import build_backbone

    unweighted = scaffold.grid_graph(3, 3)
    weighted = unweighted.with_weight(
        np.random.default_rng(3).uniform(0.2, 3.0, unweighted.num_edges)
    )
    disconnected = nx.disjoint_union(nx.cycle_graph(4), nx.complete_graph(4))
    disconnected.add_node(8)
    fixtures = {
        "grid": unweighted,
        "weighted": weighted,
        "disconnected": scaffold.normalize_graph(disconnected),
    }
    modes = {
        "exact": {},
        "candidate-cycle-sampling": {"candidate_sample_size": 3, "cycle_sample_size": 2},
        "sampled-objective": {
            "candidate_sample_size": 3,
            "cycle_sample_size": 2,
            "eval_sample_size": 5,
            "resample_eval_each_pass": True,
        },
        "tree-distance": {
            "candidate_strategy": "tree_distance",
            "candidate_sample_size": 3,
            "cycle_sample_size": 2,
        },
    }
    checked = 0
    for fixture, graph in fixtures.items():
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        for i, (u, v) in enumerate(graph.edge_index.T):
            attrs = (
                {} if graph.edge_weight is None else {"weight": float(graph.edge_weight[i])}
            )
            G.add_edge(int(u), int(v), **attrs)
        for init in ("glst", "maxst", "mst", "fast_maxst", "fast_mst", "randst", "randspt"):
            for mode, settings in modes.items():
                options = dict(seed=17, max_passes=3, init_support=init, **settings)
                actual = build_backbone(graph, "llst", **options)
                H = LocalSearchLowStretchTreeSparsifier(**options).sparsify(G)
                expected = np.array(
                    [H.has_edge(int(u), int(v)) for u, v in graph.edge_index.T]
                )
                np.testing.assert_array_equal(
                    actual, expected, err_msg=f"LLST mismatch: {fixture}, {init}, {mode}"
                )
                checked += 1
        print(f"{fixture}: all initializers and search modes match", flush=True)
    print(f"LLST research parity: {checked} identical forest masks.")


if __name__ == "__main__":
    main()
