"""Offline comparison of weighted kernels with the research checkout.

Run: python validation/compare_weighted_research.py --output /tmp/weighted.json
Requires the research environment (torch, PyG, NetworkX), but downloads no data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=PACKAGE_ROOT.parent / "ICML_SPARSIFICATION")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.research_root.resolve()
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))
    sys.path.insert(0, str(root))

    import networkx as nx
    from sparsifiers.joint_dilation_congestion import JointDilationCongestionSparsifier
    from sparsifiers.scaffold.scaffold_greedy import ScaffoldGreedySparsifier
    from sparsifiers.scaffold.spanning_tree import (
        build_fast_weighted_spanning_forest_mask,
        build_weighted_spanning_forest_mask,
    )
    from sparsifiers.scaffold.tree_score import scaffold_tree_scores

    import scaffold
    from scaffold.backbone import build_backbone
    from scaffold.scoring import PathScorer, ScoreParams, tree_scores

    rng = np.random.default_rng(137)
    graph = scaffold.grid_graph(5, 5).with_weight(rng.uniform(0.1, 3., 40))
    features = rng.normal(size=(graph.num_nodes, 7)).astype(np.float32)
    checks = []

    for metric in ("uniform", "cosine", "euclidean", "dot"):
        research = JointDilationCongestionSparsifier(support_weight_method=metric)
        expected = research._compute_feature_edge_scores(features, graph.src, graph.dst, batch_size=9).numpy()
        actual = scaffold.feature_edge_weights(graph, features, metric, batch_size=3)
        # Research returns float32 and literal zeros; the package uses float64
        # and floors at 1e-12 for positive-length low-stretch backbones.
        np.testing.assert_allclose(actual, np.maximum(expected, 1e-12), atol=2e-7, rtol=2e-6)
        checks.append(f"feature weights: {metric}")

    for maximum, name in ((True, "maxsf"), (False, "minsf")):
        for fast in (False, True):
            builder = build_fast_weighted_spanning_forest_mask if fast else build_weighted_spanning_forest_mask
            backbone = f"fast-{name}" if fast else name
            for limit in (None, 12):
                expected = builder(graph.num_nodes, graph.src, graph.dst, graph.edge_weight, maximum=maximum, max_edges=limit)
                actual = build_backbone(graph, backbone, max_edges=limit)
                np.testing.assert_array_equal(actual, expected)
                checks.append(f"forest: {backbone}, limit={limit}")

    params = ScoreParams()
    for backbone in ("randsf", "maxsf", "minsf", "spf"):
        mask = build_backbone(graph, backbone, seed=3)
        for weighted_paths in (False, True):
            expected = scaffold_tree_scores(graph.num_nodes, graph.src, graph.dst, mask,
                                            graph.edge_weight, weighted_paths=weighted_paths, workers=2)
            actual = tree_scores(graph.num_nodes, graph.src, graph.dst, mask,
                                 graph.edge_weight, weighted_paths=weighted_paths, workers=2)
            for key in ("dil", "econ_path", "vcon_path", "score", "edge_congestion", "node_congestion"):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=1e-9)
            checks.append(f"tree scores: {backbone}, weighted_paths={weighted_paths}")

    # Compare dynamic scoring after several additions: the support acquires
    # cycles, so this also tests weighted Dijkstra rather than only tree paths.
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    for u, v, weight in zip(graph.src, graph.dst, graph.edge_weight):
        G.add_edge(int(u), int(v), weight=float(weight))
    mask = build_backbone(graph, "randsf", seed=3)
    H = G.edge_subgraph([tuple(edge) for edge in graph.edge_index[:, mask].T]).copy()
    H.add_nodes_from(G)
    scorer = PathScorer(graph.num_nodes, graph.src, graph.dst, graph.edge_weight, mask.copy(), workers=2, weighted_paths=True)
    research = ScaffoldGreedySparsifier()
    for iteration in range(6):
        candidates = np.flatnonzero(~mask)
        # Both the all-candidate Greedy population and a sampled Batch population.
        for ids, label in ((candidates, "all"), (candidates[::2], "batch")):
            edges = [tuple(map(int, edge)) for edge in graph.edge_index[:, ids].T]
            expected = research._evaluate_all_candidates(G, H, edges, "weight", True)
            actual = scorer.evaluate(ids, params)
            for key, reference_key in (("dil", "dil"), ("econ_path", "eConPath"), ("vcon_path", "vConPath"), ("score", "score")):
                np.testing.assert_allclose(actual[key], [expected[edge][reference_key] for edge in edges], atol=1e-8, rtol=1e-8)
            checks.append(f"dynamic scores: round={iteration}, population={label}")
        all_scores = scorer.evaluate(candidates, params)["score"]
        chosen = candidates[np.argmax(all_scores)]
        mask[chosen] = True
        scorer.add_edge(chosen)
        u, v = map(int, graph.edge_index[:, chosen])
        H.add_edge(u, v, **G[u][v])

    source_files = [
        "sparsifiers/joint_dilation_congestion.py", "sparsifiers/scaffold/scaffold_greedy.py",
        "sparsifiers/scaffold/spanning_tree.py", "sparsifiers/scaffold/tree_score.py",
    ]
    report = {
        "seed": 137, "nodes": graph.num_nodes, "edges": graph.num_edges,
        "weight_range": [0.1, 3.0], "workers": 2, "checks_passed": len(checks),
        "checks": checks,
        "research_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "scope": "Feature formulas, weighted forest masks, tree scores, and dynamic candidate scores; not a runtime benchmark or full Heap/Batch/Sample trajectory equivalence.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Weighted research comparison: {len(checks)} checks passed ({graph.num_nodes} nodes, {graph.num_edges} edges).")


if __name__ == "__main__":
    main()
