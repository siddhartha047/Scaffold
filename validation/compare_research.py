"""Compare ``scaffold.fast`` with the ICML research implementation on Cora.

This runs the same real PyG data through both code paths at a ratio above the
connectivity floor. The research implementation is configured as one tensor
cluster with ``fast_score='tree_exact'``; that is the direct counterpart of
the package's LCA score-once / global-top-k default.

Run from the package repository::

    python validation/compare_research.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_ROOT = PACKAGE_ROOT.parent / "ICML_SPARSIFICATION"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help="Path to the ICML_SPARSIFICATION research checkout.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Research data directory (default: RESEARCH_ROOT/data).",
    )
    parser.add_argument(
        "--dataset",
        default="cora",
        choices=("cora", "citeseer", "pubmed"),
        help="Planetoid dataset to compare.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.6,
        help="Undirected edge ratio; must exceed this graph's connectivity floor.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def _load_main_data(data_dir, dataset_name):
    """Load and materialize data exactly as ``main.py`` does before sparsifying."""
    from torch_geometric.data import Data
    from utils.dataset import load_dataset

    dataset = load_dataset(str(data_dir), dataset_name)
    graph = dataset.graph
    data = Data(
        x=graph["node_feat"].detach().cpu(),
        edge_index=graph["edge_index"].detach().cpu(),
        y=dataset.label.detach().cpu(),
        num_nodes=int(graph["num_nodes"]),
    )
    if graph.get("edge_weight") is not None:
        data.edge_weight = graph["edge_weight"].detach().cpu()
    for marker in ("edge_index_is_undirected_unique", "edge_index_is_symmetric_unique"):
        if bool(graph.get(marker, False)):
            setattr(data, marker, True)
    return data


def _edge_set(edge_index):
    return {tuple(edge) for edge in np.asarray(edge_index).T.tolist()}


def main():
    args = parse_args()
    research_root = args.research_root.expanduser().resolve()
    if not (research_root / "main.py").is_file():
        raise SystemExit(f"research checkout not found: {research_root}")
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else research_root / "data"
    )

    # Use the checked-out package and research tree, even if another release is
    # installed in the current interpreter.
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))
    sys.path.insert(0, str(research_root))

    from sparsifiers.scaffold.scaffold_fast import ScaffoldFastSparsifier

    import scaffold
    from scaffold.kernels import component_count

    data = _load_main_data(data_dir, args.dataset)
    graph = scaffold.normalize_graph(data)
    base_components = component_count(
        graph.num_nodes, graph.src, graph.dst
    )
    spanning_edges = graph.num_nodes - base_components
    floor = spanning_edges / graph.num_edges if graph.num_edges else 0.0
    target_edges = int(math.ceil(args.keep_ratio * graph.num_edges - 1e-12))
    if target_edges <= spanning_edges:
        raise SystemExit(
            f"keep_ratio={args.keep_ratio} gives {target_edges} edges, but this "
            f"graph needs more than its {spanning_edges}-edge forest for the "
            f"LCA ranking to affect selection; choose keep_ratio > {floor:.6f}."
        )

    package_start = time.perf_counter()
    package_result = scaffold.fast(
        data,
        keep_ratio=args.keep_ratio,
        backbone="fast-maxst",
        selection="topk",
        seed=args.seed,
    )
    package_seconds = time.perf_counter() - package_start

    # These are the research settings corresponding exactly to the package's
    # default fast algorithm. In particular, one cluster avoids METIS changing
    # the problem into separate local jobs, and tree_exact selects one global
    # top-k instead of the research config's sampled tree_exact_loop.
    research_sparsifier = ScaffoldFastSparsifier(
        target_ratio=args.keep_ratio,
        init_support="fast_maxst",
        fast_tree_buckets=256,
        alpha=1.0,
        edge_beta=1.0,
        node_beta=1.0,
        edge_norm_p=2.0,
        node_norm_q=2.0,
        cluster_count=1,
        cluster_method="metis",
        parallel_workers=1,
        backend="tensor",
        fast_score="tree_exact",
        support_weight_method="uniform",
        support_budget_mode="early_stop",
        swap_refine=False,
        progress="false",
        seed=args.seed,
    )
    research_start = time.perf_counter()
    research_result = research_sparsifier.sparsify(data)
    research_seconds = time.perf_counter() - research_start
    research_graph = scaffold.normalize_graph(research_result)

    package_edges = _edge_set(package_result.undirected_edge_index)
    research_edges = _edge_set(research_graph.edge_index)
    only_package = package_edges - research_edges
    only_research = research_edges - package_edges
    exact_match = not only_package and not only_research

    print("\nResearch parity result")
    print(f"  dataset:              {args.dataset}")
    print(f"  nodes / input edges:   {graph.num_nodes:,} / {graph.num_edges:,}")
    print(f"  input components:      {base_components:,}")
    print(f"  connectivity floor:    {floor:.6f}")
    print(f"  requested / kept:      {args.keep_ratio:.3f} / {target_edges:,}")
    print(f"  package seconds:       {package_seconds:.6f}")
    print(f"  research seconds:      {research_seconds:.6f}")
    print(f"  only in package:       {len(only_package):,}")
    print(f"  only in research:      {len(only_research):,}")
    print(f"  exact edge-set match:  {exact_match}")

    if not exact_match:
        print(f"  package-only sample:   {sorted(only_package)[:10]}")
        print(f"  research-only sample:  {sorted(only_research)[:10]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
