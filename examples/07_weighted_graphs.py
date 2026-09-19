"""Weighted forests and growth on a small offline graph; NumPy/SciPy only.

Run: python examples/07_weighted_graphs.py
Add --sklearn to also demonstrate the optional metrics extra.
"""

import argparse

import numpy as np

import scaffold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn", action="store_true")
    args = parser.parse_args()
    graph = scaffold.grid_graph(5, 5)
    features = np.random.default_rng(7).normal(size=(graph.num_nodes, 4))
    weighted = scaffold.with_feature_weights(graph, features, metric="cosine")

    for method in scaffold.METHODS:
        backbone = "fixed-randsf" if method == "sample" else "randsf"
        result = scaffold.sparsify(
            weighted, method=method, keep_ratio=0.8, backbone=backbone,
            weighted_paths=True, seed=0, workers=2,
        )
        if method == "sample":
            result = result.draw(keep_ratio=0.8, seed=1)
        print(result.summary())
        np.testing.assert_array_equal(result.undirected_edge_weight, weighted.edge_weight[result.mask])

    # Raw Euclidean distances make small weights mean short/cheap connections.
    distances = scaffold.with_feature_weights(graph, features, "euclidean", kind="distance")
    result = scaffold.fast(distances, keep_ratio=0.8, backbone="minsf", weighted_paths=True)
    print("Euclidean distances + MinSF:", result.summary())

    if args.sklearn:
        weighted = scaffold.with_feature_weights(
            graph, features, "minkowski", kind="distance", metric_kwargs={"p": 3},
        )
        print("Minkowski weights:", weighted.edge_weight[:5])


if __name__ == "__main__":
    main()
