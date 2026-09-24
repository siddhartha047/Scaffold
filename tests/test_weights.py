"""Feature weights match the research definitions without changing topology."""

import builtins

import numpy as np
import pytest
from scipy import sparse

import scaffold


@pytest.fixture
def example():
    graph = scaffold.normalize_graph(np.array([[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]]), num_nodes=5)
    features = np.array([[1., 0.], [0., 2.], [-1., 0.], [0., 0.], [3., 4.]])
    return graph, features


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot", "uniform"])
def test_research_formulas(example, metric):
    graph, features = example
    expected = {
        "cosine": [0.5, 1e-12, 0.5, 0.5, 0.5],
        "euclidean": 1 / (1 + np.array([np.sqrt(5), 2, 1, np.sqrt(5), 2])),
        "dot": [1, 1e-12, 1, 1, 1],
        "uniform": np.ones(5),
    }
    for batch_size in (1, 2, 8192):
        np.testing.assert_allclose(
            scaffold.feature_edge_weights(graph, features, metric, batch_size=batch_size), expected[metric],
        )


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "sqeuclidean", "manhattan", "chebyshev"])
@pytest.mark.parametrize("kind", ["similarity", "distance"])
def test_dense_sparse_and_sklearn_agree(example, metric, kind):
    pairwise = pytest.importorskip("sklearn.metrics.pairwise")
    graph, features = example
    distance = pairwise.pairwise_distances(features, metric=metric)[graph.src, graph.dst]
    expected = distance if kind == "distance" else (
        1 - distance / 2 if metric == "cosine" else 1 / (1 + distance)
    )
    expected = np.maximum(expected, 1e-12)
    for features_in in (features, sparse.csr_matrix(features)):
        actual = scaffold.feature_edge_weights(graph, features_in, metric, kind=kind, batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("metric,kwargs", [("minkowski", {"p": 3}), ("canberra", {}), ("mahalanobis", {"VI": np.eye(2)})])
def test_extra_sklearn_metrics(example, metric, kwargs):
    metrics = pytest.importorskip("sklearn.metrics")
    graph, features = example
    matrix = metrics.DistanceMetric.get_metric(metric, **kwargs).pairwise(features)
    expected = matrix[graph.src, graph.dst]
    for values in (features, sparse.csr_matrix(features)):
        distances = scaffold.feature_edge_weights(graph, values, metric, kind="distance", metric_kwargs=kwargs)
        similarities = scaffold.feature_edge_weights(graph, values, metric, metric_kwargs=kwargs)
        np.testing.assert_allclose(distances, np.maximum(expected, 1e-12))
        np.testing.assert_allclose(similarities, 1 / (1 + expected))


def test_callable_distance_and_missing_optional_dependency(example, monkeypatch):
    graph, features = example
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    expected = scaffold.feature_edge_weights(graph, features, "euclidean")
    actual = scaffold.feature_edge_weights(graph, features, lambda a, b: np.linalg.norm(a - b))
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ImportError, match="python -m pip install scikit-learn"):
        scaffold.feature_edge_weights(graph, features, "minkowski")


def test_canonical_edges_and_isolated_nodes():
    edges = np.array([[2, 1, 0, 1, 2], [0, 0, 2, 0, 2]])
    features = np.arange(5.)[:, None]
    graph = scaffold.with_feature_weights(edges, features, "euclidean", kind="distance")
    assert graph.num_nodes == 5
    np.testing.assert_array_equal(graph.edge_index, [[0, 0], [1, 2]])
    np.testing.assert_array_equal(graph.edge_weight, [1, 2])


def test_networkx_labels_and_no_mutation():
    nx = pytest.importorskip("networkx")
    graph = nx.Graph()
    graph.add_nodes_from(["z", "b", "a", "isolated"])
    graph.add_edge("a", "z", weight=17)
    weighted = scaffold.with_feature_weights(graph, np.array([0, 10, 3, 0]), "euclidean", kind="distance")
    assert weighted.node_labels == ["z", "b", "a", "isolated"]
    np.testing.assert_array_equal(weighted.edge_weight, [3])
    assert graph["a"]["z"]["weight"] == 17


def test_pyg_features_and_weighted_graph_round_trip(example):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    from scaffold.pyg import ScaffoldResampler, sparsify_data

    graph, features = example
    data = Data(x=torch.tensor(features), edge_index=torch.tensor(graph.edge_index), y=torch.arange(5), num_nodes=5)
    weighted = scaffold.with_feature_weights(data, metric="euclidean")
    views = [sparsify_data(weighted, num_edges=4, weighted_paths=True)]
    views.append(ScaffoldResampler(weighted, num_edges=4, tree_count=2, weighted_paths=True, seed=1).epoch(0))
    for view in views:
        assert torch.equal(view.x, data.x)
        assert torch.equal(view.y, data.y)
        assert view.num_nodes == 5
        expected = 1 / (1 + torch.linalg.norm(data.x[view.edge_index[0]] - data.x[view.edge_index[1]], dim=1))
        torch.testing.assert_close(view.edge_weight.double(), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"kind": "invalid"}, {"batch_size": 0}, {"batch_size": 1.2}, {"batch_size": True},
    {"min_weight": 0}, {"min_weight": np.nan}, {"min_weight": 2}, {"metric": None},
    {"metric": "dot", "kind": "distance"}, {"metric_kwargs": {"p": 3}},
])
def test_invalid_options(example, kwargs):
    graph, features = example
    with pytest.raises(ValueError):
        scaffold.feature_edge_weights(graph, features, **kwargs)


@pytest.mark.parametrize("features", [None, np.ones((2, 2)), np.zeros((5, 0)), np.full((5, 1), np.nan), np.ones((5, 2), dtype=complex)])
def test_invalid_features(example, features):
    with pytest.raises(ValueError):
        scaffold.feature_edge_weights(example[0], features)


def test_uniform_and_constant_dot(example):
    graph, _ = example
    np.testing.assert_array_equal(scaffold.with_feature_weights(graph, metric="Uniform").edge_weight, np.ones(5))
    np.testing.assert_array_equal(scaffold.feature_edge_weights(graph, np.ones((5, 2)), "dot"), np.ones(5))
    with pytest.raises(ValueError, match="nonnegative"):
        scaffold.feature_edge_weights(graph, np.ones((5, 2)), lambda a, b: -1)


def test_empty_graph():
    graph = scaffold.normalize_graph(np.empty((2, 0), dtype=int), num_nodes=3)
    weighted = scaffold.with_feature_weights(graph, np.ones((3, 2)))
    assert weighted.num_nodes == 3 and weighted.num_edges == 0
    assert weighted.edge_weight.shape == (0,)
