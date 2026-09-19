"""Weighted backbones, path modes, exports and reusable Sample artifacts."""

import numpy as np
import pytest
from scipy import sparse

import scaffold
from scaffold.algorithms.sample import ScaffoldSampler
from scaffold.backbone import build_backbone
from scaffold.scoring import PathScorer, ScoreParams, path_scores, tree_scores


@pytest.mark.parametrize("backbone", ["randsf", "maxsf", "minsf", "spf"])
@pytest.mark.parametrize("weighted_paths", [True, False])
def test_weighted_tree_kernel_matches_dynamic_reference(backbone, weighted_paths):
    graph = scaffold.grid_graph(5, 5, weight="random", seed=4)
    mask = build_backbone(graph, backbone, seed=2)
    candidates = np.flatnonzero(~mask)
    params = ScoreParams()
    tree = tree_scores(graph.num_nodes, graph.src, graph.dst, mask, graph.edge_weight,
                       params=params, weighted_paths=weighted_paths)
    scorer = PathScorer(graph.num_nodes, graph.src, graph.dst, graph.edge_weight,
                        mask, workers=2, weighted_paths=weighted_paths)
    reference = scorer._evaluate_python(candidates, params)
    compiled = scorer.evaluate(candidates, params)
    for key in ("dil", "econ_path", "vcon_path", "score"):
        np.testing.assert_allclose(compiled[key], reference[key], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(tree[key][candidates], reference[key], rtol=1e-8, atol=1e-8)


def test_path_modes_choose_different_support_routes():
    graph = scaffold.normalize_graph(np.array([[0, 0, 0, 1, 2], [1, 2, 3, 2, 3]])).with_weight([1., 20., 2., 1., 1.])
    mask = np.array([True, True, False, True, True])
    weighted = path_scores(4, graph.src, graph.dst, mask, graph.edge_weight, weighted_paths=True)
    hops = path_scores(4, graph.src, graph.dst, mask, graph.edge_weight, weighted_paths=False)
    np.testing.assert_allclose(weighted["dil"], [3 / 2])
    np.testing.assert_allclose(hops["dil"], [2 / 2])
    assert len(weighted["path_edges"][0]) == 3
    assert len(hops["path_edges"][0]) == 2


@pytest.mark.parametrize("method", ["greedy", "heap", "batch", "fast", "sample"])
@pytest.mark.parametrize("weighted_paths", [True, False])
def test_weighted_budget_connectivity_and_exports(method, weighted_paths):
    graph = scaffold.grid_graph(4, 4, weight="random", seed=8)
    backbone = "fixed-randsf" if method == "sample" else "randsf"
    result = scaffold.sparsify(graph, method=method, num_edges=19, backbone=backbone, seed=3, weighted_paths=weighted_paths)
    if method == "sample":
        result = result.draw(num_edges=19, seed=0)
    assert result.sparse_edges == 19
    assert result.num_components() == 1
    assert result.metadata["weighted_paths"] == weighted_paths
    np.testing.assert_array_equal(result.undirected_edge_weight, graph.edge_weight[result.mask])
    adjacency = result.to_scipy()
    np.testing.assert_allclose(np.asarray(adjacency[graph.src[result.mask], graph.dst[result.mask]]).ravel(), graph.edge_weight[result.mask])


@pytest.mark.parametrize("method", ["greedy", "heap", "batch"])
def test_dynamic_default_still_uses_weighted_paths(method):
    graph = scaffold.grid_graph(4, 4, weight="random", seed=9)
    options = dict(method=method, num_edges=19, backbone="randsf", seed=5)
    automatic = scaffold.sparsify(graph, **options)
    explicit = scaffold.sparsify(graph, weighted_paths=True, **options)
    np.testing.assert_array_equal(automatic.mask, explicit.mask)


@pytest.mark.parametrize("method", ["greedy", "heap", "batch", "fast", "sample"])
@pytest.mark.parametrize("bad_weight", [-1., np.nan, np.inf])
def test_invalid_scoring_weights_rejected(method, bad_weight):
    graph = scaffold.grid_graph(3, 3).with_weight(np.full(12, bad_weight))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        scaffold.sparsify(graph, method=method, num_edges=10, weighted_paths=True)


def test_weighted_spf_and_minmax_forests():
    graph = scaffold.normalize_graph(np.array([[0, 0, 1, 3], [1, 2, 2, 4]]), num_nodes=6).with_weight([1., 100., 2., 3.])
    np.testing.assert_array_equal(build_backbone(graph, "spf"), [True, False, True, True])
    np.testing.assert_array_equal(build_backbone(graph, "minsf"), [True, False, True, True])
    np.testing.assert_array_equal(build_backbone(graph, "maxsf"), [False, True, True, True])
    assert build_backbone(graph, "spf", max_edges=1).sum() == 1
    assert not build_backbone(graph, "spf", max_edges=0).any()


@pytest.mark.parametrize("convert", [np.asarray, sparse.csr_matrix])
def test_near_unit_adjacency_weights_not_dropped(convert):
    matrix = np.array([[0, 1 + 1e-7, 1], [1 + 1e-7, 0, 1], [1, 1, 0.]])
    graph = scaffold.normalize_graph(convert(matrix))
    assert graph.is_weighted
    np.testing.assert_array_equal(graph.edge_weight, [1 + 1e-7, 1, 1])


def test_zero_weight_edges():
    graph = scaffold.grid_graph(3, 3).with_weight(np.zeros(12))
    for method in scaffold.METHODS:
        result = scaffold.sparsify(graph, method=method, num_edges=10, weighted_paths=True, seed=2)
        if method == "sample":
            result = result.draw(num_edges=10, seed=2)
        assert result.sparse_edges == 10 and result.num_components() == 1


def test_weighted_sample_artifact_roundtrip_and_mismatch(tmp_path):
    graph = scaffold.grid_graph(4, 4, weight="random", seed=8)
    scores = scaffold.sample(graph, tree_count=2, weighted_paths=True, seed=2)
    path = scores.sampler.save(tmp_path / "weighted.npz")
    loaded = ScaffoldSampler.load(path, graph)
    assert loaded.weighted_paths is True
    np.testing.assert_array_equal(loaded.draw(num_edges=19, seed=5).mask, scores.draw(num_edges=19, seed=5).mask)
    with pytest.raises(ValueError, match="weights"):
        ScaffoldSampler.load(path, graph.with_weight(graph.edge_weight + 1))
    with pytest.raises(ValueError, match="weights"):
        ScaffoldSampler.load(path, graph.with_weight(None))
    with pytest.raises(ValueError, match="weighted_paths"):
        ScaffoldSampler.load(path, graph, weighted_paths=False)


def test_legacy_sample_artifact(tmp_path):
    graph = scaffold.grid_graph(3, 3)
    scores = scaffold.sample(graph, tree_count=2, seed=2)
    path = scores.sampler.save(tmp_path / "current.npz")
    with np.load(path) as values:
        legacy = {key: values[key] for key in values.files if key not in ("has_edge_weight", "edge_weight", "weighted_paths")}
    legacy["version"] = np.int64(1)
    path = tmp_path / "legacy.npz"
    np.savez(path, **legacy)
    loaded = ScaffoldSampler.load(path, graph)
    np.testing.assert_array_equal(loaded.draw(num_edges=10, seed=5).mask, scores.draw(num_edges=10, seed=5).mask)
    with pytest.raises(ValueError, match="legacy artifact"):
        ScaffoldSampler.load(path, graph.with_weight(np.ones(12)))
