"""Forest names preserve seeded edges, metadata, draw plans and artifacts."""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from conftest import GREEDY_METHODS
from scaffold.algorithms.sample import ScaffoldSampler
from scaffold.backbone import (
    build_backbone,
    canonical_backbone_name,
    canonical_sample_backbone,
    register_backbone,
)
from scaffold.graph import from_edge_index

SUPPORT_PAIRS = [
    ("maxsf", "maxst"), ("msf", "mst"), ("minsf", "mst"),
    ("minst", "mst"), ("fast-maxsf", "fast-maxst"),
    ("fast-msf", "fast-mst"), ("fast-minsf", "fast-mst"),
    ("fast-minst", "fast-mst"), ("randsf", "randst"),
    ("fast-randsf", "fast-randst"), ("slsf", "slst"),
    ("glsf", "glst"), ("llsf", "llst"), ("randspf", "randspt"),
    ("sf", "maxst"), ("st", "maxst"), ("spf", "spt"),
]
SAMPLE_PAIRS = [
    ("fixed-maxsf", "fixed-maxst"), ("fixed-msf", "fixed-maxst"),
    ("fixed-slsf", "fixed-slst"), ("rotate-randsf", "rotate-randst"),
    ("fixed-sf", "fixed-maxst"), ("fixed-st", "fixed-maxst"),
    ("fixed-randsf", "fixed-randst"),
]


@pytest.mark.parametrize("alias,canonical", SUPPORT_PAIRS)
def test_support_normalization(alias, canonical):
    for name in (alias, f" {alias.upper()} ", alias.replace("-", "_"), canonical):
        assert canonical_backbone_name(name) == canonical
    assert canonical_backbone_name(canonical) == canonical


@pytest.mark.parametrize("alias,canonical", SUPPORT_PAIRS)
@pytest.mark.parametrize("weighted", [False, True])
def test_seeded_forest_masks_match(alias, canonical, disconnected, weighted):
    if canonical in ("slst", "glst", "llst", "randspt"):
        pytest.importorskip("networkx")
    graph = disconnected
    if weighted:
        graph = from_edge_index(
            graph.edge_index, num_nodes=7,
            edge_weight=np.arange(1, graph.num_edges + 1, dtype=float),
        )
    expected = build_backbone(graph, canonical, seed=17)
    actual = build_backbone(graph, f" {alias.upper().replace('-', '_')} ", seed=17)
    np.testing.assert_array_equal(actual, expected)
    assert actual.sum() == 4  # two trees, plus an isolated vertex


@pytest.mark.parametrize("method", GREEDY_METHODS)
@pytest.mark.parametrize("alias,canonical", [
    ("randsf", "randst"), ("fast-maxsf", "fast-maxst"), ("minsf", "mst"),
])
def test_public_methods_record_canonical_names(method, alias, canonical, small_grid):
    run = getattr(scaffold, method)
    options = dict(keep_ratio=0.8, seed=17, workers=1)
    old = run(small_grid, backbone=canonical, **options)
    new = run(small_grid, backbone=alias, **options)
    np.testing.assert_array_equal(new.mask, old.mask)
    assert new.metadata["backbone"] == old.metadata["backbone"] == canonical


def test_weighted_min_and_max_remain_distinct():
    graph = from_edge_index(
        np.array([[0, 0, 1], [1, 2, 2]]), num_nodes=3, edge_weight=[1.0, 3.0, 2.0],
    )
    np.testing.assert_array_equal(build_backbone(graph, "maxsf"), [False, True, True])
    np.testing.assert_array_equal(build_backbone(graph, "minsf"), [True, False, True])
    np.testing.assert_array_equal(build_backbone(graph, "sf"), build_backbone(graph, "maxst"))


@pytest.mark.parametrize("alias,canonical", SAMPLE_PAIRS)
def test_sample_aliases_preserve_draws_and_saved_names(alias, canonical, small_grid, tmp_path):
    if canonical == "fixed-slst":
        pytest.importorskip("networkx")
    for name in (alias, alias.upper().replace("-", "_"), f" {alias} ", canonical):
        assert canonical_sample_backbone(name) == canonical
    options = dict(seed=17, tree_count=2, workers=1)
    old = scaffold.sample(small_grid, backbone=canonical, **options)
    new = scaffold.sample(small_grid, backbone=alias, **options)
    assert new.metadata["backbone"] == old.metadata["backbone"] == canonical
    np.testing.assert_array_equal(new.scores, old.scores)
    np.testing.assert_array_equal(new.backbone, old.backbone)
    np.testing.assert_array_equal(new.sampler.order, old.sampler.order)
    for _ in range(4):
        np.testing.assert_array_equal(new.draw(keep_ratio=0.8).mask, old.draw(keep_ratio=0.8).mask)
        assert new.sampler._plan_key == old.sampler._plan_key
        assert new.sampler._plan_key[1] == canonical
    path = tmp_path / "weights.npz"
    new.sampler.save(path)
    with np.load(path, allow_pickle=False) as artifact:
        assert artifact["backbone"].item() == canonical
        contents = {key: artifact[key] for key in artifact.files}
    restored = ScaffoldSampler.load(path, small_grid, seed=17, workers=1)
    # Also accept artifacts written using the newer spelling.
    contents["backbone"] = np.asarray(alias)
    np.savez_compressed(path, **contents)
    restored_alias = ScaffoldSampler.load(path, small_grid, seed=17, workers=1)
    assert restored_alias.backbone == restored.backbone == canonical
    for _ in range(3):
        np.testing.assert_array_equal(
            restored_alias.draw(keep_ratio=0.8).mask, restored.draw(keep_ratio=0.8).mask,
        )


@pytest.mark.parametrize("alias,canonical", [
    pair for pair in SUPPORT_PAIRS if pair[1] != "llst"
])
def test_local_search_initializer_aliases(alias, canonical):
    pytest.importorskip("networkx")
    graph = scaffold.grid_graph(3, 3, weight="random", seed=5)
    options = dict(seed=17, max_passes=1)
    expected = build_backbone(graph, "llst", init_support=canonical, **options)
    actual = build_backbone(graph, "llsf", init_support=alias, **options)
    np.testing.assert_array_equal(actual, expected)


def test_custom_backbones_masks_and_callables_are_unchanged(small_grid, monkeypatch):
    from scaffold import backbone as module

    monkeypatch.setattr(module, "_REGISTRY", module._REGISTRY.copy())
    mask = build_backbone(small_grid, "randsf", seed=3)

    def builder(graph, **kwargs):
        return mask.copy()

    register_backbone("my-custom-forest", builder)
    assert canonical_backbone_name("my-custom-forest") == "my-custom-forest"
    assert "my-custom-forest" in scaffold.available_backbones(notation="forest")
    for value in ("my-custom-forest", mask, np.flatnonzero(mask), builder):
        np.testing.assert_array_equal(build_backbone(small_grid, value), mask)


def test_forest_listing_preserves_historical_listing(disconnected):
    old_names = scaffold.available_backbones()
    names = scaffold.available_backbones(notation="forest")
    assert {canonical_backbone_name(name) for name in names} == set(old_names)
    assert {"maxst", "llst", "randst", "fast-maxst"}.issubset(old_names)
    assert {"maxsf", "llsf", "randsf", "fast-maxsf"}.issubset(names)
    for name in names:
        if name in ("slsf", "llsf", "glsf", "randspf"):
            continue  # optional builders are exercised separately
        mask = build_backbone(disconnected, name, seed=17)
        assert mask.sum() == (0 if name == "none" else 4)
    with pytest.raises(ValueError, match="notation"):
        scaffold.available_backbones(notation="unknown")


@pytest.mark.parametrize("method", GREEDY_METHODS)
@pytest.mark.parametrize("name,legacy", [
    ("SF", "maxst"), ("MaxSF", "maxst"), ("MinSF", "mst"),
    ("RandSF", "randst"), ("FastMaxSF", "fast-maxst"),
    ("FastMinSF", "fast-mst"), ("FastRandSF", "fast-randst"),
    ("SPF", "spt"), ("RandSPF", "randspt"), ("SLSF", "slst"),
    ("GLSF", "glst"), ("LLSF", "llst"),
])
def test_paper_notation_works_end_to_end(method, name, legacy):
    if legacy in ("randspt", "slst", "glst", "llst"):
        pytest.importorskip("networkx")
    graph = scaffold.grid_graph(3, 3, weight="random", seed=5)
    options = dict(method=method, keep_ratio=0.8, seed=17, workers=1)
    old = scaffold.sparsify(graph, backbone=legacy, **options)
    new = scaffold.sparsify(graph, backbone=name, **options)
    np.testing.assert_array_equal(new.mask, old.mask)
    assert new.metadata["backbone"] == old.metadata["backbone"] == legacy


@pytest.mark.parametrize("name,legacy", [
    ("fixed-SF", "fixed-maxst"), ("fixed-MaxSF", "fixed-maxst"),
    ("fixed-RandSF", "fixed-randst"), ("rotate-RandSF", "rotate-randst"),
    ("fixed-SLSF", "fixed-slst"),
])
def test_sample_accepts_paper_notation(name, legacy, small_grid):
    if legacy == "fixed-slst":
        pytest.importorskip("networkx")
    options = dict(seed=17, tree_count=2, workers=1)
    old = scaffold.sample(small_grid, backbone=legacy, **options)
    new = scaffold.sample(small_grid, backbone=name, **options)
    np.testing.assert_array_equal(new.scores, old.scores)
    for _ in range(3):
        np.testing.assert_array_equal(new.draw(keep_ratio=0.8).mask, old.draw(keep_ratio=0.8).mask)


@pytest.mark.parametrize("name", ["slsf", "randspf"])
@pytest.mark.parametrize("budget", [0, 1, 3, 4, 100, None])
def test_new_path_backbones_obey_component_budgets(name, budget, disconnected):
    nx = pytest.importorskip("networkx")
    mask = build_backbone(disconnected, name, max_edges=budget, seed=5)
    forest = nx.Graph()
    forest.add_nodes_from(range(disconnected.num_nodes))
    forest.add_edges_from(disconnected.edge_index[:, mask].T)
    expected = 4 if budget is None else min(budget, 4)
    assert mask.sum() == expected
    assert nx.is_forest(forest)
    assert nx.number_connected_components(forest) == 7 - expected
    assert forest.degree(6) == 0


@pytest.mark.parametrize("name", ["slsf", "randspf"])
def test_new_path_backbones_handle_empty_graph(name):
    pytest.importorskip("networkx")
    graph = from_edge_index(np.empty((2, 0), dtype=int), num_nodes=4)
    assert build_backbone(graph, name).shape == (0,)


@pytest.mark.parametrize("name", ["slsf", "randspf"])
def test_new_path_backbones_settle_weighted_shortest_paths(name):
    nx = pytest.importorskip("networkx")
    # Every root discovers its last vertex before the indirect path improves
    # the expensive edge. The final forest must exclude that edge.
    graph = from_edge_index(
        np.array([[0, 0, 1], [1, 2, 2]]), num_nodes=3, edge_weight=[1.0, 100.0, 2.0],
    )
    for seed in range(6):
        mask = build_backbone(graph, name, seed=seed)
        np.testing.assert_array_equal(mask, [True, False, True])
        tree = nx.Graph()
        tree.add_edges_from(graph.edge_index[:, mask].T)
        assert nx.is_tree(tree)


@pytest.mark.parametrize("name", ["slsf", "randspf"])
def test_new_path_backbones_reject_nonpositive_lengths(name):
    pytest.importorskip("networkx")
    graph = from_edge_index(np.array([[0], [1]]), num_nodes=2, edge_weight=[0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        build_backbone(graph, name)
