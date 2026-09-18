"""LLST preserves forests and actually improves the weighted stretch objective."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scaffold
from scaffold.backbone import build_backbone, canonical_backbone_name

nx = pytest.importorskip("networkx")


def nx_graph(graph, mask=None):
    result = nx.Graph()
    result.add_nodes_from(range(graph.num_nodes))
    for i, (u, v) in enumerate(graph.edge_index.T):
        if mask is None or mask[i]:
            result.add_edge(int(u), int(v), weight=float(graph.weights_or_ones()[i]))
    return result


def total_stretch(G, tree):
    # Independent oracle: no package LCA or scoring routines.
    return sum(
        nx.shortest_path_length(tree, u, v, weight="weight") / data["weight"]
        for u, v, data in G.edges(data=True)
    )


def best_swap_value(G, tree):
    best = total_stretch(G, tree)
    for u, v, data in G.edges(data=True):
        if tree.has_edge(u, v):
            continue
        path = nx.shortest_path(tree, u, v)
        for a, b in zip(path, path[1:]):
            trial = tree.copy()
            trial.add_edge(u, v, **data)
            trial.remove_edge(a, b)
            assert nx.is_tree(trial)
            best = min(best, total_stretch(G, trial))
    return best


@pytest.mark.parametrize("weighted", [False, True])
def test_one_pass_finds_best_improving_swap(weighted):
    graph = scaffold.grid_graph(3, 3)
    if weighted:
        graph = graph.with_weight(
            np.random.default_rng(4).uniform(0.25, 3, graph.num_edges)
        )
    G = nx_graph(graph)
    initial = nx_graph(graph, build_backbone(graph, "maxst"))
    result = nx_graph(
        graph, build_backbone(graph, "llst", init_support="maxst", max_passes=1)
    )
    assert nx.is_tree(result)
    assert total_stretch(G, result) == pytest.approx(best_swap_value(G, initial))


@pytest.mark.parametrize("weighted", [False, True])
def test_exhaustive_search_reaches_one_swap_local_minimum(weighted):
    graph = scaffold.grid_graph(3, 3)
    if weighted:
        graph = graph.with_weight(
            np.random.default_rng(8).uniform(0.25, 3, graph.num_edges)
        )
    G = nx_graph(graph)
    initial = nx_graph(graph, build_backbone(graph, "maxst"))
    tree = nx_graph(
        graph, build_backbone(graph, "llst", init_support="maxst", max_passes=100)
    )
    assert nx.is_tree(tree)
    assert total_stretch(G, tree) < total_stretch(G, initial)
    assert best_swap_value(G, tree) == pytest.approx(total_stretch(G, tree))


@pytest.mark.parametrize(
    "init",
    [
        "glst",
        "maxst",
        "mst",
        "fast_maxst",
        "fast_mst",
        "randst",
        "randspt",
        "fast-randst",
        "spt",
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_initializers_produce_spanning_subtrees(init, weighted):
    graph = scaffold.grid_graph(3, 3)
    if weighted:
        graph = graph.with_weight(
            np.random.default_rng(6).uniform(0.25, 3, graph.num_edges)
        )
    mask = build_backbone(graph, "llst", init_support=init, max_passes=2, seed=17)
    assert mask.dtype == bool and mask.shape == (graph.num_edges,)
    assert mask.sum() == graph.num_nodes - 1
    assert nx.is_tree(nx_graph(graph, mask))


@pytest.mark.parametrize("strategy", ["random", "tree_distance"])
@pytest.mark.parametrize("resample", [False, True])
def test_sampled_modes_are_seeded_and_do_not_touch_global_rng(strategy, resample):
    graph = scaffold.grid_graph(4, 4)
    options = dict(
        init_support="randst",
        max_passes=3,
        candidate_strategy=strategy,
        candidate_sample_size=4,
        eval_sample_size=7,
        cycle_sample_size=2,
        resample_eval_each_pass=resample,
    )
    np_state, py_state = np.random.get_state(), random.getstate()
    first = build_backbone(graph, "llst", seed=17, **options)
    second = build_backbone(graph, "llst", seed=np.int64(17), **options)
    np.testing.assert_array_equal(first, second)
    assert nx.is_tree(nx_graph(graph, first))
    after = np.random.get_state()
    assert np_state[0] == after[0] and np_state[2:] == after[2:]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert py_state == random.getstate()


@pytest.mark.parametrize("limit", [0, 1, 3, 4, 100])
def test_partial_forests_and_disconnected_components(disconnected, limit):
    mask = build_backbone(disconnected, "llst", seed=3, max_edges=limit)
    tree = nx_graph(disconnected, mask)
    assert mask.sum() == min(limit, 4)
    assert nx.is_forest(tree)
    assert tree.number_of_nodes() == 7  # includes the isolated node
    if limit >= 4:
        assert nx.number_connected_components(tree) == 3


def test_partial_component_returns_initializer():
    graph = scaffold.grid_graph(3, 3)
    expected = build_backbone(graph, "maxst", max_edges=4)
    actual = build_backbone(graph, "llst", init_support="maxst", max_edges=4)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("num_nodes", [0, 1, 4])
def test_edgeless_graphs(num_nodes):
    graph = scaffold.normalize_graph(np.empty((2, 0), dtype=np.int64), num_nodes=num_nodes)
    assert build_backbone(graph, "llst").shape == (0,)


def test_input_forest_is_unchanged():
    G = nx.path_graph(6)
    G.add_edge(8, 9)
    G.add_node(10)
    graph = scaffold.normalize_graph(G)
    assert build_backbone(graph, "llst").all()


@pytest.mark.parametrize("method", ["greedy", "heap", "batch", "fast"])
@pytest.mark.parametrize("budget", [0, 4, 8, 10, 12])
def test_public_methods_honor_budget_with_llst(method, budget):
    graph = scaffold.grid_graph(3, 3)
    result = scaffold.sparsify(
        graph,
        method=method,
        num_edges=budget,
        backbone="llst",
        backbone_options={"max_passes": 2},
        seed=3,
    )
    assert result.sparse_edges == budget
    assert result.metadata["backbone"] == "llst"
    assert result.metadata["backbone_edges"] == 8
    assert result.metadata["below_connectivity_floor"] == (budget < 8)
    if budget >= 8:
        assert result.num_components() == 1


def test_networkx_labels_and_weights_survive_roundtrip():
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 1), ("a", "c", 5), ("b", "c", 2)])
    G.add_node("isolated")
    result = scaffold.fast(G, num_edges=2, backbone="llst")
    H = result.to_networkx()
    assert set(H) == set(G)
    assert nx.number_connected_components(H) == 2
    assert {frozenset(e) for e in H.edges()} == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
    }
    for u, v in H.edges():
        assert H[u][v]["weight"] == G[u][v]["weight"]


def test_glst_initializer_uses_weighted_path_lengths():
    graph = scaffold.normalize_graph(np.array([[0, 0, 1], [1, 2, 2]])).with_weight(
        np.array([1.0, 5.0, 2.0])
    )
    np.testing.assert_array_equal(build_backbone(graph, "glst"), [True, False, True])


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_llst_rejects_invalid_weighted_stretch(bad):
    graph = scaffold.grid_graph(2, 2).with_weight(np.array([1.0, bad, 2.0, 3.0]))
    with pytest.raises(ValueError, match="finite, strictly positive"):
        build_backbone(graph, "llst")


@pytest.mark.parametrize("bad", ["llst", "none", "missing"])
def test_invalid_initializers_fail_without_recursing(bad):
    with pytest.raises(ValueError, match="init_support"):
        build_backbone(scaffold.grid_graph(2, 2), "llst", init_support=bad)


def test_invalid_candidate_strategy_is_not_silently_ignored():
    with pytest.raises(ValueError, match="candidate_strategy"):
        build_backbone(scaffold.grid_graph(2, 2), "llst", candidate_strategy="bad")


def test_registered_aliases_and_optional_framework_isolation():
    assert "llst" in scaffold.available_backbones()
    assert canonical_backbone_name("local-search-low-stretch-tree") == "llst"
    code = """
import sys
import scaffold
from scaffold.backbone import build_backbone
assert 'networkx' not in sys.modules
build_backbone(scaffold.grid_graph(2, 2), 'local_search_low_stretch_tree')
assert 'networkx' in sys.modules
assert 'torch' not in sys.modules
assert 'torch_geometric' not in sys.modules
"""
    source = str(Path(scaffold.__file__).resolve().parents[1])
    code = f"import sys; sys.path.insert(0, {source!r})\n" + code
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
