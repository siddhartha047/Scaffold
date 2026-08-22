"""Algorithmic guarantees, checked for every variant.

These are the properties a user is entitled to rely on, so they are tested
against all four methods rather than one at a time:

* the edge budget is respected exactly,
* the output is a subgraph of the input on the same node set,
* connectivity is preserved whenever the budget allows it,
* the same seed gives the same answer,
* mandatory (bridge) edges are never dropped above the floor.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from conftest import ALL_METHODS, GREEDY_METHODS, sparsify_any
from scaffold.backbone import (
    DEFAULT_BACKBONE,
    build_backbone,
    canonical_backbone_name,
)


# ----------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("keep_ratio", [0.6, 0.75, 0.9, 1.0])
def test_budget_is_exact(grid, method, keep_ratio):
    expected = int(np.ceil(keep_ratio * grid.num_edges - 1e-12))
    result = sparsify_any(grid, method, keep_ratio)
    assert result.sparse_edges == expected


@pytest.mark.parametrize("method", ALL_METHODS)
def test_num_edges_budget(grid, method):
    result = scaffold.sparsify(grid, method=method, num_edges=90, seed=0)
    if method == "sample":
        result = result.draw(num_edges=90, seed=0)
    assert result.sparse_edges == 90


@pytest.mark.parametrize("method", ALL_METHODS)
def test_budget_arguments_are_mutually_exclusive(grid, method):
    with pytest.raises(ValueError, match="not both"):
        scaffold.sparsify(grid, method=method, keep_ratio=0.5, num_edges=50)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_target_ratio_alias_matches_keep_ratio(grid, method):
    by_keep = scaffold.sparsify(grid, method=method, keep_ratio=0.7, seed=0)
    by_target = scaffold.sparsify(grid, method=method, target_ratio=0.7, seed=0)
    if method == "sample":
        by_keep = by_keep.draw(seed=0)
        by_target = by_target.draw(seed=0)
    np.testing.assert_array_equal(by_keep.mask, by_target.mask)


def test_ratio_aliases_are_mutually_exclusive(grid):
    with pytest.raises(ValueError, match="either keep_ratio or target_ratio"):
        scaffold.fast(grid, keep_ratio=0.7, target_ratio=0.7)


def test_target_ratio_and_num_edges_are_mutually_exclusive(grid):
    with pytest.raises(ValueError, match="not both"):
        scaffold.fast(grid, target_ratio=0.7, num_edges=80)


@pytest.mark.parametrize("keep_ratio", [-0.1, 1.5])
def test_keep_ratio_must_be_a_fraction(grid, keep_ratio):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        scaffold.fast(grid, keep_ratio=keep_ratio)


# ----------------------------------------------------------------------
# structural validity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_is_a_subgraph_of_the_input(grid, method):
    result = sparsify_any(grid, method, 0.7)
    original = {(int(u), int(v)) for u, v in grid.edge_index.T}
    produced = {(int(u), int(v)) for u, v in result.undirected_edge_index.T}
    assert produced <= original
    assert result.num_nodes == grid.num_nodes


@pytest.mark.parametrize("method", ALL_METHODS)
def test_no_self_loops_or_duplicates(grid, method):
    result = sparsify_any(grid, method, 0.7)
    src, dst = result.undirected_edge_index
    assert (src != dst).all()
    assert (src < dst).all()  # canonical orientation
    assert len({(int(u), int(v)) for u, v in zip(src, dst)}) == src.size


@pytest.mark.parametrize("method", ALL_METHODS)
def test_symmetric_edge_index_is_twice_the_undirected_one(grid, method):
    result = sparsify_any(grid, method, 0.7)
    assert result.edge_index.shape == (2, 2 * result.sparse_edges)
    forward = set(map(tuple, result.edge_index.T.tolist()))
    assert all((v, u) in forward for u, v in forward)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_mask_and_edge_index_agree(grid, method):
    result = sparsify_any(grid, method, 0.7)
    np.testing.assert_array_equal(
        result.undirected_edge_index, grid.edge_index[:, result.mask]
    )
    np.testing.assert_array_equal(result.edge_ids, np.flatnonzero(result.mask))


# ----------------------------------------------------------------------
# connectivity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("keep_ratio", [0.6, 0.8, 1.0])
def test_connectivity_preserved_above_the_floor(grid, method, keep_ratio):
    floor = (grid.num_nodes - 1) / grid.num_edges
    assert keep_ratio >= floor, "fixture assumption"
    result = sparsify_any(grid, method, keep_ratio)
    assert result.num_components() == 1


@pytest.mark.parametrize("method", ALL_METHODS)
def test_component_count_preserved_on_a_disconnected_graph(disconnected, method):
    result = sparsify_any(disconnected, method, 0.9)
    assert result.num_components() == 3


@pytest.mark.parametrize("method", ALL_METHODS)
def test_below_the_floor_it_fragments_and_says_so(grid, method):
    floor = (grid.num_nodes - 1) / grid.num_edges
    result = sparsify_any(grid, method, floor / 2)
    target = int(np.ceil((floor / 2) * grid.num_edges - 1e-12))
    assert result.sparse_edges == target
    assert result.num_components() > 1
    assert result.metadata["below_connectivity_floor"] is True
    assert result.metadata["delta_min"] == pytest.approx(floor)


@pytest.mark.parametrize("method", GREEDY_METHODS)
def test_below_floor_builds_full_forest_then_randomly_trims(grid, method):
    spanning_edges = grid.num_nodes - 1
    target = spanning_edges // 2
    result = scaffold.sparsify(grid, method=method, num_edges=target, seed=17)

    assert result.metadata["backbone_edges"] == spanning_edges
    assert result.metadata["support_budget_mode"] == "full_then_random_trim"
    assert result.metadata["budget_trimmed"] == spanning_edges - target
    assert result.sparse_edges == target

    full_forest = build_backbone(grid, "fast-maxst")
    assert np.all(full_forest[result.edge_ids])


@pytest.mark.parametrize("method", GREEDY_METHODS)
def test_below_floor_random_trim_responds_to_seed(grid, method):
    target = (grid.num_nodes - 1) // 2
    first = scaffold.sparsify(grid, method=method, num_edges=target, seed=1)
    second = scaffold.sparsify(grid, method=method, num_edges=target, seed=2)
    assert not np.array_equal(first.mask, second.mask)


def test_all_methods_share_below_floor_trim_policy(grid):
    target = (grid.num_nodes - 1) // 2
    masks = [
        scaffold.sparsify(grid, method=method, num_edges=target, seed=23).mask
        for method in GREEDY_METHODS
    ]
    masks.append(scaffold.sample(grid, seed=23).draw(num_edges=target, seed=23).mask)
    for mask in masks[1:]:
        np.testing.assert_array_equal(mask, masks[0])


def test_sample_below_floor_randomly_trims_full_forced_forest(grid):
    target = (grid.num_nodes - 1) // 2
    first_scores = scaffold.sample(grid, seed=1)
    second_scores = scaffold.sample(grid, seed=2)
    first = first_scores.draw(num_edges=target, seed=1)
    repeated = scaffold.sample(grid, seed=1).draw(num_edges=target, seed=1)
    second = second_scores.draw(num_edges=target, seed=2)

    assert first.sparse_edges == target
    assert first.metadata["forced_edges"] == target
    assert first.metadata["support_budget_mode"] == "full_then_random_trim"
    assert first.metadata["sampled_edges"] == 0
    assert first.metadata["budget_trimmed"] == grid.num_nodes - 1 - target
    np.testing.assert_array_equal(first.mask, repeated.mask)
    assert not np.array_equal(first.mask, second.mask)


def test_sample_retrims_below_floor_on_each_draw(grid):
    target = (grid.num_nodes - 1) // 2
    scores = scaffold.sample(grid, seed=5)
    first = scores.draw(num_edges=target)
    second = scores.draw(num_edges=target)
    assert not np.array_equal(first.mask, second.mask)


def test_sample_below_floor_probabilities_match_uniform_trim(grid):
    target = (grid.num_nodes - 1) // 2
    scores = scaffold.sample(grid, seed=5)
    p = scores.inclusion_probabilities(num_edges=target)
    expected = target / (grid.num_nodes - 1)

    assert p.sum() == pytest.approx(target)
    np.testing.assert_allclose(p[scores.backbone], expected)
    assert np.all(p[~scores.backbone] == 0.0)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_bridges_are_kept(bridged, method):
    """Ring-of-cliques bridges are the only cross-clique paths."""
    result = sparsify_any(bridged, method, 0.5)
    assert result.num_components() == 1


@pytest.mark.parametrize("method", GREEDY_METHODS)
def test_backbone_none_may_disconnect(grid, method):
    """Opting out of the backbone means opting out of the guarantee."""
    result = scaffold.sparsify(
        grid, method=method, keep_ratio=0.3, backbone="none", seed=0
    )
    assert result.sparse_edges == int(np.ceil(0.3 * grid.num_edges))


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
def test_same_seed_same_result(grid, method):
    first = sparsify_any(grid, method, 0.7, seed=42)
    second = sparsify_any(grid, method, 0.7, seed=42)
    np.testing.assert_array_equal(first.mask, second.mask)


@pytest.mark.parametrize("method", GREEDY_METHODS)
def test_deterministic_backbone_makes_seed_irrelevant(grid, method):
    """With ``fast-maxst`` nothing is stochastic, so seeds must not matter."""
    first = scaffold.sparsify(grid, method=method, keep_ratio=0.7, seed=1)
    second = scaffold.sparsify(grid, method=method, keep_ratio=0.7, seed=999)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_random_backbone_responds_to_the_seed(grid):
    first = scaffold.fast(grid, keep_ratio=0.7, backbone="randst", seed=1)
    second = scaffold.fast(grid, keep_ratio=0.7, backbone="randst", seed=2)
    assert not np.array_equal(first.mask, second.mask)


def test_fast_randst_is_registered_with_research_alias(grid):
    assert DEFAULT_BACKBONE == "fast-maxst"
    assert "fast-randst" in scaffold.available_backbones()
    assert canonical_backbone_name("fast_randst") == "fast-randst"
    by_hyphen = build_backbone(grid, "fast-randst", seed=3)
    by_underscore = build_backbone(grid, "fast_randst", seed=3)
    np.testing.assert_array_equal(by_hyphen, by_underscore)


def test_fast_randst_is_a_seeded_spanning_forest(grid):
    first = build_backbone(grid, "fast-randst", seed=1)
    repeated = build_backbone(grid, "fast-randst", seed=1)
    second = build_backbone(grid, "fast-randst", seed=2)

    assert int(first.sum()) == grid.num_nodes - 1
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_fast_randst_is_distinct_from_randst(grid):
    fast_random = build_backbone(grid, "fast-randst", seed=7)
    full_random = build_backbone(grid, "randst", seed=7)
    assert not np.array_equal(fast_random, full_random)


def test_fast_randst_honors_partial_forest_budget(grid):
    mask = build_backbone(grid, "fast-randst", seed=7, max_edges=20)
    assert int(mask.sum()) == 20


@pytest.mark.parametrize("method", GREEDY_METHODS)
def test_fast_randst_integrates_with_greedy_methods(grid, method):
    result = scaffold.sparsify(
        grid, method=method, keep_ratio=0.7, backbone="fast-randst", seed=7
    )
    assert result.num_components() == 1
    assert result.metadata["backbone"] == "fast-randst"


def test_global_numpy_rng_is_not_touched(grid):
    np.random.seed(1234)
    before = np.random.get_state()[1][:8].copy()
    scaffold.fast(grid, keep_ratio=0.7, backbone="randst", seed=7)
    scaffold.fast(grid, keep_ratio=0.7, backbone="fast-randst", seed=7)
    scaffold.sample(grid, seed=7).draw(keep_ratio=0.7)
    np.testing.assert_array_equal(before, np.random.get_state()[1][:8])


# ----------------------------------------------------------------------
# per-variant behaviour
# ----------------------------------------------------------------------
def test_fast_topk_and_rounds_agree_on_budget(grid):
    topk = scaffold.fast(grid, keep_ratio=0.7, selection="topk", seed=0)
    rounds = scaffold.fast(grid, keep_ratio=0.7, selection="rounds", seed=0)
    assert topk.sparse_edges == rounds.sparse_edges
    assert rounds.metadata["rounds"] >= 1


def test_fast_topk_is_optimal_for_the_static_score(grid):
    """``topk`` maximizes total score among all budget-feasible selections.

    ``rounds`` samples, so it cannot beat a global top-k on the same scores.
    """
    topk = scaffold.fast(grid, keep_ratio=0.7, selection="topk", seed=0,
                         return_scores=True)
    rounds = scaffold.fast(grid, keep_ratio=0.7, selection="rounds", seed=0)
    scores = np.nan_to_num(topk.metadata["scores"], posinf=1e12)
    assert scores[topk.mask].sum() >= scores[rounds.mask].sum()


def test_exact_batch_size_reduces_rounds(small_grid):
    single = scaffold.exact(small_grid, keep_ratio=0.9, batch_size=1, seed=0)
    batched = scaffold.exact(small_grid, keep_ratio=0.9, batch_size=4, seed=0)
    assert batched.metadata["rounds"] < single.metadata["rounds"]
    assert batched.sparse_edges == single.sparse_edges


def test_heap_rescores_far_fewer_candidates_than_exact(grid):
    exact = scaffold.exact(grid, keep_ratio=0.8, seed=0)
    heap = scaffold.heap(grid, keep_ratio=0.8, seed=0)
    assert heap.metadata["rescored_candidates"] < exact.metadata["scored_candidates"]


@pytest.mark.parametrize("score_form", ["max", "product"])
def test_heap_score_forms_both_work(grid, score_form):
    result = scaffold.heap(grid, keep_ratio=0.75, score_form=score_form, seed=0)
    assert result.num_components() == 1
    assert result.metadata["score_form"] == score_form


def test_heap_with_clusters(grid):
    result = scaffold.heap(grid, keep_ratio=0.75, clusters=4, seed=0)
    assert result.metadata["clusters"] >= 1
    assert result.num_components() == 1


# ----------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
def test_empty_graph(method):
    graph = scaffold.normalize_graph(np.zeros((2, 0), dtype=np.int64), num_nodes=5)
    result = sparsify_any(graph, method, 0.5)
    assert result.sparse_edges == 0
    assert result.num_nodes == 5
    assert result.keep_ratio == 0.0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_single_edge_graph(method):
    graph = scaffold.normalize_graph(np.array([[0], [1]]), num_nodes=2)
    result = sparsify_any(graph, method, 1.0)
    assert result.sparse_edges == 1


@pytest.mark.parametrize("method", ALL_METHODS)
def test_zero_budget(grid, method):
    result = sparsify_any(grid, method, 0.0)
    assert result.sparse_edges == 0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_weighted_graph(weighted_grid, method):
    result = sparsify_any(weighted_grid, method, 0.8)
    assert result.undirected_edge_weight is not None
    assert result.undirected_edge_weight.shape == (result.sparse_edges,)
    assert np.isfinite(result.undirected_edge_weight).all()


def test_self_loops_and_duplicates_are_removed_on_input():
    edge_index = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 2]])
    graph = scaffold.normalize_graph(edge_index, num_nodes=3)
    assert graph.num_edges == 2  # (0,1) and (1,2); loops and the mirror dropped
    np.testing.assert_array_equal(graph.edge_index, np.array([[0, 1], [1, 2]]))


def test_unknown_method_lists_the_valid_ones(grid):
    with pytest.raises(ValueError, match="exact"):
        scaffold.sparsify(grid, method="nonesuch")


@pytest.mark.parametrize(
    "alias", ["fast", "FAST", "scaffold-fast", "scaffold_fast", "Scaffold-Fast"]
)
def test_method_name_aliases(grid, alias):
    result = scaffold.sparsify(grid, method=alias, keep_ratio=0.7, seed=0)
    assert result.method == "fast"
