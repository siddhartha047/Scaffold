"""SCAFFOLD-Sample: weights, inclusion probabilities and the per-epoch draw.

The guarantees under test:

* every draw has *exactly* the requested edge count,
* every draw has *exactly* the input's connected components (above the floor),
  with probability 1 rather than in expectation,
* inclusion probabilities sum to the budget and never exceed 1,
* the weights are ratio-independent -- one precompute serves every budget,
* draws differ across epochs and their union grows.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from conftest import requires_networkx
from scaffold.algorithms.sample import ScaffoldSampler, cap_and_renormalize
from scaffold.backbone import build_backbone


# ----------------------------------------------------------------------
# the weights themselves
# ----------------------------------------------------------------------
def test_sample_returns_one_weight_per_edge(grid):
    scores = scaffold.sample(grid, seed=0)
    assert scores.scores.shape == (grid.num_edges,)
    assert scores.pi is scores.scores
    assert np.isfinite(scores.scores).all()
    assert (scores.scores > 0).all()


def test_sample_does_not_sparsify(grid):
    """It scores every edge; it does not choose a subgraph."""
    scores = scaffold.sample(grid, keep_ratio=0.5, seed=0)
    assert scores.sparse_edges == grid.num_edges
    assert scores.mask.all()
    assert scores.undirected_edge_index.shape[1] == grid.num_edges


def test_edge_weight_is_symmetric_and_aligned(grid):
    scores = scaffold.sample(grid, seed=0)
    assert scores.edge_index.shape[1] == 2 * grid.num_edges
    assert scores.edge_weight.shape == (2 * grid.num_edges,)
    half = grid.num_edges
    np.testing.assert_array_equal(
        scores.edge_weight[:half], scores.edge_weight[half:]
    )


@requires_networkx
def test_to_dict_uses_original_labels():
    import networkx as nx

    G = nx.relabel_nodes(nx.cycle_graph(6), lambda i: f"n{i}")
    weights = scaffold.sample(G, seed=0).to_dict()
    assert len(weights) == 6
    assert all(isinstance(u, str) and isinstance(v, str) for u, v in weights)


def test_backbone_edges_score_highest(grid):
    """A deterministic backbone edge is structurally indispensable."""
    scores = scaffold.sample(grid, seed=0)
    assert scores.scores[scores.backbone].mean() > scores.scores[~scores.backbone].mean()


def test_weights_are_ratio_independent(grid):
    a = scaffold.sample(grid, keep_ratio=0.2, seed=0).scores
    b = scaffold.sample(grid, keep_ratio=0.9, seed=0).scores
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("tree_count", [1, 4, 16])
def test_tree_count_changes_the_weights_not_the_contract(grid, tree_count):
    scores = scaffold.sample(grid, seed=0, tree_count=tree_count)
    assert scores.metadata["tree_count"] == tree_count
    assert scores.draw(keep_ratio=0.7).num_components() == 1


def test_aggregate_lambda_zero_gives_pure_frequency(grid):
    """lambda=0 drops the score term, leaving backbone frequency only."""
    scores = scaffold.sample(grid, seed=0, aggregate_lambda=0.0, tree_count=4)
    unique = np.unique(np.round(scores.scores, 10))
    assert unique.size <= 6  # only k/4 for k = 0..4, plus the eps clip


# ----------------------------------------------------------------------
# inclusion probabilities
# ----------------------------------------------------------------------
@pytest.mark.parametrize("keep_ratio", [0.6, 0.75, 0.9])
def test_inclusion_probabilities_sum_to_the_budget(grid, keep_ratio):
    scores = scaffold.sample(grid, seed=0)
    p = scores.inclusion_probabilities(keep_ratio=keep_ratio)
    budget = int(np.ceil(keep_ratio * grid.num_edges))
    assert p.sum() == pytest.approx(budget, abs=1e-6)
    assert ((p >= 0) & (p <= 1)).all()


def test_forced_edges_have_probability_one(grid):
    scores = scaffold.sample(grid, seed=0)
    p = scores.inclusion_probabilities(keep_ratio=0.7)
    assert np.allclose(p[scores.backbone], 1.0)


def test_cap_and_renormalize_fixed_point():
    pi = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    p = cap_and_renormalize(pi, target=3)
    assert p.sum() == pytest.approx(3.0)
    assert (p <= 1.0 + 1e-12).all()
    assert p[0] == pytest.approx(1.0)  # the dominant weight is capped


def test_cap_and_renormalize_degenerate_budgets():
    pi = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cap_and_renormalize(pi, 0), np.zeros(3))
    np.testing.assert_array_equal(cap_and_renormalize(pi, 5), np.ones(3))


def test_coverage_report(grid):
    scores = scaffold.sample(grid, seed=0)
    report = scores.sampler.coverage(keep_ratio=0.7, epochs=(1, 5, 50))
    assert report["always_included"] >= int(scores.backbone.sum())
    curve = report["coverage_curve"]
    assert curve[1] < curve[5] <= curve[50] <= 1.0
    assert curve[1] == pytest.approx(0.7, abs=0.02)


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("keep_ratio", [0.6, 0.7, 0.85, 1.0])
def test_every_draw_has_the_exact_budget(grid, keep_ratio):
    scores = scaffold.sample(grid, seed=0)
    expected = int(np.ceil(keep_ratio * grid.num_edges))
    for _ in range(12):
        assert scores.draw(keep_ratio=keep_ratio).sparse_edges == expected


def test_every_draw_preserves_components(grid):
    scores = scaffold.sample(grid, seed=0)
    for _ in range(12):
        assert scores.draw(keep_ratio=0.65).num_components() == 1


def test_draws_preserve_components_on_a_disconnected_graph(disconnected):
    scores = scaffold.sample(disconnected, seed=0)
    for _ in range(8):
        assert scores.draw(keep_ratio=0.9).num_components() == 3


def test_successive_draws_differ(grid):
    scores = scaffold.sample(grid, seed=0)
    masks = [scores.draw(keep_ratio=0.7).mask for _ in range(6)]
    assert not all(np.array_equal(masks[0], m) for m in masks[1:])


def test_union_of_draws_grows_beyond_a_single_view(grid):
    scores = scaffold.sample(grid, seed=0)
    union = np.zeros(grid.num_edges, dtype=bool)
    for _ in range(25):
        union |= scores.draw(keep_ratio=0.6).mask
    assert union.mean() > 0.75


def test_explicit_seed_pins_the_draw(grid):
    first = scaffold.sample(grid, seed=0).draw(keep_ratio=0.7, seed=5)
    second = scaffold.sample(grid, seed=0).draw(keep_ratio=0.7, seed=5)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_default_budget_is_remembered(grid):
    scores = scaffold.sample(grid, keep_ratio=0.7, seed=0)
    assert scores.draw().sparse_edges == int(np.ceil(0.7 * grid.num_edges))
    assert scores.draw(keep_ratio=0.9).sparse_edges == int(np.ceil(0.9 * grid.num_edges))


@pytest.mark.parametrize(
    "backbone", ["fixed-maxst", "fixed-randst", "rotate-randst"]
)
def test_backbone_modes(grid, backbone):
    scores = scaffold.sample(grid, seed=0, backbone=backbone)
    draws = [scores.draw(keep_ratio=0.7) for _ in range(4)]
    assert all(d.num_components() == 1 for d in draws)
    if backbone.startswith("fixed-"):
        assert {d.metadata["rotation"] for d in draws} == {0}
    else:
        assert len({d.metadata["rotation"] for d in draws}) > 1


def test_fixed_randst_matches_public_backbone(grid):
    scores = scaffold.sample(grid, seed=7, backbone="fixed-randst")
    expected = build_backbone(grid, "randst", seed=7)
    np.testing.assert_array_equal(scores.backbone, expected)


def test_five_method_visual_can_share_randst(grid):
    plt = pytest.importorskip("matplotlib.pyplot")

    backbone = build_backbone(grid, "randst", seed=7)
    fig, results = scaffold.viz.compare_methods(
        grid, keep_ratio=0.7, seed=7, backbone="randst"
    )
    try:
        for result in results.values():
            assert np.all(result.mask[backbone])
    finally:
        plt.close(fig)


def test_rotating_backbone_varies_the_guaranteed_edges(grid):
    """With a fixed backbone some edges are in every draw; rotation loosens that."""
    fixed = scaffold.sample(grid, seed=0, backbone="fixed-maxst")
    rotating = scaffold.sample(grid, seed=0, backbone="rotate-randst")

    def always(scores):
        union = np.ones(grid.num_edges, dtype=bool)
        for _ in range(10):
            union &= scores.draw(keep_ratio=0.7).mask
        return int(union.sum())

    assert always(rotating) < always(fixed)


def test_invalid_backbone_mode_is_rejected(grid):
    with pytest.raises(ValueError, match="fixed-maxst"):
        scaffold.sample(grid, backbone="nonsense")


def test_invalid_scheme_is_rejected(grid):
    with pytest.raises(ValueError, match="systematic"):
        scaffold.sample(grid, scheme="alias")


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
def test_artifact_save_and_load(grid, tmp_path):
    scores = scaffold.sample(grid, seed=0)
    path = scores.sampler.save(tmp_path / "weights.npz")

    reloaded = ScaffoldSampler.load(path, grid)
    np.testing.assert_array_equal(reloaded.pi, scores.scores)
    np.testing.assert_array_equal(reloaded.det_forest, scores.backbone)
    assert reloaded.draw(keep_ratio=0.7).sparse_edges == int(
        np.ceil(0.7 * grid.num_edges)
    )


def test_artifact_preserves_fixed_randst_mode(grid, tmp_path):
    scores = scaffold.sample(grid, seed=7, backbone="fixed-randst")
    path = scores.sampler.save(tmp_path / "randst-weights.npz")
    reloaded = ScaffoldSampler.load(path, grid)

    assert reloaded.backbone == "fixed-randst"
    np.testing.assert_array_equal(reloaded.det_forest, scores.backbone)


def test_artifact_rejects_a_different_graph(grid, tmp_path):
    path = scaffold.sample(grid, seed=0).sampler.save(tmp_path / "weights.npz")
    other = scaffold.grid_graph(9, 9)
    with pytest.raises(ValueError, match="does not match"):
        ScaffoldSampler.load(path, other)


def test_sampler_requires_fit(grid):
    with pytest.raises(RuntimeError, match="fit"):
        ScaffoldSampler().draw(keep_ratio=0.5)
