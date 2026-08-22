"""The tree-prefix kernel must equal the shortest-path definition exactly.

This is the load-bearing correctness property of the whole package: SCAFFOLD-Fast
and SCAFFOLD-Sample only work because a path aggregate on a tree is a root-prefix
difference. If :func:`scaffold.tree_scores` and :func:`scaffold.path_scores`
ever disagree on a forest, the fast variants are computing a different objective
than the reference greedy does.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from scaffold.backbone import build_backbone
from scaffold.scoring import ScoreParams, path_scores, tree_scores

BACKBONES = ["fast-maxst", "maxst", "mst", "randst", "spt"]
PARAM_SETS = [
    ScoreParams(),
    ScoreParams(alpha=1.0, beta_edge=0.0, beta_node=0.0),
    ScoreParams(alpha=2.0, beta_edge=0.5, beta_node=1.5),
    ScoreParams(edge_norm_p=1.0, node_norm_q=3.0),
]


def _graphs():
    return {
        "grid": scaffold.grid_graph(7, 7),
        "grid_periodic": scaffold.grid_graph(5, 6, periodic=True),
        "cliques": scaffold.ring_of_cliques(3, 5),
        "geometric": scaffold.random_geometric(60, 0.22, seed=1)[0],
    }


@pytest.mark.parametrize("graph_name", sorted(_graphs()))
@pytest.mark.parametrize("backbone", BACKBONES)
def test_tree_scores_match_path_scores(graph_name, backbone):
    graph = _graphs()[graph_name]
    mask = build_backbone(graph, backbone, seed=11)
    params = ScoreParams()

    fast = tree_scores(
        graph.num_nodes, graph.src, graph.dst, mask,
        weight=graph.edge_weight, params=params,
    )
    candidates = np.flatnonzero(~mask)
    reference = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask,
        weight=graph.edge_weight, params=params, candidate_ids=candidates,
    )

    for key in ("dil", "econ_path", "vcon_path", "score"):
        np.testing.assert_allclose(
            fast[key][candidates], reference[key], rtol=1e-9, atol=1e-9,
            err_msg=f"{key} disagrees on {graph_name}/{backbone}",
        )
    np.testing.assert_array_equal(fast["mandatory"][candidates], reference["mandatory"])


@pytest.mark.parametrize("params", PARAM_SETS)
def test_tree_scores_match_across_objective_settings(params):
    graph = scaffold.grid_graph(6, 7)
    mask = build_backbone(graph, "randst", seed=5)
    candidates = np.flatnonzero(~mask)

    fast = tree_scores(graph.num_nodes, graph.src, graph.dst, mask, params=params)
    reference = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask,
        params=params, candidate_ids=candidates,
    )
    np.testing.assert_allclose(
        fast["score"][candidates], reference["score"], rtol=1e-9, atol=1e-9
    )


def test_tree_scores_on_weighted_graph():
    graph = scaffold.grid_graph(6, 6, weight="random", seed=2)
    mask = build_backbone(graph, "maxst")
    candidates = np.flatnonzero(~mask)

    # weighted_paths=True sums tree edge weights for the numerator, which is
    # what the shortest-path reference does on a weighted graph.
    fast = tree_scores(
        graph.num_nodes, graph.src, graph.dst, mask,
        weight=graph.edge_weight, weighted_paths=True,
    )
    reference = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask,
        weight=graph.edge_weight, candidate_ids=candidates,
    )
    np.testing.assert_allclose(
        fast["dil"][candidates], reference["dil"], rtol=1e-9, atol=1e-9
    )


def test_cross_component_candidates_are_mandatory(disconnected):
    """Edges spanning two forest components get inf score and are flagged.

    They must also contribute nothing to the congestion counters -- the trap
    that produces plausible-looking wrong numbers rather than a crash.
    """
    graph = disconnected
    # A backbone of only the first triangle's edges leaves the second
    # triangle's edges spanning separate forest components.
    mask = np.zeros(graph.num_edges, dtype=bool)
    mask[0] = True

    out = tree_scores(graph.num_nodes, graph.src, graph.dst, mask)
    candidates = np.flatnonzero(~mask)
    assert out["mandatory"][candidates].any()
    assert np.isinf(out["score"][out["mandatory"]]).all()
    assert (out["econ_path"][out["mandatory"]] == 0).all()

    reference = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask, candidate_ids=candidates
    )
    np.testing.assert_array_equal(out["mandatory"][candidates], reference["mandatory"])
    np.testing.assert_allclose(
        out["econ_path"][candidates], reference["econ_path"], rtol=1e-9, atol=1e-9
    )


def test_tree_edges_are_excluded_from_statistics():
    graph = scaffold.grid_graph(5, 5)
    mask = build_backbone(graph, "fast-maxst")
    out = tree_scores(graph.num_nodes, graph.src, graph.dst, mask)
    assert (out["score"][mask] == 0).all()
    assert (out["dil"][mask] == 0).all()
    assert not out["mandatory"][mask].any()


def test_full_backbone_leaves_no_candidates():
    """A backbone covering every edge is legal and yields an empty result."""
    graph = scaffold.grid_graph(3, 3)
    mask = np.ones(graph.num_edges, dtype=bool)
    out = tree_scores(graph.num_nodes, graph.src, graph.dst, mask)
    assert out["total_stretch"] == 0.0
    assert not out["candidate_mask"].any()


def test_infinite_norm_order_is_rejected_not_approximated():
    with pytest.raises(NotImplementedError, match="max-on-path"):
        ScoreParams(edge_norm_p=float("inf"))
    with pytest.raises(NotImplementedError):
        ScoreParams(node_norm_q=float("inf"))


def test_non_positive_norm_order_is_rejected():
    with pytest.raises(ValueError):
        ScoreParams(edge_norm_p=0.0)
    with pytest.raises(ValueError):
        ScoreParams(node_norm_q=-1.0)


def test_path_scores_work_on_a_non_forest_support():
    """The reference scorer must handle cycles, which the tree kernel cannot."""
    graph = scaffold.grid_graph(5, 5)
    mask = build_backbone(graph, "fast-maxst")
    mask[np.flatnonzero(~mask)[:5]] = True  # introduce cycles
    candidates = np.flatnonzero(~mask)

    out = path_scores(
        graph.num_nodes, graph.src, graph.dst, mask, candidate_ids=candidates
    )
    assert np.isfinite(out["dil"]).all()
    assert (out["dil"] >= 1.0).all()  # a detour is never shorter than the edge
