"""Worker count must never change a result.

Parallelism here is a pure speed knob: every kernel writes to disjoint output
slots and every floating-point reduction is done outside the parallel regions,
so a run at eight threads must agree with a serial run *bit for bit*, not
merely to some tolerance. These tests are the guard on that claim -- a
regression to approximate agreement means a reduction has leaked into a
parallel loop and the results have quietly become scheduler-dependent.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from scaffold.kernels import build_tree_index, spanning_forest_mask, tree_lca
from scaffold.scoring import ScoreParams, tree_scores
from scaffold.utils.workers import (
    DEFAULT_WORKER_CAP,
    available_cpus,
    parallel_threads,
    resolve_workers,
    split_workers,
)

from conftest import ALL_METHODS, sparsify_any

WORKER_COUNTS = (1, 2, 4, 8)


# ----------------------------------------------------------------------
# resolution policy
# ----------------------------------------------------------------------
def test_explicit_workers_beats_every_environment_variable(monkeypatch):
    monkeypatch.setenv("SCAFFOLD_NUM_WORKERS", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert resolve_workers(2) == 2


def test_scaffold_variable_beats_omp(monkeypatch):
    monkeypatch.setenv("SCAFFOLD_NUM_WORKERS", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert resolve_workers(None) == min(3, available_cpus())


def test_omp_is_honoured_when_scaffold_variable_is_absent(monkeypatch):
    monkeypatch.delenv("SCAFFOLD_NUM_WORKERS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert resolve_workers(None) == min(2, available_cpus())


def test_default_is_capped_rather_than_taking_the_whole_node(monkeypatch):
    """The cap is the point: a shared node must not be seized by default."""
    monkeypatch.delenv("SCAFFOLD_NUM_WORKERS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    resolved = resolve_workers(None)
    assert resolved == min(DEFAULT_WORKER_CAP, available_cpus())
    assert resolved <= DEFAULT_WORKER_CAP


@pytest.mark.parametrize("spelling", [0, None, "auto", ""])
def test_auto_spellings_all_resolve(monkeypatch, spelling):
    monkeypatch.delenv("SCAFFOLD_NUM_WORKERS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert resolve_workers(spelling) == min(DEFAULT_WORKER_CAP, available_cpus())


@pytest.mark.parametrize("spelling", [-1, "all"])
def test_all_cores_can_be_requested_explicitly(spelling):
    assert resolve_workers(spelling) == available_cpus()


def test_requests_are_clamped_to_the_affinity_mask():
    assert resolve_workers(10_000) == available_cpus()


def test_garbage_worker_values_are_rejected():
    with pytest.raises(ValueError):
        resolve_workers("many")


def test_malformed_environment_values_fall_through(monkeypatch):
    monkeypatch.setenv("SCAFFOLD_NUM_WORKERS", "not-a-number")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert resolve_workers(None) == min(3, available_cpus())


def test_split_workers_never_oversubscribes():
    for tasks in (1, 2, 3, 8, 16):
        for total in (1, 2, 4, 8):
            outer, inner = split_workers(tasks, total)
            assert outer >= 1 and inner >= 1
            assert outer <= tasks
            assert outer * inner <= total


def test_parallel_threads_restores_the_previous_setting():
    numba = pytest.importorskip("numba")
    before = numba.get_num_threads()
    with parallel_threads(2):
        assert numba.get_num_threads() == min(2, numba.config.NUMBA_NUM_THREADS)
    assert numba.get_num_threads() == before


def test_parallel_threads_clamps_instead_of_raising():
    numba = pytest.importorskip("numba")
    with parallel_threads(10_000) as pinned:
        assert pinned == numba.config.NUMBA_NUM_THREADS


# ----------------------------------------------------------------------
# kernel-level identity
# ----------------------------------------------------------------------
def test_parallel_lca_matches_serial(grid):
    mask = spanning_forest_mask(
        grid.num_nodes, grid.src, grid.dst,
        np.arange(grid.num_edges, dtype=np.int64),
    )
    depth, root, up, _parent, _tin = build_tree_index(
        grid.num_nodes, grid.src[mask], grid.dst[mask]
    )
    cand = np.flatnonzero(~mask)
    with parallel_threads(1):
        serial = tree_lca(grid.src[cand], grid.dst[cand], depth, root, up)
    with parallel_threads(8):
        parallel = tree_lca(grid.src[cand], grid.dst[cand], depth, root, up)
    assert np.array_equal(serial, parallel)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize(
    "params",
    [
        ScoreParams(),
        ScoreParams(alpha=2.0, beta_edge=0.0, beta_node=0.0),
        ScoreParams(edge_norm_p=3.0, node_norm_q=1.5),
    ],
    ids=["default", "dilation-only", "odd-norms"],
)
def test_tree_scores_are_bitwise_identical_across_workers(grid, workers, params):
    mask = spanning_forest_mask(
        grid.num_nodes, grid.src, grid.dst,
        np.arange(grid.num_edges, dtype=np.int64),
    )
    kwargs = dict(
        num_nodes=grid.num_nodes, src=grid.src, dst=grid.dst,
        tree_mask=mask, params=params,
    )
    reference = tree_scores(**kwargs, workers=1)
    got = tree_scores(**kwargs, workers=workers)
    for key in ("dil", "econ_path", "vcon_path", "score", "length"):
        assert np.array_equal(
            reference[key], got[key], equal_nan=True
        ), f"{key} drifted at workers={workers}"


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_tree_scores_identical_on_a_disconnected_graph(disconnected, workers):
    mask = spanning_forest_mask(
        disconnected.num_nodes, disconnected.src, disconnected.dst,
        np.arange(disconnected.num_edges, dtype=np.int64),
    )
    kwargs = dict(
        num_nodes=disconnected.num_nodes, src=disconnected.src,
        dst=disconnected.dst, tree_mask=mask,
    )
    reference = tree_scores(**kwargs, workers=1)
    got = tree_scores(**kwargs, workers=workers)
    assert np.array_equal(reference["score"], got["score"], equal_nan=True)
    assert np.array_equal(reference["mandatory"], got["mandatory"])


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_mandatory_candidates_stay_infinite_in_parallel(grid, workers):
    """A truncated forest leaves candidates with no tree path at all.

    Those carry ``dil = score = inf`` and are excluded from the congestion
    counters. It is the branch a parallel loop is most likely to get wrong --
    an infinity leaking into a shared maximum would corrupt every other score --
    so it gets its own case rather than relying on a fixture to produce it.
    """
    truncated = spanning_forest_mask(
        grid.num_nodes, grid.src, grid.dst,
        np.arange(grid.num_edges, dtype=np.int64),
        max_edges=grid.num_nodes // 3,
    )
    kwargs = dict(
        num_nodes=grid.num_nodes, src=grid.src, dst=grid.dst, tree_mask=truncated
    )
    reference = tree_scores(**kwargs, workers=1)
    got = tree_scores(**kwargs, workers=workers)

    assert reference["mandatory"].any(), "fixture no longer exercises the inf path"
    assert np.array_equal(reference["mandatory"], got["mandatory"])
    assert np.array_equal(reference["score"], got["score"], equal_nan=True)
    assert np.array_equal(reference["dil"], got["dil"], equal_nan=True)
    assert np.isinf(got["score"][got["mandatory"]]).all()
    # The maxima are finite: infinities must not have leaked into the reduction.
    assert np.isfinite(got["maxima"]).all()


@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize("weighted_paths", [False, True])
def test_tree_scores_identical_on_a_weighted_graph(
    weighted_grid, workers, weighted_paths
):
    mask = spanning_forest_mask(
        weighted_grid.num_nodes, weighted_grid.src, weighted_grid.dst,
        np.arange(weighted_grid.num_edges, dtype=np.int64),
    )
    kwargs = dict(
        num_nodes=weighted_grid.num_nodes, src=weighted_grid.src,
        dst=weighted_grid.dst, tree_mask=mask,
        weight=weighted_grid.edge_weight, weighted_paths=weighted_paths,
    )
    reference = tree_scores(**kwargs, workers=1)
    got = tree_scores(**kwargs, workers=workers)
    assert np.array_equal(reference["dil"], got["dil"], equal_nan=True)
    assert np.array_equal(reference["score"], got["score"], equal_nan=True)


# ----------------------------------------------------------------------
# end-to-end identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_selected_edges_are_identical_across_workers(grid, method, workers):
    reference = sparsify_any(grid, method, keep_ratio=0.75, seed=0, workers=1)
    got = sparsify_any(grid, method, keep_ratio=0.75, seed=0, workers=workers)
    assert np.array_equal(reference.mask, got.mask)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_identical_on_a_weighted_graph(weighted_grid, method, workers):
    reference = sparsify_any(weighted_grid, method, keep_ratio=0.7, seed=1, workers=1)
    got = sparsify_any(weighted_grid, method, keep_ratio=0.7, seed=1, workers=workers)
    assert np.array_equal(reference.mask, got.mask)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_identical_on_a_disconnected_graph(disconnected, method, workers):
    reference = sparsify_any(disconnected, method, keep_ratio=0.9, seed=2, workers=1)
    got = sparsify_any(disconnected, method, keep_ratio=0.9, seed=2, workers=workers)
    assert np.array_equal(reference.mask, got.mask)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_sample_weights_are_identical_across_workers(grid, workers):
    """pi aggregates float sums over R backbones -- the order must be fixed."""
    reference = scaffold.sample(grid, seed=0, tree_count=8, workers=1)
    got = scaffold.sample(grid, seed=0, tree_count=8, workers=workers)
    assert np.array_equal(reference.scores, got.scores)
    assert np.array_equal(reference.order, got.order)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_sample_draws_are_identical_across_workers(grid, workers):
    reference = scaffold.sample(grid, seed=0, tree_count=4, workers=1)
    got = scaffold.sample(grid, seed=0, tree_count=4, workers=workers)
    for draw_seed in (0, 5):
        left = reference.draw(keep_ratio=0.8, seed=draw_seed)
        right = got.draw(keep_ratio=0.8, seed=draw_seed)
        assert np.array_equal(left.mask, right.mask)


def test_more_backbones_than_workers_still_scores_every_one(grid):
    """The outer pool must not drop or duplicate a backbone."""
    scores = scaffold.sample(grid, seed=0, tree_count=9, workers=2)
    assert scores.metadata["tree_count"] == 9
    reference = scaffold.sample(grid, seed=0, tree_count=9, workers=1)
    assert np.array_equal(reference.scores, scores.scores)


def test_resolved_worker_count_is_reported(grid):
    assert scaffold.fast(grid, keep_ratio=0.75, workers=3).metadata["workers"] == 3
    assert scaffold.sample(grid, seed=0, workers=3).metadata["workers"] == 3


# ----------------------------------------------------------------------
# compiled path scorer vs the Python reference
# ----------------------------------------------------------------------
def _random_graph(seed, n=60, extra=90, weighted=False):
    rng = np.random.default_rng(seed)
    # A path guarantees connectivity; the extra edges make the support
    # interesting and give candidates real detours to measure.
    src = list(range(n - 1))
    dst = list(range(1, n))
    for _ in range(extra):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            src.append(int(min(u, v)))
            dst.append(int(max(u, v)))
    edge_index = np.array([src + dst, dst + src])
    weight = None
    if weighted:
        w = rng.uniform(0.5, 4.0, size=len(src))
        weight = np.concatenate([w, w])
    from scaffold.graph import from_edge_index

    return from_edge_index(edge_index, num_nodes=n, edge_weight=weight)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
@pytest.mark.parametrize("weighted", [False, True], ids=["bfs", "dijkstra"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_compiled_path_scorer_matches_the_python_reference(workers, weighted, seed):
    """The compiled scorer is a rewrite, not a refactor -- pin it to the spec.

    Both the BFS and the Dijkstra branch have to settle vertices in the same
    order as the reference, or they pick different (equally short) paths and the
    congestion counters diverge even though every distance is right.
    """
    from scaffold.backbone import build_backbone
    from scaffold.scoring import PathScorer

    graph = _random_graph(seed, weighted=weighted)
    mask = build_backbone(graph, "fast-maxst", seed=seed)
    candidates = np.flatnonzero(~mask)
    params = ScoreParams()

    def scorer():
        return PathScorer(
            graph.num_nodes, graph.src, graph.dst,
            weight=graph.edge_weight, support_mask=mask.copy(), workers=workers,
        )

    expected = scorer()._evaluate_python(candidates, params)
    got = scorer().evaluate(candidates, params)

    for key in ("dil", "econ_path", "vcon_path", "score"):
        assert np.allclose(
            expected[key], got[key], rtol=1e-12, atol=0.0, equal_nan=True
        ), f"{key} diverged (weighted={weighted}, workers={workers}, seed={seed})"
    assert np.array_equal(expected["mandatory"], got["mandatory"])


@pytest.mark.parametrize("weighted", [False, True], ids=["bfs", "dijkstra"])
def test_compiled_paths_match_the_reference_paths(weighted):
    """Heap indexes candidates by the edges and nodes their path touches."""
    from scaffold.backbone import build_backbone
    from scaffold.scoring import PathScorer

    graph = _random_graph(7, weighted=weighted)
    mask = build_backbone(graph, "fast-maxst", seed=7)
    candidates = np.flatnonzero(~mask)
    params = ScoreParams()

    def scorer():
        return PathScorer(
            graph.num_nodes, graph.src, graph.dst,
            weight=graph.edge_weight, support_mask=mask.copy(), workers=4,
        )

    expected = scorer()._evaluate_python(candidates, params)
    got = scorer().evaluate(candidates, params, need_paths=True)
    for i in range(candidates.size):
        assert sorted(expected["path_edges"][i] or ()) == sorted(
            got["path_edges"][i] or ()
        )
        assert sorted(expected["path_nodes"][i] or ()) == sorted(
            got["path_nodes"][i] or ()
        )


def test_skipping_paths_does_not_change_the_scores(grid):
    """need_paths is an output-format switch, never a scoring switch."""
    from scaffold.backbone import build_backbone
    from scaffold.scoring import PathScorer

    mask = build_backbone(grid, "fast-maxst", seed=0)
    candidates = np.flatnonzero(~mask)
    params = ScoreParams()
    scorer = PathScorer(
        grid.num_nodes, grid.src, grid.dst, support_mask=mask.copy(), workers=4
    )
    with_paths = scorer.evaluate(candidates, params, need_paths=True)
    without = scorer.evaluate(candidates, params, need_paths=False)
    for key in ("dil", "econ_path", "vcon_path", "score"):
        assert np.array_equal(with_paths[key], without[key], equal_nan=True)
    assert without["path_edges"][0] == ()
