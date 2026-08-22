"""Round-trips through every supported graph library.

The contract: whatever you put in, you can get back out; node identity is
preserved; and the algorithms never see the library-specific object.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaffold
from conftest import requires_networkx, requires_pyg, requires_scipy


# ----------------------------------------------------------------------
# canonicalization
# ----------------------------------------------------------------------
def test_symmetric_input_is_collapsed_once():
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = scaffold.normalize_graph(edge_index, num_nodes=3)
    assert graph.num_edges == 2


def test_edges_are_sorted_canonically():
    edge_index = np.array([[3, 0, 2], [1, 2, 1]])
    graph = scaffold.normalize_graph(edge_index, num_nodes=4)
    np.testing.assert_array_equal(graph.edge_index, np.array([[0, 1, 1], [2, 2, 3]]))


def test_canonical_order_is_stable_across_input_permutations():
    """Sample artifacts are indexed positionally, so this cannot drift."""
    rng = np.random.default_rng(0)
    base = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    first = scaffold.normalize_graph(base, num_nodes=5)
    shuffled = base[:, rng.permutation(base.shape[1])]
    second = scaffold.normalize_graph(shuffled, num_nodes=5)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)


def test_m_by_2_edge_list_is_accepted():
    graph = scaffold.normalize_graph(np.array([[0, 1], [1, 2], [2, 3]]), num_nodes=4)
    assert graph.num_edges == 3


def test_out_of_range_node_id_is_rejected():
    with pytest.raises(ValueError, match="num_nodes"):
        scaffold.normalize_graph(np.array([[0], [9]]), num_nodes=3)


def test_unsupported_type_names_the_alternatives():
    with pytest.raises(TypeError, match="networkx"):
        scaffold.normalize_graph("not a graph")


def test_num_nodes_is_inferred_when_absent():
    graph = scaffold.normalize_graph(np.array([[0, 5], [5, 7]]))
    assert graph.num_nodes == 8


def test_isolated_nodes_survive():
    """Dropping them would misalign the output with a feature matrix."""
    graph = scaffold.normalize_graph(np.array([[0], [1]]), num_nodes=10)
    result = scaffold.fast(graph, keep_ratio=1.0)
    assert result.num_nodes == 10
    assert result.to_scipy().shape == (10, 10)


# ----------------------------------------------------------------------
# NetworkX
# ----------------------------------------------------------------------
@requires_networkx
def test_networkx_roundtrip_preserves_labels():
    import networkx as nx

    G = nx.relabel_nodes(nx.karate_club_graph(), lambda i: f"n{i}")
    result = scaffold.fast(G, keep_ratio=0.6, seed=0)
    H = result.to_networkx()

    assert set(H.nodes()) == set(G.nodes())
    assert set(map(frozenset, H.edges())) <= set(map(frozenset, G.edges()))
    assert H.number_of_edges() == result.sparse_edges


@requires_networkx
def test_networkx_weights_are_read_and_written():
    import networkx as nx

    G = nx.path_graph(8)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 1.0 + i

    graph = scaffold.normalize_graph(G)
    assert graph.is_weighted

    result = scaffold.fast(G, keep_ratio=1.0)
    H = result.to_networkx()
    assert all("weight" in data for _, _, data in H.edges(data=True))


@requires_networkx
def test_unweighted_networkx_stays_unweighted():
    import networkx as nx

    graph = scaffold.normalize_graph(nx.cycle_graph(6))
    assert graph.edge_weight is None


@requires_networkx
def test_networkx_isolated_nodes_are_kept():
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "lonely"])
    G.add_edges_from([("a", "b"), ("b", "c")])
    H = scaffold.fast(G, keep_ratio=1.0).to_networkx()
    assert "lonely" in H.nodes()


# ----------------------------------------------------------------------
# SciPy
# ----------------------------------------------------------------------
@requires_scipy
@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "lil"])
def test_scipy_formats_all_load(fmt):
    graph = scaffold.grid_graph(6, 6)
    A = scaffold.fast(graph, keep_ratio=1.0).to_scipy()
    result = scaffold.fast(A.asformat(fmt), keep_ratio=0.5, seed=0)
    assert result.sparse_edges == int(np.ceil(0.5 * graph.num_edges))


@requires_scipy
def test_scipy_output_is_symmetric():
    graph = scaffold.grid_graph(6, 6)
    A = scaffold.fast(graph, keep_ratio=0.8, seed=0).to_scipy()
    assert (A != A.T).nnz == 0
    assert A.nnz == 2 * int(np.ceil(0.8 * graph.num_edges))


@requires_scipy
def test_non_square_matrix_is_rejected():
    import scipy.sparse as sp

    with pytest.raises(ValueError, match="square"):
        scaffold.normalize_graph(sp.csr_matrix((4, 7)))


# ----------------------------------------------------------------------
# PyTorch Geometric
# ----------------------------------------------------------------------
@requires_pyg
def _make_data(num_nodes=40, seed=0):
    import torch
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < 120:
        u, v = rng.integers(0, num_nodes, size=2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    src, dst = zip(*sorted(pairs))
    edge_index = torch.tensor(
        np.array([list(src) + list(dst), list(dst) + list(src)]), dtype=torch.long
    )
    data = Data(
        x=torch.randn(num_nodes, 6),
        y=torch.randint(0, 3, (num_nodes,)),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[:10] = True
    return data


@requires_pyg
def test_pyg_roundtrip_preserves_everything_but_topology():
    import torch

    data = _make_data()
    result = scaffold.fast(data, keep_ratio=0.5, seed=0)
    out = result.to_pyg()

    assert torch.equal(out.x, data.x)
    assert torch.equal(out.y, data.y)
    assert torch.equal(out.train_mask, data.train_mask)
    assert int(out.num_nodes) == int(data.num_nodes)
    assert out.edge_index.shape[1] == 2 * result.sparse_edges
    assert out.edge_index.dtype == torch.long


@requires_pyg
def test_pyg_output_edge_index_is_symmetric():
    data = _make_data()
    out = scaffold.fast(data, keep_ratio=0.6, seed=0).to_pyg()
    pairs = set(map(tuple, out.edge_index.t().tolist()))
    assert all((v, u) in pairs for u, v in pairs)


@requires_pyg
def test_pyg_edge_weight_is_read():
    import torch

    data = _make_data()
    data.edge_weight = torch.rand(data.edge_index.shape[1])
    graph = scaffold.normalize_graph(data)
    assert graph.is_weighted


@requires_pyg
def test_to_torch_returns_tensors():
    import torch

    data = _make_data()
    edge_index, edge_weight = scaffold.fast(data, keep_ratio=0.5, seed=0).to_torch()
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.dtype == torch.long
    assert edge_weight is None  # unweighted input


@requires_pyg
def test_pyg_transform_and_resampler():
    from scaffold.pyg import ScaffoldResampler, ScaffoldTransform

    data = _make_data()
    transform = ScaffoldTransform(method="fast", keep_ratio=0.5, seed=0)
    out = transform(data)
    assert out.edge_index.shape[1] < data.edge_index.shape[1]

    resampler = ScaffoldResampler(data, keep_ratio=0.6, seed=0)
    first = resampler.epoch(0)
    second = resampler.epoch(1)
    assert first.edge_index.shape == second.edge_index.shape
    assert resampler.edge_weight.shape == (resampler.graph.num_edges,)
    assert 0.0 <= resampler.delta_min <= 1.0


@requires_pyg
def test_pyg_resampler_caches_within_an_interval():
    import torch

    from scaffold.pyg import ScaffoldResampler

    data = _make_data()
    resampler = ScaffoldResampler(data, keep_ratio=0.6, seed=0, every=3)
    assert torch.equal(resampler.epoch(0).edge_index, resampler.epoch(2).edge_index)
    assert not torch.equal(resampler.epoch(0).edge_index, resampler.epoch(3).edge_index)


# ----------------------------------------------------------------------
# the alias package
# ----------------------------------------------------------------------
def test_scaffold_sparsify_alias():
    import scaffold_sparsify

    assert scaffold_sparsify.__version__ == scaffold.__version__
    assert scaffold_sparsify.fast is scaffold.fast
    result = scaffold_sparsify.sparsify(
        scaffold_sparsify.grid_graph(5, 5), keep_ratio=0.8
    )
    assert result.sparse_edges == int(np.ceil(0.8 * 40))
