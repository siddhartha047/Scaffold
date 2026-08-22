"""Conversions between user graph objects and the internal :class:`Graph`.

Every optional dependency is imported lazily inside the branch that needs it,
so ``import scaffold`` costs one ``numpy`` import even in an environment with
PyTorch, PyG, NetworkX and SciPy all installed.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..graph import Graph, from_edge_index

__all__ = [
    "to_graph",
    "is_networkx_graph",
    "is_pyg_data",
    "is_scipy_sparse",
    "is_torch_tensor",
    "from_networkx",
    "to_networkx",
    "from_pyg",
    "to_pyg",
    "from_scipy",
    "to_scipy",
]


# ----------------------------------------------------------------------
# type probes (duck-typed so no optional import is forced)
# ----------------------------------------------------------------------
def is_networkx_graph(obj) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("networkx") and hasattr(obj, "edges") and hasattr(obj, "nodes")


def is_pyg_data(obj) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("torch_geometric") and hasattr(obj, "edge_index")


def is_scipy_sparse(obj) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("scipy.sparse") and hasattr(obj, "tocoo")


def is_torch_tensor(obj) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("torch") and hasattr(obj, "detach")


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------
def to_graph(G, num_nodes: Optional[int] = None, reduce: str = "first") -> Graph:
    """Normalize any supported input into a canonical :class:`Graph`."""
    if isinstance(G, Graph):
        return G
    if is_pyg_data(G):
        return from_pyg(G, num_nodes=num_nodes, reduce=reduce)
    if is_networkx_graph(G):
        return from_networkx(G, reduce=reduce)
    if is_scipy_sparse(G):
        return from_scipy(G, reduce=reduce)
    if isinstance(G, tuple) and len(G) == 2:
        # (edge_index, num_nodes) is a common shorthand.
        return from_edge_index(G[0], num_nodes=G[1], reduce=reduce, source=G)
    if is_torch_tensor(G) or isinstance(G, np.ndarray) or isinstance(G, (list, tuple)):
        array = np.asarray(G.detach().cpu().numpy() if is_torch_tensor(G) else G)
        if array.ndim == 2 and array.shape[0] == array.shape[1] and array.shape[0] > 2:
            return from_dense(array, reduce=reduce)
        return from_edge_index(array, num_nodes=num_nodes, reduce=reduce, source=G)
    raise TypeError(
        f"Unsupported graph type: {type(G)!r}. Pass a scaffold.Graph, "
        "networkx.Graph, torch_geometric.data.Data, scipy.sparse matrix, or a "
        "(2, m) edge_index array."
    )


# ----------------------------------------------------------------------
# NetworkX
# ----------------------------------------------------------------------
def from_networkx(G, weight: str = "weight", reduce: str = "first") -> Graph:
    """Convert a NetworkX graph, preserving the original node labels."""
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    rows = list(G.edges(data=True))
    m = len(rows)
    src = np.empty(m, dtype=np.int64)
    dst = np.empty(m, dtype=np.int64)
    values = np.empty(m, dtype=np.float64)
    weighted = False
    for i, (u, v, data) in enumerate(rows):
        src[i] = index[u]
        dst[i] = index[v]
        if weight in data:
            weighted = True
            values[i] = float(data[weight])
        else:
            values[i] = 1.0
    return from_edge_index(
        np.stack((src, dst)),
        num_nodes=len(nodes),
        edge_weight=values if weighted else None,
        reduce=reduce,
        source=G,
        node_labels=nodes,
    )


def to_networkx(graph: Graph, edge_index=None, edge_weight=None, weight_key: str = "weight"):
    """Build a ``networkx.Graph`` for a selected edge set.

    Node labels from the original input are restored when the graph was created
    from NetworkX; otherwise nodes are ``0..num_nodes-1``.
    """
    import networkx as nx

    edge_index = graph.edge_index if edge_index is None else np.asarray(edge_index)
    labels = graph.node_labels
    H = nx.Graph()
    if labels is not None:
        H.add_nodes_from(labels)
    else:
        H.add_nodes_from(range(graph.num_nodes))

    src, dst = edge_index[0], edge_index[1]
    if labels is not None:
        pairs = ((labels[int(u)], labels[int(v)]) for u, v in zip(src, dst))
    else:
        pairs = ((int(u), int(v)) for u, v in zip(src, dst))

    if edge_weight is None:
        H.add_edges_from(pairs)
    else:
        values = np.asarray(edge_weight, dtype=float).reshape(-1)
        H.add_edges_from(
            (u, v, {weight_key: float(w)}) for (u, v), w in zip(pairs, values)
        )
    return H


# ----------------------------------------------------------------------
# PyTorch Geometric
# ----------------------------------------------------------------------
def from_pyg(data, num_nodes: Optional[int] = None, reduce: str = "first") -> Graph:
    """Convert a ``torch_geometric.data.Data`` object.

    ``edge_attr`` is used as the edge weight only when it is 1-D or has a single
    column; multi-dimensional attributes are ignored (use
    ``scaffold.feature_edge_weights`` to derive scalars from node features).
    """
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is None:
        attr = getattr(data, "edge_attr", None)
        if attr is not None and (attr.ndim == 1 or attr.shape[-1] == 1):
            edge_weight = attr.reshape(-1)
    if num_nodes is None:
        num_nodes = int(data.num_nodes)
    return from_edge_index(
        data.edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        reduce=reduce,
        source=data,
    )


def to_pyg(graph: Graph, edge_index, edge_weight=None, copy_from: Any = None):
    """Build a new ``Data`` object carrying the sparsified topology.

    Every non-edge attribute of ``copy_from`` (features, labels, masks) is
    forwarded unchanged; only ``edge_index`` / ``edge_weight`` / ``edge_attr``
    are replaced, so downstream training code keeps working.
    """
    import torch
    from torch_geometric.data import Data

    source = copy_from if copy_from is not None else graph.source
    edge_index_t = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)

    out = Data()
    if source is not None and is_pyg_data(source):
        for key, value in source:
            if key in ("edge_index", "edge_weight", "edge_attr"):
                continue
            out[key] = value
    out.edge_index = edge_index_t
    out.num_nodes = int(graph.num_nodes)
    if edge_weight is not None:
        out.edge_weight = torch.as_tensor(
            np.asarray(edge_weight, dtype=np.float32), dtype=torch.float
        )
    return out


# ----------------------------------------------------------------------
# SciPy sparse / dense
# ----------------------------------------------------------------------
def from_scipy(matrix, reduce: str = "first") -> Graph:
    """Convert any ``scipy.sparse`` adjacency matrix."""
    coo = matrix.tocoo()
    if coo.shape[0] != coo.shape[1]:
        raise ValueError(
            f"adjacency matrix must be square, got shape {coo.shape}"
        )
    data = np.asarray(coo.data, dtype=np.float64)
    weighted = data.size > 0 and not np.allclose(data, 1.0)
    return from_edge_index(
        np.stack((coo.row.astype(np.int64), coo.col.astype(np.int64))),
        num_nodes=int(coo.shape[0]),
        edge_weight=data if weighted else None,
        reduce=reduce,
        source=matrix,
    )


def to_scipy(graph: Graph, edge_index, edge_weight=None, fmt: str = "csr"):
    """Build a symmetric ``scipy.sparse`` adjacency for a selected edge set."""
    import scipy.sparse as sp

    edge_index = np.asarray(edge_index)
    src, dst = edge_index[0], edge_index[1]
    if edge_weight is None:
        values = np.ones(src.size, dtype=np.float64)
    else:
        values = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
    # The canonical edge list is one-directional, so symmetrize unless the
    # caller already handed us both directions.
    if src.size and np.all(src <= dst):
        rows = np.concatenate((src, dst))
        cols = np.concatenate((dst, src))
        values = np.concatenate((values, values))
    else:
        rows, cols = src, dst
    matrix = sp.coo_matrix(
        (values, (rows, cols)), shape=(graph.num_nodes, graph.num_nodes)
    )
    return matrix.asformat(fmt)


def from_dense(array, reduce: str = "first") -> Graph:
    """Convert a dense square adjacency matrix."""
    array = np.asarray(array)
    rows, cols = np.nonzero(array)
    values = array[rows, cols].astype(np.float64)
    weighted = values.size > 0 and not np.allclose(values, 1.0)
    return from_edge_index(
        np.stack((rows.astype(np.int64), cols.astype(np.int64))),
        num_nodes=int(array.shape[0]),
        edge_weight=values if weighted else None,
        reduce=reduce,
        source=array,
    )
