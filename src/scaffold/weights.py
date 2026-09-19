"""Feature-derived weights on existing edges, with bounded temporary memory.

The similarity defaults match the research Scaffold feature-weight formulas.
No all-pairs node matrix is constructed. scikit-learn is an optional, lazy
backend for additional distances; the common metrics use NumPy/SciPy only.
"""

from __future__ import annotations

import operator

import numpy as np

from .graph import normalize_graph

__all__ = ["feature_edge_weights", "with_feature_weights"]

_BUILTINS = {"uniform", "cosine", "euclidean", "sqeuclidean", "manhattan", "chebyshev", "dot"}
_ALIASES = {"cosine_similarity": "cosine", "l2": "euclidean", "l1": "manhattan",
            "cityblock": "manhattan", "infinity": "chebyshev"}


def _prepare_features(G, features, allow_missing=False):
    from scipy import sparse

    if features is None:
        features = getattr(G, "x", None)
    if features is None:
        features = getattr(getattr(G, "source", None), "x", None)
    if features is None:
        if allow_missing:
            return normalize_graph(G), None
        raise ValueError("Node features are required; pass features= or a PyG Data with x.")
    if hasattr(features, "detach"):
        features = features.detach().cpu()
        if str(features.dtype) == "torch.bfloat16":
            features = features.float()
        features = features.numpy()
    if sparse.issparse(features):
        features = features.tocsr()
    else:
        features = np.asarray(features)
        if features.ndim == 1:
            features = features[:, None]
    if features.ndim != 2 or features.shape[1] == 0 or features.dtype.kind not in "biuf":
        raise ValueError("features must be a real numeric (num_nodes, num_features) matrix")
    graph = normalize_graph(G, num_nodes=features.shape[0])
    if features.shape[0] != graph.num_nodes:
        raise ValueError("features must have one row per graph node, in normalized node order")
    return graph, features


def _sum_rows(array):
    return np.asarray(array.sum(axis=1)).reshape(-1)


def _paired_builtin(left, right, metric, sparse):
    if metric == "cosine":
        # Separate norm clamps reproduce torch cosine_similarity(eps=1e-12),
        # including zero vectors (cosine 0 -> shifted similarity 0.5).
        if sparse:
            dot = _sum_rows(left.multiply(right))
            lnorm = np.sqrt(_sum_rows(left.multiply(left)))
            rnorm = np.sqrt(_sum_rows(right.multiply(right)))
            return np.clip((dot / np.maximum(lnorm, 1e-12)) / np.maximum(rnorm, 1e-12), -1, 1)
        lnorm = np.linalg.norm(left, axis=1)
        rnorm = np.linalg.norm(right, axis=1)
        left = left / np.maximum(lnorm, 1e-12)[:, None]
        right = right / np.maximum(rnorm, 1e-12)[:, None]
        return np.clip(np.einsum("ij,ij->i", left, right), -1, 1)
    if metric == "dot":
        return _sum_rows(left.multiply(right)) if sparse else np.einsum("ij,ij->i", left, right)
    difference = left - right
    if metric in ("euclidean", "sqeuclidean"):
        squared = _sum_rows(difference.multiply(difference)) if sparse else np.einsum("ij,ij->i", difference, difference)
        return np.sqrt(squared) if metric == "euclidean" else squared
    if metric == "manhattan":
        return _sum_rows(abs(difference))
    maximum = abs(difference).max(axis=1)
    return np.asarray(maximum.toarray() if sparse else maximum).reshape(-1)


def feature_edge_weights(
    G, features=None, metric="cosine", *, kind="similarity", batch_size=8192,
    metric_kwargs=None, min_weight=1e-12,
):
    """Return float64 weights in the normalized graph's canonical edge order.

    ``kind='similarity'`` matches research weighting: shifted cosine
    ``(1+cos)/2``, Euclidean affinity ``1/(1+d)``, and globally min-max-scaled
    dot products. Other distances also become ``1/(1+d)``. With
    ``kind='distance'``, use raw distances (cosine becomes ``1-cos``).
    Values are floored at ``min_weight`` so shortest-path/low-stretch backbones
    can use the result even for identical or opposite feature vectors.

    Common metrics need only NumPy/SciPy. Additional sklearn DistanceMetric
    names (e.g. ``minkowski``, ``canberra``, ``mahalanobis``) require the
    ``metrics`` extra. Pass their parameters through ``metric_kwargs``.
    A callable must be a symmetric, finite, nonnegative distance function.
    Additional metrics/callables evaluate one edge pair at a time and can be
    slower; no n-by-n or batch-by-batch distance matrix is allocated.

    Dense NumPy/Torch and SciPy sparse features are accepted. Feature rows
    follow ``normalize_graph(G).node_labels`` for a NetworkX input. PyG ``x``
    is inferred when features are omitted. The input graph is not mutated.
    """
    from scipy import sparse

    if kind not in ("similarity", "distance"):
        raise ValueError("kind must be 'similarity' or 'distance'")
    if isinstance(batch_size, (bool, np.bool_)):
        raise ValueError("batch_size must be a positive integer")
    try:
        batch_size = operator.index(batch_size)
    except TypeError as exc:
        raise ValueError("batch_size must be a positive integer") from exc
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    min_weight = float(min_weight)
    if not np.isfinite(min_weight) or not 0 < min_weight <= 1:
        raise ValueError("min_weight must be finite and in (0, 1]")
    name = metric.lower().strip() if isinstance(metric, str) else None
    name = _ALIASES.get(name, name)
    if name is None and not callable(metric):
        raise ValueError("metric must be a distance name or callable")
    if name == "dot" and kind != "similarity":
        raise ValueError("dot is a similarity; choose a distance metric for kind='distance'")
    params = dict(metric_kwargs or {})
    if name in _BUILTINS and params:
        raise ValueError(f"metric_kwargs are not supported for built-in metric {name!r}")
    graph, X = _prepare_features(G, features, allow_missing=name == "uniform")
    values = np.empty(graph.num_edges, dtype=np.float64)
    if name == "uniform":
        values.fill(1.0)
        return values
    is_sparse = sparse.issparse(X)
    # Cap dense temporaries by feature dimension, including finite checks.
    # The n-by-d input itself is not copied to float64.
    chunk_size = min(batch_size, max(1, (32 * 1024 * 1024) // (8 * X.shape[1])))
    for start in range(0, X.shape[0], chunk_size):
        part = X[start:start + chunk_size]
        data = part.data if is_sparse else part
        if not np.isfinite(data).all():
            raise ValueError("features must contain only finite values")
    backend = None
    if name not in _BUILTINS and name is not None:
        try:
            from sklearn.metrics import DistanceMetric
        except ImportError as exc:
            raise ImportError("This distance requires scikit-learn; install scaffold-sparse[metrics].") from exc
        backend = DistanceMetric.get_metric(name, **params)

    # Roughly 32 MiB per gathered endpoint block (at least one feature row).
    for start in range(0, graph.num_edges, chunk_size):
        stop = min(graph.num_edges, start + chunk_size)
        left = X[graph.src[start:stop]].astype(np.float64, copy=False)
        right = X[graph.dst[start:stop]].astype(np.float64, copy=False)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if name in _BUILTINS:
                part = _paired_builtin(left, right, name, is_sparse)
            else:
                if is_sparse:
                    left, right = left.toarray(), right.toarray()
                part = np.empty(stop - start, dtype=np.float64)
                for i in range(stop - start):
                    part[i] = (backend.pairwise(left[i:i + 1], right[i:i + 1])[0, 0]
                               if backend is not None else metric(left[i], right[i], **params))
        if not np.isfinite(part).all():
            raise ValueError("Feature weights/distances are non-finite; rescale features or check the metric.")
        if name == "cosine":
            part = (1 + part) / 2 if kind == "similarity" else 1 - part
        elif name != "dot":
            if np.any(part < 0):
                raise ValueError("Distance metrics must return nonnegative values")
            if kind == "similarity":
                part = 1 / (1 + part)
        values[start:stop] = part
    if name == "dot" and values.size:
        low, high = values.min(), values.max()
        span = high - low
        if not np.isfinite(span):
            raise ValueError("Dot-product range overflowed; rescale features")
        values = np.ones_like(values) if span <= 1e-12 else (values - low) / span
    return np.maximum(values, min_weight)


def with_feature_weights(G, features=None, metric="cosine", **kwargs):
    """Return a new canonical Graph weighted from node features on its edges.

    Accepts the same metric/kind/batching options as ``feature_edge_weights``.
    Preserves topology, isolated nodes, node labels, and the PyG source object
    for round-trip conversion. Existing edge weights are explicitly replaced.
    """
    uniform = isinstance(metric, str) and metric.lower().strip() == "uniform"
    graph, features = _prepare_features(G, features, allow_missing=uniform)
    return graph.with_weight(feature_edge_weights(graph, features, metric, **kwargs))
