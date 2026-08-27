"""Argument validation and budget arithmetic.

The edge budget is resolved in exactly one place so that all five algorithms
agree on what ``keep_ratio=0.2`` means, down to the rounding.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def resolve_budget(
    num_edges: int,
    keep_ratio: Optional[float] = None,
    num_edges_target: Optional[int] = None,
    default_ratio: float = 0.1,
) -> int:
    """Turn ``keep_ratio`` / ``num_edges`` into an exact undirected edge count.

    Exactly one of the two may be given. ``keep_ratio`` maps to
    ``ceil(keep_ratio * m)``, matching the paper's ``M = ceil(delta |E|)``, and
    the result is clamped to ``[0, m]``.
    """
    if keep_ratio is not None and num_edges_target is not None:
        raise ValueError("Specify either keep_ratio or num_edges, not both.")

    total = int(num_edges)
    if num_edges_target is not None:
        target = int(num_edges_target)
        if target < 0:
            raise ValueError(f"num_edges must be non-negative, got {num_edges_target}")
        return min(total, target)

    ratio = default_ratio if keep_ratio is None else float(keep_ratio)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"keep_ratio must lie in [0, 1], got {keep_ratio}")
    if total == 0:
        return 0
    return int(min(total, math.ceil(ratio * total - 1e-12)))


def connectivity_floor(num_nodes: int, num_edges: int, num_components: int) -> float:
    """Smallest ``keep_ratio`` at which the component count can be preserved.

    A spanning forest of a graph with ``c`` components has ``n - c`` edges, so
    below ``(n - c) / m`` no sparsifier of any kind can avoid fragmenting the
    graph. Reported as ``delta_min`` in the algorithms' metadata.
    """
    if num_edges <= 0:
        return 0.0
    return (int(num_nodes) - int(num_components)) / float(num_edges)


def validate_norm_order(value: float, name: str) -> float:
    """Norm orders must be finite and positive.

    ``inf`` (a max-on-path query) is rejected rather than silently approximated:
    the root-prefix formulation computes sums, and a max would need a different
    data structure.
    """
    value = float(value)
    if math.isinf(value):
        raise NotImplementedError(
            f"{name}=inf requires a max-on-path query rather than a root-prefix "
            "sum, which the tree-scoring kernel does not implement. Use a finite "
            "order (the default is 2.0)."
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_seed(seed) -> Optional[int]:
    """Normalize a seed argument to ``int`` or ``None``."""
    if seed is None:
        return None
    return int(seed)


def as_rng(seed) -> np.random.Generator:
    """A dedicated ``Generator``; never touches the global numpy RNG state."""
    return np.random.default_rng(seed)
