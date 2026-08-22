"""Plotting helpers, built around the grid-graph demo.

Requires matplotlib (``pip install "scaffold-sparse[viz]"``). Imported lazily
by ``scaffold.viz``, so it costs nothing unless you use it.

The point of drawing a *grid* is that a lattice has no interesting structure to
find -- every edge is equivalent by symmetry -- so what you see is purely the
sparsifier's preference. Kept edges are drawn solid, dropped edges as faint
dashes, so the retained skeleton reads at a glance.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .datasets import grid_positions
from .graph import Graph
from .result import ScaffoldResult

__all__ = [
    "layout",
    "draw_graph",
    "draw_result",
    "compare_methods",
    "draw_edge_scores",
]

_KEPT = "#1f5fbf"
_DROPPED = "#c8ccd4"
_BACKBONE = "#d1495b"


def layout(graph: Graph, positions=None) -> np.ndarray:
    """Node coordinates for ``graph``.

    Uses the lattice layout when the graph came from :func:`scaffold.grid_graph`,
    an explicit ``positions`` array when given, and a spring layout otherwise.
    """
    if positions is not None:
        return np.asarray(positions, dtype=float)
    shape = getattr(graph, "grid_shape", None)
    if shape is not None:
        return grid_positions(shape[0], shape[1])
    import networkx as nx

    from .adapters import to_networkx

    H = to_networkx(graph)
    if graph.node_labels is None:
        pos = nx.spring_layout(H, seed=0)
        return np.asarray([pos[i] for i in range(graph.num_nodes)], dtype=float)
    pos = nx.spring_layout(H, seed=0)
    return np.asarray([pos[label] for label in graph.node_labels], dtype=float)


def _segments(graph: Graph, positions: np.ndarray, mask=None):
    src, dst = graph.src, graph.dst
    if mask is not None:
        src, dst = src[mask], dst[mask]
    return np.stack((positions[src], positions[dst]), axis=1)


def draw_graph(
    graph: Graph,
    ax=None,
    positions=None,
    mask=None,
    highlight=None,
    title: Optional[str] = None,
    node_size: float = 14.0,
    linewidth: float = 1.6,
    show_dropped: bool = True,
):
    """Draw ``graph``, optionally showing which edges a ``mask`` keeps.

    When ``highlight`` (typically the backbone) is given, the kept edges are
    split in two: the backbone is drawn thin and muted underneath, and the
    edges the algorithm *chose to spend its budget on* are drawn bold on top.
    That split is the interesting one -- the backbone is the same for every
    method, so the bold edges are where they actually differ.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    positions = layout(graph, positions)

    if mask is None:
        kept = np.ones(graph.num_edges, dtype=bool)
    else:
        kept = np.ascontiguousarray(mask, dtype=bool)

    if show_dropped and (~kept).any():
        ax.add_collection(
            LineCollection(
                _segments(graph, positions, ~kept),
                colors=_DROPPED,
                linewidths=linewidth * 0.6,
                linestyles=(0, (2, 2)),
                zorder=1,
            )
        )

    if highlight is None:
        base, added = None, kept
    else:
        base = np.ascontiguousarray(highlight, dtype=bool) & kept
        added = kept & ~base

    if base is not None and base.any():
        ax.add_collection(
            LineCollection(
                _segments(graph, positions, base),
                colors=_BACKBONE,
                linewidths=linewidth * 0.85,
                alpha=0.55,
                zorder=2,
            )
        )
    if added.any():
        ax.add_collection(
            LineCollection(
                _segments(graph, positions, added),
                colors=_KEPT,
                linewidths=linewidth * (1.45 if base is not None else 1.0),
                zorder=3,
            )
        )
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=node_size, c="#22262e", zorder=4, linewidths=0,
    )
    _finish(ax, positions, title)
    return ax


def draw_result(result: ScaffoldResult, ax=None, positions=None, title=None, **kwargs):
    """Draw a :class:`~scaffold.result.ScaffoldResult` over its input graph."""
    if title is None:
        title = (
            f"scaffold.{result.method}  "
            f"{result.sparse_edges}/{result.original_edges} edges "
            f"({result.keep_ratio:.0%})"
        )
    return draw_graph(
        result.graph, ax=ax, positions=positions, mask=result.mask, title=title, **kwargs
    )


def compare_methods(
    graph: Graph,
    keep_ratio: float = 0.5,
    methods: Sequence[str] = ("exact", "heap", "fast", "sample"),
    positions=None,
    seed=0,
    show_backbone: bool = True,
    figsize=None,
    **kwargs,
):
    """Side-by-side panel: the input graph, then one panel per method.

    Returns ``(figure, results)`` where ``results`` maps method name to its
    :class:`~scaffold.result.ScaffoldResult`.
    """
    import matplotlib.pyplot as plt

    from .api import sparsify
    from .backbone import DEFAULT_BACKBONE, build_backbone
    from .utils.validation import resolve_budget

    positions = layout(graph, positions)
    panels = 1 + len(methods)
    figsize = figsize or (3.4 * panels, 3.8)
    fig, axes = plt.subplots(1, panels, figsize=figsize)
    axes = np.atleast_1d(axes)

    draw_graph(
        graph,
        ax=axes[0],
        positions=positions,
        title=f"input\n{graph.num_edges} edges",
        **kwargs,
    )

    # Recomputed rather than carried in metadata: a mask over every edge is
    # cheap here and unwelcome baggage on a hundred-million-edge graph.
    highlight = None
    if show_backbone:
        highlight = build_backbone(
            graph,
            DEFAULT_BACKBONE,
            max_edges=resolve_budget(graph.num_edges, keep_ratio),
            seed=seed,
        )

    results = {}
    for ax, method in zip(axes[1:], methods):
        result = sparsify(graph, method=method, keep_ratio=keep_ratio, seed=seed)
        if method == "sample":
            result = result.draw(keep_ratio=keep_ratio, seed=seed)
        results[method] = result
        components = result.num_components()
        draw_graph(
            result.graph,
            ax=ax,
            positions=positions,
            mask=result.mask,
            highlight=highlight,
            title=(
                f"scaffold.{method}\n{result.sparse_edges} edges "
                f"({result.keep_ratio:.0%}), {components} comp."
            ),
            **kwargs,
        )
    fig.tight_layout()
    return fig, results


def draw_edge_scores(
    graph: Graph,
    scores,
    ax=None,
    positions=None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    linewidth: float = 2.4,
    colorbar: bool = True,
    label: str = "edge weight",
):
    """Colour every edge by a per-edge score -- the natural view of ``scaffold.sample``.

    Infinite scores (mandatory, cross-component edges) are clamped to the
    finite maximum so one bridge does not flatten the whole colour scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4))
    positions = layout(graph, positions)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size != graph.num_edges:
        raise ValueError(
            f"expected one score per undirected edge ({graph.num_edges}), "
            f"got {values.size}"
        )
    finite = values[np.isfinite(values)]
    ceiling = float(finite.max()) if finite.size else 1.0
    values = np.where(np.isfinite(values), values, ceiling)

    collection = LineCollection(
        _segments(graph, positions),
        array=values,
        cmap=cmap,
        linewidths=linewidth,
        zorder=2,
    )
    ax.add_collection(collection)
    ax.scatter(
        positions[:, 0], positions[:, 1], s=12, c="#22262e", zorder=3, linewidths=0
    )
    if colorbar:
        ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label=label)
    _finish(ax, positions, title)
    return ax


def _finish(ax, positions, title):
    pad = 0.05 * max(1.0, float(np.ptp(positions)))
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
