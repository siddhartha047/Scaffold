"""Visual walkthrough of SCAFFOLD on a grid graph.

Run::

    python examples/01_grid_demo.py --out docs/images

Why a grid? A 2-D lattice has no community structure, no hubs, and no
"important" edges -- every edge is equivalent by symmetry. So whatever pattern
survives sparsification is the *algorithm's* preference, not the graph's
structure. On Cora you cannot see that; here you can.

Produces five figures:

1. ``grid_backbones.png``  -- what each support backbone looks like
2. ``grid_methods.png``    -- the five algorithms at the same budget
3. ``grid_ratios.png``     -- SCAFFOLD-Fast as the budget tightens
4. ``grid_scores.png``     -- SCAFFOLD-Sample's per-edge weights
5. ``grid_coverage.png``   -- what per-epoch resampling covers over time
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import scaffold
from scaffold import viz
from scaffold.backbone import build_backbone

ROWS = COLS = 12
KEEP_RATIO = 0.72  # a 12x12 grid needs 143/264 = 0.542 just to stay connected
SEED = 0
METHODS_FIGURE_BACKBONE = "randst"


def figure_backbones(graph, positions, out_dir):
    """Every SCAFFOLD run starts from a spanning forest. They differ a lot."""
    import matplotlib.pyplot as plt

    names = ["fast-maxst", "maxst", "fast-randst", "randst", "spt", "glst"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names):
        mask = build_backbone(graph, name, seed=SEED)
        viz.draw_graph(
            graph,
            ax=ax,
            positions=positions,
            mask=mask,
            title=f"backbone='{name}'\n{int(mask.sum())} edges",
        )
    fig.suptitle(
        "Support backbones: the spanning forest SCAFFOLD grows from "
        "(default: fast-maxst)",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "grid_backbones.png")


def figure_methods(graph, positions, out_dir):
    """The five algorithms at one budget. Red = the backbone they share."""
    fig, results = viz.compare_methods(
        graph,
        keep_ratio=KEEP_RATIO,
        positions=positions,
        seed=SEED,
        backbone=METHODS_FIGURE_BACKBONE,
    )
    fig.suptitle(
        f"{ROWS}x{COLS} grid, keep_ratio={KEEP_RATIO:.0%} "
        f"(pale red = shared {METHODS_FIGURE_BACKBONE} backbone, "
        "bold blue = edges the method chose)",
        y=1.03,
    )
    fig.tight_layout()
    path = _save(fig, out_dir, "grid_methods.png")
    for name, result in results.items():
        print(f"  {name:7s} {result.summary()}  components={result.num_components()}")
    return path


def figure_ratios(graph, positions, out_dir):
    """Tightening the budget. Below delta_min the graph must fragment."""
    import matplotlib.pyplot as plt

    delta_min = (graph.num_nodes - 1) / graph.num_edges
    ratios = [0.95, 0.85, 0.75, 0.65, 0.45]
    fig, axes = plt.subplots(1, len(ratios), figsize=(3.2 * len(ratios), 3.6))
    for ax, ratio in zip(np.atleast_1d(axes), ratios):
        result = scaffold.fast(graph, keep_ratio=ratio, seed=SEED)
        components = result.num_components()
        note = "" if ratio >= delta_min else "  (below floor)"
        viz.draw_graph(
            graph,
            ax=ax,
            positions=positions,
            mask=result.mask,
            title=f"keep_ratio={ratio:.0%}{note}\n"
            f"{result.sparse_edges} edges, {components} comp.",
        )
    fig.suptitle(
        f"scaffold.fast as the budget tightens "
        f"(connectivity floor delta_min={delta_min:.1%})",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "grid_ratios.png")


def figure_scores(graph, positions, out_dir):
    """scaffold.sample returns weights, not a subgraph."""
    import matplotlib.pyplot as plt

    scores = scaffold.sample(graph, seed=SEED)
    probabilities = scores.inclusion_probabilities(keep_ratio=KEEP_RATIO)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    viz.draw_edge_scores(
        graph,
        scores.scores,
        ax=axes[0],
        positions=positions,
        title="pi: SCAFFOLD edge weight\n(ratio-independent, computed once)",
        label="pi",
    )
    viz.draw_edge_scores(
        graph,
        probabilities,
        ax=axes[1],
        positions=positions,
        title=f"inclusion probability at keep_ratio={KEEP_RATIO:.0%}\n"
        f"(sums to the budget; core p=1 in yellow)",
        label="p",
    )
    draw = scores.draw(keep_ratio=KEEP_RATIO, seed=SEED)
    viz.draw_graph(
        graph,
        ax=axes[2],
        positions=positions,
        mask=draw.mask,
        title=f"one draw\n{draw.sparse_edges} edges, "
        f"{draw.num_components()} component(s)",
    )
    fig.suptitle(
        "scaffold.sample: score every edge once, then draw a fresh graph per epoch",
        y=1.02,
    )
    fig.tight_layout()
    print(f"  deterministic core: {int((probabilities >= 1 - 1e-12).sum())} edges "
          f"always present out of {graph.num_edges}")
    return _save(fig, out_dir, "grid_scores.png")


def figure_coverage(graph, positions, out_dir):
    """Any one epoch sees keep_ratio of the graph. Many epochs see nearly all of it."""
    import matplotlib.pyplot as plt

    scores = scaffold.sample(graph, seed=SEED)
    seen = np.zeros(graph.num_edges, dtype=bool)
    snapshots = {}
    curve = []
    for epoch in range(1, 51):
        seen |= scores.draw(keep_ratio=KEEP_RATIO).mask
        curve.append(seen.mean())
        if epoch in (1, 3, 10):
            snapshots[epoch] = seen.copy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (epoch, mask) in zip(axes, snapshots.items()):
        viz.draw_graph(
            graph,
            ax=ax,
            positions=positions,
            mask=mask,
            title=f"union after {epoch} epoch(s)\n{mask.mean():.0%} of edges seen",
        )
    axes[-1].plot(range(1, 51), np.asarray(curve) * 100, color="#1f5fbf", linewidth=2)
    axes[-1].axhline(KEEP_RATIO * 100, color="#c8ccd4", linestyle="--", linewidth=1.2)
    axes[-1].text(
        26, KEEP_RATIO * 100 - 6, "a single fixed sparsifier", fontsize=8, color="#6b7280"
    )
    axes[-1].set_xlabel("epochs")
    axes[-1].set_ylabel("% of edges seen at least once")
    axes[-1].set_ylim(0, 104)
    axes[-1].spines[["top", "right"]].set_visible(False)
    axes[-1].set_title("coverage over training", fontsize=10)
    fig.suptitle(
        f"Per-epoch resampling at keep_ratio={KEEP_RATIO:.0%}: "
        "each view is small, the union is not",
        y=1.03,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "grid_coverage.png")


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/images", help="output directory")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--cols", type=int, default=COLS)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    graph = scaffold.grid_graph(args.rows, args.cols)
    positions = scaffold.grid_positions(args.rows, args.cols)
    print(
        f"grid {args.rows}x{args.cols}: {graph.num_nodes} nodes, "
        f"{graph.num_edges} undirected edges, "
        f"delta_min={(graph.num_nodes - 1) / graph.num_edges:.3f}"
    )

    print("\n[1/5] backbones")
    figure_backbones(graph, positions, args.out)
    print("\n[2/5] methods")
    figure_methods(graph, positions, args.out)
    print("\n[3/5] budgets")
    figure_ratios(graph, positions, args.out)
    print("\n[4/5] sample weights")
    figure_scores(graph, positions, args.out)
    print("\n[5/5] coverage")
    figure_coverage(graph, positions, args.out)
    print("\ndone.")


if __name__ == "__main__":
    main()
