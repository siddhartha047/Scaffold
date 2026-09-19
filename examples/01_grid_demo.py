"""Visual walkthrough of SCAFFOLD on a grid graph.

Run::

    python examples/01_grid_demo.py --out docs/images

Why a grid? Its regular layout makes retained paths and omitted edges easy
to compare, without the visual clutter of a large irregular graph.
The method and sampling demos share a seeded RandSF backbone. The opening
budget demo uses Scaffold-Greedy with RandSF; the backbone comparison shows
each named construction separately.

Produces five grid figures, backbone/growth/resampling GIFs, and measurements
in ``grid_backbones.json``, ``grid_ratios.json``, and ``grid_coverage.json``:

1. ``grid_backbones.png``  -- what each support backbone looks like
2. ``grid_methods.png``    -- the five algorithms at the same budget
3. ``grid_ratios.png/gif`` -- four budgets and RandSF growing one edge at a time
4. ``grid_scores.png``     -- SCAFFOLD-Sample's per-edge weights
5. ``grid_coverage.png``   -- what per-epoch resampling covers over time
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import scaffold
from scaffold import viz
from scaffold.backbone import (
    DEFAULT_LLST_MAX_INPUT_EDGES,
    build_backbone,
    canonical_backbone_name,
)
from scaffold.scoring import tree_scores

ROWS = COLS = 12
KEEP_RATIO = 0.72  # a 12x12 grid needs 143/264 = 0.542 just to stay connected
SEED = 0
DEMO_BACKBONE = "RandSF"
SAMPLE_DEMO_BACKBONE = "fixed-RandSF"
BACKBONE_LABELS = {
    "fast-maxst": "Fast-MaxSF",
    "maxst": "MaxSF",
    "fast-randst": "Fast-RandSF",
    "randst": "RandSF",
    "spt": "SPF",
    "glst": "GLSF",
    "llst": "LLSF",
}
BACKBONE_NAMES = {
    "fast-maxst": "Fast maximum-weight\nspanning forest",
    "maxst": "Maximum-weight\nspanning forest",
    "fast-randst": "Fast randomized\nspanning forest",
    "randst": "Random spanning forest",
    "spt": "Shortest-path forest",
    "glst": "Greedy low-stretch forest",
    "llst": "Local-search\nlow-stretch forest",
}


def figure_backbones(graph, positions, out_dir, quick=False):
    """Every SCAFFOLD run starts from a spanning forest. They differ a lot."""
    names = ["fast-maxst", "maxst", "fast-randst", "randst", "spt", "glst", "llst"]
    llst_options = {"init_support": "GLSF", "max_passes": 10}
    if quick:
        llst_options.update(max_passes=2, candidate_sample_size=16, cycle_sample_size=4)
    measurements = {}
    masks = {}
    for name in names:
        options = llst_options if name == "llst" else {}
        # Use forest notation at the API boundary, retaining historical JSON keys.
        print(f"  building {BACKBONE_LABELS[name]}...", flush=True)
        mask = build_backbone(graph, BACKBONE_LABELS[name], seed=SEED, **options)
        stats = tree_scores(graph.num_nodes, graph.src, graph.dst, mask)
        stretch = float(stats["total_stretch"])
        measurements[name] = {
            "options": options,
            "forest_edges": int(mask.sum()),
            "components": graph.num_nodes - int(mask.sum()),
            "omitted_edge_stretch": stretch,
            "retained_edge_ids": np.flatnonzero(mask).tolist(),
        }
        masks[name] = mask
        print(f"    {int(mask.sum())} forest edges; omitted-edge stretch {stretch:.0f}")
    path = _draw_backbones(graph, positions, out_dir, masks, measurements)
    _animate_backbones(graph, positions, out_dir, masks, measurements)
    metrics_path = os.path.join(out_dir, "grid_backbones.json")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "seed": SEED,
                "backbones": measurements,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"  wrote {metrics_path}")
    return path


def _backbone_panel(graph, positions, ax, label, detail, mask=None, accent=False):
    """Keep method names large and separate from the compact metric line."""
    viz.draw_graph(graph, ax=ax, positions=positions, mask=mask, node_size=12)
    _panel_title(ax, label, detail, accent=accent)


def _panel_title(ax, label, detail, accent=False):
    ax.set_title(
        label,
        fontsize=16,
        fontweight="semibold",
        pad=29,
        color="#1f5fbf" if accent else "#22262e",
    )
    ax.text(
        0.5,
        1.015,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#4b5563",
    )


def _count_label(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _grid_heading(fig, title, subtitle, footer=""):
    fig.suptitle(title, fontsize=22, fontweight="semibold", y=0.98)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=12, color="#4b5563")
    if footer:
        fig.text(0.5, 0.025, footer, ha="center", fontsize=11, color="#4b5563")
    fig.subplots_adjust(
        left=0.055, right=0.97, bottom=0.11, top=0.83, wspace=0.23, hspace=0.52
    )


def _draw_backbones(graph, positions, out_dir, masks, measurements):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 8.8))
    axes = axes.ravel()
    _backbone_panel(
        graph,
        positions,
        axes[0],
        "Input grid",
        f"{graph.num_nodes} nodes · {graph.num_edges} edges",
    )
    for ax, (name, mask) in zip(axes[1:], masks.items()):
        stats = measurements[name]
        label = BACKBONE_LABELS[name]
        if name == "fast-maxst":
            label += " · default"
        _backbone_panel(
            graph,
            positions,
            ax,
            label,
            f"{stats['forest_edges']} edges · stretch {stats['omitted_edge_stretch']:,.0f}",
            mask=mask,
            accent=name == "llst",
        )
        ax.text(
            0.5,
            -0.055,
            BACKBONE_NAMES[name],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            linespacing=1.15,
            color="#1f5fbf" if name == "llst" else "#4b5563",
        )
    reduction = 1 - measurements["llst"]["omitted_edge_stretch"] / max(
        1.0, measurements["glst"]["omitted_edge_stretch"]
    )
    fig.suptitle("Support backbones", fontsize=22, fontweight="semibold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Unweighted graph. Same forest size. Different supporting paths.",
        ha="center",
        fontsize=13,
        color="#4b5563",
    )
    fig.text(
        0.5,
        0.047,
        f"LLSF refines GLSF: {reduction:.1%} less omitted-edge stretch "
        "with the same edge count.",
        ha="center",
        fontsize=12,
        color="#1f5fbf",
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.015,
        "Blue: retained. Gray dashes: omitted. Stretch: total omitted-edge detour length. "
        f"LLSF: {_llst_search_label(measurements)}; seed {SEED}.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, bottom=0.15, top=0.83, wspace=0.08, hspace=0.74
    )
    return _save(fig, out_dir, "grid_backbones.png")


def _animate_backbones(graph, positions, out_dir, masks, measurements):
    """Cycle through the measured forests, reusing the PNG's actual outputs.

    These are completed-backbone comparisons, not algorithm insertion steps.
    """
    import matplotlib.pyplot as plt

    explanations = {
        "fast-maxst": "Default: bucketed weight ordering, followed by a union-find scan.",
        "maxst": "Exact maximum-weight forest; tied weights give the same forest here.",
        "fast-randst": "A seeded coprime-stride edge scan builds a randomized forest.",
        "randst": "A full random edge permutation gives a different randomized forest.",
        "spt": "A breadth-first forest keeps short paths from its starting roots.",
        "glst": "GLSF grows a forest using projected stretch and cut size.",
        "llst": "LLSF refines GLSF through improving cycle swaps; the forest size stays fixed.",
    }
    frames = []
    for index, (name, mask) in enumerate(masks.items(), start=1):
        stats = measurements[name]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.6), dpi=120)
        _backbone_panel(
            graph,
            positions,
            axes[0],
            "Unweighted input",
            f"{graph.num_nodes} nodes · {graph.num_edges} edges",
        )
        _backbone_panel(
            graph,
            positions,
            axes[1],
            BACKBONE_LABELS[name],
            f"{stats['forest_edges']} edges · {_count_label(stats['components'], 'component')} · "
            f"stretch {stats['omitted_edge_stretch']:,.0f}",
            mask=mask,
            accent=True,
        )
        fig.suptitle(
            "Support backbone comparison", fontsize=20, fontweight="semibold", y=0.97
        )
        fig.text(
            0.5,
            0.91,
            f"{index} / {len(masks)}     ·     {BACKBONE_NAMES[name].replace(chr(10), ' ')}",
            ha="center",
            fontsize=12,
            color="#4b5563",
        )
        fig.text(0.5, 0.105, explanations[name], ha="center", fontsize=12)
        if name == "llst":
            initial = measurements["glst"]["omitted_edge_stretch"]
            final = stats["omitted_edge_stretch"]
            note = (
                f"GLSF → LLSF: {initial:,.0f} → {final:,.0f} stretch "
                f"({1 - final / max(1.0, initial):.1%} reduction); "
                f"{_llst_search_label(measurements)}."
            )
        else:
            note = (
                "Blue: retained edges. Gray dashes: omitted edges. Lower stretch is better."
            )
        fig.text(0.5, 0.047, note, ha="center", fontsize=11, color="#1f5fbf")
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.19, top=0.75, wspace=0.2)
        frames.append(_gif_frame(fig))
        plt.close(fig)

    return _save_gif(frames, out_dir, "grid_backbones.gif", duration=2200, final_hold=4500)


def _llst_search_label(measurements):
    options = measurements["llst"]["options"]
    sampled = any(options.get(key, 0) for key in (
        "candidate_sample_size", "cycle_sample_size", "eval_sample_size"
    ))
    mode = "sampled" if sampled else "exhaustive"
    return f"up to {options.get('max_passes', 10)} {mode} swaps"


def _gif_frame(fig):
    from PIL import Image

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def _save_gif(frames, out_dir, name, duration=500, final_hold=2500,
              first_hold=None, disposal=2):
    # A shared palette prevents color flicker across frames.
    first = frames[0] if frames[0].mode == "P" else frames[0].quantize(colors=128)
    palette = first.getpalette()
    frames = [first] + [
        frame if frame.mode == "P" and frame.getpalette() == palette
        else frame.convert("RGB").quantize(palette=first, dither=0)
        for frame in frames[1:]
    ]
    durations = [duration] * (len(frames) - 1) + [final_hold]
    if first_hold is not None and len(frames) > 1:
        durations[0] = first_hold
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    first.save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=disposal,
        **({"optimize": False} if disposal == 1 else {}),
    )
    print(f"  wrote {path}")
    return path


def figure_methods(graph, positions, out_dir):
    """The five algorithms at one budget. Red = the backbone they share."""
    fig, results = viz.compare_methods(
        graph,
        keep_ratio=KEEP_RATIO,
        positions=positions,
        seed=SEED,
        backbone=DEMO_BACKBONE,
        ncols=3,
        figsize=(11.4, 8.8),
    )
    _panel_title(
        fig.axes[0], "Input grid", f"{graph.num_nodes} nodes · {graph.num_edges} edges"
    )
    for ax, (name, result) in zip(fig.axes[1:], results.items()):
        _panel_title(
            ax,
            f"Scaffold-{name.capitalize()}",
            f"{result.sparse_edges} edges · {_count_label(result.num_components(), 'component')}",
        )
    _grid_heading(
        fig,
        "Five algorithms, one edge budget",
        f"Target retention {KEEP_RATIO:.0%} · shared {BACKBONE_LABELS[canonical_backbone_name(DEMO_BACKBONE)]} backbone · seed {SEED}",
        "Pale red: shared backbone. Bold blue: added edges. Gray dashes: omitted edges.",
    )
    path = _save(fig, out_dir, "grid_methods.png")
    for name, result in results.items():
        print(f"  {name:7s} {result.summary()}  components={result.num_components()}")
    return path


def figure_ratios(graph, positions, out_dir):
    """Four fixed budgets plus RandSF growing through actual Greedy insertions."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    backbone = build_backbone(graph, DEMO_BACKBONE, seed=SEED)
    forest_edges = int(backbone.sum())
    forest_fraction = forest_edges / graph.num_edges
    # Record one real run instead of repeatedly rescoring every earlier prefix.
    complete = scaffold.greedy(
        graph, num_edges=graph.num_edges, backbone=backbone,
        seed=SEED, return_trace=True,
    )
    additions = np.asarray(complete.metadata["added_edge_ids"], dtype=np.int64)
    assert len(additions) == graph.num_edges - forest_edges
    assert len(np.unique(additions)) == len(additions) and not backbone[additions].any()
    assert complete.mask.all()

    ratios = [0.85, 0.75, 0.65, 0.55]
    panels = [("Input grid", f"{graph.num_edges} edges", None, False)]
    measurements = []
    for ratio in ratios:
        budget = int(np.ceil(ratio * graph.num_edges - 1e-12))
        if budget >= forest_edges:
            mask = backbone.copy()
            mask[additions[:budget - forest_edges]] = True
            components = graph.num_nodes - forest_edges
        else:
            # Small --quick grids can put the 55% panel below the forest budget.
            partial = scaffold.greedy(graph, num_edges=budget, backbone=backbone, seed=SEED)
            mask, components = partial.mask, partial.num_components()
        detail = "connected" if components == 1 else _count_label(components, "component")
        panels.append((f"{ratio:.0%} retained", f"{budget} edges · {detail}",
                       mask, budget < forest_edges))
        measurements.append({"target_ratio": ratio, "edges": budget,
                             "components": components,
                             "retained_edge_ids": np.flatnonzero(mask).tolist()})
        print(f"  {ratio:.0%}: {budget} edges, {components} components", flush=True)
    panels.append(("Random spanning forest",
                   f"{forest_edges} edges · {forest_fraction:.1%} retained",
                   backbone.copy(), False))

    fig = plt.figure(figsize=(12, 8.4), dpi=120, facecolor="white")
    fig.text(.035, .955, "Scaffold", fontsize=29, fontweight="bold", color="#1f5fbf")
    fig.text(.035, .915, "Reduce the edge budget. Keep every node.",
             fontsize=14, color="#4b5563")
    fig.text(.965, .954, f"{graph.num_nodes} nodes · Scaffold-Greedy · RandSF backbone",
             ha="right", fontsize=11, color="#4b5563")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#1f5fbf", lw=2, label="Retained"),
            Line2D([0], [0], color="#c8ccd4", lw=1.4,
                   linestyle=(0, (2, 2)), label="Omitted"),
            Line2D([0], [0], color="#e07829", lw=2.5, label="Added this step"),
        ],
        loc="upper right", bbox_to_anchor=(.972, .938), frameon=False,
        ncol=3, fontsize=10, handlelength=2, columnspacing=1.2,
    )

    for index, (title, detail, mask, below_floor) in enumerate(panels):
        row, col = divmod(index, 3)
        left, bottom = .028 + .322 * col, .493 - .417 * row
        width, height = .300, .390
        fig.add_artist(FancyBboxPatch(
            (left, bottom), width, height, transform=fig.transFigure,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor="#f4f7fc" if index in (0, 5) else "#fafbfc",
            edgecolor="#b8ccec" if index == 5 else "#dce3ed",
            linewidth=1.1 if index == 5 else .8, zorder=0,
        ))
        color = "#b45309" if below_floor else "#1f5fbf"
        title_artist = fig.text(left + width / 2, bottom + .352, title,
                               ha="center", fontsize=18 if index == 5 else 20,
                               fontweight="semibold", color=color)
        detail_artist = fig.text(left + width / 2, bottom + .320, detail,
                                ha="center", fontsize=11,
                                color="#b45309" if below_floor else "#4b5563")
        ax = fig.add_axes([left + .012, bottom + .010, width - .024, .299])
        viz.draw_graph(graph, positions=positions, ax=ax, mask=mask,
                       node_size=10, linewidth=1.7)
        if index == 5:
            growth_ax, growth_title, growth_detail = ax, title_artist, detail_artist
            growth_box = [left, bottom, left + width, bottom + height]

    fig.text(.5, .024,
             f"RandSF starts at {forest_edges} edges ({forest_fraction:.1%}). "
             "Greedy adds one edge per step in the bottom-right panel.",
             ha="center", fontsize=11, color="#4b5563")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "grid_ratios.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  wrote {path}", flush=True)

    # Only these artists change; the five reference panels stay pixel-identical.
    for collection in list(growth_ax.collections):
        if isinstance(collection, LineCollection):
            collection.remove()
    positions_array = np.asarray(positions)
    segments = np.stack((positions_array[graph.src], positions_array[graph.dst]), axis=1)
    omitted = LineCollection(segments[~backbone], colors="#c8ccd4", linewidths=1.02,
                             linestyles=(0, (2, 2)), zorder=1)
    retained = LineCollection(segments[backbone], colors="#1f5fbf", linewidths=1.7, zorder=3)
    newest = LineCollection([], colors="#e07829", linewidths=2.7, zorder=3.5)
    for collection in (omitted, retained, newest):
        growth_ax.add_collection(collection)
    first = _gif_frame(fig).quantize(colors=256)
    frames = [first]
    pixel_width, pixel_height = first.size
    growth_pixels = (
        int(np.floor(growth_box[0] * pixel_width)),
        int(np.floor((1 - growth_box[3]) * pixel_height)),
        int(np.ceil(growth_box[2] * pixel_width)),
        int(np.ceil((1 - growth_box[1]) * pixel_height)),
    )
    mask = backbone.copy()
    for step, edge_id in enumerate(additions, 1):
        mask[edge_id] = True
        omitted.set_segments(segments[~mask])
        retained.set_segments(segments[mask])
        newest.set_segments(segments[[edge_id]])
        growth_title.set_text("Full graph" if step == len(additions) else "Greedy growth")
        count = forest_edges + step
        growth_detail.set_text(f"Step {step} · {count} edges · {count / graph.num_edges:.1%}")
        # Quantize immediately to keep 122 frames comfortably below RGB memory cost.
        rendered = _gif_frame(fig).quantize(palette=first, dither=0)
        # Palette lookup can slightly alter static pixels; freeze them explicitly.
        frame = first.copy()
        frame.paste(rendered.crop(growth_pixels), growth_pixels)
        frames.append(frame)
    plt.close(fig)
    _save_gif(frames, out_dir, "grid_ratios.gif", duration=100,
              first_hold=1600, final_hold=2400, disposal=1)

    data = {
        "method": "greedy", "backbone": canonical_backbone_name(DEMO_BACKBONE),
        "num_nodes": graph.num_nodes, "num_edges": graph.num_edges, "seed": SEED,
        "backbone_edges": forest_edges, "backbone_fraction": forest_fraction,
        "backbone_edge_ids": np.flatnonzero(backbone).tolist(),
        "added_edge_ids": additions.tolist(), "fixed_panels": measurements,
        "frame_edge_counts": list(range(forest_edges, graph.num_edges + 1)),
        "frame_count": len(frames), "gif_size": list(first.size),
        "animated_panel_figure_bounds": growth_box,
    }
    data_path = os.path.join(out_dir, "grid_ratios.json")
    with open(data_path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print(f"  wrote {data_path}: {len(additions)} single-edge additions", flush=True)
    return path


def figure_scores(graph, positions, out_dir):
    """scaffold.sample returns weights, not a subgraph."""
    import matplotlib.pyplot as plt

    scores = scaffold.sample(graph, backbone=SAMPLE_DEMO_BACKBONE, seed=SEED)
    probabilities = scores.inclusion_probabilities(keep_ratio=KEEP_RATIO)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2))
    axes = axes.ravel()
    _backbone_panel(
        graph,
        positions,
        axes[0],
        "Input grid",
        f"{graph.num_nodes} nodes · {graph.num_edges} edges",
    )
    viz.draw_edge_scores(
        graph,
        scores.scores,
        ax=axes[1],
        positions=positions,
        colorbar=False,
    )
    _panel_title(axes[1], "Sampling weights", "Computed once, reused across budgets")
    viz.draw_edge_scores(
        graph,
        probabilities,
        ax=axes[2],
        positions=positions,
        colorbar=False,
    )
    _panel_title(
        axes[2],
        "Inclusion probabilities",
        f"Target {KEEP_RATIO:.0%} · yellow edges have p = 1",
    )
    for ax, label in [(axes[1], "weight"), (axes[2], "probability")]:
        color_ax = ax.inset_axes([0.1, -0.075, 0.8, 0.035])
        bar = fig.colorbar(ax.collections[0], cax=color_ax, orientation="horizontal")
        bar.set_label(label, fontsize=10, labelpad=1)
        bar.ax.tick_params(labelsize=9, pad=2)
    draw = scores.draw(keep_ratio=KEEP_RATIO, seed=SEED)
    _backbone_panel(
        graph,
        positions,
        axes[3],
        "One sparse draw",
        f"{draw.sparse_edges} edges · {_count_label(draw.num_components(), 'component')}",
        mask=draw.mask,
    )
    _grid_heading(
        fig,
        "Score once, sample repeatedly",
        "Scaffold-Sample · fixed RandSF backbone · exact-budget draws",
        "Weights and probabilities use separate color scales. Blue edges form one draw.",
    )
    print(
        f"  deterministic core: {int((probabilities >= 1 - 1e-12).sum())} edges "
        f"always present out of {graph.num_edges}"
    )
    return _save(fig, out_dir, "grid_scores.png")


def figure_coverage(graph, positions, out_dir):
    """Any one epoch sees keep_ratio of the graph. Many epochs see nearly all of it."""
    import matplotlib.pyplot as plt

    scores = scaffold.sample(graph, backbone=SAMPLE_DEMO_BACKBONE, seed=SEED)
    seen = np.zeros(graph.num_edges, dtype=bool)
    snapshots = {}
    curve = []
    draws = []
    unions = []
    components = []
    for epoch in range(1, 51):
        draw = scores.draw(keep_ratio=KEEP_RATIO)
        draws.append(draw.mask.copy())
        components.append(draw.num_components())
        seen |= draw.mask
        unions.append(seen.copy())
        curve.append(float(seen.mean()))
        if epoch in (1, 3, 10):
            snapshots[epoch] = seen.copy()

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2))
    axes = axes.ravel()
    for ax, (epoch, mask) in zip(axes, snapshots.items()):
        _backbone_panel(
            graph,
            positions,
            ax,
            f"After {_count_label(epoch, 'epoch')}",
            f"{int(mask.sum())} / {graph.num_edges} edges seen ({mask.mean():.1%})",
            mask=mask,
        )
    _coverage_curve(axes[-1], curve, curve[0])
    _grid_heading(
        fig,
        "Small views, growing coverage",
        f"Scaffold-Sample · RandSF backbone · target {KEEP_RATIO:.0%} per epoch · seed {SEED}",
        "Panels show accumulated coverage. Each individual draw still has the same edge budget.",
    )
    path = _save(fig, out_dir, "grid_coverage.png")
    _animate_coverage(graph, positions, out_dir, draws, unions, curve, components)
    with open(os.path.join(out_dir, "grid_coverage.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": SEED,
                "backbone": SAMPLE_DEMO_BACKBONE,
                "backbone_edge_ids": np.flatnonzero(scores.backbone).tolist(),
                "keep_ratio": KEEP_RATIO,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "epochs": [
                    {
                        "epoch": i + 1,
                        "retained_edge_ids": np.flatnonzero(mask).tolist(),
                        "components": components[i],
                        "coverage": curve[i],
                    }
                    for i, mask in enumerate(draws)
                ],
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        f"  coverage after 50 epochs: {curve[-1]:.1%}; "
        f"{int(draws[0].sum())} edges in every draw"
    )
    return path


def _coverage_curve(ax, curve, fixed_fraction):
    ax.plot(
        range(1, len(curve) + 1),
        np.asarray(curve) * 100,
        color="#1f5fbf",
        linewidth=2.5,
        label="resampled supports",
    )
    ax.axhline(
        fixed_fraction * 100,
        color="#9ca3af",
        linestyle="--",
        linewidth=1.5,
        label="one fixed support",
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Edges seen (%)", fontsize=11)
    ax.scatter([len(curve)], [curve[-1] * 100], s=20, color="#1f5fbf", zorder=3)
    ax.set_xlim(0.5, 50.5)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _panel_title(
        ax,
        "Coverage over time",
        f"{curve[-1]:.1%} seen after {_count_label(len(curve), 'epoch')}",
    )


def _animate_coverage(graph, positions, out_dir, draws, unions, curve, components):
    """A 2x2 view of real draws, their cumulative union, and the coverage curve."""
    import matplotlib.pyplot as plt

    frames = []
    # Show every early draw, then sample the later epochs to keep the GIF small.
    indices = list(range(10)) + [14, 19, 29, 39, 49]
    for i in indices:
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2), dpi=110)
        axes = axes.ravel()
        _backbone_panel(graph, positions, axes[0], "Input grid", f"{graph.num_edges} edges")
        _backbone_panel(
            graph,
            positions,
            axes[1],
            f"Epoch {i + 1}: current view",
            f"{int(draws[i].sum())} edges · {_count_label(components[i], 'component')}",
            mask=draws[i],
            accent=True,
        )
        _backbone_panel(
            graph,
            positions,
            axes[2],
            "Edges seen so far",
            f"{int(unions[i].sum())} / {graph.num_edges} edges ({curve[i]:.1%})",
            mask=unions[i],
        )
        _coverage_curve(axes[3], curve[: i + 1], curve[0])
        _grid_heading(
            fig,
            "A fresh sparse view each epoch",
            f"Scaffold-Sample · RandSF backbone · target retention {KEEP_RATIO:.0%} · seed {SEED}",
            "Top right: one sparse draw. Bottom left: its union with all preceding draws.",
        )
        frames.append(_gif_frame(fig))
        plt.close(fig)
    return _save_gif(frames, out_dir, "grid_coverage.gif", duration=750, final_hold=3000)


def _save(fig, out_dir, name, dpi=150):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/images", help="output directory")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument(
        "--quick", action="store_true",
        help="Use a 6x6 grid unless dimensions are given, and two sampled LLSF swaps.",
    )
    parser.add_argument(
        "--only",
        choices=("backbones", "methods", "ratios", "scores", "coverage"),
        help="regenerate just one figure",
    )
    args = parser.parse_args()
    args.rows = args.rows if args.rows is not None else (6 if args.quick else ROWS)
    args.cols = args.cols if args.cols is not None else (6 if args.quick else COLS)

    graph = scaffold.grid_graph(args.rows, args.cols)
    if args.only in (None, "backbones") and graph.num_edges > DEFAULT_LLST_MAX_INPUT_EDGES:
        parser.error(
            f"The LLSF backbone demo is limited to {DEFAULT_LLST_MAX_INPUT_EDGES:,} "
            f"undirected input edges; this grid has {graph.num_edges:,}. "
            "Exhaustive local search can take many minutes or longer. Use a "
            "smaller grid, or select a RandSF demo with --only methods, "
            "--only ratios, --only scores, or --only coverage."
        )

    import matplotlib

    matplotlib.use("Agg")

    positions = scaffold.grid_positions(args.rows, args.cols)
    print(
        f"grid {args.rows}x{args.cols}: {graph.num_nodes} nodes, "
        f"{graph.num_edges} undirected edges, "
        f"delta_min={(graph.num_nodes - 1) / graph.num_edges:.3f}"
    )

    figures = {
        "backbones": figure_backbones,
        "methods": figure_methods,
        "ratios": figure_ratios,
        "scores": figure_scores,
        "coverage": figure_coverage,
    }
    selected = [args.only] if args.only else list(figures)
    for index, name in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {name}")
        if name == "backbones":
            figures[name](graph, positions, args.out, quick=args.quick)
        else:
            figures[name](graph, positions, args.out)
    print("\ndone.")


if __name__ == "__main__":
    main()
