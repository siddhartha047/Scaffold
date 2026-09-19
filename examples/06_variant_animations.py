"""Animate five Scaffold methods on one seeded RandSF backbone.

Run from the repository root:
    python examples/06_variant_animations.py --out docs/images/variants

Writes a fixed input PNG, five standalone GIFs, a 2x3 combined GIF/PNG,
and a JSON record of every displayed state. Fast reveals its one-pass
selection in groups; these are display steps, not algorithmic growth rounds.
Animation timing is illustrative, not a runtime comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import scaffold
from scaffold import viz
from scaffold.backbone import build_backbone

BLUE = "#1f5fbf"
ORANGE = "#e07829"
GRAY = "#c8ccd4"
METHODS = ("greedy", "heap", "batch", "fast", "sample")
DESCRIPTIONS = {
    "input": "One graph for all five methods",
    "greedy": "Rescore all candidates; add the best edge",
    "heap": "Refresh cached scores; add the best edge",
    "batch": "Score 32 sampled edges; add the top 4",
    "fast": "Score once; reveal selected edges in groups",
    "sample": "Reuse weights; draw a fresh sparse graph",
}


def _state(mask, new=(), label="RandSF backbone"):
    return {"mask": mask.copy(), "new": np.asarray(new, dtype=np.int64), "label": label}


def collect_states(graph, keep_ratio, seed, workers, draw_count=12):
    """Collect actual API results and exact insertion boundaries, without rendering."""
    backbone = build_backbone(graph, "RandSF", seed=seed)
    budget = int(np.ceil(keep_ratio * graph.num_edges - 1e-12))
    if not int(backbone.sum()) < budget < graph.num_edges:
        raise ValueError("Choose a budget above the RandSF forest size and below the full graph.")
    states, settings = {}, {}
    for method in ("greedy", "heap", "batch"):
        options = {"clusters": 1} if method != "greedy" else {}
        if method == "batch":
            options.update(sample_size=32, add_per_round=4)
        result = getattr(scaffold, method)(
            graph, num_edges=budget, backbone=backbone, seed=seed,
            workers=workers, return_trace=True, **options,
        )
        ids = result.metadata["added_edge_ids"]
        sizes = result.metadata.get("addition_round_sizes", [1] * len(ids))
        mask, offset = backbone.copy(), 0
        sequence = [_state(mask)]
        for step, size in enumerate(sizes, 1):
            new = ids[offset:offset + size]
            assert size > 0 and len(set(new)) == size and not mask[new].any()
            mask[new] = True
            sequence.append(_state(mask, new, f"Round {step} · +{size} {'edge' if size == 1 else 'edges'}"))
            offset += size
        assert offset == len(ids) and np.array_equal(mask, result.mask)
        assert result.sparse_edges == budget and result.num_components() == 1
        states[method] = sequence
        settings[method] = options
        if method == "heap":
            settings[method].update(top_k=16, add_per_round=1, score_form="max")
        print(f"{method}: {len(sizes)} real insertion rounds, {budget} final edges", flush=True)

    result = scaffold.fast(graph, num_edges=budget, backbone=backbone,
                           seed=seed, workers=workers, return_scores=True)
    ids = np.flatnonzero(result.mask & ~backbone)
    order = np.lexsort((ids, -result.metadata["scores"][ids]))
    ranked = ids[order]
    group_size = 8
    mask = backbone.copy()
    states["fast"] = [_state(mask)]
    for step, start in enumerate(range(0, len(ranked), group_size), 1):
        new = ranked[start:start + group_size]
        mask[new] = True
        states["fast"].append(_state(mask, new, f"Reveal {step} · +{len(new)} edges"))
    assert np.array_equal(mask, result.mask)
    settings["fast"] = {"reveal_group_size": group_size, "algorithm_scoring_passes": 1}

    scores = scaffold.sample(graph, backbone="fixed-randsf", seed=seed,
                             workers=workers, tree_count=8)
    assert np.array_equal(scores.backbone, backbone)
    states["sample"] = []
    previous = None
    for i in range(draw_count):
        draw = scores.draw(num_edges=budget, seed=seed + i)
        assert draw.mask[backbone].all() and draw.sparse_edges == budget
        assert draw.num_components() == 1
        new = [] if previous is None else np.flatnonzero(draw.mask & ~previous)
        states["sample"].append(_state(draw.mask, new, f"Draw {i + 1} · same edge budget"))
        previous = draw.mask
    settings["sample"] = {"backbone": "fixed-randsf", "tree_count": 8,
                          "draw_seeds": list(range(seed, seed + draw_count))}
    return backbone, budget, states, settings


def _canvas(fig):
    from PIL import Image

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def render_panel(graph, positions, method, states, budget, palette=None):
    """Render one card; only its graph, edge count, and step label can move."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(4, 4.4), dpi=120, facecolor="white")
    fig.add_artist(FancyBboxPatch(
        (.025, .02), .95, .96, transform=fig.transFigure,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor="#f4f7fc" if method == "input" else "#fafbfc",
        edgecolor="#dce3ed", linewidth=1, zorder=0,
    ))
    title = "Input graph" if method == "input" else f"Scaffold-{method.capitalize()}"
    fig.text(.5, .9, title, ha="center", fontsize=20, fontweight="semibold", color=BLUE)
    fig.text(.5, .851, DESCRIPTIONS[method], ha="center", fontsize=10, color="#4b5563")
    counter = fig.text(.5, .795, "", ha="center", fontsize=11, color="#4b5563")
    status = fig.text(.5, .105, "", ha="center", fontsize=10, color=BLUE)
    ax = fig.add_axes([.08, .15, .84, .62])
    viz.draw_graph(graph, positions=positions, ax=ax, mask=states[0]["mask"],
                   node_size=10, linewidth=1.7)
    for collection in list(ax.collections):
        if isinstance(collection, LineCollection):
            collection.remove()
    segments = np.stack((positions[graph.src], positions[graph.dst]), axis=1)
    omitted = LineCollection([], colors=GRAY, linewidths=1.02,
                             linestyles=(0, (2, 2)), zorder=1)
    retained = LineCollection([], colors=BLUE, linewidths=1.7, zorder=3)
    newest = LineCollection([], colors=ORANGE, linewidths=2.7, zorder=3.5)
    for artist in (omitted, retained, newest):
        ax.add_collection(artist)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label="Kept"),
                        Line2D([0], [0], color=ORANGE, lw=2.7, label="New"),
                        Line2D([0], [0], color=GRAY, lw=1.5, linestyle="--", label="Omitted")],
               loc="lower center", bbox_to_anchor=(.5, .025), frameon=False,
               ncol=3, fontsize=8.5, handlelength=1.5, columnspacing=1.2)
    frames = []
    for state in states:
        mask = state["mask"]
        count = int(mask.sum())
        omitted.set_segments(segments[~mask])
        retained.set_segments(segments[mask])
        newest.set_segments(segments[state["new"]])
        counter.set_text(f"{count} edges · {count / graph.num_edges:.1%} retained")
        label = state["label"]
        if method in ("greedy", "heap", "batch", "fast") and count == budget:
            label += " · done"
        status.set_text(label)
        rgb = _canvas(fig)
        if palette is None:
            palette = rgb.quantize(colors=256, dither=0)
        rendered = rgb.quantize(palette=palette, dither=0)
        if frames:
            # Preserve titles and legend exactly despite GIF palette quantization.
            box = (0, int(.17 * rendered.height), rendered.width, int(.93 * rendered.height))
            frame = frames[0].copy()
            frame.paste(rendered.crop(box), box)
        else:
            frame = rendered
        frames.append(frame)
    plt.close(fig)
    return frames, palette


def save_gif(path, frames, durations):
    # Shared palette, full opaque frames, and disposal=1 retain unchanged pixels.
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, disposal=1, optimize=False)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", flush=True)


def write_animations(graph, positions, backbone, budget, states, settings, out, seed):
    import matplotlib.pyplot as plt
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    cards = {}
    cards["greedy"], palette = render_panel(graph, positions, "greedy", states["greedy"], budget)
    input_state = _state(np.ones(graph.num_edges, dtype=bool),
                         label=f"{graph.num_nodes} nodes · unweighted · fixed input")
    cards["input"], _ = render_panel(graph, positions, "input", [input_state], budget, palette)
    cards["input"][0].save(out / "input.png")
    for method in METHODS[1:]:
        cards[method], _ = render_panel(graph, positions, method, states[method], budget, palette)

    ticks = max(len(states[method]) - 1 for method in METHODS)
    durations = [1400] + [160] * (ticks - 1) + [2200]
    timeline = {}
    for method in METHODS:
        count = len(states[method])
        timeline[method] = [min(count - 1, t * count // (ticks + 1)) if method == "sample"
                            else t * (count - 1) // ticks for t in range(ticks + 1)]
        per_state = [0] * count
        for t, state_id in enumerate(timeline[method]):
            per_state[state_id] += durations[t]
        save_gif(out / f"{method}.gif", cards[method], per_state)

    width, height = cards["input"][0].size
    header = plt.figure(figsize=(width * 3 / 120, 1), dpi=120, facecolor="white")
    header.text(.025, .62, "Scaffold", fontsize=27, fontweight="bold", color=BLUE)
    header.text(.025, .19, "One input. Five ways to build sparse supports.",
                fontsize=13, color="#4b5563")
    header.text(.975, .59, f"Shared RandSF · {int(backbone.sum())} backbone edges",
                ha="right", fontsize=12, color="#4b5563")
    header.text(.975, .19, f"Target: {budget} / {graph.num_edges} edges ({budget / graph.num_edges:.1%})",
                ha="right", fontsize=12, color="#4b5563")
    header_image = _canvas(header).quantize(palette=palette, dither=0)
    plt.close(header)
    footer = plt.figure(figsize=(width * 3 / 120, .4), dpi=120, facecolor="white")
    footer.text(.5, .36, "Orange: newly added or newly sampled edges. Fast uses one scoring pass; its groups are a reveal. Animation speed is illustrative.",
                ha="center", fontsize=10, color="#4b5563")
    footer_image = _canvas(footer).quantize(palette=palette, dither=0)
    plt.close(footer)
    white = Image.new("RGB", (1, 1), "white").quantize(palette=palette, dither=0).getpixel((0, 0))
    canvas = Image.new("P", (3 * width, header_image.height + 2 * height + footer_image.height), white)
    canvas.putpalette(palette.getpalette())
    canvas.paste(header_image, (0, 0))
    canvas.paste(footer_image, (0, header_image.height + 2 * height))
    canvas.paste(cards["input"][0], (0, header_image.height))
    combined = []
    for t in range(ticks + 1):
        frame = canvas.copy()
        for index, method in enumerate(METHODS, 1):
            row, col = divmod(index, 3)
            frame.paste(cards[method][timeline[method][t]], (col * width, header_image.height + row * height))
        combined.append(frame)
    save_gif(out / "variants.gif", combined, durations)
    combined[-1].save(out / "variants.png")

    record = {
        "num_nodes": graph.num_nodes, "num_edges": graph.num_edges,
        "seed": seed, "backbone": "RandSF", "backbone_edge_ids": np.flatnonzero(backbone).tolist(),
        "target_edges": budget, "settings": settings,
        "fast_display": "Static top-k ranked once, then revealed in groups; no rescoring.",
        "timing": "Illustrative; not a runtime comparison.", "durations_ms": durations,
        "timeline": timeline, "input_pixel_box": [0, header_image.height, width, header_image.height + height],
        "methods": {method: [{"retained_edge_ids": np.flatnonzero(state["mask"]).tolist(),
                               "new_edge_ids": state["new"].tolist(), "label": state["label"]}
                              for state in sequence] for method, sequence in states.items()},
    }
    (out / "variants.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'variants.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/images/variants"))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--quick", action="store_true", help="Use a 6x6 grid.")
    parser.add_argument("--keep-ratio", type=float, default=.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = args.rows if args.rows is not None else (6 if args.quick else 12)
    cols = args.cols if args.cols is not None else (6 if args.quick else 12)
    if rows < 2 or cols < 2:
        parser.error("Use at least two rows and two columns.")
    if 2 * rows * cols - rows - cols > 1000:
        parser.error("The Greedy/Heap animation demo is limited to 1,000 input edges; use a smaller grid.")
    import matplotlib

    matplotlib.use("Agg")
    graph = scaffold.grid_graph(rows, cols)
    try:
        backbone, budget, states, settings = collect_states(graph, args.keep_ratio, args.seed, args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    write_animations(graph, scaffold.grid_positions(rows, cols), backbone, budget,
                     states, settings, args.out, args.seed)


if __name__ == "__main__":
    main()
