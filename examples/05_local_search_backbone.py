"""Use the research LLST as a standalone forest or a SCAFFOLD backbone.

Run after installing scaffold-sparse[networkx]::

    python examples/05_local_search_backbone.py

LLST is a small-graph reference and rejects inputs above 1,000 undirected
edges by default. Use randst, fast-randst, or fast-maxst for larger graphs;
see docs/backbones.md for the deliberate max_input_edges override.
"""

from __future__ import annotations

import scaffold
from scaffold.backbone import build_backbone
from scaffold.scoring import tree_scores

graph = scaffold.grid_graph(4, 4)
options = {"init_support": "maxst", "max_passes": 5}
initial = build_backbone(graph, "maxst")
refined = build_backbone(graph, "llst", seed=0, **options)

for label, mask in (("initial MaxST", initial), ("LLST", refined)):
    stats = tree_scores(graph.num_nodes, graph.src, graph.dst, mask)
    print(
        f"{label}: {mask.sum()} forest edges; omitted-edge stretch {stats['total_stretch']:.0f}"
    )

result = scaffold.fast(
    graph, num_edges=20, backbone="llst", backbone_options=options, seed=0
)
print(result.summary())
print(f"Components: {result.num_components()}")
