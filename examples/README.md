# Examples

Run any of these from the repository root.

| | what it shows | needs |
|---|---|---|
| [`01_grid_demo.py`](01_grid_demo.py) | Visual walkthrough on a grid graph; includes LLST in the README backbone comparison | matplotlib, networkx |
| [`02_standalone.py`](02_standalone.py) | NetworkX / SciPy / numpy usage, no PyTorch | networkx, scipy |
| [`03_pytorch_geometric.py`](03_pytorch_geometric.py) | PyG integration end to end, with a trained GCN | torch, torch-geometric |
| [`04_backbones_and_tuning.py`](04_backbones_and_tuning.py) | Comparing backbone stretch (including LLST) and tuning the objective | networkx |
| [`05_local_search_backbone.py`](05_local_search_backbone.py) | LLST forest refinement and Scaffold growth from that forest | networkx |

```bash
python examples/01_grid_demo.py --out docs/images
python examples/02_standalone.py
python examples/03_pytorch_geometric.py                  # synthetic, offline
python examples/03_pytorch_geometric.py --dataset Cora   # downloads Planetoid
python examples/04_backbones_and_tuning.py
python examples/05_local_search_backbone.py
```

## Why a grid graph?

A 2-D lattice has a regular layout that makes supporting paths, omitted edges,
and differences between backbones easy to inspect. The examples use the same
graph and seed so that the effects of each method are visible.
The method, smaller-budget, and sampling demos share a seeded `randst`
backbone (`fixed-randst` for Sample). The backbone comparison still shows
each named construction.

`01_grid_demo.py` writes five PNG figures and two GIFs to `docs/images/`:

| file | shows |
|---|---|
| `grid_backbones.png` | input grid and seven backbones, including GLST and its LLST refinement, with measured stretch |
| `grid_backbones.gif` | animated comparison of the same seven completed forests; LLST is the final frame |
| `grid_methods.png` | input and five algorithms at one budget in a 2×3 grid |
| `grid_ratios.png` | input and five Fast retention ratios in a 2×3 grid |
| `grid_scores.png` | input, Sample weights, inclusion probabilities, and one draw in a 2×2 grid |
| `grid_coverage.png` | three union snapshots and the coverage curve in a 2×2 grid |
| `grid_coverage.gif` | a 2×2 animation of input, current view, cumulative union, and coverage |

To regenerate only the backbone comparison:

```bash
python examples/01_grid_demo.py --only backbones --out docs/images
```

It also writes `grid_backbones.json` with the seed, LLST options, retained edge
IDs, edge counts, component counts, and omitted-edge stretch for each forest. LLST uses the
default GLST initializer and up to 10 exhaustive search passes. All panels
use the same graph and seed; LLST preserves the forest size while reducing
the total stretch of its initializer. Exact LLST adds computation to these
small-grid examples; the other four figures can be regenerated separately
with `--only methods`, `--only ratios`, `--only scores`, or `--only coverage`.
The script rejects a backbone comparison above 1,000 undirected input edges
before building any trees, because LLST's runtime can be very large. For
larger grids, choose one of those RandST demos. The LLST API has an explicit
`max_input_edges` override for deliberate experiments; see
[the backbone guide](../docs/backbones.md#llst).

`--only coverage` generates both the static coverage grid and the resampling
animation. `grid_coverage.json` records the RandST mode and backbone edge IDs,
all 50 draws, their component counts, and cumulative coverage; the GIF shows
every early epoch and selected later epochs. It depicts actual sampled
supports, with the same edge budget in every
frame. The backbone GIF instead compares completed forests.
