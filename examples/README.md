# Examples

Run any of these from the repository root.

| | what it shows | needs |
|---|---|---|
| [`01_grid_demo.py`](01_grid_demo.py) | Visual walkthrough on a grid graph; includes LLSF in the README backbone comparison | matplotlib, networkx |
| [`02_standalone.py`](02_standalone.py) | NetworkX / SciPy / numpy usage, no PyTorch | networkx, scipy |
| [`03_pytorch_geometric.py`](03_pytorch_geometric.py) | PyG integration end to end, with a trained GCN | torch, torch-geometric |
| [`04_backbones_and_tuning.py`](04_backbones_and_tuning.py) | Comparing backbone stretch (including LLSF) and tuning the objective | networkx |
| [`05_local_search_backbone.py`](05_local_search_backbone.py) | LLSF forest refinement and Scaffold growth from that forest | networkx |

```bash
python examples/01_grid_demo.py --out docs/images
python examples/02_standalone.py
python examples/03_pytorch_geometric.py                  # synthetic, offline
python examples/03_pytorch_geometric.py --dataset Cora   # downloads Planetoid
python examples/04_backbones_and_tuning.py
python examples/05_local_search_backbone.py
```

For a fast preview, use `python examples/01_grid_demo.py --quick --out demo-preview`.
This uses a 6×6 grid and two sampled LLSF swaps; the standard command still
reproduces the README's 12×12 grid and exhaustive LLSF settings.
`04_backbones_and_tuning.py` now defaults to an 8×8 grid and two sampled LLSF
swaps; `--side 14 --exhaustive-llsf` restores the larger exhaustive comparison.
The old `--exhaustive-llst` flag remains an alias.
The 1,000-edge LLSF input guard applies in both modes.

The runtime benchmark also avoids long reference runs by default:

```bash
python benchmarks/bench_methods.py --sizes 16 32 64
# Opt into Greedy/Heap above 1,000 edges, or pin Batch's previous sizes:
python benchmarks/bench_methods.py --sizes 24 --reference-max-edges 0
python benchmarks/bench_methods.py --batch-sample-size 64 --batch-add-per-round 8
```

## Why a grid graph?

A 2-D lattice has a regular layout that makes supporting paths, omitted edges,
and differences between backbones easy to inspect. The examples use the same
graph and seed so that the effects of each method are visible.
The method and sampling demos share a seeded `randsf` backbone (`fixed-randsf`
for Sample). The opening edge-budget demo uses Scaffold-Greedy with an LLSF
backbone, built once and reused across all five ratios. The backbone comparison
still shows each named construction.

These grid inputs are unweighted and connected, so each forest in the figures
is a single spanning tree. The display labels use RandSF, MaxSF, SPF, GLSF,
and LLSF; example calls use these forest spellings too. Historical names such
as `maxst` and `llst` still work, and saved backbone keys stay compatible. The methods
compute **spanning forests** on disconnected inputs: one tree per component, with isolated
vertices preserved. Only the backbone must be acyclic; the final sparse
support can include additional edges and cycles.

`01_grid_demo.py` writes five PNG figures and two GIFs to `docs/images/`:

| file | shows |
|---|---|
| `grid_backbones.png` | input grid and seven backbones, with full method names and measured stretch |
| `grid_backbones.gif` | animated comparison of the same seven completed forests; each frame expands the name and explains construction; LLSF is last |
| `grid_methods.png` | input and five algorithms at one budget in a 2×3 grid |
| `grid_ratios.png` | README opening visual: input, then 85%, 75%, 65%, 55%, and 45% retention with Greedy and LLSF, in a 2×3 grid |
| `grid_scores.png` | input, Sample weights, inclusion probabilities, and one draw in a 2×2 grid |
| `grid_coverage.png` | three union snapshots and the coverage curve in a 2×2 grid |
| `grid_coverage.gif` | a 2×2 animation of input, current view, cumulative union, and coverage |

To regenerate only the backbone comparison:

```bash
python examples/01_grid_demo.py --only backbones --out docs/images
```

Regenerate the README's opening budget visual with
`python examples/01_grid_demo.py --only ratios --out docs/images`.
Every panel keeps all 144 nodes. The 45% panel has 25 components because
its budget is below this grid's connectivity floor (143 edges, or 54.2%).

It also writes `grid_backbones.json` with the seed, LLSF options, retained edge
IDs, edge counts, component counts, and omitted-edge stretch for each forest. LLSF uses the
default GLSF initializer and up to 10 exhaustive search passes. All panels
use the same graph and seed; LLSF preserves the forest size while reducing
the total stretch of its initializer. Exact LLSF adds computation to these
small-grid examples. The opening `--only ratios` demo also uses LLSF with these
settings; `--quick` uses two sampled swaps for both LLSF demos. Other figures
can be regenerated separately with `--only methods`, `--only scores`, or
`--only coverage`. The script rejects backbone and edge-budget demos above
1,000 undirected input edges before building any trees. For larger grids,
choose one of the RandSF demos. The LLSF API has an explicit
`max_input_edges` override for deliberate experiments; see
[the backbone guide](../docs/backbones.md#llst).

`--only coverage` generates both the static coverage grid and the resampling
animation. `grid_coverage.json` records the RandSF mode and backbone edge IDs,
all 50 draws, their component counts, and cumulative coverage; the GIF shows
every early epoch and selected later epochs. It depicts actual sampled
supports, with the same edge budget in every
frame. The backbone GIF instead compares completed forests.
