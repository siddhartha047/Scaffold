# Examples

Run any of these from the repository root.

| | what it shows | needs |
|---|---|---|
| [`01_grid_demo.py`](01_grid_demo.py) | Visual walkthrough on a grid graph; writes the README figures | matplotlib |
| [`02_standalone.py`](02_standalone.py) | NetworkX / SciPy / numpy usage, no PyTorch | networkx, scipy |
| [`03_pytorch_geometric.py`](03_pytorch_geometric.py) | PyG integration end to end, with a trained GCN | torch, torch-geometric |
| [`04_backbones_and_tuning.py`](04_backbones_and_tuning.py) | Picking a backbone and tuning the objective | — |

```bash
python examples/01_grid_demo.py --out docs/images
python examples/02_standalone.py
python examples/03_pytorch_geometric.py                  # synthetic, offline
python examples/03_pytorch_geometric.py --dataset Cora   # downloads Planetoid
python examples/04_backbones_and_tuning.py
```

## Why a grid graph?

A 2-D lattice has no communities, no hubs and no important edges — every edge
is equivalent by symmetry. So whatever pattern survives sparsification is the
*algorithm's* preference, not the graph's structure. On Cora you cannot see
that; here you can.

`01_grid_demo.py` writes five figures to `docs/images/`:

| file | shows |
|---|---|
| `grid_backbones.png` | the spanning forest each backbone produces |
| `grid_methods.png` | the four algorithms at one budget |
| `grid_ratios.png` | `scaffold.fast` as the budget tightens past the connectivity floor |
| `grid_scores.png` | `scaffold.sample`'s weights, inclusion probabilities, and one draw |
| `grid_coverage.png` | what per-epoch resampling covers over 50 epochs |
