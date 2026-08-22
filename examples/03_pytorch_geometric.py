"""PyTorch Geometric: sparsify a dataset and train a GCN on the result.

Run::

    python examples/03_pytorch_geometric.py                 # synthetic graph
    python examples/03_pytorch_geometric.py --dataset Cora  # downloads Planetoid

Requires ``pip install "scaffold-sparsify[torch]"``. With no ``--dataset`` it
builds a synthetic graph so the script runs anywhere, offline.

Covers:

1. ``Data`` in, ``Data`` out -- features, labels and masks preserved
2. ``ScaffoldTransform`` in a dataset pipeline
3. edge weights from ``scaffold.sample``
4. per-epoch resparsification with ``ScaffoldResampler``
5. a GCN trained on the full vs. the sparsified graph
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import scaffold
from scaffold.pyg import ScaffoldResampler, ScaffoldTransform, sparsify_data


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def synthetic_data(num_nodes=1200, num_classes=4, degree=8, seed=0):
    """A planted-partition graph: dense within class, sparse across."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_nodes)
    src, dst = [], []
    for u in range(num_nodes):
        same = np.flatnonzero(labels == labels[u])
        peers = rng.choice(same, size=min(degree, same.size), replace=False)
        for v in peers:
            if u < v:
                src.append(u)
                dst.append(int(v))
        for v in rng.integers(0, num_nodes, size=2):
            if u < v:
                src.append(u)
                dst.append(int(v))

    edge_index = torch.tensor(np.array([src + dst, dst + src]), dtype=torch.long)
    features = torch.randn(num_nodes, 32)
    features += torch.tensor(labels, dtype=torch.long).unsqueeze(1) * 0.55

    data = Data(
        x=features,
        y=torch.tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    permutation = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    for name, span in (
        ("train_mask", permutation[: num_nodes // 10]),
        ("val_mask", permutation[num_nodes // 10 : num_nodes // 5]),
        ("test_mask", permutation[num_nodes // 5 :]),
    ):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[span] = True
        data[name] = mask
    return data


def load_data(name):
    if name is None:
        return synthetic_data(), "synthetic planted-partition"
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=f"/tmp/scaffold-{name}", name=name)
    return dataset[0], name


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def train_model(data, epochs=100, resampler=None, seed=0, hidden=64):
    """Train a GCN. With ``resampler``, the topology is redrawn every epoch."""
    torch.manual_seed(seed)
    model = GCN(data.num_features, hidden, int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_test = 0.0, 0.0
    for epoch in range(epochs):
        view = data if resampler is None else resampler.epoch(epoch)
        model.train()
        optimizer.zero_grad()
        out = model(view.x, view.edge_index)
        loss = F.cross_entropy(out[view.train_mask], view.y[view.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(view.x, view.edge_index).argmax(dim=1)
            val = float((prediction[view.val_mask] == view.y[view.val_mask]).float().mean())
            test = float((prediction[view.test_mask] == view.y[view.test_mask]).float().mean())
        if val > best_val:
            best_val, best_test = val, test
    return best_val, best_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Cora / CiteSeer / PubMed")
    parser.add_argument("--keep-ratio", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    data, name = load_data(args.dataset)
    print(f"dataset: {name}")
    print(f"  {data}")

    # ------------------------------------------------------------------
    section("1. Data in, Data out")
    # ------------------------------------------------------------------
    result = scaffold.fast(data, keep_ratio=args.keep_ratio, seed=0)
    sparse = result.to_pyg()
    print(f"  {result.summary()}")
    print(f"  original: {data}")
    print(f"  sparse:   {sparse}")
    print(f"  x preserved:          {torch.equal(sparse.x, data.x)}")
    print(f"  train_mask preserved: {torch.equal(sparse.train_mask, data.train_mask)}")
    print(f"  components: {result.num_components()} "
          f"(input had {result.metadata['base_components']})")

    # Or in one call:
    sparse = sparsify_data(data, method="fast", keep_ratio=args.keep_ratio, seed=0)
    print(f"  sparsify_data(...) -> {sparse}")

    # ------------------------------------------------------------------
    section("2. As a dataset transform")
    # ------------------------------------------------------------------
    transform = ScaffoldTransform(method="fast", keep_ratio=args.keep_ratio, seed=0)
    print(f"  {transform}")
    print(f"  transform(data) -> {transform(data)}")
    print("\n  With a real dataset:")
    print("    Planetoid(root=..., name='Cora', transform=ScaffoldTransform(keep_ratio=0.2))")

    # ------------------------------------------------------------------
    section("3. scaffold.sample gives edge weights, not a subgraph")
    # ------------------------------------------------------------------
    scores = scaffold.sample(data, seed=0)
    edge_index, edge_weight = scores.to_torch()
    print(f"  {scores.summary()}")
    print(f"  edge_index {tuple(edge_index.shape)}, edge_weight {tuple(edge_weight.shape)}")
    print(f"  weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
    print("\n  Assign them straight onto the Data object:")
    print("    data.edge_index, data.edge_weight = scores.to_torch()")

    probabilities = scores.inclusion_probabilities(keep_ratio=args.keep_ratio)
    core = int((probabilities >= 1 - 1e-12).sum())
    print(f"\n  At keep_ratio={args.keep_ratio}: {core:,} edges are in every draw "
          f"(the deterministic core), the rest are sampled.")
    coverage = scores.sampler.coverage(keep_ratio=args.keep_ratio)
    print("  Expected coverage: " + ", ".join(
        f"{e} epochs -> {v:.1%}" for e, v in coverage["coverage_curve"].items()
    ))

    # ------------------------------------------------------------------
    section("4. Per-epoch resparsification")
    # ------------------------------------------------------------------
    resampler = ScaffoldResampler(
        data, keep_ratio=args.keep_ratio, seed=0, backbone="rotate-randst"
    )
    print(f"  {resampler}")
    print(f"  delta_min (connectivity floor): {resampler.delta_min:.4f}")
    seen = np.zeros(resampler.graph.num_edges, dtype=bool)
    for epoch in range(5):
        view = resampler.epoch(epoch)
        seen |= resampler.scores.sampler.draw(keep_ratio=args.keep_ratio).mask
        print(f"    epoch {epoch}: edge_index {tuple(view.edge_index.shape)}, "
              f"union coverage {seen.mean():.1%}")

    # ------------------------------------------------------------------
    section("5. Training a GCN")
    # ------------------------------------------------------------------
    print(f"  {args.epochs} epochs, hidden=64, keep_ratio={args.keep_ratio}\n")
    print(f"  {'setting':32s} {'val':>7s} {'test':>7s}")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 7}")

    val, test = train_model(data, epochs=args.epochs, seed=0)
    print(f"  {'full graph':32s} {val:7.4f} {test:7.4f}")

    sparse = sparsify_data(data, method="fast", keep_ratio=args.keep_ratio, seed=0)
    val, test = train_model(sparse, epochs=args.epochs, seed=0)
    print(f"  {f'scaffold.fast @ {args.keep_ratio}':32s} {val:7.4f} {test:7.4f}")

    val, test = train_model(data, epochs=args.epochs, resampler=resampler, seed=0)
    print(f"  {'scaffold.sample, redrawn/epoch':32s} {val:7.4f} {test:7.4f}")

    print(
        "\n  On a synthetic graph these numbers are illustrative only. The point is"
        "\n  that the API is a drop-in: same model, same loop, one different graph."
    )


if __name__ == "__main__":
    main()
