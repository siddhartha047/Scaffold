# Contributing

## Setup

```bash
git clone git@github.com:siddhartha047/Scaffold.git
cd Scaffold
pip install -e ".[dev]"
pytest
```

## Layout

```
src/scaffold/
├── api.py              entry points: exact / heap / fast / sample / sparsify
├── result.py           ScaffoldResult, ScaffoldScores
├── graph.py            the canonical internal Graph
├── kernels.py          numba-optional array kernels (union-find, LCA, prefix sums)
├── scoring.py          the objective: tree_scores (fast) and path_scores (reference)
├── backbone.py         spanning-forest builders + registry
├── clustering.py       node partitioning for the cluster-parallel loops
├── datasets.py         demo graphs
├── viz.py              plotting (matplotlib)
├── pyg.py              PyTorch Geometric integration
├── adapters/           nx / pyg / scipy / array conversions
├── algorithms/         one module per variant
└── utils/              validation, budget arithmetic
```

**The layering rule:** `algorithms/` never imports an adapter and never sees a
NetworkX, PyG or SciPy object. Everything below `api.py` operates on a
`Graph`. Keeping format handling out of the hot loops is what lets there be one
copy of each algorithm.

## Adding an algorithm

1. Write `src/scaffold/algorithms/<name>.py` exposing
   `run(graph, keep_ratio=None, num_edges=None, backbone=..., params=..., seed=None, **kwargs)`
   returning `(mask, metadata)`.
2. Use `GrowthContext` from `algorithms/base.py` for the budget, backbone and
   bookkeeping — that is what keeps the variants agreeing on what
   `keep_ratio=0.2` means, down to the rounding.
3. Add a wrapper in `api.py` and register it in `METHODS`.
4. Add it to `ALL_METHODS` in `tests/conftest.py`. The shared guarantee tests in
   `tests/test_algorithms.py` will then run against it automatically.

## Testing

```bash
pytest                                  # everything
pytest tests/test_scoring.py            # the tree-kernel equivalence checks
pytest --cov=scaffold --cov-report=term-missing
```

Tests requiring optional dependencies are skipped, not failed — see the
`requires_*` markers in `conftest.py`.

### What a good test looks like here

Prefer algorithmic properties to smoke tests. "It runs" tells you almost
nothing; these tell you something:

- the edge budget is exact,
- the output is a subgraph on the same node set,
- component count is preserved above `delta_min`,
- the same seed gives the same answer,
- the global NumPy RNG is never touched,
- `tree_scores` equals `path_scores` on a forest.

That last one is the load-bearing property of the package. `scaffold.fast` and
`scaffold.sample` are only correct because a path aggregate on a tree is a
root-prefix difference; if the two scorers ever disagree, the fast variants are
computing a different objective than the reference. `tests/test_scoring.py`
checks it across five backbones, four parameter sets, weighted and unweighted
graphs, and disconnected inputs, to `rtol=1e-9`.

## Style

- `ruff check .` and `ruff format .`
- Match the surrounding code. Comments explain *why*, not *what*.
- Docstrings on anything public, with the complexity where it is non-obvious.
- Type hints on public signatures; `from __future__ import annotations` at the
  top of every module (the package supports Python 3.9).

## Performance

The kernels in `kernels.py` are written in explicit-loop style so a single
source works both as pure Python and as a numba JIT kernel. If you touch them:

- keep the loop style — no fancy NumPy inside a `@jit` function;
- verify both paths (`pip uninstall numba` to test the fallback);
- prefer `O(m)` array passes over per-edge Python loops in the hot path.

## Pull requests

1. Branch off `main`.
2. Add tests for the behaviour you changed.
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Run `pytest` and `ruff check .`.
