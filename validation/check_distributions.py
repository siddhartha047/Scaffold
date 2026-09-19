"""Install built artifacts in clean virtual environments and exercise the API.

Run after ``python -m build``::

    python validation/check_distributions.py --dist dist
    python validation/check_distributions.py --dist dist --sdist

Works on Windows, macOS, and Linux. Each check installs with pip, then runs
outside the checkout in isolated mode so source-tree imports cannot hide
missing wheel contents or undeclared dependencies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

SMOKE = """
import importlib
import math
import sys
from pathlib import Path

import scaffold
import numpy as np

package = Path(scaffold.__file__).resolve()
assert Path(sys.prefix).resolve() in package.parents, package
for name in ('networkx', 'matplotlib', 'torch', 'torch_geometric', 'numba'):
    assert name not in sys.modules, name
for alias in ('scaffold_sparse', 'scaffold_sparsify'):
    assert importlib.import_module(alias).fast is scaffold.fast

graph = scaffold.grid_graph(4, 4)
target = math.ceil(0.8 * graph.num_edges)
for method in scaffold.METHODS:
    result = scaffold.sparsify(graph, method=method, keep_ratio=0.8, seed=0)
    if method == 'sample':
        result = result.draw(keep_ratio=0.8, seed=0)
    assert result.sparse_edges == target, method
    assert result.num_components() == 1, method
    assert result.to_scipy().shape == (16, 16), method
    assert result.mask.dtype == np.bool_, method
    print(result.summary())

raw = graph.edge_index
assert scaffold.fast(raw, num_edges=target).sparse_edges == target
print('Installed-package smoke checks passed:', scaffold.__version__)
"""


def check(artifact):
    with tempfile.TemporaryDirectory(prefix="scaffold-install-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(artifact)],
            cwd=root, check=True,
        )
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--sdist", action="store_true", help="also install the source archive")
    args = parser.parse_args()
    patterns = ["*.whl"] + (["*.tar.gz"] if args.sdist else [])
    for pattern in patterns:
        matches = list(args.dist.resolve().glob(pattern))
        if len(matches) != 1:
            parser.error(f"expected exactly one {pattern} in {args.dist}, found {len(matches)}")
        print(f"Checking {matches[0].name}", flush=True)
        check(matches[0])


if __name__ == "__main__":
    main()
