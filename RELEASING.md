# Distribution and releases

One source serves both the ICLR 2027 review copy on anonymous.4open.science
and the public GitHub repository linked from arXiv. Installation does not
require separate branches or an anonymous export script.

| Channel | Installation |
|---|---|
| Anonymous review copy | Download the source, then `python -m pip install .` |
| Public GitHub | Install from the downloaded source or a Git URL |
| PyPI, planned after review | `python -m pip install scaffold-sparsifier` |

The distribution name is **`scaffold-sparsifier`**; the Python import is
**`import scaffold`** in every channel. The existing `scaffold_sparse` and
`scaffold_sparsify` imports remain compatibility aliases. Scaffold-GNN is a
separate repository with its own installation and experiment runner.

## Installing from source or Git

From the repository root:

```bash
python -m pip install ".[speed]"
```

For development, use `python -m pip install -e ".[dev]"`.
For public GitHub, replace `OWNER` with the repository owner:

```bash
python -m pip install "scaffold-sparsifier[speed] @ git+https://github.com/OWNER/Scaffold.git"
```

Append `@COMMIT` to the Git URL to install a specific revision. Source
installation remains the review workflow; no PyPI release is needed for it.
Mirror anonymization is configured separately from Python installation;
include package metadata and documentation in that review, and preserve
required license attribution.

## Checking a release candidate

From a development environment, build into a fresh output directory so old
artifacts cannot be mistaken for the release:

```bash
python -m pytest
python -m build --outdir dist/release-candidate
python -m twine check dist/release-candidate/*
python validation/check_distributions.py --dist dist/release-candidate --sdist
```

The last command installs both the wheel and source archive in clean virtual
environments, checks the distribution metadata and legacy imports, and runs
all five algorithms with only the core dependencies. Use an empty output
directory for each release candidate.

## Publishing after review

Before publishing, verify that the `scaffold-sparsifier` project name is
available or owned by the release account. Keep the version in
`pyproject.toml` and `src/scaffold/_version.py` synchronized and record the
release in `CHANGELOG.md`.

The existing `.github/workflows/publish.yml` supports:

- Manual dispatch to TestPyPI or PyPI.
- A pushed `v*` version tag to PyPI.

Configure a Trusted Publisher for the repository, `publish.yml`, and the
appropriate `testpypi` or `pypi` environment before using that workflow.
Ordinary commits and branch pushes do not publish packages.

Once the PyPI release is available:

```bash
python -m pip install scaffold-sparsifier
# Recommended for medium/large graphs:
python -m pip install "scaffold-sparsifier[speed]"
```

```python
import scaffold
```
