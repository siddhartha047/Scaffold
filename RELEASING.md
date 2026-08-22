# Releasing

Two phases, deliberately. **Phase 1 keeps the package private** while you test
it; **Phase 2** opens it to the public. Nothing in the code changes between
them — only where it is installed from.

---

## Phase 1 — private (now)

While `siddhartha047/Scaffold` is a private GitHub repository, pip can install
straight from it. No package index is involved, so nothing is published
anywhere, and only people with repo access can install it.

Plain `pip install scaffold-sparse` is intentionally **not available during
this phase**: that command asks the configured package index, and PyPI is
public. While testing privately, use the Git URL or a wheel below. The plain
command becomes available only after the public PyPI release (or after setting
up a separate authenticated private Python package index).

### For yourself

```bash
git clone git@github.com:siddhartha047/Scaffold.git
cd Scaffold
pip install -e ".[dev]"
pytest
```

### For a collaborator you have added to the repo

Over SSH (they need an SSH key on their GitHub account):

```bash
pip install "git+ssh://git@github.com/siddhartha047/Scaffold.git"

# a specific commit, which is what you want during private testing
pip install "git+ssh://git@github.com/siddhartha047/Scaffold.git@<commit-sha>"

# with extras
pip install "scaffold-sparse[pyg] @ git+ssh://git@github.com/siddhartha047/Scaffold.git"
```

Over HTTPS with a personal access token (repo scope), for CI or machines
without SSH keys:

```bash
pip install "git+https://${GITHUB_TOKEN}@github.com/siddhartha047/Scaffold.git"
```

In a `requirements.txt`:

```
scaffold-sparse @ git+ssh://git@github.com/siddhartha047/Scaffold.git@<commit-sha>
```

### Distributing a built wheel

If a collaborator has no repo access, hand them a wheel directly:

```bash
python -m build            # -> dist/scaffold_sparse-0.1.0-py3-none-any.whl
pip install dist/scaffold_sparse-0.1.0-py3-none-any.whl
```

### What *not* to do while private

- **Do not upload to PyPI** — that is irreversible and public.
- **Do not upload to TestPyPI either.** TestPyPI is publicly browsable. It is
  the right rehearsal step, but only once you are ready to be seen.

---

## Verifying a release candidate

Before either phase, in a clean environment:

```bash
# 1. tests pass, including the tree-kernel equivalence checks
pytest

# 2. the examples run
python examples/01_grid_demo.py --out /tmp/scaffold-figures
python examples/02_standalone.py
python examples/03_pytorch_geometric.py
python examples/04_backbones_and_tuning.py

# optional but recommended when the sibling research checkout is available
python validation/compare_research.py

# 3. it builds
rm -rf dist/ build/ src/*.egg-info
python -m build

# 4. the metadata is valid
python -m twine check dist/*

# 5. the built wheel actually works, in a fresh venv
python -m venv /tmp/scaffold-test
/tmp/scaffold-test/bin/pip install "dist/scaffold_sparse-0.1.0-py3-none-any.whl[all]"
/tmp/scaffold-test/bin/python -c "
import scaffold
print(scaffold.__version__)
r = scaffold.fast(scaffold.grid_graph(20, 20), keep_ratio=0.6)
print(r.summary(), r.num_components())
"
```

Step 5 matters more than it looks: it is the only check that the *packaged*
files are complete. A missing `package-data` entry or a stray import of a dev-only
module shows up here and nowhere else.

---

## Phase 2 — public

### Before flipping the switch

- [ ] Tests pass on every supported Python version (see `.github/workflows/test.yml`)
- [ ] `CHANGELOG.md` has a dated entry for the version
- [ ] The version in `src/scaffold/_version.py` matches the tag you are about to push
- [ ] `README.md` figures are regenerated and committed
- [ ] The name `scaffold-sparse` is still free on PyPI
- [ ] Repo visibility changed to public
- [ ] A citation entry is in `README.md` if the paper is out

### 2a. TestPyPI first

```bash
rm -rf dist/
python -m build
python -m twine upload --repository testpypi dist/*
```

Then install from it in a clean venv. `--extra-index-url` is required because
NumPy and SciPy are not mirrored on TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            scaffold-sparse
```

### 2b. PyPI

```bash
python -m twine upload dist/*
```

Then:

```bash
pip install scaffold-sparse
```

### 2c. Tag it

```bash
git tag -a v0.1.0 -m "scaffold-sparse 0.1.0"
git push origin v0.1.0
```

---

## Automating it later

`.github/workflows/publish.yml` publishes on a pushed tag. It is written for
**PyPI Trusted Publishing**, which avoids storing a long-lived API token as a
repository secret: PyPI verifies the GitHub Actions workflow identity directly.

To enable it, on PyPI go to your project → *Publishing* → *Add a new publisher*:

| field | value |
|---|---|
| Owner | `siddhartha047` |
| Repository | `Scaffold` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The workflow is otherwise inert — it only triggers on `v*` tags.

---

## Versioning

`MAJOR.MINOR.PATCH`, bumped in `src/scaffold/_version.py` and recorded in
`CHANGELOG.md`. Both `pyproject.toml` and `_version.py` carry the number; keep
them in sync (there is a test for this in `tests/test_package.py`).

- **PATCH** — bug fixes, no API change
- **MINOR** — new features, backwards compatible
- **MAJOR** — breaking changes

Planned:

| version | contents |
|---|---|
| `0.1.0` | The four algorithms, all adapters, PyG integration, docs |
| `0.2.0` | Parallel `fast`; artifact caching helpers; benchmark suite |
| `0.3.0` | Out-of-core path for graphs that do not fit in RAM |
| `1.0.0` | Stable public API |

## About the name

`scaffold` on PyPI is taken by an unrelated project, so the distribution is
`scaffold-sparse` while the canonical import name stays `scaffold`.
`import scaffold_sparse` also works for anyone who expects the import to match
the distribution. The earlier private-development alias
`import scaffold_sparsify` remains compatible but is not the documented API.

If you ever want the bare name, PyPI has a formal process for abandoned
projects under PEP 541 — but do not build a release schedule around it.
