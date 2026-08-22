# Source installation and distribution checks

This project is distributed as a self-contained source tree. The distribution
name is **`scaffold-sparsifier`** and the Python import is **`import scaffold`**.
The `scaffold_sparse` and `scaffold_sparsify` imports are compatibility aliases.

## Install from source

Run from the folder containing `pyproject.toml`:

```bash
python -m pip install ".[speed]"
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Check built distributions

Build into a fresh directory:

```bash
python -m build --outdir dist/check
python -m twine check dist/check/*
python validation/check_distributions.py --dist dist/check --sdist
```

The last command installs the wheel and source archive in clean virtual
environments, checks distribution metadata and compatibility imports, and
runs all five algorithms with the core dependencies.

Keep the version in `pyproject.toml` and `src/scaffold/_version.py` synchronized
and record changes in `CHANGELOG.md`. Source archives use neutral ownership
metadata. Retain the included license notices with redistributed source.
