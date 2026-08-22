"""Packaging invariants and the public surface.

Cheap checks that catch the mistakes which only show up after a release: a
version bumped in one place but not the other, an optional dependency imported
at module scope, a name promised in ``__all__`` that does not exist.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

import scaffold

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_exported():
    assert re.fullmatch(r"\d+\.\d+\.\d+", scaffold.__version__)


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)
    assert declared == scaffold.__version__, (
        "pyproject.toml and src/scaffold/_version.py disagree"
    )


@pytest.mark.parametrize("name", sorted(scaffold.__all__))
def test_everything_in_all_exists(name):
    assert hasattr(scaffold, name)


def test_entry_points_are_callable():
    for name in ("sparsify", "exact", "heap", "fast", "sample"):
        assert callable(getattr(scaffold, name))
    assert set(scaffold.METHODS) == {"exact", "heap", "fast", "sample"}


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        scaffold.definitely_not_a_real_name  # noqa: B018


def test_optional_dependencies_are_not_imported_at_module_scope():
    """``import scaffold`` must not pull in torch, matplotlib or networkx.

    Run in a subprocess: this process has already imported them via other
    tests, so checking ``sys.modules`` here would prove nothing.
    """
    code = (
        "import sys, scaffold;"
        "leaked = [m for m in ('torch', 'torch_geometric', 'matplotlib', 'networkx')"
        " if m in sys.modules];"
        "print(','.join(leaked))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert output.stdout.strip() == "", (
        f"import scaffold leaked optional dependencies: {output.stdout.strip()}"
    )


def test_viz_and_pyg_are_lazy_but_reachable():
    """They are not imported eagerly, but ``scaffold.viz`` still resolves."""
    pytest.importorskip("matplotlib")
    assert scaffold.viz is importlib.import_module("scaffold.viz")


def test_readme_and_license_are_shipped():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "CHANGELOG.md").is_file()


def test_py_typed_marker_exists():
    assert (ROOT / "src" / "scaffold" / "py.typed").is_file()


def test_examples_are_syntactically_valid():
    import ast

    scripts = sorted((ROOT / "examples").glob("*.py"))
    assert scripts, "no examples found"
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))


def test_public_functions_have_docstrings():
    for name in ("sparsify", "exact", "heap", "fast", "sample"):
        doc = getattr(scaffold, name).__doc__
        assert doc and len(doc.strip()) > 40, f"{name} needs a real docstring"
