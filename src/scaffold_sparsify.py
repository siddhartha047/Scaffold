"""Alias for the :mod:`scaffold` package.

The distribution is named ``scaffold-sparsify`` because ``scaffold`` is taken
on PyPI by an unrelated project, so ``import scaffold_sparsify`` is accepted
too and means exactly the same thing::

    import scaffold_sparsify as scaffold

    result = scaffold.fast(G, keep_ratio=0.2)

``scaffold`` remains the canonical import name; this module just forwards.
"""

import scaffold as _scaffold
from scaffold import *  # noqa: F401,F403
from scaffold import __all__, __version__  # noqa: F401


def __getattr__(name):
    """Forward anything not re-exported above, including lazy submodules."""
    return getattr(_scaffold, name)
