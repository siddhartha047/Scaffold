"""Import alias for the :mod:`scaffold` package.

The distribution is named ``scaffold-sparsifier``. The canonical import is::

    import scaffold

This alias remains available for existing code::

    import scaffold_sparse as scaffold

    result = scaffold.fast(G, keep_ratio=0.2)
"""

import scaffold as _scaffold
from scaffold import *  # noqa: F401,F403
from scaffold import __all__, __version__  # noqa: F401


def __getattr__(name):
    """Forward anything not re-exported above, including lazy submodules."""
    return getattr(_scaffold, name)
