"""Legacy import alias for the :mod:`scaffold` package.

The distribution is named ``scaffold-sparsifier``. This module remains a
compatibility alias for existing code::

    import scaffold_sparsify as scaffold

    result = scaffold.fast(G, keep_ratio=0.2)

New code should use ``import scaffold``.
"""

import scaffold as _scaffold
from scaffold import *  # noqa: F401,F403
from scaffold import __all__, __version__  # noqa: F401


def __getattr__(name):
    """Forward anything not re-exported above, including lazy submodules."""
    return getattr(_scaffold, name)
