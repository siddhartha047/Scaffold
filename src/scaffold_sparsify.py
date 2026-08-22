"""Legacy import alias for the :mod:`scaffold` package.

The distribution was called ``scaffold-sparsify`` during private development.
It is now ``scaffold-sparse``; this module remains as a harmless compatibility
alias for early testers::

    import scaffold_sparsify as scaffold

    result = scaffold.fast(G, keep_ratio=0.2)

New code should use ``import scaffold`` (canonical) or ``import scaffold_sparse``.
"""

import scaffold as _scaffold
from scaffold import *  # noqa: F401,F403
from scaffold import __all__, __version__  # noqa: F401


def __getattr__(name):
    """Forward anything not re-exported above, including lazy submodules."""
    return getattr(_scaffold, name)
