"""Top-level *hackathon* package initialisation.

The package re-exports frequently used sub-modules at the global scope so
that they can be imported plainly, e.g. ``import api_client`` instead of
``from hackathon import api_client``.  This is mostly for backward
compatibility with the existing test-suite and third-party notebooks.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import List

# List of sub-packages we want to promote to top-level modules.  The names
# correspond to directories inside this package.
_PROMOTED_SUBMODULES: List[str] = [
    "api_client",
    "forecasting",
    "game_theory",
    "control",
    "dispatch",
]

for _name in _PROMOTED_SUBMODULES:
    # Import the sub-package relatively and register it under the plain
    # name – *and* under ``hackathon.<name>`` to avoid duplication.
    full_name = f"{__name__}.{_name}"
    module: ModuleType = importlib.import_module(full_name)
    sys.modules[_name] = module  # exposed as top-level package

__all__ = _PROMOTED_SUBMODULES  # Re-export for *from hackathon import * 