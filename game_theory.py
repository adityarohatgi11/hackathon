"""Alias module forwarding to `hackathon.game_theory` subpackage."""

from importlib import import_module
import sys

_pkg = import_module('hackathon.game_theory')
# Ensure child modules are visible at top-level import path
for _sub in list(getattr(_pkg, '__all__', [])):
    try:
        sys.modules[f'game_theory.{_sub}'] = import_module(f'hackathon.game_theory.{_sub}')
    except ImportError:
        # Skip if submodule cannot be imported – may not exist yet.
        pass

sys.modules[__name__] = _pkg
# Re-export attributes
globals().update(_pkg.__dict__) 