"""Alias module forwarding to `hackathon.control` package."""

from importlib import import_module
import sys

_module = import_module('hackathon.control')
# Re-export
globals().update(_module.__dict__)
# Minimise confusion for submodule imports
sys.modules[__name__] = _module 