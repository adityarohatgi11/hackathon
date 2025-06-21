"""Alias module forwarding to `hackathon.dispatch` package."""

from importlib import import_module
import sys

_module = import_module('hackathon.dispatch')
globals().update(_module.__dict__)
sys.modules[__name__] = _module 