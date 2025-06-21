"""Alias to 'hackathon.main' for backward compatibility with test-suite."""

from importlib import import_module
import sys

_module = import_module('hackathon.main')
globals().update(_module.__dict__)
sys.modules[__name__] = _module 