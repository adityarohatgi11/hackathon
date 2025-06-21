"""Alias module forwarding to `hackathon.forecasting` package."""

from importlib import import_module
import sys

_module = import_module('hackathon.forecasting')
globals().update(_module.__dict__)
sys.modules[__name__] = _module 