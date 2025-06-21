"""Thin alias for `hackathon.api_client`.

Auto-generated to keep backward compatibility with test-suites that
expect the sub-package to be importable directly at the project root.
"""

from importlib import import_module
import sys

_module = import_module('hackathon.api_client')
# Re-export public attributes
globals().update(_module.__dict__)
# Ensure subsequent `import api_client.*` statements resolve correctly
sys.modules[__name__] = _module 