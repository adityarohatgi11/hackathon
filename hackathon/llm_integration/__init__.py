"""
LLM Integration Module
Provides chat interface and decision explanation capabilities using Claude API and mock interface.
"""

from .mock_interface import MockLLMInterface
from .unified_interface import UnifiedLLMInterface

__all__ = ['MockLLMInterface', 'UnifiedLLMInterface'] 