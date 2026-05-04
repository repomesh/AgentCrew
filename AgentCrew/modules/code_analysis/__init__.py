"""
Code Analysis Module

This module provides code structure analysis using tree-sitter and file search
capabilities with platform-specific optimizations.
"""

from .service import CodeAnalysisService
from .tree_sitter_runtime import TreeSitterRuntime

__all__ = [
    "CodeAnalysisService",
    "TreeSitterRuntime",
]
