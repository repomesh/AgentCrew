"""
Code Analysis Module

This module provides code structure analysis using tree-sitter and file search
capabilities with platform-specific optimizations.
"""

from .service import CodeAnalysisService
from .tree_sitter_runtime import TreeSitterRuntime
from .parsers import (
    BaseLanguageParser,
    get_parser_for_language,
    LANGUAGE_PARSER_MAP,
)

__all__ = [
    "CodeAnalysisService",
    "TreeSitterRuntime",
    "BaseLanguageParser",
    "get_parser_for_language",
    "LANGUAGE_PARSER_MAP",
]
