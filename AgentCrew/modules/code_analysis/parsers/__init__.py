"""
Language-specific parsers for code analysis.

This module provides a unified interface for parsing different programming languages
using tree-sitter.
"""

from .base import BaseLanguageParser
from .python_parser import PythonParser
from .javascript_parser import JavaScriptParser
from .java_parser import JavaParser
from .cpp_parser import CppParser
from .ruby_parser import RubyParser
from .go_parser import GoParser
from .rust_parser import RustParser
from .php_parser import PhpParser
from .csharp_parser import CSharpParser
from .kotlin_parser import KotlinParser
from .generic_parser import GenericParser

LANGUAGE_PARSER_MAP = {
    "python": PythonParser,
    "javascript": JavaScriptParser,
    "typescript": JavaScriptParser,
    "tsx": JavaScriptParser,
    "java": JavaParser,
    "cpp": CppParser,
    "ruby": RubyParser,
    "go": GoParser,
    "rust": RustParser,
    "php": PhpParser,
    "csharp": CSharpParser,
    "c_sharp": CSharpParser,
    "kotlin": KotlinParser,
}


def get_parser_for_language(language: str) -> BaseLanguageParser:
    """
    Get the appropriate parser for a given language.

    Args:
        language: The programming language name

    Returns:
        A parser instance for the language
    """
    parser_class = LANGUAGE_PARSER_MAP.get(language)
    if parser_class:
        return parser_class()
    return GenericParser(language)


__all__ = [
    "BaseLanguageParser",
    "PythonParser",
    "JavaScriptParser",
    "JavaParser",
    "CppParser",
    "RubyParser",
    "GoParser",
    "RustParser",
    "PhpParser",
    "CSharpParser",
    "KotlinParser",
    "GenericParser",
    "LANGUAGE_PARSER_MAP",
    "get_parser_for_language",
]
