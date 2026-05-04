"""
Python language parser for code analysis.
"""

from typing import Any, Dict, Optional

from .base import BaseLanguageParser


class PythonParser(BaseLanguageParser):
    """Parser for Python source code."""

    @property
    def language_name(self) -> str:
        return "python"

    @staticmethod
    def _is_inside_function(node) -> bool:
        """Check if a node is inside a function body by walking the parent chain."""
        parent = node.parent
        while parent:
            if parent.type == "function_definition":
                return True
            parent = parent.parent
        return False

    def process_node(
        self, node, source_code: bytes, process_children_callback
    ) -> Optional[Dict[str, Any]]:
        result = self._create_base_result(node)

        if node.type in ["class_definition", "function_definition"]:
            for child in node.children:
                if child.type == "identifier":
                    result["name"] = self.extract_node_text(child, source_code)
                elif child.type == "parameters":
                    params = []
                    for param in child.children:
                        if "parameter" in param.type or param.type == "identifier":
                            params.append(self.extract_node_text(param, source_code))
                    if params:
                        result["parameters"] = params

        elif node.type == "assignment":
            if self._is_inside_function(node):
                return None

            var_name = None
            var_type = None
            for child in node.children:
                if child.type == "identifier" and var_name is None:
                    var_name = self.extract_node_text(child, source_code)
                elif child.type == "type":
                    var_type = self.extract_node_text(child, source_code)

            if var_name:
                result["type"] = "variable_declaration"
                if var_type:
                    result["name"] = f"{var_name}: {var_type}"
                else:
                    result["name"] = var_name
                return result

        children = []
        for child in node.children:
            child_result = process_children_callback(child)
            if child_result and self._is_significant_node(child_result):
                children.append(child_result)

        if children:
            result["children"] = children

        return result
