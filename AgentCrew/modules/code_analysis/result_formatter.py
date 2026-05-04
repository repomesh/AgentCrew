from typing import Any, Dict, List, Set

from .text_map_formatter import TextMapFormatter
from .file_tree_formatter import FileTreeFormatter


class ResultFormatter:
    """Formats code analysis results into structured text reports."""

    def __init__(
        self,
        text_map_formatter: TextMapFormatter,
        file_tree_formatter: FileTreeFormatter,
        class_types: Set[str],
        function_types: Set[str],
        max_files_to_analyze: int,
    ):
        self._text_map_formatter = text_map_formatter
        self._file_tree_formatter = file_tree_formatter
        self._class_types = class_types
        self._function_types = function_types
        self._max_files_to_analyze = max_files_to_analyze

    @staticmethod
    def _count_nodes(structure: Dict[str, Any], node_types: Set[str]) -> int:
        """Recursively count nodes of specific types in the tree structure."""
        count = 0

        if structure.get("type") in node_types:
            count += 1

        for child in structure.get("children", []):
            count += ResultFormatter._count_nodes(child, node_types)

        return count

    def format_analysis_results(
        self,
        analysis_results: List[Dict[str, Any]],
        analyzed_files: List[str],
        errors: List[Dict[str, str]],
        non_analyzed_files: List[str] = [],
        total_supported_files: int = 0,
    ) -> str:
        """Format the analysis results into a clear text format.

        Args:
            analysis_results: List of analysis results for each file
            analyzed_files: List of files that were analyzed
            errors: List of errors encountered during analysis
            non_analyzed_files: List of files that were skipped due to file limit
            total_supported_files: Total number of supported files in the repository
        """

        total_files = len(analyzed_files)
        classes = sum(
            self._count_nodes(f["structure"], self._class_types)
            for f in analysis_results
        )
        functions = sum(
            self._count_nodes(f["structure"], self._function_types)
            for f in analysis_results
        )
        decorated_functions = sum(
            self._count_nodes(f["structure"], {"decorated_definition"})
            for f in analysis_results
        )
        error_count = len(errors)
        non_analyzed_count = len(non_analyzed_files)

        sections = []

        sections.append("\n===ANALYSIS STATISTICS===\n")
        sections.append(f"Total files analyzed: {total_files}")
        if non_analyzed_count > 0:
            sections.append(
                f"Total files skipped (repository too large): {non_analyzed_count}"
            )
            sections.append(
                f"Total supported files in repository: {total_supported_files}"
            )
        sections.append(f"Total errors: {error_count}")
        sections.append(f"Total classes found: {classes}")
        sections.append(f"Total functions found: {functions}")
        sections.append(f"Total decorated functions: {decorated_functions}")

        if errors:
            sections.append("\n===ERRORS===")
            for error in errors:
                error_first_line = error["error"].split("\n")[0]
                sections.append(f"{error['path']}: {error_first_line}")

        sections.append("\n===REPOSITORY STRUCTURE===")
        sections.append(self._text_map_formatter.generate_text_map(analysis_results))

        if non_analyzed_files:
            sections.append("\n===NON-ANALYZED FILES (repository too large)===")
            sections.append(
                f"The following {non_analyzed_count} files were not analyzed due to the {self._max_files_to_analyze} file limit:"
            )
            max_non_analyzed_to_show = int(self._max_files_to_analyze / 2)
            non_analyzed_tree = self._file_tree_formatter.build_file_tree(
                sorted(non_analyzed_files)[:max_non_analyzed_to_show]
            )
            non_analyzed_tree_lines = self._file_tree_formatter.format_file_tree(
                non_analyzed_tree
            )
            sections.extend(non_analyzed_tree_lines)
            if len(non_analyzed_files) > max_non_analyzed_to_show:
                sections.append(
                    f"...and {len(non_analyzed_files) - max_non_analyzed_to_show} more files."
                )

        return "\n".join(sections)
