import os
import fnmatch
import subprocess
import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from loguru import logger

from .tree_sitter_runtime import TreeSitterRuntime, EXTENSION_TO_LANGUAGE
from .parsers import get_parser_for_language, BaseLanguageParser, LANGUAGE_PARSER_MAP
import mimetypes

IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
]

if TYPE_CHECKING:
    from AgentCrew.modules.llm.base import BaseLLMService

MAX_ITEMS_OUT = 30
MAX_FILES_TO_ANALYZE = 400


class CodeAnalysisService:
    """Service for analyzing code structure using tree-sitter."""

    LANGUAGE_MAP = EXTENSION_TO_LANGUAGE

    CUSTOM_PARSER_LANGUAGES = set(LANGUAGE_PARSER_MAP.keys())

    def __init__(self, llm_service: Optional["BaseLLMService"] = None):
        """Initialize the code analysis service with tree-sitter.

        Args:
            llm_service: Optional LLM service for intelligent file selection when
                        analyzing large repositories (>500 files).
        """
        self.llm_service = llm_service
        self.file_handler = None
        if self.llm_service:
            if self.llm_service.provider_name == "google":
                self.llm_service.model = "gemini-2.5-flash-lite"
            elif self.llm_service.provider_name == "claude":
                self.llm_service.model = "claude-3-5-haiku-latest"
            elif self.llm_service.provider_name == "openai":
                self.llm_service.model = "gpt-4.1-nano"
            elif self.llm_service.provider_name == "deepinfra":
                self.llm_service.model = "google/gemma-4-31B-it"
            elif self.llm_service.provider_name == "github_copilot":
                self.llm_service.model = "gpt-5-mini"
            elif self.llm_service.provider_name == "copilot_response":
                self.llm_service.model = "gpt-5.4-mini"
            elif self.llm_service.provider_name == "openai_codex":
                self.llm_service.model = "gpt-5.1-codex-mini"
            elif self.llm_service.provider_name == "together":
                self.llm_service.model = "Qwen/Qwen3.5-9B"

        self._runtime = TreeSitterRuntime.get_instance()
        self._language_parser_cache: Dict[str, BaseLanguageParser] = {}

        self.class_types = {
            "class_definition",
            "class_declaration",
            "class_specifier",
            "struct_specifier",
            "struct_declaration",
            "struct_item",
            "interface_declaration",
            "object_declaration",
        }

        self.function_types = {
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "constructor_declaration",
            "arrow_function",
            "fn_item",
            "method",
            "singleton_method",
            "primary_constructor",
        }

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language based on file extension."""
        lang = self._runtime.detect_language_for_file(file_path)
        return lang if lang else "unknown"

    def _has_custom_parser(self, language: str) -> bool:
        """Check if a custom rich parser exists for this language."""
        resolved = self._runtime._resolve_name(language)
        return (
            resolved in self.CUSTOM_PARSER_LANGUAGES
            or language in self.CUSTOM_PARSER_LANGUAGES
        )

    def _get_tree_sitter_parser(self, language: str):
        """Get the appropriate tree-sitter parser for a language (lazy, cached)."""
        return self._runtime.get_parser(language)

    def _get_language_parser(self, language: str) -> BaseLanguageParser:
        """Get the appropriate language parser for processing nodes."""
        if language not in self._language_parser_cache:
            self._language_parser_cache[language] = get_parser_for_language(language)
        return self._language_parser_cache[language]

    def _analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single file using tree-sitter."""
        try:
            with open(file_path, "rb") as f:
                source_code = f.read()

            language = self._detect_language(file_path)
            if language == "unknown":
                return {
                    "error": f"Unsupported file type: {os.path.splitext(file_path)[1]}"
                }

            if not self._runtime.is_in_manifest(language):
                return {
                    "error": f"Language '{language}' not available in tree-sitter pack"
                }

            tree_sitter_parser = self._get_tree_sitter_parser(language)

            tree = tree_sitter_parser.parse(source_code)
            root_node = tree.root_node

            if not root_node:
                return {"error": "Failed to parse file - no root node"}

            language_parser = self._get_language_parser(language)

            def process_node(node) -> Optional[Dict[str, Any]]:
                if not node:
                    return None
                return language_parser.process_node(node, source_code, process_node)

            return process_node(root_node)

        except Exception as e:
            return {"error": f"Error analyzing file: {str(e)}"}

    def _count_nodes(self, structure: Dict[str, Any], node_types: set[str]) -> int:
        """Recursively count nodes of specific types in the tree structure."""
        count = 0

        if structure.get("type") in node_types:
            count += 1

        for child in structure.get("children", []):
            count += self._count_nodes(child, node_types)

        return count

    async def _select_files_with_llm(
        self,
        files: List[str],
        max_files: int = MAX_FILES_TO_ANALYZE,
        feature_scope: Optional[str] = None,
    ) -> List[str]:
        """Use LLM to intelligently select which files to analyze from a large repository.

        Args:
            files: List of relative file paths to select from
            max_files: Maximum number of files to select

        Returns:
            List of selected file paths that should be analyzed
        """
        if not self.llm_service:
            return files[:max_files]

        feature_scope_instruction = ""
        if feature_scope:
            feature_scope_instruction = f"""
Feature focus:
- Prioritize files that are most relevant to this feature scope: {feature_scope}
- Prefer keeping files whose paths, modules, or responsibilities are closely related to that feature scope
- If tradeoffs are needed, keep feature-relevant files even when they would otherwise be lower priority than generic core files
- Still preserve critical shared/base/core files needed to understand the feature in context
"""

        prompt = f"""You are analyzing a code repository with {len(files)} files.
The analysis system can only process {max_files} files at a time.

Generate glob patterns to EXCLUDE less important files. The goal is to keep around {max_files} most important files after exclusion.
{feature_scope_instruction}
Files to EXCLUDE (generate patterns for these):
1. Test files
2. Generated/build files
3. Vendor/dependency files
4. Documentation files (e.g., **/docs/**, **/*.md)
5. Configuration duplicates and environment files
6. Migration files
7. Static assets (images, fonts, etc.)
8. Example/sample files

Files to KEEP (NEVER exclude) - ordered by priority:
1. Shared functions, utilities, and helper modules (e.g., utils/, helpers/, common/, shared/, lib/)
2. Base classes, abstract classes, and interfaces that other modules inherit from
3. Core application logic (main entry points, core modules)
4. Business features logic and domain models
5. API endpoints and controllers
6. Service classes and middleware
7. Key configuration files that define app structure
8. Files directly relevant to the requested feature scope, if one is provided

Here is the complete list of files:
{chr(10).join(files)}

Current file count: {len(files)}
Target file count: ~{max_files}
Files to exclude: ~{max(0, len(files) - max_files)}

Return ONLY a JSON array of glob patterns to exclude. Be strategic - use broad patterns when possible.

Example response format:
["**/tests/**", "**/test_*", "**/*.test.*", "**/docs/**", "**/migrations/**", "**/__pycache__/**"]"""

        try:
            response = await self.llm_service.process_message(prompt, temperature=0)

            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            exclude_patterns = json.loads(response)

            if isinstance(exclude_patterns, list):
                filtered_files = []
                for file_path in files:
                    excluded = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(file_path, pattern):
                            excluded = True
                            break
                    if not excluded:
                        filtered_files.append(file_path)

                logger.info(
                    f"LLM exclusion patterns reduced files from {len(files)} to {len(filtered_files)}"
                )

                return filtered_files[:max_files]
        except Exception as e:
            logger.warning(f"Cannot extract exclusion patterns from LLM response: {e}")

        return files[:max_files]

    KNOWN_RULE_FILES = [
        ".cursorrules",
        "CLAUDE.md",
        ".github/copilot-instructions.md",
        "CONVENTIONS.md",
        ".windsurfrules",
        "AGENTS.md",
        ".editorconfig",
        "CONTRIBUTING.md",
        ".ai/rules.md",
    ]

    async def extract_project_notes(self, analysis_result: str, repo_path: str) -> str:
        """Extract project notes, rules, and conventions from the analysis result using LLM.

        Sends the analyzed code structure to the LLM with a prompt to extract
        project-specific patterns, conventions, and rules. Also checks for
        known rule/instruction files in the repository.

        Args:
            analysis_result: The formatted analysis result string from analyze_code_structure
            repo_path: The root path of the repository being analyzed

        Returns:
            Structured project notes string for the agent to use as context
        """
        if not self.llm_service:
            return self._fallback_project_notes(repo_path)

        found_rule_files = []
        for rule_file in self.KNOWN_RULE_FILES:
            full_path = os.path.join(repo_path, rule_file)
            if os.path.isfile(full_path):
                found_rule_files.append(rule_file)

        rule_files_section = ""
        if found_rule_files:
            rule_files_section = f"""\n\nIMPORTANT: The following project rule/instruction files were detected in the repository:
{chr(10).join(f"- {f}" for f in found_rule_files)}

You MUST read these files using the get_file tool to understand project-specific rules and conventions before making any changes."""

        prompt = f"""You are analyzing a codebase structure to extract project notes and rules for a development assistant.

Based on the following code structure analysis, extract:

1. **Technology Stack**: Languages, frameworks, key libraries detected
2. **Architecture Pattern**: How the project is organized (e.g., MVC, modular, monorepo, microservices)
3. **Naming Conventions**: File naming, module naming, any patterns observed
4. **Key Entry Points**: Main files, configuration files, app bootstrapping
5. **Development Patterns**: Dependency injection, service layers, middleware patterns, etc.
6. **Project-Specific Rules**: Any conventions that a developer MUST follow when working on this codebase (e.g., where to place new files, how modules are registered, import patterns)

Code Structure Analysis:
{analysis_result}

Return a concise, structured summary in plain text (NOT JSON). Use clear headings and bullet points.
Focus only on actionable insights that help a developer understand how to work within this codebase.
Keep it under 500 words."""

        try:
            response = await self.llm_service.process_message(prompt, temperature=0)

            notes = response.strip()
            if rule_files_section:
                notes += rule_files_section

            logger.info("Successfully extracted project notes from analysis result")
            return notes
        except Exception as e:
            logger.warning(f"Failed to extract project notes via LLM: {e}")
            return self._fallback_project_notes(repo_path)

    def _fallback_project_notes(self, repo_path: str) -> str:
        """Generate minimal project notes when LLM is unavailable."""
        found_rule_files = []
        for rule_file in self.KNOWN_RULE_FILES:
            full_path = os.path.join(repo_path, rule_file)
            if os.path.isfile(full_path):
                found_rule_files.append(rule_file)

        notes = "Based on the code analysis, learn about the patterns and development flows, adapt project behaviors if possible for better response."
        if found_rule_files:
            notes += "\n\nIMPORTANT: The following project rule/instruction files were detected:\n"
            notes += chr(10).join(f"- {f}" for f in found_rule_files)
            notes += "\n\nYou MUST read these files using the get_file tool before making any changes."
        return notes

    async def analyze_code_structure(
        self,
        path: str,
        exclude_patterns: Optional[List[str]] = None,
        feature_scope: Optional[str] = None,
    ) -> Dict[str, Any] | str:
        """
        Build a tree-sitter based structural map of source code files in a git repository.

        Args:
            path: Root directory to analyze (must be a git repository)

        Returns:
            Dictionary containing analysis results for each file or formatted string
        """
        try:
            if exclude_patterns is None:
                exclude_patterns = []

            if not os.path.exists(path):
                return {"error": f"Path does not exist: {path}"}

            try:
                result = subprocess.run(
                    ["git", "ls-files"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                files = result.stdout.strip().split("\n")
            except subprocess.CalledProcessError:
                return {
                    "error": f"Failed to run git ls-files on {path}. Make sure it's a git repository."
                }

            supported_files_rel = []
            for file_path in files:
                excluded = False
                if file_path.strip():
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(file_path, pattern):
                            excluded = True
                            break
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in self.LANGUAGE_MAP and not excluded:
                        supported_files_rel.append(file_path)

            non_analyzed_files = []
            files_to_analyze = supported_files_rel

            if len(supported_files_rel) > MAX_FILES_TO_ANALYZE:
                selected_files = await self._select_files_with_llm(
                    supported_files_rel,
                    MAX_FILES_TO_ANALYZE,
                    feature_scope=feature_scope,
                )
                non_analyzed_files = [
                    f for f in supported_files_rel if f not in selected_files
                ]
                files_to_analyze = selected_files

            supported_files = [os.path.join(path, f) for f in files_to_analyze]

            analysis_results = []
            errors = []
            for file_path in supported_files:
                rel_path = os.path.relpath(file_path, path)
                try:
                    language = self._detect_language(file_path)

                    if language == "config":
                        if os.path.basename(file_path) == "package-lock.json":
                            continue
                        result = {"type": "config", "name": os.path.basename(file_path)}
                    else:
                        result = self._analyze_file(file_path)

                    if result and isinstance(result, dict) and "error" not in result:
                        analysis_results.append(
                            {
                                "path": rel_path,
                                "language": language,
                                "structure": result,
                            }
                        )
                    elif result and isinstance(result, dict) and "error" in result:
                        errors.append({"path": rel_path, "error": result["error"]})
                except Exception as e:
                    errors.append({"path": rel_path, "error": str(e)})

            if not analysis_results:
                return "Analysis completed but no valid results. This may due to excluded patterns is not correct"
            return self._format_analysis_results(
                analysis_results,
                supported_files,
                errors,
                non_analyzed_files,
                len(supported_files_rel),
            )

        except Exception as e:
            return {"error": f"Error analyzing directory: {str(e)}"}

    def _generate_text_map(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate a hierarchical text representation of the code structure analysis."""

        def format_node(
            node: Dict[str, Any], prefix: str = "", is_last: bool = True
        ) -> List[str]:
            lines = []

            node_type = node.get("type", "")
            node_name = node.get("name", "")
            node_lines = (
                f" //L: {node.get('start_line', '')}-{node.get('end_line', '')}"
            )

            if node_type == "decorated_definition" and "children" in node:
                for child in node.get("children", []):
                    if child.get("type") in {
                        "function_definition",
                        "method_definition",
                        "member_function_definition",
                    }:
                        return format_node(child, prefix, is_last)

            if not node_name and node_type in {
                "class_body",
                "block",
                "declaration_list",
                "body",
                "namespace_declaration",
                "lexical_declaration",
                "variable_declarator",
            }:
                return process_children(node.get("children", []), prefix, is_last)
            elif not node_name:
                return lines

            branch = "  "
            if node_type in {
                "class_definition",
                "class_declaration",
                "class_specifier",
                "class",
                "interface_declaration",
                "struct_specifier",
                "struct_declaration",
                "struct_item",
                "trait_item",
                "trait_declaration",
                "module",
                "type_declaration",
            }:
                node_info = f"class {node_name}{node_lines}"
            elif node_type in {
                "function_definition",
                "function_declaration",
                "method_definition",
                "method_declaration",
                "fn_item",
                "method",
                "singleton_method",
                "constructor_declaration",
                "member_function_definition",
                "constructor",
                "destructor",
                "public_method_definition",
                "private_method_definition",
                "protected_method_definition",
                "arrow_function",
                "lexical_declaration",
            }:
                if "first_line" in node:
                    node_info = node["first_line"] + node_lines
                else:
                    params = []
                    modfilers = ""
                    if "parameters" in node and node["parameters"]:
                        params = node["parameters"]
                    elif "children" in node:
                        for child in node["children"]:
                            if child.get("type") in {
                                "parameter_list",
                                "parameters",
                                "formal_parameters",
                                "argument_list",
                            }:
                                for param in child.get("children", []):
                                    if param.get("type") in {"identifier", "parameter"}:
                                        param_name = param.get("name", "")
                                        if param_name:
                                            params.append(param_name)

                    params_str = ", ".join(params) if params else ""
                    params_str = params_str.replace("\n", "")
                    if "modifiers" in node:
                        modfilers = " ".join(node["modifiers"]) + " "
                    node_info = f"{modfilers}{node_name}({params_str}){node_lines}"
            else:
                if "first_line" in node:
                    node_info = node["first_line"]
                else:
                    node_info = node_name

            if len(node_info) > 300:
                node_info = node_info[:297] + "... (REDACTED due to long content)"

            lines.append(f"{prefix}{branch}{node_info}")

            if "children" in node:
                new_prefix = prefix + "  "
                child_lines = process_children(node["children"], new_prefix, is_last)
                if child_lines:
                    lines.extend(child_lines)

            return lines

        def process_children(
            children: List[Dict], prefix: str, is_last: bool
        ) -> List[str]:
            if not children:
                return []

            lines = []
            significant_children = [
                child
                for child in children
                if child.get("type")
                in {
                    "arrow_function",
                    "call_expression",
                    "lexical_declaration",
                    "decorated_definition",
                    "class_definition",
                    "class_declaration",
                    "class_specifier",
                    "class",
                    "interface_declaration",
                    "struct_specifier",
                    "struct_declaration",
                    "struct_item",
                    "trait_item",
                    "trait_declaration",
                    "module",
                    "type_declaration",
                    "impl_item",
                    "function_definition",
                    "function_declaration",
                    "method_definition",
                    "method_declaration",
                    "fn_item",
                    "method",
                    "singleton_method",
                    "constructor_declaration",
                    "member_function_definition",
                    "constructor",
                    "destructor",
                    "public_method_definition",
                    "private_method_definition",
                    "protected_method_definition",
                    "class_body",
                    "block",
                    "declaration_list",
                    "body",
                    "impl_block",
                    "property_declaration",
                    "field_declaration",
                    "variable_declaration",
                    "const_declaration",
                }
            ]

            for i, child in enumerate(significant_children):
                is_last_child = i == len(significant_children) - 1
                child_lines = format_node(child, prefix, is_last_child)
                if child_lines:
                    lines.extend(child_lines)
                if i >= MAX_ITEMS_OUT:
                    lines.append(
                        f"{prefix}  ...({len(significant_children) - MAX_ITEMS_OUT} more items)"
                    )
                    break

            return lines

        def get_file_code_content(
            result: Dict[str, Any], file_indent: str
        ) -> List[str]:
            """Generate code structure content for a single file."""
            lines = []
            structure = result.get("structure")
            if not structure:
                return lines

            if not structure.get("children"):
                if structure.get("type"):
                    return [f"{file_indent}  {structure['type']}"]
                return lines

            significant_nodes = [
                child
                for child in structure["children"]
                if child.get("type")
                in {
                    "arrow_function",
                    "lexical_declaration",
                    "call_expression",
                    "decorated_definition",
                    "class_definition",
                    "class_declaration",
                    "class_specifier",
                    "class",
                    "interface_declaration",
                    "struct_specifier",
                    "struct_declaration",
                    "struct_item",
                    "trait_item",
                    "trait_declaration",
                    "module",
                    "type_declaration",
                    "impl_item",
                    "function_definition",
                    "function_declaration",
                    "method_definition",
                    "method_declaration",
                    "fn_item",
                    "method",
                    "singleton_method",
                    "constructor_declaration",
                    "member_function_definition",
                    "constructor",
                    "destructor",
                    "public_method_definition",
                    "private_method_definition",
                    "protected_method_definition",
                    "property_declaration",
                    "field_declaration",
                    "variable_declaration",
                    "const_declaration",
                    "namespace_declaration",
                }
            ]

            for i, node in enumerate(significant_nodes):
                is_last = i == len(significant_nodes) - 1
                node_lines = format_node(node, file_indent, is_last)
                if node_lines:
                    lines.extend(node_lines)
                if i >= MAX_ITEMS_OUT:
                    lines.append(
                        f"{file_indent}  ...({len(significant_nodes) - MAX_ITEMS_OUT} more items)"
                    )
                    break
            return lines

        sorted_results = sorted(analysis_results, key=lambda x: x["path"])

        results_by_path = {result["path"]: result for result in sorted_results}

        tree: Dict[str, Any] = {}
        for result in sorted_results:
            path = result["path"].replace("\\", "/")
            parts = path.split("/")
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = {"__is_file__": True, "__path__": result["path"]}
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        output_lines = []

        def format_tree(node: Dict[str, Any], indent: str = "") -> None:
            items = sorted(node.keys())
            for name in items:
                child = node[name]
                if isinstance(child, dict) and child.get("__is_file__"):
                    output_lines.append(f"{indent}{name}")
                    file_path = child["__path__"]
                    if file_path in results_by_path:
                        file_content = get_file_code_content(
                            results_by_path[file_path], indent
                        )
                        output_lines.extend(file_content)
                elif isinstance(child, dict):
                    output_lines.append(f"{indent}{name}/")
                    format_tree(child, indent + "  ")

        format_tree(tree)

        return (
            "\n".join(output_lines)
            if output_lines
            else "No significant code structure found."
        )

    def get_file_content(
        self,
        file_path,
        start_line=None,
        end_line=None,
    ) -> Union[Tuple[str, str], Tuple[str, Dict[str, Any]]]:
        """
        Return the content of a file, optionally reading only a specific line range.
        For document files (PDF, DOCX, XLSX, PPTX), uses Docling to convert
        to text/markdown and ignores start_line/end_line parameters.
        For image files, returns base64 encoded data in image_url format.

        Args:
            file_path: Path to the file to read
            start_line: Optional starting line number (1-indexed) - ignored for document files
            end_line: Optional ending line number (1-indexed, inclusive) - ignored for document files

        Returns:
            Tuple of (file_path, content) where content is either:
            - str: text content for text/document files
            - dict: {"type": "image_url", "image_url": {"url": "data:mime;base64,..."}} for images
        """

        from AgentCrew.modules.utils.file_handler import (
            FileHandler,
            ALLOWED_MIME_TYPES,
        )

        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type and mime_type in IMAGE_MIME_TYPES:
            with open(file_path, "rb") as file:
                binary_data = file.read()
            base64_data = base64.b64encode(binary_data).decode("utf-8")
            return file_path, {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            }

        if mime_type and mime_type in ALLOWED_MIME_TYPES:
            if self.file_handler is None:
                self.file_handler = FileHandler()
            result = self.file_handler.process_file(file_path)
            if result and "text" in result:
                return file_path, result["text"]
            elif result is None:
                raise ValueError(f"Failed to process document file: {file_path}")

        with open(file_path, "rb") as file:
            content = file.read()

        decoded_content = content.decode("utf-8")

        if start_line is not None and end_line is not None:
            if start_line < 1:
                raise ValueError("start_line must be >= 1")
            if end_line < start_line:
                raise ValueError("end_line must be >= start_line")

            lines = decoded_content.split("\n")
            total_lines = len(lines)

            if start_line > total_lines:
                raise ValueError(
                    f"start_line {start_line} exceeds file length ({total_lines} lines)"
                )
            if end_line > total_lines:
                end_line = total_lines

            selected_lines = lines[start_line - 1 : end_line]
            return file_path, "\n".join(selected_lines)

        return file_path, decoded_content

        return file_path, decoded_content

    def _build_file_tree(self, file_paths: List[str]) -> Dict[str, Any]:
        """Build a hierarchical tree structure from flat file paths.

        Args:
            file_paths: List of relative file paths

        Returns:
            Nested dictionary representing the file tree
        """
        tree: Dict[str, Any] = {}
        for path in sorted(file_paths):
            parts = path.replace("\\", "/").split("/")
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = None
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        return tree

    def _format_file_tree(self, tree: Dict[str, Any], indent: str = "") -> List[str]:
        """Format a file tree dictionary into indented lines.

        Args:
            tree: Nested dictionary representing file tree
            indent: Current indentation string

        Returns:
            List of formatted lines
        """
        lines = []
        items = sorted(tree.keys())
        for name in items:
            subtree = tree[name]
            if subtree is None:
                lines.append(f"{indent}{name}")
            else:
                lines.append(f"{indent}{name}/")
                child_lines = self._format_file_tree(subtree, indent + "  ")
                lines.extend(child_lines)
        return lines

    def _format_analysis_results(
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
            self._count_nodes(f["structure"], self.class_types)
            for f in analysis_results
        )
        functions = sum(
            self._count_nodes(f["structure"], self.function_types)
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
        sections.append(self._generate_text_map(analysis_results))

        if non_analyzed_files:
            sections.append("\n===NON-ANALYZED FILES (repository too large)===")
            sections.append(
                f"The following {non_analyzed_count} files were not analyzed due to the {MAX_FILES_TO_ANALYZE} file limit:"
            )
            max_non_analyzed_to_show = int(MAX_FILES_TO_ANALYZE / 2)
            non_analyzed_tree = self._build_file_tree(
                sorted(non_analyzed_files)[:max_non_analyzed_to_show]
            )
            non_analyzed_tree_lines = self._format_file_tree(non_analyzed_tree)
            sections.extend(non_analyzed_tree_lines)
            if len(non_analyzed_files) > max_non_analyzed_to_show:
                sections.append(
                    f"...and {len(non_analyzed_files) - max_non_analyzed_to_show} more files."
                )

        return "\n".join(sections)
