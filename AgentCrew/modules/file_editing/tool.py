"""
File editing tool definitions and handlers for AgentCrew.

Provides file_write_or_edit tool for intelligent file editing with search/replace blocks.
"""

from typing import Any, Callable
from .service import FileEditingService


def convert_blocks_to_string(blocks: list[dict[str, str]]) -> str:
    result_parts = []
    for block in blocks:
        search_text = block.get("search", "")
        replace_text = block.get("replace", "")
        block_str = (
            f"<<<<<<< SEARCH\n{search_text}\n=======\n{replace_text}\n>>>>>>> REPLACE"
        )
        result_parts.append(block_str)
    return "\n".join(result_parts)


def is_full_content_mode(blocks: list[dict[str, str]]) -> bool:
    if len(blocks) == 1:
        block = blocks[0]
        search_text = block.get("search", "")
        return search_text == ""
    return False


def get_file_write_or_edit_tool_definition() -> dict[str, Any]:
    tool_description = """Write/edit files via search/replace blocks.

FORMAT: Array of {"search": "...", "replace": "..."} objects
- Empty search + replace with content = write full file content
- Non-empty search + replace = search/replace operation
- Non-empty search + empty replace = delete matched content

RULES:
1. SEARCH must match exactly (character-perfect)
2. Include changing lines +0-3 context
3. Preserve whitespace/indentation

Auto syntax check (30+ langs) with rollback on error
"""

    tool_arguments = {
        "file_path": {
            "type": "string",
            "description": "Path (absolute/relative). Use ~ for home. Ex: './src/main.py'",
        },
        "text_or_search_replace_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Exact content to find. Empty string means full file write mode.",
                    },
                    "replace": {
                        "type": "string",
                        "description": "Replacement content (empty string to delete)",
                    },
                },
                "required": ["search", "replace"],
            },
            "description": 'Array of search/replace blocks. For full file write: [{"search": "", "replace": "full content"}]. For edits: [{"search": "exact match", "replace": "replacement"}]',
        },
    }

    tool_required = [
        "file_path",
        "text_or_search_replace_blocks",
    ]

    return {
        "type": "function",
        "function": {
            "name": "write_or_edit_file",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        },
    }


def get_file_write_or_edit_tool_handler(
    file_editing_service: FileEditingService,
) -> Callable:
    async def handle_file_write_or_edit(**params) -> str:
        file_path = params.get("file_path")
        blocks = params.get("text_or_search_replace_blocks")

        if not file_path:
            raise ValueError("Error: No file path provided.")

        if blocks is None:
            raise ValueError("Error: No search/replace blocks provided.")

        if not isinstance(blocks, list):
            raise ValueError(
                "Error: text_or_search_replace_blocks must be an array of search/replace objects."
            )

        full_content_mode = is_full_content_mode(blocks)

        if full_content_mode:
            content = blocks[0].get("replace", None)
            if content is None:
                raise ValueError(
                    "Error: text_or_search_replace_blocks must be an array of search/replace objects."
                )
            result = file_editing_service.write_or_edit_file(
                file_path=file_path,
                is_search_replace=False,
                text_or_search_replace_blocks=content,
            )
        else:
            blocks_string = convert_blocks_to_string(blocks)
            result = file_editing_service.write_or_edit_file(
                file_path=file_path,
                is_search_replace=True,
                text_or_search_replace_blocks=blocks_string,
            )

        if result["status"] == "success":
            parts = [f"{result['file_path']}"]
            parts.append(f"{result.get('changes_applied', 1)} change(s)")
            syntax = result.get("syntax_check", {})
            if syntax.get("is_valid"):
                parts.append(f"syntax OK ({syntax.get('language', '?')})")
            elif syntax.get("warnings"):
                warnings = "\n".join(
                    f"  L{w['line']}:C{w['column']} {w['message']}"
                    for w in syntax["warnings"][:5]
                )
                extra = (
                    f"\n  +{len(syntax['warnings']) - 5} more"
                    if len(syntax["warnings"]) > 5
                    else ""
                )
                parts.append(
                    f"syntax WARNING ({syntax.get('language', '?')}):\n{warnings}{extra}"
                )
            if result.get("backup_created"):
                parts.append("backup OK")
            return " | ".join(parts)

        elif result["status"] in ["no_match", "ambiguous"]:
            return f"{result['status'].title()}: {result.get('error', '?')} (block {result.get('block_index', '?')})"

        elif result["status"] == "denied":
            return f"Access denied: {result.get('error', 'Permission error')}"

        elif result["status"] == "parse_error":
            return f"Parse: {result.get('error', 'Invalid block format')}"

        else:
            return f"{result.get('error', 'Unknown error')}"

    return handle_file_write_or_edit


def register(service_instance: FileEditingService | None = None, agent=None):
    from AgentCrew.modules.tools.registration import register_tool

    if service_instance is None:
        service_instance = FileEditingService()

    register_tool(
        get_file_write_or_edit_tool_definition,
        get_file_write_or_edit_tool_handler,
        service_instance,
        agent,
    )
