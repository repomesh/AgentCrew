"""Pure data transformation functions for MCP configuration.

No QWidget dependency — only dict/list manipulation.
"""


def normalize_include_tools(include_tools) -> list[str]:
    """Normalize and deduplicate includeTools entries."""
    if not isinstance(include_tools, list):
        return []

    normalized_tools: list[str] = []
    seen_tools: set[str] = set()
    for tool_name in include_tools:
        if not isinstance(tool_name, str):
            continue
        normalized_tool_name = tool_name.strip()
        if not normalized_tool_name or normalized_tool_name in seen_tools:
            continue
        normalized_tools.append(normalized_tool_name)
        seen_tools.add(normalized_tool_name)

    return normalized_tools


def form_data_to_dict(
    name: str,
    streaming_server: bool,
    url: str,
    command: str,
    include_tools_str: str,
    args_list: list[str],
    env_dict: dict,
    headers_dict: dict,
    enabled_agents_list: list[str],
) -> dict:
    """Convert form field values into an MCP server config dict."""
    include_tools = normalize_include_tools(include_tools_str.split(","))
    return {
        "name": name.strip(),
        "command": command.strip(),
        "args": args_list,
        "env": env_dict,
        "enabledForAgents": enabled_agents_list,
        "streaming_server": streaming_server,
        "url": url.strip(),
        "headers": headers_dict,
        "includeTools": include_tools,
    }


def dict_to_form_fields(server_config: dict) -> dict:
    """Extract form-field values from an MCP server config dict.

    Returns a flat dict suitable for populating MCPForm.
    """
    return {
        "name": server_config.get("name", ""),
        "streaming_server": server_config.get("streaming_server", False),
        "url": server_config.get("url", ""),
        "command": server_config.get("command", ""),
        "include_tools_str": ", ".join(
            normalize_include_tools(server_config.get("includeTools"))
        ),
        "args": server_config.get("args", []),
        "env": server_config.get("env", {}),
        "headers": server_config.get("headers", {}),
        "enabled_agents": server_config.get("enabledForAgents", []),
    }
