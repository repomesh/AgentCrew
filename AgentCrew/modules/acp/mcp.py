import re
from typing import Any

from loguru import logger

from AgentCrew.modules.mcpclient.config import MCPServerConfig

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_acp_mcp_servers(
    session_id: str,
    agent_name: str,
    mcp_servers: list[Any] | None,
) -> list[MCPServerConfig]:
    configs: list[MCPServerConfig] = []
    if not mcp_servers:
        return configs

    for index, server in enumerate(mcp_servers):
        command = _field(server, "command")
        url = _field(server, "url")
        server_type = str(_field(server, "type") or "").lower()
        raw_name = str(_field(server, "name") or f"server_{index + 1}")
        safe_name = _unique_server_name(session_id, raw_name, index)

        if server_type in {"http", "sse"}:
            if not url:
                logger.warning(
                    f"ACP MCP server '{raw_name}' ignored: http/sse server missing url"
                )
                continue
            configs.append(
                MCPServerConfig(
                    name=safe_name,
                    command="",
                    args=[],
                    env=None,
                    enabledForAgents=[agent_name],
                    streaming_server=True,
                    url=str(url),
                    headers=_normalize_headers(_field(server, "headers")),
                )
            )
        else:
            if not command:
                logger.warning(
                    f"ACP MCP server '{raw_name}' ignored: stdio server missing command"
                )
                continue
            configs.append(
                MCPServerConfig(
                    name=safe_name,
                    command=str(command),
                    args=[str(arg) for arg in (_field(server, "args") or [])],
                    env=_normalize_env(_field(server, "env")),
                    enabledForAgents=[agent_name],
                    streaming_server=False,
                )
            )

    return configs


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_env(env: Any) -> dict[str, str] | None:
    if not env:
        return None

    if isinstance(env, dict):
        return {str(key): str(value) for key, value in env.items()}

    if isinstance(env, list):
        normalized: dict[str, str] = {}
        for item in env:
            name = _field(item, "name")
            value = _field(item, "value")
            if name is None or value is None:
                continue
            normalized[str(name)] = str(value)
        return normalized or None

    return None


def _normalize_headers(headers: Any) -> dict[str, str] | None:
    if not headers:
        return None

    if isinstance(headers, dict):
        return {str(key): str(value) for key, value in headers.items()}

    if isinstance(headers, list):
        normalized: dict[str, str] = {}
        for item in headers:
            name = _field(item, "name")
            value = _field(item, "value")
            if name is None or value is None:
                continue
            normalized[str(name)] = str(value)
        return normalized or None

    return None


def _unique_server_name(session_id: str, raw_name: str, index: int) -> str:
    session_part = _safe_name(session_id).replace("agentcrew-", "")[:12] or "session"
    name_part = _safe_name(raw_name)[:32] or f"server_{index + 1}"
    return f"acp_{session_part}_{index + 1}_{name_part}"


def _safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value).strip("_")
