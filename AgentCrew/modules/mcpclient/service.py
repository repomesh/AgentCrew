from __future__ import annotations

import asyncio
import base64
import os
import random
import tempfile
import threading
from collections.abc import Iterable
from urllib.parse import unquote, urlparse

from loguru import logger
from typing import TYPE_CHECKING

from AgentCrew.modules.utils.file_handler import optimize_image_data_uri
from mcp import ClientSession, StdioServerParameters
from mcp.types import (
    BlobResourceContents,
    ImageContent,
    Resource,
    PaginatedRequestParams,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from AgentCrew.modules.agents import LocalAgent, AgentManager
from .auth import OAuthClientResolver, FileTokenStorage, InlineTokenStorage
from AgentCrew.modules import FileLogIO

if TYPE_CHECKING:
    from typing import Any, Callable
    from mcp.types import ContentBlock, Prompt, Tool
    from AgentCrew.modules.utils.file_handler import FileHandler
    from .config import MCPServerConfig

# Initialize the logger
mcp_log_io = FileLogIO("mcpclient_agentcrew")
MCP_OPERATION_TIMEOUT_SECONDS = 300.0


class MCPService:
    """Service for interacting with Model Context Protocol (MCP) servers."""

    def __init__(self, ping_interval: int = 30):
        """Initialize the MCP service.

        Args:
            ping_interval: Interval in seconds between keep-alive pings (default: 30)
        """
        self.sessions: dict[str, ClientSession] = {}
        self.connected_servers: dict[str, bool] = {}
        self.tools_cache: dict[str, dict[str, Tool]] = {}
        self.loop = asyncio.new_event_loop()
        self._server_connection_tasks: dict[str, asyncio.Task] = {}
        self._server_shutdown_events: dict[str, asyncio.Event] = {}
        self._server_keepalive_tasks: dict[str, asyncio.Task] = {}
        self.server_prompts: dict[str, list[Prompt]] = {}
        self.server_resources: dict[str, list[dict[str, Any]]] = {}
        self.tokens_storage_cache: dict[str, FileTokenStorage] = {}
        self.file_handler: FileHandler | None = None
        self.ping_interval = ping_interval

    def _get_file_handler(self) -> FileHandler:
        if self.file_handler is None:
            from AgentCrew.modules.utils.file_handler import FileHandler

            self.file_handler = FileHandler()
        return self.file_handler

    async def _manage_single_connection(
        self, server_config: MCPServerConfig, agent_name: str | None = None
    ):
        """Manages the lifecycle of a single MCP server connection."""
        server_name = server_config.name
        combined_server_id = self._get_server_id_format(server_name, agent_name)
        shutdown_event = asyncio.Event()
        self._server_shutdown_events[combined_server_id] = shutdown_event
        logger.info(f"MCPService: Starting connection management for {server_name}")

        try:
            if server_config.streaming_server:
                # Import here to avoid import errors if not available

                logger.info(
                    f"MCPService: Using streaming HTTP client for {server_name}"
                )
                # Prepare headers for the streamable HTTP client
                headers = server_config.headers if server_config.headers else {}

                # Backward compatible with SSE
                token_storage = self._build_token_storage(server_config)
                client_info = await token_storage.get_client_info()
                if client_info and client_info.redirect_uris:
                    port = client_info.redirect_uris[0].port
                else:
                    port = random.randint(14100, 14200)

                oauth_resolver = OAuthClientResolver(port=port)

                if server_config.url.endswith("/sse"):
                    session_context = sse_client(
                        server_config.url,
                        headers=headers,
                        auth=oauth_resolver.get_oauth_client_provider(
                            server_config.url, token_storage
                        ),
                        sse_read_timeout=60 * 60 * 24,
                    )
                else:
                    from httpx import AsyncClient, Timeout

                    session_context = streamable_http_client(
                        server_config.url,
                        http_client=AsyncClient(
                            headers=headers,
                            auth=oauth_resolver.get_oauth_client_provider(
                                server_config.url, token_storage
                            ),
                            timeout=Timeout(60, read=60 * 60 * 24),
                        ),
                    )

                async with session_context as stream_context:
                    logger.info(
                        f"MCPService: streamablehttp_client established for {server_name}"
                    )
                    read_stream = stream_context[0]
                    write_stream = stream_context[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        logger.info(
                            f"MCPService: ClientSession established for {server_name}"
                        )
                        server_info = await session.initialize()
                        self.sessions[combined_server_id] = session
                        self.connected_servers[combined_server_id] = True
                        logger.info(
                            f"MCPService: {server_name} connected. Registering tools..."
                        )

                        target_agent_names = self._target_agent_names(
                            agent_name, server_config
                        )
                        if server_info.capabilities.resources:
                            await self.register_server_resources(
                                combined_server_id,
                                server_config,
                                target_agent_names,
                            )

                        for enabled_agent_name in target_agent_names:
                            await self.register_server_tools(
                                combined_server_id,
                                server_config,
                                enabled_agent_name,
                            )

                        if server_info.capabilities.prompts:
                            prompts = await self.sessions[
                                combined_server_id
                            ].list_prompts()
                            self.server_prompts[server_name] = prompts.prompts

                        logger.info(
                            f"MCPService: {server_name} setup complete. Starting keep-alive task."
                        )

                        keepalive_task = asyncio.create_task(
                            self._keepalive_worker(combined_server_id, shutdown_event)
                        )
                        self._server_keepalive_tasks[combined_server_id] = (
                            keepalive_task
                        )

                        logger.info(
                            f"MCPService: {server_name} keep-alive task started. Waiting for shutdown signal."
                        )
                        await shutdown_event.wait()

                        if combined_server_id in self._server_keepalive_tasks:
                            keepalive_task.cancel()
                            try:
                                await keepalive_task
                            except asyncio.CancelledError:
                                logger.info(
                                    f"MCPService: Keep-alive task for {server_name} cancelled."
                                )
            else:
                # Original stdio client logic
                server_params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env,
                )

                async with stdio_client(server_params, errlog=mcp_log_io) as (
                    read_stream,
                    write_stream,
                ):
                    logger.info(
                        f"MCPService: stdio_client established for {server_name}"
                    )
                    async with ClientSession(read_stream, write_stream) as session:
                        logger.info(
                            f"MCPService: ClientSession established for {server_name}"
                        )
                        retried = 0
                        server_info = None
                        while retried < 3:
                            try:
                                server_info = await session.initialize()
                                break
                            except Exception:
                                print("Retrying MCP connection...")
                                await asyncio.sleep(retried * 2)
                                retried += 1

                        if not server_info:
                            raise Exception("Failed to initialize MCP session.")

                        self.sessions[combined_server_id] = session
                        self.connected_servers[combined_server_id] = (
                            True  # Mark as connected before tool registration
                        )
                        logger.info(
                            f"MCPService: {combined_server_id} connected. Registering tools..."
                        )

                        target_agent_names = self._target_agent_names(
                            agent_name, server_config
                        )
                        if server_info.capabilities.resources:
                            await self.register_server_resources(
                                combined_server_id,
                                server_config,
                                target_agent_names,
                            )
                        for enabled_agent_name in target_agent_names:
                            await self.register_server_tools(
                                combined_server_id,
                                server_config,
                                enabled_agent_name,
                            )

                        if server_info.capabilities.prompts:
                            prompts = await self.sessions[
                                combined_server_id
                            ].list_prompts()
                            self.server_prompts[server_name] = prompts.prompts

                        logger.info(
                            f"MCPService: {server_name} setup complete. Starting keep-alive task."
                        )

                        keepalive_task = asyncio.create_task(
                            self._keepalive_worker(combined_server_id, shutdown_event)
                        )
                        self._server_keepalive_tasks[combined_server_id] = (
                            keepalive_task
                        )

                        logger.info(
                            f"MCPService: {server_name} keep-alive task started. Waiting for shutdown signal."
                        )
                        await shutdown_event.wait()

                        if combined_server_id in self._server_keepalive_tasks:
                            keepalive_task.cancel()
                            try:
                                await keepalive_task
                            except asyncio.CancelledError:
                                logger.info(
                                    f"MCPService: Keep-alive task for {server_name} cancelled."
                                )

        except asyncio.CancelledError:
            logger.info(f"MCPService: Connection task for {server_name} was cancelled.")
        except Exception:
            logger.exception(
                f"MCPService: Error in connection management for '{server_name}'"
            )
            agent_manager = AgentManager.get_instance()
            if agent_name:
                agent = agent_manager.get_local_agent(agent_name)
                if agent and combined_server_id in agent.mcps_loading:
                    agent.mcps_loading.remove(combined_server_id)
            else:
                for enabled_agent_name in server_config.enabledForAgents:
                    enabled_agent = agent_manager.get_local_agent(enabled_agent_name)
                    if (
                        enabled_agent
                        and combined_server_id in enabled_agent.mcps_loading
                    ):
                        enabled_agent.mcps_loading.remove(combined_server_id)
        finally:
            logger.info(f"MCPService: Cleaning up connection for {server_name}.")
            self.sessions.pop(combined_server_id, None)
            self.connected_servers.pop(combined_server_id, False)
            self.tools_cache.pop(combined_server_id, None)
            self._server_shutdown_events.pop(combined_server_id, None)
            self._server_keepalive_tasks.pop(combined_server_id, None)
            logger.info(f"MCPService: Cleanup for {server_name} complete.")

    async def _keepalive_worker(self, server_id: str, shutdown_event: asyncio.Event):
        """
        Periodically send ping requests to keep the session alive and refresh tokens.

        Args:
            server_id: ID of the server to send pings to
            shutdown_event: Event that signals when to stop the keep-alive loop
        """
        logger.info(f"MCPService: Keep-alive worker started for {server_id}")

        try:
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(), timeout=self.ping_interval
                    )
                    break
                except asyncio.TimeoutError:
                    if server_id in self.sessions and self.connected_servers.get(
                        server_id
                    ):
                        try:
                            session = self.sessions[server_id]
                            await session.send_ping()
                            logger.debug(f"MCPService: Sent ping to {server_id}")
                        except Exception as e:
                            logger.warning(
                                f"MCPService: Failed to send ping to {server_id}: {e}"
                            )
                    else:
                        logger.warning(
                            f"MCPService: Session for {server_id} not available, stopping keep-alive"
                        )
                        break
        except asyncio.CancelledError:
            logger.info(f"MCPService: Keep-alive worker for {server_id} cancelled")
            raise
        except Exception as e:
            logger.exception(
                f"MCPService: Error in keep-alive worker for {server_id}: {e}"
            )
        finally:
            logger.info(f"MCPService: Keep-alive worker for {server_id} stopped")

    def _get_server_id_format(
        self, server_name: str, agent_name: str | None = None
    ) -> str:
        """Format server ID with optional agent name prefix."""
        return f"{agent_name}__{server_name}" if agent_name else server_name

    def _target_agent_names(
        self, agent_name: str | None, server_config: MCPServerConfig
    ) -> list[str]:
        if agent_name:
            return [agent_name]
        return list(server_config.enabledForAgents)

    async def _list_all_resources(self, session: ClientSession) -> list[Resource]:
        resources: list[Resource] = []
        cursor = None
        for _ in range(50):
            response = await session.list_resources(
                params=PaginatedRequestParams(cursor=cursor)
            )
            resources.extend(response.resources)
            cursor = response.nextCursor
            if not cursor:
                return resources
        return resources

    async def register_server_resources(
        self,
        combined_server_id: str,
        server_config: MCPServerConfig,
        agent_names: Iterable[str],
    ) -> None:

        try:
            resources = [
                self._format_resource_for_agent(resource)
                for resource in await self._list_all_resources(
                    self.sessions[combined_server_id]
                )
            ]
            self.server_resources[server_config.name] = resources

            for agent_name in agent_names:
                agent = AgentManager.get_instance().get_local_agent(agent_name)
                if not agent:
                    continue
                agent.mcp_resources[server_config.name] = resources
                self._register_get_resource_tool(
                    agent, combined_server_id, server_config.name
                )

            logger.info(
                f"MCPService: Registered {len(resources)} resources for server '{combined_server_id}'"
            )
        except Exception:
            logger.exception(
                f"Error listing resources from server '{combined_server_id}'"
            )

    def _format_resource_for_agent(self, resource: Resource) -> dict[str, Any]:
        data = {"uri": str(resource.uri), "name": resource.name}
        if resource.title:
            data["title"] = resource.title
        if resource.description:
            data["description"] = resource.description
        if resource.mimeType:
            data["mimeType"] = resource.mimeType
        return data

    def _get_resource_tool_definition(self, server_name: str) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": f"{server_name}__get_resource",
                "description": f"Fetch the content of an available MCP resource from server '{server_name}' by URI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "The exact MCP resource URI to fetch.",
                        }
                    },
                    "required": ["uri"],
                },
            },
        }

    def _register_get_resource_tool(
        self, agent: LocalAgent, combined_server_id: str, server_name: str
    ) -> None:
        agent.register_tool(
            lambda: self._get_resource_tool_definition(server_name),
            self._create_get_resource_handler(combined_server_id, server_name),
            self,
        )

    def _create_get_resource_handler(
        self, combined_server_id: str, server_name: str
    ) -> Callable:
        def handler_factory(service_instance=None):
            async def get_resource(uri: str) -> list[dict[str, Any]]:
                from pydantic import AnyUrl

                resource_uri = uri.strip() if uri else ""
                if not resource_uri:
                    raise ValueError("Resource URI is required")
                if (
                    combined_server_id not in self.sessions
                    or not self.connected_servers.get(combined_server_id)
                ):
                    raise ValueError(
                        f"Cannot read resource: Server '{server_name}' is not connected"
                    )
                resource = next(
                    (
                        res
                        for res in self.server_resources.get(server_name, [])
                        if res.get("uri") == resource_uri
                    ),
                    None,
                )
                if not resource:
                    raise ValueError(
                        f"Resource URI is not available on MCP server '{server_name}': {resource_uri}"
                    )

                resource_link = ResourceLink(
                    type="resource_link",
                    uri=AnyUrl(resource_uri),
                    name=resource.get("name", resource_uri),
                    description=resource.get("description"),
                    mimeType=resource.get("mimeType"),
                )
                return await self._format_resource_link_async(
                    resource_link, self.sessions[combined_server_id]
                )

            return get_resource

        return handler_factory

    def _get_or_create_token_storage(self, server_name: str) -> FileTokenStorage:
        """
        Get or create a FileTokenStorage instance for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            FileTokenStorage instance for the server
        """
        if server_name not in self.tokens_storage_cache:
            self.tokens_storage_cache[server_name] = FileTokenStorage(server_name)
            logger.info(
                f"MCPService: Created new FileTokenStorage for server '{server_name}'"
            )
        return self.tokens_storage_cache[server_name]

    def _build_token_storage(self, server_config: MCPServerConfig):
        """Build effective token storage using cached file storage plus optional config overrides."""
        base_storage = self._get_or_create_token_storage(server_config.name)
        oauth_override = getattr(server_config, "oauth", None)
        if not oauth_override:
            return base_storage

        return InlineTokenStorage(
            base_storage=base_storage,
            tokens_override=oauth_override.tokens,
            client_info_override=oauth_override.client_info,
        )

    async def start_server_connection_management(
        self, server_config: MCPServerConfig, agent_name: str | None = None
    ):
        """Starts and manages the connection for a single MCP server."""
        combined_server_id = self._get_server_id_format(server_config.name, agent_name)
        if (
            combined_server_id in self._server_connection_tasks
            and not self._server_connection_tasks[combined_server_id].done()
        ):
            logger.info(
                f"MCPService: Connection management for {combined_server_id} already in progress."
            )
            return

        agent_manager = AgentManager.get_instance()
        if agent_name:
            agent = agent_manager.get_local_agent(agent_name)
            if agent:
                agent.mcps_loading.append(combined_server_id)
        else:
            for enabled_agent_name in server_config.enabledForAgents:
                enabled_agent = agent_manager.get_local_agent(enabled_agent_name)
                if enabled_agent:
                    enabled_agent.mcps_loading.append(combined_server_id)

        logger.info(
            f"MCPService: Creating task for _manage_single_connection for {combined_server_id}"
        )
        if self.loop.is_closed():
            logger.warning(
                "MCPService: Loop is closed, cannot create task for server connection."
            )
            return
        task = self.loop.create_task(
            self._manage_single_connection(server_config, agent_name)
        )
        self._server_connection_tasks[combined_server_id] = task

    async def shutdown_all_server_connections(self, agent_name: str | None = None):
        """Signals all active server connections to shut down and waits for them."""
        logger.info("MCPService: Shutting down all server connections...")
        active_tasks = []
        for server_id, event in list(self._server_shutdown_events.items()):
            if agent_name and agent_name not in server_id:
                continue  # Skip servers not matching the agent name
            extracted_server_id = server_id.replace(
                f"{agent_name}__", ""
            )  # Remove prefix if present
            await self.deregister_server_tools(extracted_server_id, agent_name)
            logger.info(f"MCPService: Signaling shutdown for {server_id}")
            event.set()
            if server_id in self._server_connection_tasks:
                task = self._server_connection_tasks[server_id]
                if task and not task.done():
                    active_tasks.append(task)

        if active_tasks:
            logger.info(
                f"MCPService: Waiting for {len(active_tasks)} connection tasks to complete..."
            )
            await asyncio.gather(*active_tasks, return_exceptions=True)

        self._server_connection_tasks.clear()
        logger.info("MCPService: All server connections shut down process initiated.")

    async def shutdown_single_server_connection(self, server_id: str):
        """Signals a specific server connection to shut down and waits for it."""
        logger.info(f"MCPService: Shutting down connection for server {server_id}...")
        if server_id in self._server_shutdown_events:
            event = self._server_shutdown_events[server_id]
            event.set()
            logger.info(f"MCPService: Shutdown signal sent to {server_id}.")

            task = self._server_connection_tasks.get(server_id)
            if task and not task.done():
                logger.info(
                    f"MCPService: Waiting for connection task of {server_id} to complete..."
                )
                try:
                    await asyncio.wait_for(
                        task, timeout=10.0
                    )  # Wait for task to finish
                    logger.info(
                        f"MCPService: Connection task for {server_id} completed."
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"MCPService: Timeout waiting for {server_id} connection task to complete. It might be stuck."
                    )
                    task.cancel()  # Force cancel if it didn't finish
                except Exception as e:
                    logger.error(f"MCPService: Error waiting for {server_id} task: {e}")
            else:
                logger.info(
                    f"MCPService: No active task found for {server_id} or task already done."
                )
        else:
            logger.warning(
                f"MCPService: No shutdown event found for server {server_id}. It might not be running or already shut down."
            )

        # Clean up entries related to this server, though _manage_single_connection's finally should handle most
        self._server_connection_tasks.pop(server_id, None)
        logger.info(
            f"MCPService: Shutdown process for {server_id} initiated/completed."
        )

    def _filter_tools_for_registration(
        self,
        tools: list[Tool],
        include_tools: list[str] | None,
        combined_server_id: str,
    ) -> list[Tool]:
        if not include_tools:
            return tools

        tool_map = {tool.name: tool for tool in tools}
        filtered_tools = [tool_map[name] for name in include_tools if name in tool_map]
        unmatched_tools = [name for name in include_tools if name not in tool_map]

        if unmatched_tools:
            logger.warning(
                f"MCPService: MCP server '{combined_server_id}' includeTools contains unknown tools: {', '.join(unmatched_tools)}"
            )

        if not filtered_tools:
            logger.warning(
                f"MCPService: MCP server '{combined_server_id}' matched 0 tools from includeTools filter. Server remains connected but no tools will be registered."
            )

        return filtered_tools

    async def register_server_tools(
        self,
        combined_server_id: str,
        server_config: MCPServerConfig,
        agent_name: str,
    ) -> None:
        """
        Register all tools from a connected server.

        Args:
            server_id: ID of the server to register tools from
        """

        if combined_server_id not in self.sessions or not self.connected_servers.get(
            combined_server_id
        ):
            logger.warning(
                f"Cannot register tools: Server '{combined_server_id}' is not connected"
            )
            return

        try:
            response = await self.sessions[combined_server_id].list_tools()
            server_name = server_config.name

            self.tools_cache[combined_server_id] = {
                tool.name: tool for tool in response.tools
            }
            filtered_tools = self._filter_tools_for_registration(
                response.tools,
                server_config.includeTools,
                combined_server_id,
            )

            logger.info(
                f"MCPService: Registering {len(filtered_tools)}/{len(response.tools)} tools for server '{combined_server_id}' on agent '{agent_name}'"
            )

            if agent_name:
                agent_manager = AgentManager.get_instance()
                registry = agent_manager.get_local_agent(agent_name)
            else:
                registry = None
            for tool in filtered_tools:

                def tool_definition_factory(tool_info=tool, srv_id=server_name):
                    def get_definition():
                        return self._format_tool_definition(tool_info, srv_id)

                    return get_definition

                handler_factory = self._create_tool_handler(
                    combined_server_id, tool.name
                )

                if registry:
                    registry.register_tool(
                        tool_definition_factory(), handler_factory, self
                    )
            if (
                isinstance(registry, LocalAgent)
                and combined_server_id in registry.mcps_loading
            ):
                registry.mcps_loading.remove(combined_server_id)

        except Exception:
            logger.exception(
                f"Error registering tools from server '{combined_server_id}'"
            )
            self.connected_servers[combined_server_id] = False

    def _remove_agent_server_tool_definitions(
        self, local_agent: LocalAgent, server_name: str
    ) -> None:
        namespaced_prefix = f"{server_name}__"
        tool_names_to_remove = [
            tool_name
            for tool_name in local_agent.tool_definitions.keys()
            if tool_name.startswith(namespaced_prefix)
        ]

        if not tool_names_to_remove:
            return

        if local_agent.is_active:
            local_agent._clear_tools_from_llm()

        for tool_name in tool_names_to_remove:
            local_agent.tool_definitions.pop(tool_name, None)
        local_agent.mcp_resources.pop(server_name, None)

        if local_agent.is_active:
            local_agent.resync_tools_to_llm()

    async def deregister_server_tools(
        self, server_name: str, agent_name: str | None = None
    ):
        agent_manager = AgentManager.get_instance()
        if agent_name:
            local_agent = agent_manager.get_local_agent(agent_name)
            if not local_agent:
                return
            combined_server_id = self._get_server_id_format(server_name, agent_name)
            if combined_server_id in self.tools_cache:
                self._remove_agent_server_tool_definitions(local_agent, server_name)
        else:
            for current_agent_name in agent_manager.agents.keys():
                local_agent = agent_manager.get_local_agent(current_agent_name)
                if not local_agent:
                    continue

                combined_server_id = self._get_server_id_format(
                    server_name, current_agent_name
                )
                if combined_server_id in self.tools_cache:
                    self._remove_agent_server_tool_definitions(local_agent, server_name)

    def _format_tool_definition(self, tool: Tool, server_id: str) -> dict[str, Any]:
        """
        Format a tool definition for the tool registry.

        Args:
            tool: Tool information from the server
            server_id: ID of the server the tool belongs to
            provider: LLM provider to format for (if None, uses default format)

        Returns:
            Formatted tool definition
        """
        # Create namespaced tool name
        namespaced_name = f"{server_id}__{tool.name}"

        from jsonref import replace_refs
        import json

        merged_inputSchema_string = json.dumps(
            replace_refs(
                tool.inputSchema,
                merge_props=True,
                jsonschema=True,
            ),
            indent=2,
        )
        merged_inputSchema = json.loads(merged_inputSchema_string)
        if "$defs" in merged_inputSchema:
            del merged_inputSchema["$defs"]

        # Format for different providers
        return {
            "type": "function",
            "function": {
                "name": namespaced_name,
                "description": tool.description,
                "parameters": merged_inputSchema,
            },
        }

    def _run_async(self, coro):
        """Helper method to run coroutines in the service's event loop"""
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    async def _await_on_service_loop(
        self, coro, timeout: float | None = MCP_OPERATION_TIMEOUT_SECONDS
    ):
        """Await a coroutine on the MCP service loop without blocking the caller loop."""
        if self.loop.is_closed():
            raise RuntimeError("MCP service event loop is closed")

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self.loop:
            if timeout is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=timeout)

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        wrapped = asyncio.wrap_future(future)
        try:
            if timeout is None:
                return await wrapped
            return await asyncio.wait_for(wrapped, timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    def _create_tool_handler(self, server_id: str, tool_name: str) -> Callable:
        """
        Create an asynchronous handler function for a tool.

        Args:
            server_id: ID of the server the tool belongs to
            tool_name: Name of the tool

        Returns:
            Asynchronous handler function for the tool
        """

        def handler_factory(
            service_instance=None,
        ):  # service_instance will be self (MCPService)
            # This is the actual async handler the agent will await.
            async def actual_tool_executor(
                **params,
            ) -> list[dict[str, Any]]:
                if server_id not in self.sessions or not self.connected_servers.get(
                    server_id
                ):
                    raise Exception(
                        f"Cannot call tool: Server '{server_id}' is not connected"
                    )

                session = self.sessions[server_id]
                result = await self._await_on_service_loop(
                    session.call_tool(tool_name, params)
                )
                return await self._format_contents_async(result.content, session)

            return actual_tool_executor  # Return the async function

        return handler_factory  # Return the factory

    async def list_tools(self, server_id: str | None = None) -> list[dict[str, Any]]:
        """
        list all available tools from a connected MCP server or all servers.

        Args:
            server_id: Optional ID of the server to list tools from. If None, lists tools from all servers.

        Returns:
            list of tools with their metadata
        """
        if server_id is not None:
            if server_id not in self.sessions or not self.connected_servers.get(
                server_id
            ):
                return []

            try:
                response = await self.sessions[server_id].list_tools()
                self.tools_cache[server_id] = {
                    tool.name: tool for tool in response.tools
                }

                return [
                    {
                        "name": f"{server_id}.{tool.name}",
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]
            except Exception:
                logger.exception(f"Error listing tools from server '{server_id}'")
                self.connected_servers[server_id] = False
                return []
        else:
            # list tools from all connected servers
            all_tools = []
            for srv_id in list(self.sessions.keys()):
                all_tools.extend(await self.list_tools(srv_id))
            return all_tools

    async def get_prompt(
        self,
        server_id: str,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific prompt from a connected MCP server.

        Args:
            server_id: ID of the server to get the prompt from
            prompt_name: Name of the prompt to retrieve

        Returns:
            Prompt object if found, None otherwise
        """

        if server_id not in self.sessions or not self.connected_servers.get(server_id):
            return {
                "content": f"Cannot call tool: Server '{server_id}' is not connected",
                "status": "error",
            }

        try:
            session = self.sessions[server_id]
            result = await self._await_on_service_loop(
                session.get_prompt(prompt_name, arguments)
            )
            return {"content": result.messages, "status": "success"}
        except Exception as e:
            logger.error(
                f"Error retrieving prompt '{prompt_name}' from server '{server_id}': {str(e)}"
            )
            self.connected_servers[server_id] = False
            return {
                "content": f"Error calling tool '{prompt_name}' on server '{server_id}': {str(e)}",
                "status": "error",
            }

    def _format_contents(
        self,
        content: list[ContentBlock],
        session: ClientSession | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for c in content:
            if isinstance(c, TextContent):
                result.append(
                    {
                        "type": "text",
                        "text": c.text,
                    }
                )
            elif isinstance(c, ImageContent):
                data_uri = optimize_image_data_uri(f"data:{c.mimeType};base64,{c.data}")
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )
            elif isinstance(c, ResourceLink):
                result.extend(self._format_resource_link(c, session))

        return result

    async def _format_contents_async(
        self,
        content: list[ContentBlock],
        session: ClientSession | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for c in content:
            if isinstance(c, TextContent):
                result.append(
                    {
                        "type": "text",
                        "text": c.text,
                    }
                )
            elif isinstance(c, ImageContent):
                data_uri = optimize_image_data_uri(f"data:{c.mimeType};base64,{c.data}")
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )
            elif isinstance(c, ResourceLink):
                result.extend(await self._format_resource_link_async(c, session))

        return result

    def _format_resource_link(
        self, resource_link: ResourceLink, session: ClientSession | None
    ) -> list[dict[str, Any]]:
        if session is None:
            return [
                self._resource_link_fallback(resource_link, "No MCP session available")
            ]

        try:
            resource_result = self._run_async(session.read_resource(resource_link.uri))
        except Exception as e:
            return [self._resource_link_fallback(resource_link, str(e))]

        return self._format_resource_result(resource_link, resource_result)

    async def _format_resource_link_async(
        self, resource_link: ResourceLink, session: ClientSession | None
    ) -> list[dict[str, Any]]:
        if session is None:
            return [
                self._resource_link_fallback(resource_link, "No MCP session available")
            ]

        try:
            resource_result = await self._await_on_service_loop(
                session.read_resource(resource_link.uri)
            )
        except Exception as e:
            return [self._resource_link_fallback(resource_link, str(e))]

        return self._format_resource_result(resource_link, resource_result)

    def _format_resource_result(
        self, resource_link: ResourceLink, resource_result
    ) -> list[dict[str, Any]]:
        formatted = []
        for resource_content in resource_result.contents:
            error_str = None
            try:
                processed = self._process_resource_content(
                    resource_link, resource_content
                )
            except Exception as e:
                processed = None
                error_str = str(e)

            if processed:
                formatted.append(processed)
            else:
                image_fallback = self._resource_image_fallback(
                    resource_link, resource_content
                )
                formatted.append(
                    image_fallback
                    or self._resource_link_fallback(
                        resource_link,
                        error_str
                        or f"Unsupported MIME type: {self._resource_mime_type(resource_link, resource_content)}",
                    )
                )
        return formatted

    def _process_resource_content(
        self,
        resource_link: ResourceLink,
        resource_content: TextResourceContents | BlobResourceContents,
    ) -> dict[str, Any] | None:
        suffix = self._resource_temp_suffix(resource_link, resource_content)
        temp_path = None

        try:
            if isinstance(resource_content, TextResourceContents):
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=suffix, delete=False
                ) as temp_file:
                    temp_file.write(resource_content.text)
                    temp_path = temp_file.name
            elif isinstance(resource_content, BlobResourceContents):
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=suffix, delete=False
                ) as temp_file:
                    temp_file.write(base64.b64decode(resource_content.blob))
                    temp_path = temp_file.name

            return self._get_file_handler().process_file(temp_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _resource_temp_suffix(
        self,
        resource_link: ResourceLink,
        resource_content: TextResourceContents | BlobResourceContents,
    ) -> str:
        uri_path = unquote(urlparse(str(resource_link.uri)).path)
        _, ext = os.path.splitext(uri_path)

        if not ext:
            name = getattr(resource_link, "name", "")
            _, ext = os.path.splitext(name)

        if not ext:
            mime_type = self._resource_mime_type(resource_link, resource_content)
            if mime_type.startswith("text/"):
                ext = ".txt"

        return ext

    def _resource_mime_type(
        self,
        resource_link: ResourceLink,
        resource_content: TextResourceContents | BlobResourceContents,
    ) -> str:
        return (
            getattr(resource_content, "mimeType", None)
            or getattr(resource_link, "mimeType", None)
            or "unknown"
        )

    def _resource_image_fallback(
        self,
        resource_link: ResourceLink,
        resource_content: TextResourceContents | BlobResourceContents,
    ) -> dict[str, Any] | None:
        mime_type = self._resource_mime_type(resource_link, resource_content)
        if not mime_type.startswith("image/") or not isinstance(
            resource_content, BlobResourceContents
        ):
            return None

        data_uri = optimize_image_data_uri(
            f"data:{mime_type};base64,{resource_content.blob}"
        )
        return {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }

    def _resource_link_fallback(
        self, resource_link: ResourceLink, reason: str
    ) -> dict[str, Any]:
        name = getattr(resource_link, "name", "resource")
        uri = str(getattr(resource_link, "uri", ""))
        mime_type = getattr(resource_link, "mimeType", None) or "unknown"
        return {
            "type": "text",
            "text": (
                f"MCP resource link could not be processed: {name}\n"
                f"URI: {uri}\n"
                f"MIME type: {mime_type}\n"
                f"Reason: {reason}"
            ),
        }

    async def call_tool(
        self, server_id: str, tool_name: str, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call a tool on an MCP server.

        Args:
            server_id: ID of the server to call the tool on
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            dict containing the tool's response
        """
        if server_id not in self.sessions or not self.connected_servers.get(server_id):
            return {
                "content": f"Cannot call tool: Server '{server_id}' is not connected",
                "status": "error",
            }

        if server_id not in self.tools_cache or tool_name not in self.tools_cache.get(
            server_id, {}
        ):
            # Refresh tools cache
            await self.list_tools(server_id)
            if (
                server_id not in self.tools_cache
                or tool_name not in self.tools_cache.get(server_id, {})
            ):
                return {
                    "content": f"Tool '{tool_name}' is not available on server '{server_id}'",
                    "status": "error",
                }

        try:
            result = await self._await_on_service_loop(
                self.sessions[server_id].call_tool(tool_name, tool_args)
            )
            return {
                "content": await self._format_contents_async(
                    result.content, self.sessions[server_id]
                ),
                "status": "success",
            }
        except Exception as e:
            self.connected_servers[server_id] = False
            return {
                "content": f"Error calling tool '{tool_name}' on server '{server_id}': {str(e)}",
                "status": "error",
            }

    def start(self):
        """Start the service's event loop in a separate thread"""

        def run_loop():
            logger.info("MCPService: Event loop thread started.")
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            finally:
                logger.info("MCPService: Event loop stopping...")
                # This block executes when loop.stop() is called or run_forever() exits.
                # Attempt to cancel any remaining tasks.
                # shutdown_all_server_connections should ideally handle most task terminations gracefully.
                try:
                    all_tasks = asyncio.all_tasks(loop=self.loop)
                    # Exclude the current task if this finally block is run by a task on the loop
                    current_task = (
                        asyncio.current_task(loop=self.loop)
                        if self.loop.is_running()
                        else None
                    )  # Check if loop is running

                    tasks_to_cancel = [
                        t for t in all_tasks if t is not current_task and not t.done()
                    ]
                    if tasks_to_cancel:
                        logger.info(
                            f"MCPService: Cancelling {len(tasks_to_cancel)} outstanding tasks in event loop thread."
                        )
                        for task in tasks_to_cancel:
                            task.cancel()
                        # Give tasks a chance to process cancellation
                        # This needs to run on the loop, but run_forever has exited.
                        # We can run_until_complete for these specific tasks.
                        if self.loop.is_running():  # Should not be, but as a safeguard
                            self.loop.run_until_complete(
                                asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                            )
                        else:  # If loop not running, create a temporary runner for cleanup

                            async def gather_cancel_tasks():
                                await asyncio.gather(
                                    *tasks_to_cancel, return_exceptions=True
                                )

                            self.loop.run_until_complete(gather_cancel_tasks())

                except RuntimeError as e:
                    logger.error(
                        f"MCPService: Runtime error during task cancellation in run_loop finally: {e}"
                    )
                except Exception as e_final:
                    logger.error(
                        f"MCPService: General error during task cancellation in run_loop finally: {e_final}"
                    )

                if not self.loop.is_closed():
                    self.loop.close()
                logger.info("MCPService: Event loop thread stopped and loop closed.")

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

    def stop(self):
        """Stop the service's event loop"""
        logger.info("MCPService: Stopping event loop...")
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            if self.loop.is_running():
                logger.info("MCPService: Requesting event loop to stop.")
                self.loop.call_soon_threadsafe(self.loop.stop)
            # The finally block in run_loop should handle task cleanup and closing the loop.
        else:
            logger.info(
                "MCPService: Loop not available or already closed when stop() called."
            )

        if hasattr(self, "loop_thread") and self.loop_thread.is_alive():
            logger.info("MCPService: Waiting for event loop thread to join...")
            self.loop_thread.join(timeout=10)
            if self.loop_thread.is_alive():
                logger.warning(
                    "MCPService: Event loop thread did not join in time. Loop might be stuck or tasks not yielding."
                )
        else:
            logger.info(
                "MCPService: Loop thread not available or not alive when stop() called."
            )

        # Fallback: Ensure loop is closed if thread exited but loop wasn't closed by run_loop's finally
        if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
            logger.warning(
                "MCPService: Loop was not closed by thread's run_loop, attempting to close now."
            )
            # This is a fallback, ideally run_loop's finally handles this.
            try:
                # Minimal cleanup if loop is in a weird state
                # Ensure all tasks are finished or cancelled before closing loop
                if (
                    not self.loop.is_running()
                ):  # If not running, we can try to run_until_complete for cleanup
                    tasks = [
                        t for t in asyncio.all_tasks(loop=self.loop) if not t.done()
                    ]
                    if tasks:
                        logger.info(
                            f"MCPService: Running {len(tasks)} pending tasks to completion before closing loop in stop()."
                        )

                        async def finalize_tasks():
                            await asyncio.gather(*tasks, return_exceptions=True)

                        self.loop.run_until_complete(finalize_tasks())
            except (
                RuntimeError
            ) as e:  # e.g. "cannot call run_until_complete() on a running loop"
                logger.error(
                    f"MCPService: Runtime error during final loop cleanup in stop(): {e}"
                )
            except Exception as e_final_stop:
                logger.error(
                    f"MCPService: General error during final loop cleanup in stop(): {e_final_stop}"
                )
            finally:
                if not self.loop.is_closed():  # Check again before closing
                    self.loop.close()
                    logger.info("MCPService: Loop closed in stop() fallback.")

        logger.info("MCPService: Stop process complete.")
