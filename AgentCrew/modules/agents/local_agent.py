from __future__ import annotations

import asyncio
from datetime import datetime
import os
import copy
from typing import TYPE_CHECKING, Literal

from .base import BaseAgent, MessageType
from AgentCrew.modules.llm.token_usage import TokenUsage
from loguru import logger

if TYPE_CHECKING:
    from AgentCrew.modules.llm import BaseLLMService
    from typing import Any, Callable, Union


def normalize_voice_enabled(value) -> Literal["enabled", "disabled"]:
    if value in (True, "enabled", "full", "partial"):
        return "enabled"
    return "disabled"


class LocalAgent(BaseAgent):
    """Base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_service: BaseLLMService,
        services: dict[str, Any],
        tools: list[str],
        temperature: float | None = None,
        is_remoting_mode: bool = False,
        voice_enabled: Literal["enabled", "disabled"] = "disabled",
        voice_id: str | None = None,
    ):
        """
        Initialize a new agent.

        Args:
            name: The name of the agent
            description: A description of the agent's capabilities
            llm_service: The LLM service to use for this agent
            services: Dictionary of available services
            voice_enabled: Whether voice features are enabled for this agent
            voice_id: Voice ID to use for text-to-speech
        """
        super().__init__(name, description)
        self.llm = llm_service
        self.temperature = temperature
        self.services = services
        self.tools: list[str] = tools  # list of tool names that the agent needs
        self.system_prompt = None
        self.custom_system_prompt = None
        self.tool_prompts = []
        self.is_remoting_mode: bool = is_remoting_mode
        self.pinned_model_id: str | None = None
        self.token_usage = TokenUsage()
        self.voice_enabled: Literal["enabled", "disabled"] = normalize_voice_enabled(
            voice_enabled
        )
        self.voice_id: str | None = voice_id

        self.tool_definitions = {}  # {tool_name: (definition_func, handler_factory, service_instance)}
        self.registered_tools = (
            set()
        )  # Set of tool names that are registered with the LLM
        self._defer_tool_registration = False
        self.mcps_loading = []

        from AgentCrew.modules.agents.manager import AgentMode

        self._colaboration_mode = AgentMode.TRANSFER

        from .tool_registrar import AgentToolRegistrar
        from .context_manager import AgentContextManager

        self._tool_registrar = AgentToolRegistrar(self)
        self._context_manager = AgentContextManager(self)

    @property
    def input_tokens_usage(self) -> int:
        return self.token_usage.input_tokens

    @input_tokens_usage.setter
    def input_tokens_usage(self, value: int):
        self.token_usage = TokenUsage(
            input_tokens=value,
            output_tokens=self.token_usage.output_tokens,
            cached_tokens=self.token_usage.cached_tokens,
            cache_creation_tokens=self.token_usage.cache_creation_tokens,
        )

    @property
    def output_tokens_usage(self) -> int:
        return self.token_usage.output_tokens

    @output_tokens_usage.setter
    def output_tokens_usage(self, value: int):
        self.token_usage = TokenUsage(
            input_tokens=self.token_usage.input_tokens,
            output_tokens=value,
            cached_tokens=self.token_usage.cached_tokens,
            cache_creation_tokens=self.token_usage.cache_creation_tokens,
        )

    def _extract_tool_name(self, tool_def: Any) -> str:
        """Extract tool name from definition regardless of format."""
        from AgentCrew.modules.tools.utils import extract_tool_name

        return extract_tool_name(tool_def)

    def append_message(self, messages: Union[dict, list[dict]]):
        copy_messages = copy.deepcopy(messages)
        if isinstance(copy_messages, list):
            self.history.extend(copy_messages)
        else:
            self.history.append(copy_messages)

    def register_tools(self):
        """Register tools for this agent using the services dictionary."""
        self._tool_registrar.register_tools()

    def register_tool(self, definition_func, handler_factory, service_instance=None):
        """Register a tool with this agent."""
        self._tool_registrar.register_tool(
            definition_func, handler_factory, service_instance
        )

    def set_system_prompt(self, prompt: str):
        """
        Set the system prompt for this agent.

        Args:
            prompt: The system prompt
        """
        self.system_prompt = prompt

    def _parse_system_prompt(self, prompt: str) -> str:
        """
        Parse the system prompt to ensure it is in the correct format.

        Args:
            prompt: The system prompt
        """
        return (
            prompt.replace("{current_date}", datetime.today().strftime("%A, %d %B %Y"))
            .replace("{cwd}", os.getcwd())
            .replace("{current_agent_name}", self.name)
            .replace("{current_agent_description}", self.description)
        )

    def set_custom_system_prompt(self, prompt: str):
        """
        Set the system prompt for this agent.

        Args:
            prompt: The system prompt
        """
        self.custom_system_prompt = prompt

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            The system prompt
        """
        return self.system_prompt or ""

    def activate(self):
        """
        Activate this agent by registering all tools with the LLM service.

        Returns:
            True if activation was successful, False otherwise
        """
        if not self.llm:
            return False

        if self.is_active:
            return True  # Already active

        self.register_tools()

        # Reinitialize MCP session manager for the current agent
        if not self.is_remoting_mode:
            from AgentCrew.modules.mcpclient import MCPSessionManager

            mcp_manager = MCPSessionManager.get_instance()
            if mcp_manager.initialized:
                mcp_manager.initialize_for_agent(self.name)

        system_prompt = (
            f"<Agent_Instructions>\n{self.get_system_prompt()}\n</Agent_Instructions>"
        )
        if self.custom_system_prompt:
            system_prompt = f"{system_prompt}\n\n{self.custom_system_prompt}"
        if self.tool_prompts:
            system_prompt = f"{system_prompt}\n\n{'\n\n'.join(self.tool_prompts)}"

        self.llm.set_system_prompt(self._parse_system_prompt(system_prompt))
        self.llm.temperature = self.temperature if self.temperature is not None else 0.4
        self._defer_tool_registration = True
        self.is_active = True
        return True

    def deactivate(self):
        """
        Deactivate this agent by clearing all tools from the LLM service.

        Returns:
            True if deactivation was successful, False otherwise
        """
        if not self.llm:
            return False

        self._clear_tools_from_llm()
        self.tool_definitions = {}
        self.tool_prompts = []
        self.is_active = False
        self.mcps_loading = []
        # Reinitialize MCP session manager for the current agent
        if not self.is_remoting_mode:
            from AgentCrew.modules.mcpclient import MCPSessionManager

            mcp_manager = MCPSessionManager.get_instance()
            if mcp_manager.initialized:
                mcp_manager.cleanup_for_agent(self.name)
        return True

    def _register_tools_with_llm(self):
        """Register all of this agent's tools with the LLM service."""
        self._tool_registrar.sync_to_llm()

    def _clear_tools_from_llm(self):
        """Clear all tools from the LLM service."""
        self._tool_registrar._clear_from_llm()

    def resync_tools_to_llm(self):
        self._register_tools_with_llm()
        self._defer_tool_registration = False

    @property
    def clean_history(self):
        return self.history

    def get_provider(self) -> str:
        return self.llm.provider_name if self.llm else ""

    def is_streaming(self) -> bool:
        return self.llm.is_stream if self.llm else False

    def _format_tool_result(
        self,
        tool_use: dict,
        tool_result: Any,
        is_error: bool = False,
        is_rejected: bool = False,
    ) -> dict[str, Any]:
        """
        Format a tool result for OpenAI API.

        Args:
            tool_use: The tool use details
            tool_result: The result from the tool execution
            is_error: Whether the result is an error

        Returns:
            A formatted message for tool response
        """
        # OpenAI format for tool responses
        message = {
            "role": "tool",
            "agent": self.name,
            "tool_call_id": tool_use["id"],
            "tool_name": tool_use["name"],
            "content": tool_result,
        }

        # Add error indication if needed
        if is_error:
            message["content"] = f"ERROR: {str(message['content'])}"
        if is_rejected:
            message["is_rejected"] = True

        return message

    def _format_assistant_message(
        self, assistant_response: str, tool_uses: list[dict] | None = None
    ) -> dict[str, Any]:
        """
        Format the assistant's response into the appropriate message format for the LLM provider.

        Args:
            assistant_response (str): The text response from the assistant
            tool_use (dict, optional): Tool use information if a tool was used

        Returns:
            dict[str, Any]: A properly formatted message to append to the messages list
        """
        valid_tool_uses = [
            tool_use
            for tool_use in (tool_uses or [])
            if tool_use.get("id") and tool_use.get("name")
        ]
        if valid_tool_uses:
            return {
                "role": "assistant",
                "agent": self.name,
                "content": assistant_response,
                "tool_calls": [
                    {
                        "id": tool_use["id"],
                        "name": tool_use["name"],
                        "arguments": tool_use["input"],
                        "type": tool_use.get("type", "tool_call"),
                    }
                    for tool_use in valid_tool_uses
                ],
            }
        else:
            return {
                "agent": self.name,
                "role": "assistant",
                "content": assistant_response,
            }

    def _format_thinking_message(self, thinking_data) -> dict[str, Any] | None:
        """
        Format thinking content into the appropriate message format for Claude.

        Args:
            thinking_data: Tuple containing (thinking_content, thinking_signature)
                or None if no thinking data is available

        Returns:
            dict[str, Any]: A properly formatted message containing thinking blocks
        """
        if not thinking_data:
            return None

        thinking_content, thinking_signature = thinking_data

        if not thinking_content:
            return None

        # For Claude, thinking blocks need to be preserved in the assistant's message
        thinking_block = {"type": "thinking", "thinking": thinking_content}

        # Add signature if available
        if thinking_signature:
            thinking_block["signature"] = thinking_signature

        return {"role": "assistant", "agent": self.name, "content": [thinking_block]}

    def format_message(
        self, message_type: MessageType, message_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if message_type == MessageType.Assistant:
            return self._format_assistant_message(
                message_data.get("message", ""), message_data.get("tool_uses", None)
            )
        elif message_type == MessageType.Thinking:
            return self._format_thinking_message(message_data.get("thinking", None))
        elif message_type == MessageType.ToolResult:
            return self._format_tool_result(
                message_data.get("tool_use", {}),
                message_data.get("tool_result", ""),
                message_data.get("is_error", False),
                message_data.get("is_rejected", False),
            )
        elif message_type == MessageType.FileContent:
            return (
                self.llm.process_file_for_message(message_data.get("file_uri", ""))
                if self.llm
                else message_data
            )

    def configure_think(self, think_setting):
        if self.llm:
            self.llm.set_think(think_setting)

    async def execute_tool_call(self, tool_name: str, tool_input: dict) -> Any:
        return await self.llm.execute_tool(tool_name, tool_input) if self.llm else None

    def calculate_usage_cost(
        self, input_tokens, output_tokens, cached_tokens=0
    ) -> float:
        return (
            self.llm.calculate_cost(input_tokens, output_tokens, cached_tokens)
            if self.llm
            else 0
        )

    def get_model(self) -> str:
        return f"{self.llm.provider_name}/{self.llm.model}" if self.llm else ""

    def update_llm_service(self, new_llm_service: BaseLLMService) -> bool:
        """
        Update the LLM service used by this agent.

        Args:
            new_llm_service: The new LLM service to use

        Returns:
            True if the update was successful, False otherwise
        """
        was_active = self.is_active

        # Deactivate with the current LLM if active
        if was_active:
            self.deactivate()

        # Update the LLM service
        self.llm = new_llm_service

        # Reactivate with the new LLM if it was active before
        if was_active:
            self.activate()

        return True

    def _build_adaptive_behavior_context(self) -> dict[str, Any]:
        return self._context_manager.build_adaptive_context()

    def _get_directory_structure(self) -> str:
        return self._context_manager._get_directory_structure()

    def _enhance_agent_context_messages(
        self, final_messages: list[dict[str, Any]]
    ) -> None:
        self._context_manager.enhance_messages(final_messages)

    def _filter_invalid_tool_uses(
        self, tool_uses: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        filtered_tool_uses = []
        for tool_use in tool_uses:
            if isinstance(tool_use.get("name"), str) and bool(
                tool_use.get("name", "").strip()
            ):
                filtered_tool_uses.append(tool_use)
            elif tool_use.get("id") or tool_use.get("args_json"):
                logger.warning(
                    "Dropping malformed parsed tool call without a usable name"
                )
        return filtered_tool_uses

    def _clean_shrinkable_tool_result(
        self, final_messages: list[dict[str, Any]]
    ) -> None:
        self._context_manager.shrink_tool_results(final_messages)

    def _extract_last_user_message_for_memory(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", [])
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    return normalized
                continue
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    normalized = part.strip()
                    if normalized:
                        text_parts.append(normalized)
                elif isinstance(part, dict) and part.get("type") == "text":
                    normalized = str(part.get("text", "")).strip()
                    if normalized:
                        text_parts.append(normalized)
            if text_parts:
                return " ".join(text_parts)
        return ""

    def _extract_assistant_messages_for_memory(
        self, messages: list[dict], current_response: str = ""
    ) -> list[str]:
        assistant_messages: list[str] = []
        last_user_idx = -1
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_idx = index

        for message in messages[last_user_idx + 1 :]:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    assistant_messages.append(normalized)
        normalized_current_response = current_response.strip()
        if normalized_current_response and (
            not assistant_messages
            or assistant_messages[-1] != normalized_current_response
        ):
            assistant_messages.append(normalized_current_response)
        return assistant_messages

    def store_memory_if_available(
        self,
        user_message: str,
        messages: list[dict],
        current_response: str,
        session_id: str | None = None,
    ) -> None:
        from AgentCrew.modules.memory.base_service import BaseMemoryService

        memory_service = self.services.get("memory")
        if not memory_service or not isinstance(memory_service, BaseMemoryService):
            return
        assistant_messages = self._extract_assistant_messages_for_memory(
            messages, current_response
        )
        try:
            memory_service.store_conversation(
                user_message,
                assistant_messages,
                self.name,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning(f"Failed to store conversation in memory: {e}")

    async def process_messages(
        self,
        messages: list[dict[str, Any]] | None = None,
        callback: Callable | None = None,
    ):
        """
        Process messages using this agent.

        Args:
            messages: The messages to process

        Returns:
            The processed messages with the agent's response
        """
        if not self.llm:
            return
        if self._defer_tool_registration:
            while len(self.mcps_loading) > 0:
                await asyncio.sleep(0.2)
            self._register_tools_with_llm()

        assistant_response = ""
        _tool_uses = []
        _token_usage = TokenUsage()
        # Ensure the first message is a system message with the agent's prompt
        self._clean_shrinkable_tool_result(messages or self.history)
        final_messages = messages[:] if messages else self.history[:]
        self._enhance_agent_context_messages(final_messages)
        try:
            async with await self.llm.stream_assistant_response(
                copy.deepcopy(
                    final_messages
                )  # This will prevent llm converting message break the original format
            ) as stream:
                async for chunk in stream:
                    (
                        assistant_response,
                        tool_uses,
                        chunk_token_usage,
                        chunk_text,
                        thinking_chunk,
                    ) = self.llm.process_stream_chunk(
                        chunk, assistant_response, _tool_uses
                    )
                    yield (assistant_response, chunk_text, thinking_chunk)

                    if tool_uses:
                        _tool_uses = tool_uses
                    if chunk_token_usage:
                        _token_usage = _token_usage.merge(chunk_token_usage)

            self.token_usage = _token_usage
            if callback:
                callback(
                    self._filter_invalid_tool_uses(_tool_uses),
                    _token_usage,
                )
            else:
                self.tool_uses = _tool_uses

        except GeneratorExit as e:
            logger.warning(f"Stream processing interrupted: {e}")
            return
        except Exception as e:
            logger.error(f"Error during message processing: {e}")
            logger.debug(f"Final messages at error time: {final_messages}")
            raise e
