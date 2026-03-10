from __future__ import annotations

import asyncio
from datetime import datetime
import os
import copy
from typing import List, TYPE_CHECKING

from .base import BaseAgent, MessageType
from loguru import logger

if TYPE_CHECKING:
    from AgentCrew.modules.llm import BaseLLMService
    from typing import Dict, Any, Optional, Callable, Literal, Union


class LocalAgent(BaseAgent):
    """Base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_service: BaseLLMService,
        services: Dict[str, Any],
        tools: List[str],
        temperature: Optional[float] = None,
        is_remoting_mode: bool = False,
        voice_enabled: Literal["full", "partial", "disabled"] = "disabled",
        voice_id: Optional[str] = None,
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
        self.tools: List[str] = tools  # List of tool names that the agent needs
        self.system_prompt = None
        self.custom_system_prompt = None
        self.tool_prompts = []
        self.is_remoting_mode: bool = is_remoting_mode
        self.input_tokens_usage = 0
        self.output_tokens_usage = 0
        self.voice_enabled: Literal["full", "partial", "disabled"] = voice_enabled
        self.voice_id: Optional[str] = voice_id

        self.tool_definitions = {}  # {tool_name: (definition_func, handler_factory, service_instance)}
        self.registered_tools = (
            set()
        )  # Set of tool names that are registered with the LLM
        self._defer_tool_registration = False
        self.mcps_loading = []

        from .tool_registrar import AgentToolRegistrar
        from .context_manager import AgentContextManager

        self._tool_registrar = AgentToolRegistrar(self)
        self._context_manager = AgentContextManager(self)

    def _extract_tool_name(self, tool_def: Any) -> str:
        """Extract tool name from definition regardless of format."""
        from AgentCrew.modules.tools.utils import extract_tool_name

        return extract_tool_name(tool_def)

    def append_message(self, messages: Union[Dict, List[Dict]]):
        copy_messages = copy.deepcopy(messages)
        if isinstance(copy_messages, List):
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

    @property
    def clean_history(self):
        return self.history

    def get_provider(self) -> str:
        return self.llm.provider_name

    def is_streaming(self) -> bool:
        return self.llm.is_stream

    def _format_tool_result(
        self,
        tool_use: Dict,
        tool_result: Any,
        is_error: bool = False,
        is_rejected: bool = False,
    ) -> Dict[str, Any]:
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
        self, assistant_response: str, tool_uses: list[Dict] | None = None
    ) -> Dict[str, Any]:
        """
        Format the assistant's response into the appropriate message format for the LLM provider.

        Args:
            assistant_response (str): The text response from the assistant
            tool_use (Dict, optional): Tool use information if a tool was used

        Returns:
            Dict[str, Any]: A properly formatted message to append to the messages list
        """
        if tool_uses and any(tu.get("id") for tu in tool_uses):
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
                    for tool_use in tool_uses
                    if tool_use.get("id")  # Only include tool calls with valid IDs
                ],
            }
        else:
            return {
                "agent": self.name,
                "role": "assistant",
                "content": assistant_response,
            }

    def _format_thinking_message(self, thinking_data) -> Optional[Dict[str, Any]]:
        """
        Format thinking content into the appropriate message format for Claude.

        Args:
            thinking_data: Tuple containing (thinking_content, thinking_signature)
                or None if no thinking data is available

        Returns:
            Dict[str, Any]: A properly formatted message containing thinking blocks
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
        self, message_type: MessageType, message_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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
            return self.llm.process_file_for_message(message_data.get("file_uri", ""))

    def configure_think(self, think_setting):
        self.llm.set_think(think_setting)

    async def execute_tool_call(self, tool_name: str, tool_input: Dict) -> Any:
        return await self.llm.execute_tool(tool_name, tool_input)

    def calculate_usage_cost(self, input_tokens, output_tokens) -> float:
        return self.llm.calculate_cost(input_tokens, output_tokens)

    def get_model(self) -> str:
        return f"{self.llm.provider_name}/{self.llm.model}"

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

    def _build_adaptive_behavior_context(self) -> Dict[str, Any]:
        return self._context_manager.build_adaptive_context()

    def _get_directory_structure(self) -> str:
        return self._context_manager._get_directory_structure()

    def _enhance_agent_context_messages(
        self, final_messages: List[Dict[str, Any]]
    ) -> None:
        self._context_manager.enhance_messages(final_messages)

    def _clean_shrinkable_tool_result(
        self, final_messages: List[Dict[str, Any]]
    ) -> None:
        self._context_manager.shrink_tool_results(final_messages)

    async def process_messages(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Process messages using this agent.

        Args:
            messages: The messages to process

        Returns:
            The processed messages with the agent's response
        """

        if self._defer_tool_registration:
            while len(self.mcps_loading) > 0:
                await asyncio.sleep(0.2)
            self._register_tools_with_llm()

        assistant_response = ""
        _tool_uses = []
        _input_tokens_usage = 0
        _output_tokens_usage = 0
        # Ensure the first message is a system message with the agent's prompt
        final_messages = messages[:] if messages else self.history[:]
        self._enhance_agent_context_messages(final_messages)
        self._clean_shrinkable_tool_result(final_messages)
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
                        chunk_input_tokens,
                        chunk_output_tokens,
                        chunk_text,
                        thinking_chunk,
                    ) = self.llm.process_stream_chunk(
                        chunk, assistant_response, _tool_uses
                    )
                    yield (assistant_response, chunk_text, thinking_chunk)

                    if tool_uses:
                        _tool_uses = tool_uses
                    if chunk_input_tokens > 0:
                        _input_tokens_usage = chunk_input_tokens
                    if chunk_output_tokens > 0:
                        _output_tokens_usage = chunk_output_tokens

            self.input_tokens_usage = _input_tokens_usage
            self.output_tokens_usage = _output_tokens_usage
            if callback:
                callback(_tool_uses, _input_tokens_usage, _output_tokens_usage)
            else:
                self.tool_uses = _tool_uses

        except GeneratorExit as e:
            logger.warning(f"Stream processing interrupted: {e}")
            return
        except Exception as e:
            logger.error(f"Error during message processing: {e}")
            logger.debug(f"Final messages at error time: {final_messages}")
            raise e

    def get_process_result(self):
        """
        @DEPRECATED: Use the callback in process_messages instead.
        """
        return (self.tool_uses, self.input_tokens_usage, self.output_tokens_usage)
