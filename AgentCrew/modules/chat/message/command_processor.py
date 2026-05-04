from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import os

from AgentCrew.modules.utils.file_handler import FileHandler
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.service_manager import ServiceManager
from AgentCrew.modules.chat.consolidation import ConversationConsolidator
from AgentCrew.modules.config.global_config import GlobalConfig
from AgentCrew.modules.agents.local_agent import LocalAgent
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AgentCrew.modules.mcpclient import MCPService
    from AgentCrew.modules.chat.message import MessageHandler


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool = False
    exit_flag: bool = False
    clear_flag: bool = False
    message: str = ""


class CommandProcessor:
    """Handles command processing for the message handler."""

    def __init__(self, message_handler: MessageHandler):
        self.message_handler = message_handler

    async def process_command(self, user_input: str) -> CommandResult:
        """Process a command and return the result."""
        if self._is_exit_command(user_input):
            self.message_handler._notify("exit_requested")
            return CommandResult(handled=True, exit_flag=True)
        elif user_input.lower() == "/clear":
            self.message_handler.start_new_conversation()
            return CommandResult(handled=True, clear_flag=True)
        elif user_input.lower().startswith("/copy"):
            return await self._handle_copy_command(user_input)
        elif user_input.lower().startswith("/debug"):
            return self._handle_debug_command(user_input)
        elif user_input.lower().startswith("/think"):
            return self._handle_think_command(user_input)
        elif user_input.lower().startswith("/consolidate"):
            return await self._handle_consolidate_command(user_input)
        elif user_input.lower().startswith("/evolve"):
            return await self._handle_evolve_command(user_input)
        elif user_input.lower().startswith("/unconsolidate"):
            return await self._handle_unconsolidate_command(user_input)
        elif user_input.lower().startswith("/jump"):
            result = self._handle_jump_command(user_input)
            return CommandResult(handled=result, clear_flag=True)
        elif user_input.lower().startswith("/fork"):
            return self._handle_fork_command(user_input)
        elif user_input.lower().startswith("/agent_mode"):
            return self._handle_agent_mode_command(user_input)
        elif user_input.lower().startswith("/agent"):
            success, message = self._handle_agent_command(user_input)
            self.message_handler._notify(
                "agent_command_result", {"success": success, "message": message}
            )
            return CommandResult(handled=True, clear_flag=True)
        elif user_input.lower().startswith("/model"):
            exit_flag, clear_flag = self._handle_model_command(user_input)
            return CommandResult(
                handled=True, exit_flag=exit_flag, clear_flag=clear_flag
            )
        elif user_input.lower().startswith("/mcp"):
            exit_flag, clear_flag = await self._handle_mcp_command(user_input)
            return CommandResult(
                handled=True, exit_flag=exit_flag, clear_flag=clear_flag
            )
        elif user_input.startswith("/file"):
            return self._handle_file_command(user_input)
        elif user_input.startswith("/drop"):
            return self._handle_drop_command(user_input)
        elif user_input.lower() == "/voice":
            return await self._handle_voice_command(user_input)
        elif user_input.lower() == "/end_voice":
            return await self._handle_end_voice_command(user_input)
        elif user_input.lower() == "/toggle_transfer":
            return self._handle_toggle_transfer_command(user_input)

        # Catch-all: any unrecognised /command should not fall through to the LLM
        if user_input.startswith("/"):
            self.message_handler._notify(
                "error",
                "Invalid command: type /help to view all available commands",
            )
            return CommandResult(handled=True, clear_flag=True)

        return CommandResult(handled=False)

    def _is_exit_command(self, user_input: str) -> bool:
        return user_input.lower() in ["/exit", "/quit"]

    def _handle_think_command(self, user_input: str) -> CommandResult:
        """Handle the /think command to set or show thinking budget.

        Usage:
            /think          - Show current thinking budget
            /think <budget> - Set thinking budget (0 to disable)
        """
        parts = user_input.split()

        if len(parts) == 1:
            current_budget = getattr(
                self.message_handler.agent.llm, "thinking_budget", None
            )
            if current_budget is not None:
                if current_budget == 0:
                    self.message_handler._notify(
                        "system_message", "Thinking mode is currently disabled."
                    )
                else:
                    self.message_handler._notify(
                        "system_message",
                        f"Thinking budget is currently set to {current_budget} tokens.",
                    )
            else:
                reasoning_effort = getattr(
                    self.message_handler.agent.llm, "reasoning_effort", None
                )
                if reasoning_effort:
                    self.message_handler._notify(
                        "system_message",
                        f"Reasoning effort is currently set to: {reasoning_effort}",
                    )
                else:
                    self.message_handler._notify(
                        "system_message",
                        "Thinking mode is not available for the current model.",
                    )
            self.message_handler._notify(
                "system_message", "Usage: /think <budget> (0 to disable)"
            )
            return CommandResult(handled=True, clear_flag=True)

        try:
            budget = parts[1]
            self.message_handler.agent.configure_think(budget)
            self.message_handler._notify("think_budget_set", budget)
        except ValueError:
            self.message_handler._notify(
                "error", "Invalid budget value. Please provide a number."
            )
        return CommandResult(handled=True, clear_flag=True)

    async def _handle_copy_command(self, user_input: str) -> CommandResult:
        copy_idx = user_input[5:].strip() or 1
        user_input_idxs = [
            turn.message_index for turn in self.message_handler.conversation_turns
        ]

        asssistant_messages_iterator = reversed(
            [
                msg
                for i, msg in enumerate(self.message_handler.streamline_messages)
                if msg.get("role") == "assistant"
                and (
                    i + 1 in user_input_idxs
                    if i + 1 < len(self.message_handler.streamline_messages)
                    else True
                )
            ]
        )
        latest_assistant_blk = None
        try:
            for _ in range(int(copy_idx)):
                latest_assistant_blk = next(asssistant_messages_iterator, None)

            if latest_assistant_blk:
                latest_content_blk = latest_assistant_blk.get("content", "")
                if isinstance(latest_content_blk, list):
                    latest_content = next(
                        (
                            c.get("text", "")
                            for c in latest_content_blk
                            if isinstance(c, dict) and c.get("type", "") == "text"
                        ),
                        "",
                    )
                else:
                    latest_content = latest_content_blk

                self.message_handler._notify("copy_requested", latest_content)
        except Exception as e:
            self.message_handler._notify("error", f"Failed to copy message: {str(e)}")
            return CommandResult(handled=True, clear_flag=True)

        return CommandResult(handled=True, clear_flag=True)

    async def _handle_consolidate_command(self, user_input: str) -> CommandResult:
        """Handle consolidate command."""
        try:
            parts = user_input.split()
            if len(parts) == 1:
                preserve_count = 10
            else:
                preserve_count = int(parts[1])

            if isinstance(self.message_handler.agent, LocalAgent):
                consolidator = ConversationConsolidator(self.message_handler.agent.llm)

                result = await consolidator.consolidate(
                    self.message_handler.streamline_messages, preserve_count
                )

                if result["success"]:
                    self.message_handler.agent_manager.rebuild_agents_messages(
                        self.message_handler.streamline_messages
                    )

                    self.message_handler._notify("consolidation_completed", result)

                    if (
                        self.message_handler.current_conversation_id
                        and self.message_handler.persistent_service
                    ):
                        try:
                            self.message_handler.persistent_service.append_conversation_messages(
                                self.message_handler.current_conversation_id,
                                self.message_handler.streamline_messages,
                                True,  # Replace all messages with the consolidated version
                            )
                        except Exception as e:
                            self.message_handler._notify(
                                "error",
                                f"Failed to save consolidated conversation: {str(e)}",
                            )

                    message = (
                        f"Consolidated {result['messages_consolidated']} messages, "
                        f"preserving {result['messages_preserved']} recent messages. "
                        f"Token savings: ~{result['original_token_count'] - result['consolidated_token_count']}"
                    )
                    self.message_handler._notify("system_message", message)
                else:
                    self.message_handler._notify(
                        "system_message",
                        f"Consolidation skipped: {result['reason']}",
                    )

                return CommandResult(handled=True, clear_flag=True)
            else:
                self.message_handler._notify(
                    "error",
                    "Consolidation is only supported with LocalAgent.",
                )
                return CommandResult(handled=False, clear_flag=False)
        except ValueError as e:
            self.message_handler._notify(
                "error",
                f"Invalid consolidation parameter: {str(e)}. Use /consolidate [number]",
            )
            return CommandResult(handled=True, clear_flag=True)
        except Exception as e:
            self.message_handler._notify(
                "error", f"Error during consolidation: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)

    async def _handle_evolve_command(self, user_input: str) -> CommandResult:
        await self.message_handler.start_evolution_review()
        return CommandResult(handled=True, clear_flag=True)

    async def _handle_unconsolidate_command(self, user_input: str) -> CommandResult:
        """Handle unconsolidate command to remove last consolidated message."""
        try:
            if isinstance(self.message_handler.agent, LocalAgent):
                consolidator = ConversationConsolidator(self.message_handler.agent.llm)

                result = await consolidator.unconsolidate(
                    self.message_handler.streamline_messages
                )

                if result["success"]:
                    self.message_handler.agent_manager.rebuild_agents_messages(
                        self.message_handler.streamline_messages
                    )

                    self.message_handler._notify("unconsolidation_completed", result)

                    if (
                        self.message_handler.current_conversation_id
                        and self.message_handler.persistent_service
                    ):
                        try:
                            self.message_handler.persistent_service.append_conversation_messages(
                                self.message_handler.current_conversation_id,
                                self.message_handler.streamline_messages,
                                True,
                            )
                        except Exception as e:
                            self.message_handler._notify(
                                "error",
                                f"Failed to save unconsolidated conversation: {str(e)}",
                            )

                    message = (
                        f"Unconsolidated last consolidated message containing "
                        f"{result['messages_restored']} original messages."
                    )
                    self.message_handler._notify("system_message", message)
                else:
                    self.message_handler._notify(
                        "system_message",
                        f"Unconsolidation skipped: {result['reason']}",
                    )

                return CommandResult(handled=True, clear_flag=True)
            else:
                self.message_handler._notify(
                    "error",
                    "Unconsolidation is only supported with LocalAgent.",
                )
                return CommandResult(handled=False, clear_flag=False)
        except Exception as e:
            self.message_handler._notify(
                "error", f"Error during unconsolidation: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)

    async def _handle_mcp_command(self, command: str) -> Tuple[bool, bool]:
        """
        Handle the /mcp command: list prompts or fetch a specific prompt content.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        parts = command.strip().split()
        mcp_service: Optional[MCPService] = self.message_handler.mcp_manager.mcp_service
        # /mcp with no args: list all prompts
        if len(parts) == 1:
            prompts = []
            if mcp_service:
                for server_id, prompt_list in mcp_service.server_prompts.items():
                    for prompt in prompt_list:
                        prompt_name = prompt.name
                        if prompt_name:
                            prompts.append(f"{server_id}/{prompt_name}")
            msg = (
                "Available MCP prompts:\n" + "\n".join(prompts)
                if prompts
                else "No MCP prompts found."
            )
            self.message_handler._notify("system_message", msg)
            return False, True
        # /mcp <server_id.prompt_name>: fetch and show the prompt
        elif len(parts) == 2:
            full_name = parts[1]
            if "/" not in full_name:
                self.message_handler._notify(
                    "error", "Please use format: /mcp server_id/prompt_name"
                )
                return False, True
            server_id, prompt_name = full_name.split("/", 1)
            try:
                prompt = await mcp_service.get_prompt(server_id, prompt_name)
                prompt_content = prompt.get("content", [])
                if len(prompt_content) > 0:
                    prompt_text = prompt_content[0].content.text
                    self.message_handler._notify(
                        "mcp_prompt",
                        {"name": prompt_name, "content": f"{prompt_text}"},
                    )
                else:
                    self.message_handler._notify(
                        "error", f"Prompt {server_id}.{prompt_name} not found."
                    )
            except Exception as e:
                self.message_handler._notify(
                    "error", f"Error fetching prompt: {str(e)}"
                )
            return False, True
        else:
            self.message_handler._notify("error", "Usage: /mcp [server_id.prompt_name]")
            return False, True

    def _handle_jump_command(self, command: str) -> bool:
        """Handle the /jump command to rewind conversation to a previous turn.

        Usage:
            /jump          - Show available turns for jumping
            /jump <turn>   - Jump back to the specified turn
        """
        try:
            parts = command.split()

            # Show available turns if no turn number provided
            if len(parts) == 1:
                if not self.message_handler.conversation_turns:
                    self.message_handler._notify(
                        "system_message",
                        "No conversation turns available for jumping.",
                    )
                    return True

                turns_info = []
                for i, turn in enumerate(self.message_handler.conversation_turns, 1):
                    preview = turn.get_preview(50)
                    turns_info.append(f"  {i}. {preview}")

                message = (
                    "📋 Available turns for jumping:\n"
                    + "\n".join(turns_info)
                    + "\n\nUsage: /jump <turn_number>"
                )
                self.message_handler._notify("system_message", message)
                return True

            turn_number = int(parts[1])

            if turn_number < 1 or turn_number > len(
                self.message_handler.conversation_turns
            ):
                self.message_handler._notify(
                    "error",
                    f"Invalid turn number. Available turns: 1-{len(self.message_handler.conversation_turns)}",
                )
                return False

            selected_turn = self.message_handler.conversation_turns[turn_number - 1]

            selected_message = self.message_handler.streamline_messages[
                selected_turn.message_index
            ]

            selected_message_agent = selected_message.get("agent", "")

            self.message_handler.streamline_messages = (
                self.message_handler.streamline_messages[: selected_turn.message_index]
            )
            if (
                self.message_handler.current_conversation_id
                and self.message_handler.persistent_service
            ):
                self.message_handler.persistent_service.append_conversation_messages(
                    self.message_handler.current_conversation_id,
                    self.message_handler.streamline_messages,
                    True,
                )

            last_message = next(
                (msg for msg in reversed(self.message_handler.streamline_messages)),
                None,
            )
            if last_message and last_message.get("agent", ""):
                self._handle_agent_command(f"/agent {last_message['agent']}")
            elif selected_message_agent:
                self._handle_agent_command(f"/agent {selected_message_agent}")

            self.message_handler.agent_manager.rebuild_agents_messages(
                self.message_handler.streamline_messages
            )
            self.message_handler.conversation_turns = (
                self.message_handler.conversation_turns[: turn_number - 1]
            )
            self.message_handler.last_assisstant_response_idx = len(
                self.message_handler.streamline_messages
            )
            if isinstance(selected_message.get("content"), list):
                selected_content = next(
                    (
                        c.get("text", "")
                        for c in selected_message.get("content", [])
                        if c.get("type", "") == "text"
                    ),
                    "",
                )

            else:
                selected_content = selected_message.get("content", "")

            self.message_handler._notify(
                "jump_performed",
                {
                    "turn_number": turn_number,
                    "preview": selected_turn.get_preview(100),
                    "message": selected_content,
                },
            )

            return True

        except ValueError:
            self.message_handler._notify(
                "error", "Invalid turn number. Please provide a number."
            )
            return False

    def _handle_fork_command(self, command: str) -> CommandResult:
        """Handle the /fork command to create a conversation fork at a specific turn.

        Usage:
            /fork          - Show available turns for forking
            /fork <turn>   - Fork at the specified turn and switch to the new conversation
        """
        try:
            parts = command.split()

            # Show available turns if no turn number provided
            if len(parts) == 1:
                if not self.message_handler.conversation_turns:
                    self.message_handler._notify(
                        "system_message",
                        "No conversation turns available for forking.",
                    )
                    return CommandResult(handled=True, clear_flag=True)

                # Build list of available turns
                turns_info = []
                for i, turn in enumerate(self.message_handler.conversation_turns, 1):
                    preview = turn.get_preview(50)
                    turns_info.append(f"  {i}. {preview}")

                message = (
                    "📋 Available turns for forking:\n"
                    + "\n".join(turns_info)
                    + "\n\nUsage: /fork <turn_number>"
                )
                self.message_handler._notify("system_message", message)
                return CommandResult(handled=True, clear_flag=True)

            # Parse turn number
            turn_arg = parts[1]
            turn_number = int(turn_arg)

            # Validate turn number
            if turn_number < 1 or turn_number > len(
                self.message_handler.conversation_turns
            ):
                self.message_handler._notify(
                    "error",
                    f"Invalid turn number. Available turns: 1-{len(self.message_handler.conversation_turns)}",
                )
                return CommandResult(handled=True, clear_flag=True)

            # Get the selected turn for preview
            selected_turn = self.message_handler.conversation_turns[turn_number - 1]
            preview = selected_turn.get_preview(100)

            # Create the fork and switch to it
            success = self.message_handler.conversation_manager.fork_and_switch(
                turn_number
            )
            if success:
                self.message_handler._notify(
                    "fork_and_switch_performed",
                    {
                        "turn_number": turn_number,
                        "preview": preview,
                    },
                )
            return CommandResult(handled=success, clear_flag=True)

        except ValueError:
            self.message_handler._notify(
                "error", "Invalid turn number. Please provide a number."
            )
            return CommandResult(handled=True, clear_flag=True)
        except Exception as e:
            self.message_handler._notify(
                "error", f"Failed to fork conversation: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)

    def _handle_model_command(self, command: str) -> Tuple[bool, bool]:
        """
        Handle the /model command to switch models or list available models.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        model_id = command[7:].strip()
        registry = ModelRegistry.get_instance()
        manager = ServiceManager.get_instance()

        # If no model ID is provided, list available models
        if not model_id:
            models_by_provider = {}
            for provider in registry.get_providers():
                models = registry.get_models_by_provider(provider)
                if models:
                    models_by_provider[provider] = []
                    for model in models:
                        current = (
                            registry.current_model
                            and registry.current_model.id == model.id
                        )
                        models_by_provider[provider].append(
                            {
                                "id": f"{model.provider}/{model.id}",
                                "name": model.name,
                                "description": model.description,
                                "capabilities": model.capabilities,
                                "current": current,
                            }
                        )

            self.message_handler._notify("models_listed", models_by_provider)
            return False, True

        if registry.set_current_model(model_id):
            model = registry.get_current_model()
            if model:
                manager.set_model_for_model(model)

                new_llm_service = manager.get_service_for_model(model)

                self.message_handler.agent_manager.update_llm_service(new_llm_service)

                try:
                    GlobalConfig().set_last_used_model(model_id, model.provider)
                except Exception as e:
                    print(f"Warning: Failed to save last used model: {e}")

                self.message_handler._notify(
                    "model_changed",
                    {"id": model.id, "name": model.name, "provider": model.provider},
                )
            else:
                self.message_handler._notify("error", "Failed to switch model.")
        else:
            self.message_handler._notify("error", f"Unknown model: {model_id}")

        return False, True

    def _handle_agent_command(self, command: str) -> Tuple[bool, str]:
        """
        Handle the /agent command to switch agents or list available agents.

        Returns:
            Tuple of (success, message)
        """
        parts = command.split()

        if len(parts) == 1:
            agents_info = {"current": self.message_handler.agent.name, "available": {}}

            for agent_name, agent in self.message_handler.agent_manager.agents.items():
                agents_info["available"][agent_name] = {
                    "description": agent.description,
                    "current": (
                        self.message_handler.agent
                        and self.message_handler.agent.name == agent_name
                    ),
                }

            self.message_handler._notify("agents_listed", agents_info)
            return True, "Listed available agents"

        agent_name = parts[1]
        old_agent_name = self.message_handler.agent_manager.get_current_agent().name
        if old_agent_name == agent_name:
            return (False, f"Already using {agent_name} agent")
        if self.message_handler.agent_manager.select_agent(agent_name):
            self.message_handler.agent = (
                self.message_handler.agent_manager.get_current_agent()
            )
            old_agent = self.message_handler.agent_manager.get_agent(old_agent_name)
            if old_agent:
                self.message_handler.agent.history = list(old_agent.history)
                old_agent.history = []

            try:
                GlobalConfig().set_last_used_agent(agent_name)
            except Exception as e:
                # Don't fail the command if config save fails, just log it
                print(f"Warning: Failed to save last used agent: {e}")

            self.message_handler._notify("agent_changed", agent_name)
            return True, f"Switched to {agent_name} agent"
        else:
            available_agents = ", ".join(
                self.message_handler.agent_manager.agents.keys()
            )
            self.message_handler._notify(
                "error",
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )
            return (
                False,
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )

    def _handle_file_command(self, user_input: str) -> CommandResult:
        """Handle file command with support for multiple files."""
        # Extract file paths from user input (space-separated)
        file_paths_str: str = user_input[6:].strip()
        file_paths: List[str] = [
            os.path.expanduser(path.strip())
            for path in shlex.split(file_paths_str)
            if path.strip()
        ]

        if not file_paths:
            self.message_handler._notify("error", "No file paths provided")
            return CommandResult(handled=True, clear_flag=True)

        processed_files: List[str] = []
        failed_files: List[str] = []
        all_file_contents: List[Dict[str, str]] = []

        # Process each file
        for file_path in file_paths:
            # self.message_handler._notify("file_processing", {"file_path": file_path})

            # Process file with the file handling service
            if self.message_handler.file_handler is None:
                self.message_handler.file_handler = FileHandler()
            file_content = self.message_handler.file_handler.process_file(file_path)

            # Fallback to llm handle
            if not file_content:
                from AgentCrew.modules.agents.base import MessageType

                file_content = self.message_handler.agent.format_message(
                    MessageType.FileContent, {"file_uri": file_path}
                )

            if file_content:
                all_file_contents.append(file_content)
                processed_files.append(file_path)
                self.message_handler._notify(
                    "file_processed",
                    {
                        "file_path": file_path,
                        "message": file_content,
                    },
                )
            else:
                failed_files.append(file_path)
                self.message_handler._notify(
                    "error",
                    f"Failed to process file {file_path} Or Model is not supported",
                )

        # Add all successfully processed file contents to messages
        if all_file_contents:
            self.message_handler._messages_append(
                {
                    "role": "user",
                    "agent": self.message_handler.agent.name,
                    "content": all_file_contents,
                }
            )

            # Notify about overall processing results
            if failed_files:
                self.message_handler._notify(
                    "error", f"Failed to process: {', '.join(failed_files)}"
                )
            self.message_handler._notify(
                "system_message",
                f"✅ Successfully processed {len(processed_files)} files: {', '.join(processed_files)}",
            )

        return CommandResult(handled=True, clear_flag=True)

    def _handle_drop_command(self, user_input: str) -> CommandResult:
        """Handle drop command to remove queued files."""
        # Extract file identifier from user input
        file_path = user_input[6:].strip()  # Remove "/drop " prefix

        if not file_path:
            # Show available files if no argument provided
            if not self.message_handler._queued_attached_files:
                self.message_handler._notify(
                    "error", "No files are currently queued for processing"
                )
                return CommandResult(handled=True, clear_flag=True)

            self.message_handler._notify(
                "system_message",
                "📋 Queued files:\n"
                + "\n".join(self.message_handler._queued_attached_files)
                + "\nUsage: /drop <file_id>",
            )
            return CommandResult(handled=True, clear_flag=True)

        try:
            # Get the file command and parse its file paths
            try:
                self.message_handler._queued_attached_files.remove(file_path)
            except Exception:
                self.message_handler._notify(
                    "error", f"Cannot unqueue file: {file_path}"
                )

            # Update or remove the command
            self.message_handler._notify(
                "system_message", f"🗑️ Removed file from queue: {file_path}"
            )

            # Notify about the file being removed for UI updates
            self.message_handler._notify("file_dropped", {"file_path": file_path})

            return CommandResult(handled=True, clear_flag=True)

        except ValueError as e:
            self.message_handler._notify("error", f"Invalid file ID format: {str(e)}")
            return CommandResult(handled=True, clear_flag=True)
        except Exception as e:
            self.message_handler._notify("error", f"Error removing file: {str(e)}")
            return CommandResult(handled=True, clear_flag=True)

    async def _handle_voice_command(self, command: str) -> CommandResult:
        """Handle /voice command to start voice recording."""
        try:
            # Check if already recording
            if self.message_handler.voice_service is None:
                self.message_handler._notify(
                    "error",
                    "Voice service not available. Start AgentCrew with --with-voice and set ELEVENLABS_API_KEY or DEEPINFRA_API_KEY.",
                )
                return CommandResult(handled=True, clear_flag=True)

            if self.message_handler.voice_service.is_recording():
                self.message_handler._notify(
                    "error",
                    "Already recording. Use /end_voice to stop current recording.",
                )
                return CommandResult(handled=True, clear_flag=True)

            if self.message_handler.has_active_stream():
                self.message_handler._notify(
                    "system_message",
                    "🎤 Voice input is unavailable while the assistant is still responding.",
                )
                return CommandResult(handled=True, clear_flag=True)

            async def submit_active_voice(audio_data: Any, sample_rate: int):
                if self.message_handler.has_active_stream():
                    return
                transcript = await self._voice_transcript(audio_data, sample_rate)
                if self.message_handler.has_active_stream():
                    return
                self.message_handler._notify("voice_activate", transcript)

            # Start recording
            result = self.message_handler.voice_service.start_voice_recording(
                voice_completed_cb=submit_active_voice
            )

            if result["success"]:
                self.message_handler._notify("voice_recording_started", None)
                self.message_handler._notify(
                    "system_message",
                    "🎤 Recording started. Press Enter to stop.",
                )
            else:
                self.message_handler._notify("error", result["error"])

            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify("error", f"Voice command failed: {str(e)}")
            return CommandResult(handled=True, clear_flag=True)

    async def _voice_transcript(self, audio_data: Any, sample_rate: int):
        transcribed_text = None
        if audio_data is not None and self.message_handler.voice_service:
            transcribe_result = await self.message_handler.voice_service.speech_to_text(
                audio_data, sample_rate
            )
            if transcribe_result["success"]:
                transcribed_text = transcribe_result["text"]
                confidence = transcribe_result.get("confidence", 1.0)

                # Notify about transcription
                self.message_handler._notify(
                    "system_message",
                    f"✅ Transcribed (confidence: {confidence:.0%}): {transcribed_text}",
                )

            else:
                self.message_handler._notify("error", transcribe_result["error"])

            return transcribed_text

    async def _handle_end_voice_command(self, command: str) -> CommandResult:
        """Handle /end_voice command to stop recording and transcribe."""
        try:
            # Check if voice service exists and is recording
            if self.message_handler.voice_service is None:
                # self.message_handler._notify(
                #     "error",
                #     "No voice service initialized. Use /voice to start recording.",
                # )
                return CommandResult(handled=True, clear_flag=True)

            if not self.message_handler.voice_service.is_recording():
                # self.message_handler._notify("error", "No recording in progress.")
                return CommandResult(handled=True, clear_flag=True)

            # Stop recording
            self.message_handler._notify("voice_recording_stopping", None)
            stop_result = self.message_handler.voice_service.stop_voice_recording()

            if not stop_result["success"]:
                # self.message_handler._notify("error", stop_result["error"])
                return CommandResult(handled=True, clear_flag=True)

            # Transcribe
            # self.message_handler._notify("system_message", "🔄 Transcribing audio...")

            # transcribed_text = await self._voice_transcript(
            #     stop_result["audio_data"], stop_result["sample_rate"]
            # )

            self.message_handler._notify("voice_recording_completed", None)
            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify("error", f"End voice command failed: {str(e)}")
            self.message_handler._notify("voice_recording_completed", None)
            return CommandResult(handled=True, clear_flag=True)

    def _handle_debug_command(self, user_input: str) -> CommandResult:
        """Handle /debug command with optional filtering.

        Usage:
            /debug         - Show both agent and chat messages
            /debug agent   - Show only agent messages
            /debug chat    - Show only chat/streamline messages
        """
        parts = user_input.lower().split()
        filter_type = parts[1] if len(parts) > 1 else None

        if filter_type and filter_type not in ("agent", "chat"):
            self.message_handler._notify(
                "error", f"Invalid filter '{filter_type}'. Use 'agent' or 'chat'."
            )
            return CommandResult(handled=True, clear_flag=True)

        if filter_type is None or filter_type == "agent":
            self.message_handler._notify(
                "debug_requested",
                {"type": "agent", "messages": self.message_handler.agent.clean_history},
            )

        if filter_type is None or filter_type == "chat":
            self.message_handler._notify(
                "debug_requested",
                {"type": "chat", "messages": self.message_handler.streamline_messages},
            )

        return CommandResult(handled=True, clear_flag=True)

    def _handle_toggle_transfer_command(self, user_input: str) -> CommandResult:
        """Handle /toggle_transfer command — backward-compat alias toggling between transfer and none."""
        try:
            from AgentCrew.modules.agents.manager import AgentMode

            current_mode = self.message_handler.agent_manager.agent_mode
            new_mode = (
                AgentMode.NONE
                if current_mode == AgentMode.TRANSFER
                else AgentMode.TRANSFER
            )
            self.message_handler.agent_manager.agent_mode = new_mode

            self.message_handler.agent.deactivate()
            self.message_handler.agent.activate()

            status = "enabled" if new_mode == AgentMode.TRANSFER else "disabled"
            self.message_handler._notify(
                "system_message", f"🔄 Transfer enforcement is now {status}."
            )
            self.message_handler._notify("transfer_enforce_toggled", status)

            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify(
                "error", f"Failed to toggle transfer enforcement: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)

    def _handle_agent_mode_command(self, user_input: str) -> CommandResult:
        """Handle /agent_mode command to view or switch agent interaction mode."""
        try:
            from AgentCrew.modules.agents.manager import AgentMode

            parts = user_input.strip().split(maxsplit=1)

            if len(parts) == 1:
                current = self.message_handler.agent_manager.agent_mode.value
                self.message_handler._notify(
                    "system_message",
                    f"🤖 Current agent mode: **{current}**\n"
                    f"Options: transfer, delegate, none",
                )
                return CommandResult(handled=True, clear_flag=True)

            mode_str = parts[1].lower()
            try:
                new_mode = AgentMode(mode_str)
            except ValueError:
                self.message_handler._notify(
                    "error",
                    f"Invalid mode '{mode_str}'. Options: transfer, delegate, none",
                )
                return CommandResult(handled=True, clear_flag=True)

            self.message_handler.agent_manager.agent_mode = new_mode

            self.message_handler.agent.deactivate()
            self.message_handler.agent.activate()

            self.message_handler._notify(
                "system_message",
                f"🔄 Agent mode switched to: **{new_mode.value}**",
            )

            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify(
                "error", f"Failed to switch agent mode: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)
