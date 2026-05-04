"""
Main console UI class that orchestrates all console functionality.
Refactored to use separate modules for different responsibilities.
"""

from __future__ import annotations
import asyncio
import queue
import threading
import time
import sys
import signal
from typing import Any
from rich.console import Console
from rich.text import Text
from AgentCrew.modules.chat.message_handler import Observer
from AgentCrew.modules.chat.agent_evaluation import parse_agent_evaluation
from loguru import logger

from .constants import (
    RICH_STYLE_GREEN,
    RICH_STYLE_BLUE,
    RICH_STYLE_YELLOW,
    RICH_STYLE_YELLOW_BOLD,
    PROMPT_CHAR,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AgentCrew.modules.chat.message_handler import MessageHandler


class ConsoleUI(Observer):
    """
    A console-based UI for the interactive chat that implements the Observer interface
    to receive updates from the MessageHandler.
    """

    def __init__(self, message_handler: MessageHandler, swap_enter: bool = False):
        """
        Initialize the ConsoleUI.

        Args:
            message_handler: The MessageHandler instance that this UI will observe.
        """

        from .display_handlers import DisplayHandlers
        from .tool_display import ToolDisplayHandlers
        from .input_handler import InputHandler
        from .ui_effects import UIEffects
        from .confirmation_handler import ConfirmationHandler
        from .conversation_handler import ConversationHandler
        from .command_handlers import CommandHandlers

        self.message_handler = message_handler
        self.message_handler.attach(self)

        self._is_resizing = False

        self.console = Console()
        self._last_ctrl_c_time = 0
        self.session_cost = 0.0

        # Initialize component handlers
        self.display_handlers = DisplayHandlers(self)
        self.tool_display = ToolDisplayHandlers(self)
        self.ui_effects = UIEffects(self)
        self.input_handler = InputHandler(self, swap_enter=swap_enter)
        self.confirmation_handler = ConfirmationHandler(self)
        self.conversation_handler = ConversationHandler(self)
        self.command_handlers = CommandHandlers(self)
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0

    def _set_voice_processing_state(self, is_processing: bool):
        voice_service = self.message_handler.voice_service
        if voice_service and hasattr(voice_service, "audio_handler"):
            voice_service.audio_handler.is_processing = is_processing
            if is_processing:
                voice_service.audio_handler.clear_buffered_audio()

    def _clear_pending_input_queue(self):
        while True:
            try:
                self.input_handler._input_queue.get_nowait()
            except queue.Empty:
                break

    def _process_voice_activation(self, transcript: str):
        assistant_response = None

        try:
            self.input_handler.is_message_processing = True
            should_exit, was_cleared = asyncio.run(
                self.message_handler.process_user_input(transcript)
            )

            if should_exit or was_cleared or not self.message_handler.agent.history:
                return

            assistant_response, self._input_tokens, self._output_tokens = asyncio.run(
                self.message_handler.get_assistant_response()
            )
        except Exception as e:
            self.message_handler._notify("error", f"Voice activation failed: {str(e)}")
        finally:
            self.input_handler.is_message_processing = False
            self._clear_pending_input_queue()
            self._set_voice_processing_state(False)

        self._calculate_token_usage(self._input_tokens, self._output_tokens)

        if assistant_response:
            self.display_token_usage(
                self._input_tokens,
                self._output_tokens,
                self._total_cost,
                self.session_cost,
            )

    def listen(self, event: str, data: Any = None):
        """
        Update method required by the Observer interface. Handles events from the MessageHandler.

        Args:
            event: The type of event that occurred.
            data: The data associated with the event.
        """

        if event == "thinking_started":
            self.ui_effects.stop_loading_animation()  # Stop loading on first chunk
            # self.display_handlers.display_thinking_started(data)  # data is agent_name
        elif event == "thinking_chunk":
            if data.strip():
                self.ui_effects.update_live_display(data, is_thinking=True)
        elif event == "thinking_completed":
            self.ui_effects.finish_response(
                self.ui_effects.updated_text, is_thinking=True
            )
        elif event == "user_message_created":
            pass
        elif event == "response_chunk":
            _, assistant_response = data
            parsed = parse_agent_evaluation(assistant_response)

            self.ui_effects.stop_loading_animation()
            self.ui_effects.update_live_display(
                parsed["visible_content"],
                planning_content=parsed["planning_content"],
            )
        elif event == "tool_use":
            self.ui_effects.stop_loading_animation()  # Stop loading on first chunk
            if data.get("name") == "delegate":
                self.tool_display.display_delegate_started(data)
                params = data.get("input") or data.get("arguments", {})
                agent_name = (
                    params.get("target_agent", "Agent")
                    if isinstance(params, dict)
                    else "Agent"
                )
                self.ui_effects.start_delegate_animation(
                    data.get("id", agent_name), agent_name
                )
            else:
                self.tool_display.display_tool_use(data)  # data is the tool use object
        elif event == "tool_result":
            if data.get("tool_use", {}).get("name") == "delegate":
                tool_use = data["tool_use"]
                params = tool_use.get("input") or tool_use.get("arguments", {})
                agent_name = (
                    params.get("target_agent", "Agent")
                    if isinstance(params, dict)
                    else "Agent"
                )
                self.ui_effects.stop_delegate_animation(tool_use.get("id", agent_name))
                self.tool_display.display_delegate_completed(tool_use)
            else:
                self.ui_effects.start_loading_animation()
        elif event == "tool_error":
            self.tool_display.display_tool_error(
                data
            )  # data is dict with tool_use and error
        elif event == "tool_confirmation_required":
            self.ui_effects.stop_loading_animation()  # Stop loading on first chunk
            self.confirmation_handler.display_tool_confirmation_request(
                data, self.message_handler
            )  # data is the tool use with confirmation ID
        elif event == "tool_denied":
            self.tool_display.display_tool_denied(
                data
            )  # data is the tool use that was denied
        elif event == "response_completed" or event == "assistant_message_added":
            parsed = parse_agent_evaluation(data)
            self.ui_effects.finish_response(
                parsed["visible_content"],
                planning_content=parsed["planning_content"],
            )
            if event == "response_completed":
                self._set_voice_processing_state(False)

        elif event == "stream_cancel_requested":
            self.display_handlers.display_message(
                Text("Stopping current stream...", style=RICH_STYLE_YELLOW)
            )
        elif event == "stream_canceled":
            self.ui_effects.cleanup()
            self.display_handlers.display_message(
                Text("Stream canceled.", style=RICH_STYLE_YELLOW_BOLD)
            )
        elif event == "stream_open_timeout":
            self.ui_effects.cleanup()
            self.display_handlers.display_message(
                Text(
                    "Stream timed out before first chunk.", style=RICH_STYLE_YELLOW_BOLD
                )
            )
        elif event == "error":
            self.display_handlers.display_error(
                data
            )  # data is the error message or dict
            self.ui_effects.cleanup()
        elif event == "clear_requested":
            self.display_handlers.display_message(
                Text("🎮 Chat history cleared.", style=RICH_STYLE_YELLOW_BOLD)
            )
            self.display_handlers.clear_files()
            self.session_cost = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._total_cost = 0
        elif event == "copy_requested":
            self.copy_to_clipboard(data)  # data is the text to copy
        elif event == "debug_requested":
            self.display_handlers.display_debug_info(
                data
            )  # data is the debug information
        elif event == "think_budget_set":
            thinking_text = Text("Thinking budget set to ", style=RICH_STYLE_YELLOW)
            thinking_text.append(f"{data} tokens.")
            self.display_handlers.display_message(thinking_text)
        elif event == "models_listed":
            self.display_handlers.display_models(
                data
            )  # data is dict of models by provider
        elif event == "model_changed":
            model_text = Text("Switched to ", style=RICH_STYLE_YELLOW)
            model_text.append(f"{data['name']} ({data['id']})")
            self.display_handlers.display_message(model_text)
        elif event == "agents_listed":
            self.display_handlers.display_agents(data)  # data is dict of agent info
        elif event == "agent_changed":
            agent_text = Text("Switched to ", style=RICH_STYLE_YELLOW)
            agent_text.append(f"{data} agent")
            self.display_handlers.display_message(agent_text)
        elif event == "system_message":
            self.display_handlers.display_message(data)
        elif event == "mcp_prompt":
            self.confirmation_handler.display_mcp_prompt_confirmation(
                data, self.input_handler._input_queue
            )
        elif event == "agent_changed_by_transfer":
            transfer_text = Text("Transfered to ", style=RICH_STYLE_YELLOW)
            transfer_text.append(
                f"{data['agent_name'] if 'agent_name' in data else 'other'} agent"
            )
            self.display_handlers.display_message(transfer_text)
        elif event == "jump_performed":
            jump_text = Text(
                f"🕰️ Jumping to turn {data['turn_number']}...\n",
                style=RICH_STYLE_YELLOW_BOLD,
            )
            preview_text = Text("Conversation rewound to: ", style=RICH_STYLE_YELLOW)
            preview_text.append(data["preview"])
            self._clear_and_reprint_chat()

            self.display_handlers.display_message(jump_text)
            self.display_handlers.display_message(preview_text)
            self.input_handler.set_current_buffer(data["message"])
        elif event == "fork_and_switch_performed":
            fork_text = Text(
                f"🍴 Forked at turn {data['turn_number']}...\n",
                style=RICH_STYLE_YELLOW_BOLD,
            )
            preview_text = Text("Switched to fork: ", style=RICH_STYLE_YELLOW)
            preview_text.append(data["preview"])
            self._clear_and_reprint_chat()

            self.display_handlers.display_message(fork_text)
            self.display_handlers.display_message(preview_text)
        elif event == "evolution_summary_ready":
            self.ui_effects.stop_evolution_animation()
            self.display_handlers.display_evolution_summary(data)
            self.input_handler._stop_input_thread()
            choice = self.input_handler.get_choice_input(
                "Review prompt evolution proposal:",
                ["accept", "edit", "decline"],
                default="accept",
            )
            if choice == "accept":
                asyncio.run(
                    self.message_handler.submit_pending_evolution_review("accept")
                )
            elif choice == "edit":
                edited_summary = self.input_handler.get_prompt_input(
                    "Edit approved summary (Alt+Enter or Ctrl+S to submit):",
                    default=data.get("user_editable_summary", ""),
                )
                if edited_summary.strip():
                    asyncio.run(
                        self.message_handler.submit_pending_evolution_review(
                            "edit", edited_summary.strip()
                        )
                    )
                else:
                    asyncio.run(
                        self.message_handler.submit_pending_evolution_review("decline")
                    )
            else:
                asyncio.run(
                    self.message_handler.submit_pending_evolution_review("decline")
                )
            self.input_handler._start_input_thread()
        elif event == "evolution_applied":
            self.ui_effects.stop_evolution_animation()
            result_text = Text(
                "🧬 Prompt evolution applied for ", style=RICH_STYLE_YELLOW
            )
            result_text.append(data["agent_name"], style=RICH_STYLE_GREEN)
            self.display_handlers.display_message(result_text)
            self.display_handlers.display_prompt_evolution_result(
                data,
                max_width=max(30, (self.console.width // 2) - 6),
            )
        elif event == "evolution_declined":
            self.display_handlers.display_message(
                Text("Prompt evolution declined.", style=RICH_STYLE_YELLOW)
            )
        elif event == "evolution_started":
            self.ui_effects.start_evolution_animation(
                data.get("agent_name", "Agent") if data else "Agent"
            )
        elif event == "evolution_finished":
            self.ui_effects.stop_evolution_animation()
        elif event == "file_processing":
            self.ui_effects.stop_loading_animation()  # Stop loading on first chunk
            self.display_handlers.add_file(data["file_path"])
        elif event == "file_dropped":
            self.display_handlers._added_files.remove(data["file_path"])
        elif event == "consolidation_completed":
            self.display_handlers.display_consolidation_result(data)
            self.display_handlers.display_loaded_conversation(
                self.message_handler.streamline_messages,
                self.message_handler.agent.name,
            )

        elif event == "unconsolidation_completed":
            self.display_handlers.display_loaded_conversation(
                self.message_handler.streamline_messages,
                self.message_handler.agent.name,
            )
        elif event == "conversations_listed":
            self.display_handlers.display_conversations(
                data,
                get_history_callback=self.conversation_handler.get_conversation_history,
                delete_callback=self.conversation_handler.delete_conversations,
            )
            self.conversation_handler.update_cached_conversations(data)
        elif event == "conversation_loaded":
            loaded_text = Text("Loaded conversation: ", style=RICH_STYLE_YELLOW)
            loaded_text.append(data.get("id", "N/A"))
            self.display_handlers.display_message(loaded_text)
        elif event == "conversation_saved":
            logger.info(f"Conversation saved: {data.get('id', 'N/A')}")
        elif event == "update_token_usage":
            self._input_tokens = data.get("input_tokens", 0)
            self._output_tokens = data.get("output_tokens", 0)
            self._calculate_token_usage(self._input_tokens, self._output_tokens)
        elif event == "voice_recording_started":
            self.display_handlers.display_message(
                Text("Start recording. Press Enter to stop...", style="bold yellow")
            )
        elif event == "voice_activate":
            if data:
                self._set_voice_processing_state(True)
                threading.Thread(
                    target=self._process_voice_activation,
                    args=(data,),
                    daemon=True,
                ).start()

        elif event == "voice_recording_stopping":
            self.display_handlers.display_message(
                Text("⏹️  Stopping recording...", style="bold yellow")
            )
        elif event == "voice_recording_completed":
            pass

    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard and show confirmation."""
        try:
            import pyperclip
        except ImportError:
            pyperclip = None
        if text:
            if pyperclip:
                pyperclip.copy(text)
                self.console.print(
                    Text("\n✓ Text copied to clipboard!", style=RICH_STYLE_YELLOW)
                )
            else:
                self.console.print(
                    Text(
                        "\n! Clipboard functionality not available (pyperclip not installed)",
                        style=RICH_STYLE_YELLOW,
                    )
                )
        else:
            self.console.print(Text("\n! No text to copy.", style=RICH_STYLE_YELLOW))

    def _handle_terminal_resize(self, signum, frame):
        """
        Signal handler for SIGWINCH.
        This function is called when the terminal window is resized.
        """
        import time

        if self.input_handler.is_message_processing or self._is_resizing:
            return  # Ignore resize during message processing
        self._is_resizing = True
        time.sleep(0.5)  # brief pause to allow resize to complete
        self._clear_and_reprint_chat()

        self.display_token_usage(
            self._input_tokens,
            self._output_tokens,
            self._total_cost,
            self.session_cost,
        )

        self.display_handlers.print_prompt_prefix(
            self.message_handler.agent.name,
            self.message_handler.agent.get_model(),
            self.message_handler.tool_manager.get_effective_yolo_mode(),
        )

        self.display_handlers.print_divider("👤 YOU: ", with_time=True)
        prompt = Text(
            PROMPT_CHAR,
            style=RICH_STYLE_BLUE,
        )
        if self.input_handler._current_prompt_session:
            prompt.append(
                self.input_handler._current_prompt_session.default_buffer.text,
                style="white",
            )

        self.console.print(prompt, end="")
        self._is_resizing = False

    def _clear_and_reprint_chat(self):
        """Clear and reprint the chat display."""

        import os

        os.system("cls" if os.name == "nt" else "printf '\033c'")
        self.display_handlers.display_loaded_conversation(
            self.message_handler.streamline_messages, self.message_handler.agent.name
        )

    def start_streaming_response(self, agent_name: str):
        """Start streaming the assistant's response."""
        self.ui_effects.start_streaming_response(agent_name)

    def update_live_display(self, chunk: str):
        """Update the live display with a new chunk of the response."""
        if not self.ui_effects.live:
            self.start_streaming_response(self.message_handler.agent.name)
        self.ui_effects.update_live_display(chunk)

    def finish_live_update(self):
        """Stop the live update display."""
        self.ui_effects.finish_live_update()

    def start_loading_animation(self):
        """Start the loading animation."""
        self.ui_effects.start_loading_animation()

    def stop_loading_animation(self):
        """Stop the loading animation."""
        self.ui_effects.stop_loading_animation()

    def get_user_input(self):
        """Get user input using the input handler."""
        return self.input_handler.get_user_input()

    def _handle_keyboard_interrupt(self):
        """Handle Ctrl+C pressed during streaming or other operations."""
        self.ui_effects.stop_loading_animation()
        self.message_handler.request_stop_stream()

        current_time = time.time()
        if (
            hasattr(self, "_last_ctrl_c_time")
            and current_time - self._last_ctrl_c_time < 2
        ):
            self.console.print(
                Text(
                    "\n🎮 Confirmed exit. Goodbye!",
                    style=RICH_STYLE_YELLOW_BOLD,
                )
            )
            self.input_handler.stop()
            raise SystemExit(0)
        else:
            self._last_ctrl_c_time = current_time
            self.console.print(
                Text(
                    "\n🎮 Chat interrupted. Press Ctrl+C again within 2 seconds to exit.",
                    style=RICH_STYLE_YELLOW_BOLD,
                )
            )

    def print_welcome_message(self):
        """Print the welcome message for the chat."""
        import AgentCrew

        version = getattr(AgentCrew, "__version__", "Unknown")
        self.display_handlers.print_welcome_message(version)

    def print_logo(self):
        self.console.print(
            Text(
                """
  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗  ██████╗ ██████╗  ███████╗ ██╗    ██╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██║    ██║
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ██║      ██████╔╝ █████╗   ██║ █╗ ██║
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║      ██╔══██╗ ██╔══╝   ██║███╗██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ╚██████╗ ██║  ██║ ███████╗ ╚███╔███╔╝
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝     ╚═════╝ ╚═╝  ╚═╝ ╚══════╝  ╚══╝╚══╝ 
        """,
                RICH_STYLE_GREEN,
            )
        )

    def display_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_cost: float,
        session_cost: float,
    ):
        """Display token usage and cost information."""
        self.display_handlers.display_token_usage(
            input_tokens, output_tokens, total_cost, session_cost
        )

    def _calculate_token_usage(self, input_tokens: int, output_tokens: int):
        """Calculate token usage and update session cost."""
        self._total_cost = self.message_handler.agent.calculate_usage_cost(
            input_tokens, output_tokens
        )
        self.session_cost += self._total_cost

    def start(self):
        """Start the console UI main loop."""
        self.print_logo()
        self.print_welcome_message()

        self.session_cost = 0.0

        try:
            while True:
                if sys.platform != "win32":
                    if (
                        not signal.getsignal(signal.SIGWINCH)
                        or signal.getsignal(signal.SIGWINCH) == signal.SIG_DFL
                    ):
                        signal.signal(signal.SIGWINCH, self._handle_terminal_resize)
                try:
                    # Get user input (now in separate thread)
                    self.input_handler.is_message_processing = False
                    self.stop_loading_animation()  # Stop if any
                    user_input = self.get_user_input()

                    # Handle list command directly
                    if user_input.strip() in ["/exit", "/quit"]:
                        self.display_handlers.display_message(
                            Text(
                                "🎮 Ending chat session. Goodbye!",
                                style=RICH_STYLE_YELLOW_BOLD,
                            )
                        )
                        self.input_handler.stop()
                        raise SystemExit(0)
                    elif user_input.strip() == "/list":
                        conversations = (
                            self.message_handler.list_conversations_with_forks()
                        )
                        self.conversation_handler.update_cached_conversations(
                            conversations
                        )
                        self.input_handler._stop_input_thread()
                        try:
                            selected_id = self.display_handlers.display_conversations(
                                conversations,
                                get_history_callback=self.conversation_handler.get_conversation_history,
                                delete_callback=self.conversation_handler.delete_conversations,
                            )
                            if selected_id:
                                self.conversation_handler.handle_load_conversation(
                                    selected_id, self.message_handler
                                )
                        finally:
                            self.input_handler._start_input_thread()
                        continue

                    # Handle load command directly
                    elif user_input.strip().startswith("/load"):
                        load_arg = user_input.strip()[
                            5:
                        ].strip()  # Extract argument after "/load"
                        if load_arg:
                            self.conversation_handler.handle_load_conversation(
                                load_arg, self.message_handler
                            )
                        else:
                            # No argument: show conversation list like /list
                            conversations = (
                                self.message_handler.list_conversations_with_forks()
                            )
                            self.conversation_handler.update_cached_conversations(
                                conversations
                            )
                            self.input_handler._stop_input_thread()
                            try:
                                selected_id = self.display_handlers.display_conversations(
                                    conversations,
                                    get_history_callback=self.conversation_handler.get_conversation_history,
                                    delete_callback=self.conversation_handler.delete_conversations,
                                )
                                if selected_id:
                                    self.conversation_handler.handle_load_conversation(
                                        selected_id, self.message_handler
                                    )
                            finally:
                                self.input_handler._start_input_thread()
                        continue

                    elif user_input.strip() == "/help":
                        self.console.print("\n")
                        self.print_welcome_message()
                        continue

                    elif user_input.strip() == "/visual":
                        try:
                            self.input_handler._stop_input_thread()
                            from .visual_mode import VisualModeViewer

                            viewer = VisualModeViewer(
                                console=self.console,
                                on_copy=self.copy_to_clipboard,
                            )
                            viewer.set_messages(
                                self.message_handler.streamline_messages
                            )
                            viewer.show()
                        finally:
                            self._clear_and_reprint_chat()
                            self.input_handler._start_input_thread()
                        continue

                    # Handle toggle_session_yolo command directly (console only, session-based)
                    elif user_input.strip() == "/toggle_session_yolo":
                        self.command_handlers.handle_toggle_session_yolo_command()
                        continue

                    elif user_input.strip().startswith("/export_agent"):
                        # Extract arguments after "/export_agent"
                        args = user_input.strip()[13:].strip()
                        if args:
                            # Split into agent names and output file
                            # Expected format: /export_agent <agent1,agent2,...> <output_file>
                            parts = args.rsplit(maxsplit=1)
                            if len(parts) == 2:
                                agent_names, output_file = parts
                                self.command_handlers.handle_export_agent_command(
                                    agent_names, output_file
                                )
                            else:
                                self.console.print(
                                    Text(
                                        "Usage: /export_agent <agent_names> <output_file>\n"
                                        "Export selected agents to a TOML file.\n"
                                        "Agent names should be comma-separated.\n"
                                        "Example: /export_agent Agent1,Agent2 ./my_agents.toml",
                                        style=RICH_STYLE_YELLOW,
                                    )
                                )
                        else:
                            self.console.print(
                                Text(
                                    "Usage: /export_agent <agent_names> <output_file>\n"
                                    "Export selected agents to a TOML file.\n"
                                    "Agent names should be comma-separated.\n"
                                    "Example: /export_agent Agent1,Agent2 ./my_agents.toml",
                                    style=RICH_STYLE_YELLOW,
                                )
                            )
                        continue

                    elif user_input.strip().startswith("/import_agent"):
                        file_or_url = user_input.strip()[
                            13:
                        ].strip()  # Extract argument after "/import_agent"
                        if file_or_url:
                            self.command_handlers.handle_import_agent_command(
                                file_or_url
                            )
                        else:
                            self.console.print(
                                Text(
                                    "Usage: /import_agent <file_path_or_url>\nImport/replace agents from file or URL.\nExample: /import_agent ./agents.toml or /import_agent https://example.com/agents.toml",
                                    style=RICH_STYLE_YELLOW,
                                )
                            )
                        continue

                    # Handle edit_agent command directly
                    elif user_input.strip() == "/edit_agent":
                        self.command_handlers.handle_edit_agent_command()
                        continue

                    # Handle edit_mcp command directly
                    elif user_input.strip() == "/edit_mcp":
                        self.command_handlers.handle_edit_mcp_command()
                        continue

                    # Handle edit_config command directly
                    elif user_input.strip() == "/edit_config":
                        self.command_handlers.handle_edit_config_command()
                        continue

                    # Handle list_behaviors command
                    elif user_input.strip() == "/list_behaviors":
                        self.command_handlers.handle_list_behaviors_command()
                        continue

                    # Handle update_behavior command
                    elif user_input.strip().startswith("/update_behavior"):
                        args = user_input.strip()[16:].strip()
                        if args:
                            parts = args.split(maxsplit=2)
                            if len(parts) == 3:
                                scope, behavior_id, behavior_text = parts
                                self.command_handlers.handle_update_behavior_command(
                                    behavior_id, behavior_text, scope
                                )
                            else:
                                self.console.print(
                                    Text(
                                        "Usage: /update_behavior <scope> <id> <behavior_text>\n"
                                        "Example: /update_behavior project my_behavior_id when user asks about X, do provide detailed examples",
                                        style=RICH_STYLE_YELLOW,
                                    )
                                )
                        else:
                            self.console.print(
                                Text(
                                    "Usage: /update_behavior <scope> <id> <behavior_text>\n"
                                    "Example: /update_behavior project my_behavior_id when user asks about X, do provide detailed examples",
                                    style=RICH_STYLE_YELLOW,
                                )
                            )
                        continue

                    # Handle delete_behavior command
                    elif user_input.strip().startswith("/delete_behavior"):
                        args = user_input.strip()[16:].strip()
                        parts = args.split(maxsplit=1)
                        if len(parts) == 2:
                            scope, behavior_id = parts
                            self.command_handlers.handle_delete_behavior_command(
                                behavior_id, scope
                            )
                        else:
                            self.console.print(
                                Text(
                                    "Usage: /delete_behavior <scope> <id>\n"
                                    "Example: /delete_behavior <scope> my_behavior_id",
                                    style=RICH_STYLE_YELLOW,
                                )
                            )
                        continue

                    # Start loading animation while waiting for response
                    if (
                        not user_input.startswith("/")
                        or user_input.startswith("/file ")
                        or user_input.startswith("/consolidate ")
                        or user_input.startswith("/agent ")
                        or user_input.startswith("/model ")
                        or user_input.startswith("/retry")
                    ):
                        self.start_loading_animation()

                    # Process user input and commands
                    should_exit, was_cleared = asyncio.run(
                        self.message_handler.process_user_input(user_input)
                    )

                    if should_exit:
                        break

                    # Skip to next iteration if messages were cleared
                    if was_cleared:
                        continue

                    # Skip to next iteration if no messages to process
                    if not self.message_handler.agent.history:
                        continue

                    # Get assistant response
                    assistant_response, input_tokens, output_tokens = asyncio.run(
                        self.message_handler.get_assistant_response()
                    )

                    self._is_resizing = False
                    self._input_tokens = input_tokens
                    self._output_tokens = output_tokens

                    # Ensure loading animation is stopped
                    self.stop_loading_animation()

                    if assistant_response:
                        # Calculate and display token usage
                        self._calculate_token_usage(
                            self._input_tokens, self._output_tokens
                        )
                        self.display_token_usage(
                            self._input_tokens,
                            self._output_tokens,
                            self._total_cost,
                            self.session_cost,
                        )
                except KeyboardInterrupt:
                    self._handle_keyboard_interrupt()
                    continue  # Continue the loop instead of breaking
        finally:
            # Clean up input thread when exiting
            self.input_handler.stop()
            self.ui_effects.cleanup()
