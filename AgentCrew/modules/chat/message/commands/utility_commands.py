from __future__ import annotations

from typing import TYPE_CHECKING

from AgentCrew.modules.chat.message.commands.base import CommandResult

if TYPE_CHECKING:
    from AgentCrew.modules.chat.message import MessageHandler


class UtilityCommands:
    """Handles utility slash commands."""

    def __init__(self, message_handler: MessageHandler):
        self.message_handler = message_handler

    def handle_think(self, user_input: str) -> CommandResult:
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

    async def handle_copy(self, user_input: str) -> CommandResult:
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

    def handle_debug(self, user_input: str) -> CommandResult:
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
