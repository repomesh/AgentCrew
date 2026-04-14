from typing import Any

from PySide6.QtWidgets import QApplication
from AgentCrew.modules.chat.agent_evaluation import parse_agent_evaluation


class MessageEventHandler:
    """Handles message-related events in the chat UI."""

    def __init__(self, chat_window):
        from AgentCrew.modules.gui import ChatWindow

        if isinstance(chat_window, ChatWindow):
            self.chat_window = chat_window
        self.chat_window.thinking_content = ""

    def handle_event(self, event: str, data: Any):
        """Handle a message-related event."""
        if event == "response_chunk":
            self.handle_response_chunk(data)
        elif event == "user_message_created":
            self.handle_user_message_created(data)
        elif event == "response_completed" or event == "assistant_message_added":
            self.handle_response_completed(data)
        elif event == "thinking_started":
            self.handle_thinking_started(data)
        elif event == "thinking_chunk":
            self.handle_thinking_chunk(data)
        elif event == "thinking_completed":
            self.handle_thinking_completed()
        elif event == "stream_cancel_requested":
            self.handle_stream_cancel_requested()
        elif event == "stream_canceled":
            self.handle_stream_canceled(data)
        elif event == "stream_open_timeout":
            self.handle_stream_open_timeout(data)
        elif event == "user_context_request":
            self.handle_user_context_request()

    def _update_planning_widget(self, planning_content: str):
        if not planning_content.strip():
            return
        if self.chat_window.current_planning_widget is None:
            self.chat_window.current_planning_widget = (
                self.chat_window.chat_components.add_planning_message(planning_content)
            )
        else:
            self.chat_window.current_planning_widget.set_text(
                f"🧭 Agent plan\n\n{planning_content}"
            )
        self.chat_window.current_planning_content = planning_content

    def handle_response_chunk(self, data):
        """Handle response chunks with smooth streaming."""
        _, full_response = data
        parsed = parse_agent_evaluation(full_response)

        visible_content = parsed["visible_content"]
        planning_content = parsed["planning_content"]

        if planning_content:
            self._update_planning_widget(planning_content)

        if visible_content.strip():
            if (
                self.chat_window.expecting_response
                and self.chat_window.current_response_bubble is None
            ):
                self.chat_window.current_response_bubble = (
                    self.chat_window.chat_components.append_message("", False)
                )

        if self.chat_window.current_response_bubble:
            self.chat_window.current_response_bubble.update_streaming_text(
                visible_content
            )

    def handle_user_message_created(self, data):
        """Handle user message creation."""
        if self.chat_window.current_user_bubble:
            self.chat_window.current_user_bubble.message_index = (
                self.chat_window.message_handler.current_user_input_idx
            )
            self.chat_window.current_user_bubble = None
            self.chat_window.chat_scroll.verticalScrollBar().setValue(
                self.chat_window.chat_scroll.verticalScrollBar().maximum()
            )

    def handle_response_completed(self, data):
        """Handle response completion."""
        parsed = parse_agent_evaluation(data)
        visible_content = parsed["visible_content"]
        planning_content = parsed["planning_content"]

        if planning_content:
            self._update_planning_widget(planning_content)

        if visible_content.strip() and self.chat_window.current_response_bubble is None:
            self.chat_window.current_response_bubble = (
                self.chat_window.chat_components.append_message("", False)
            )

        if self.chat_window.current_response_bubble:
            self.chat_window.current_response_bubble.raw_text_buffer = visible_content
            self.chat_window.current_response_bubble.raw_text = visible_content
            self.chat_window.current_response_bubble._finalize_streaming()
            self.chat_window.current_response_bubble.message_index = (
                len(self.chat_window.message_handler.streamline_messages) - 1
            )
        self.chat_window.expecting_response = False
        QApplication.processEvents()
        self.chat_window.chat_scroll.repaint()

    def handle_thinking_started(self, data):
        """Handle thinking process started."""
        agent_name = data
        self.chat_window.chat_components.add_system_message(
            f"💭 {agent_name.upper()}'s thinking process started"
        )

        # Create a new thinking bubble
        self.chat_window.current_thinking_bubble = (
            self.chat_window.chat_components.append_thinking_message("", agent_name)
        )
        self.chat_window.thinking_content = ""  # Initialize thinking content

    def handle_thinking_chunk(self, chunk):
        """Handle a chunk of the thinking process."""
        self.chat_window.thinking_content += chunk
        # Use smooth streaming for thinking chunks too
        if self.chat_window.current_thinking_bubble:
            self.chat_window.current_thinking_bubble.update_streaming_text(
                self.chat_window.thinking_content
            )

    def handle_thinking_completed(self):
        """Handle thinking process completion."""
        self.chat_window.display_status_message("Thinking completed.")
        # Finalize thinking stream if active
        if self.chat_window.current_thinking_bubble:
            self.chat_window.current_thinking_bubble.raw_text_buffer = (
                self.chat_window.thinking_content
            )
            self.chat_window.current_thinking_bubble._finalize_streaming()
        # Reset thinking bubble reference
        self.chat_window.current_thinking_bubble = None
        self.chat_window.thinking_content = ""

    def handle_stream_cancel_requested(self):
        self.chat_window.display_status_message("Stopping current stream...")

    def handle_stream_canceled(self, data):
        if self.chat_window.current_response_bubble:
            self.chat_window.current_response_bubble.stop_streaming()
        if self.chat_window.current_thinking_bubble:
            self.chat_window.current_thinking_bubble.stop_streaming()
            self.chat_window.current_thinking_bubble = None
        self.chat_window.current_planning_widget = None
        self.chat_window.current_planning_content = ""
        self.chat_window.display_status_message("Stream canceled.")
        self.chat_window.ui_state_manager.set_input_controls_enabled(True)
        QApplication.processEvents()

    def handle_stream_open_timeout(self, data):
        if self.chat_window.current_response_bubble:
            self.chat_window.current_response_bubble.stop_streaming()
        if self.chat_window.current_thinking_bubble:
            self.chat_window.current_thinking_bubble.stop_streaming()
            self.chat_window.current_thinking_bubble = None
        self.chat_window.current_planning_widget = None
        self.chat_window.current_planning_content = ""
        self.chat_window.display_status_message("Stream timed out before first chunk.")
        self.chat_window.ui_state_manager.set_input_controls_enabled(True)
        QApplication.processEvents()

    def handle_user_context_request(self):
        """Handle user context request."""
        self.chat_window.chat_components.add_system_message("Refreshing my memory...")
