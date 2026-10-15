from types import SimpleNamespace
from unittest.mock import Mock, patch

from AgentCrew.modules.console.console_ui import ConsoleUI
from AgentCrew.modules.gui.components.message_handlers import MessageEventHandler


def test_console_cancel_cleans_live_effects_and_blocks_late_progress():
    ui = ConsoleUI.__new__(ConsoleUI)
    ui._stream_cancel_pending = False
    ui.ui_effects = Mock()
    ui.display_handlers = Mock()

    ui._on_stream_cancel_requested()

    assert ui._stream_cancel_pending is True
    ui.ui_effects.cleanup.assert_called_once_with()
    assert ui.display_handlers.display_message.call_args.args[0].plain == (
        "Stopping current stream..."
    )

    ui._on_delegate_progress("run", "agent", "start", "")
    ui._on_thinking_chunk("late thinking")
    ui._on_response_chunk("late", "late")
    ui._on_tool_result(tool_result=[])
    ui.ui_effects.update_delegate_progress.assert_not_called()
    ui.ui_effects.update_live_display.assert_not_called()
    ui.ui_effects.start_loading_animation.assert_not_called()

    ui._on_stream_canceled()
    assert ui.ui_effects.cleanup.call_count == 2
    assert ui.display_handlers.display_message.call_args.args[0].plain == (
        "Stream canceled."
    )


def test_gui_cancel_stops_bubbles_and_ignores_late_stream_events():
    response = Mock()
    thinking = Mock()
    state = SimpleNamespace(
        current_response_bubble=response,
        current_thinking_bubble=thinking,
        current_user_bubble=None,
        current_file_bubble=None,
    )
    stream = SimpleNamespace(delegated_user_input=None, thinking_content="old")
    window = SimpleNamespace(
        bubble_state=state,
        stream_state=stream,
        chat_components=Mock(),
        display_status_message=Mock(),
        reset_bubble_state=Mock(),
        reset_stream_state=Mock(),
        ui_state_manager=Mock(),
    )
    handler = MessageEventHandler.__new__(MessageEventHandler)
    handler.chat_window = window
    handler._stream_cancel_pending = False

    handler.handle_stream_cancel_requested()

    response.stop_streaming.assert_called_once_with()
    thinking.stop_streaming.assert_called_once_with()
    window.display_status_message.assert_called_with("Stopping current stream...")
    window.reset_stream_state.assert_not_called()
    window.ui_state_manager.set_input_controls_enabled.assert_not_called()

    handler.handle_thinking_started("agent")
    handler.handle_thinking_chunk("late")
    handler.handle_response_chunk(("late", "late"))
    with patch(
        "AgentCrew.modules.gui.components.message_handlers.QApplication.processEvents"
    ):
        handler.handle_response_completed("late")
    assert stream.thinking_content == "old"
    window.chat_components.append_message.assert_not_called()
    response.update_streaming_text.assert_not_called()
    thinking.update_streaming_text.assert_not_called()

    with patch(
        "AgentCrew.modules.gui.components.message_handlers.QApplication.processEvents"
    ):
        handler.handle_stream_canceled(None)
    window.reset_stream_state.assert_called_once_with(delegated_user_input=None)
    window.ui_state_manager.set_input_controls_enabled.assert_called_once_with(True)
    window.display_status_message.assert_called_with("Stream canceled.")

    handler.handle_user_message_created(None)
    assert handler._stream_cancel_pending is False
    handler.handle_thinking_chunk("next turn")
    thinking.update_streaming_text.assert_called_once_with("oldnext turn")
