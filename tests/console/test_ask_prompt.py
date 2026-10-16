from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from prompt_toolkit.formatted_text import HTML

from AgentCrew.modules.console.confirmation_handler import ConfirmationHandler
from AgentCrew.modules.console.input_handler import InputHandler

PROMPT_TEXT = '2>&1 <placeholder> "double" \'single\'\n```sh\necho "hello"\n```'


def rendered(html):
    return "".join(text for _, text in html.__pt_formatted_text__())


def test_choice_prompt_escapes_dynamic_text_without_changing_style_or_spacing():
    handler = InputHandler.__new__(InputHandler)
    app = Mock(is_running=False, is_done=True)
    app.run.return_value = "yes"
    with patch(
        "prompt_toolkit.shortcuts.choice_input.ChoiceInput"
    ) as choice_input:
        choice_input.return_value._create_application.return_value = app
        assert handler.get_choice_input(PROMPT_TEXT, ["yes"]) == "yes"

    message = choice_input.call_args.kwargs["message"]
    assert isinstance(message, HTML)
    assert rendered(message) == PROMPT_TEXT + " "
    assert message.__pt_formatted_text__() == [("class:ansiyellow", PROMPT_TEXT), ("", " ")]


def test_custom_prompt_escapes_dynamic_text_without_changing_style_or_spacing():
    handler = InputHandler.__new__(InputHandler)
    handler.swap_enter = False
    handler.message_handler = Mock()
    with patch("prompt_toolkit.prompt", return_value="answer") as prompt:
        assert handler.get_prompt_input(PROMPT_TEXT) == "answer"

    message = prompt.call_args.args[0]
    assert isinstance(message, HTML)
    assert rendered(message) == PROMPT_TEXT + " "
    assert message.__pt_formatted_text__() == [("class:ansiblue", PROMPT_TEXT), ("", " ")]


def make_ask(questions=None):
    ui = SimpleNamespace(console=Mock(width=80), input_handler=Mock(), start_loading_animation=Mock())
    ui.input_handler._stop_input_thread.return_value = True
    ui.input_handler.prompt_submit_hint = "submit hint"
    return ConfirmationHandler(ui), Mock(), {
        "name": "ask",
        "input": {"questions": questions or [{"question": "First?", "guided_answers": ["A"]}]},
        "confirmation_id": 17,
    }


def assert_cleanup(ask, resolved, payload, restart=True):
    resolved.resolve_tool_confirmation.assert_called_once_with(17, payload)
    if restart:
        ask._ui.start_loading_animation.assert_called_once_with()
        ask.input_handler._start_input_thread.assert_called_once_with()
    else:
        ask._ui.start_loading_animation.assert_not_called()
        ask.input_handler._start_input_thread.assert_not_called()


def test_ask_answers_multiple_questions_and_resolves_once():
    ask, resolved, tool = make_ask([
        {"question": "First?", "guided_answers": ["A"]},
        {"question": "Second?", "guided_answers": ["B"]},
    ])
    ask.input_handler.get_choice_input.side_effect = ["A", "← Back", "Next →", "Custom your answer"]
    ask.input_handler.get_prompt_input.return_value = "custom"

    ask.display_tool_confirmation_request(tool, resolved)

    assert_cleanup(ask, resolved, {"action": "answer", "answer": "Answers:\n- First?: A\n- Second?: custom"})
    assert ask.input_handler.get_choice_input.call_count == 4
    ask.input_handler.get_prompt_input.assert_called_once()


def test_ask_cancel_resolves_once_and_restores_input():
    ask, resolved, tool = make_ask()
    ask.input_handler.get_choice_input.return_value = ""

    ask.display_tool_confirmation_request(tool, resolved)

    assert_cleanup(ask, resolved, {"action": "answer", "answer": "Cancelled by user"})


def test_empty_custom_answer_cancels_entire_ask_without_advancing():
    ask, resolved, tool = make_ask([
        {"question": "First?", "guided_answers": ["A"]},
        {"question": "Second?", "guided_answers": ["B"]},
        {"question": "Third?", "guided_answers": ["C"]},
    ])
    ask.input_handler.get_choice_input.side_effect = ["A", "Custom your answer"]
    ask.input_handler.get_prompt_input.return_value = ""

    ask.display_tool_confirmation_request(tool, resolved)

    assert_cleanup(ask, resolved, {"action": "answer", "answer": "Cancelled by user"})
    assert ask.input_handler.get_choice_input.call_count == 2
    ask.input_handler.get_prompt_input.assert_called_once()


@pytest.mark.parametrize("failure", ["stop", "display", "choice", "custom"])
def test_ask_ui_failures_resolve_cancellation_and_restore_safe_lifecycle(failure):
    ask, resolved, tool = make_ask()
    if failure == "stop":
        ask.input_handler._stop_input_thread.side_effect = RuntimeError("stop failed")
    elif failure == "display":
        ask.console.print.side_effect = RuntimeError("display failed")
    elif failure == "choice":
        ask.input_handler.get_choice_input.side_effect = RuntimeError("choice failed")
    else:
        ask.input_handler.get_choice_input.return_value = "Custom your answer"
        ask.input_handler.get_prompt_input.side_effect = RuntimeError("prompt failed")

    ask.display_tool_confirmation_request(tool, resolved)

    assert_cleanup(
        ask, resolved, {"action": "answer", "answer": "Cancelled by user"},
        restart=failure != "stop",
    )
    if failure == "stop":
        ask.input_handler.get_choice_input.assert_not_called()
        ask.input_handler.get_prompt_input.assert_not_called()


def test_failed_stop_never_opens_competing_prompt_or_restarts_thread():
    ask, resolved, tool = make_ask()
    ask.input_handler._stop_input_thread.return_value = False

    ask.display_tool_confirmation_request(tool, resolved)

    assert_cleanup(
        ask, resolved, {"action": "answer", "answer": "Cancelled by user"}, restart=False
    )
    ask.input_handler.get_choice_input.assert_not_called()
    ask.input_handler.get_prompt_input.assert_not_called()
