from __future__ import annotations

from types import SimpleNamespace

import pytest
from prompt_toolkit.keys import Keys
from rich.console import Console

from AgentCrew.modules.console.visual_mode.viewer_input_handler import (
    VisualModeInputHandler,
)
from AgentCrew.modules.console.visual_mode.viewer_ui import VisualModeUI


def _viewer() -> tuple[VisualModeUI, VisualModeInputHandler, list[str]]:
    ui = VisualModeUI(Console(width=80, height=30))
    ui.set_messages([{"role": "user", "content": "first line\n\nsecond line\nlast"}])
    ui._cursor_line = 3
    ui._cursor_col = 4
    copied: list[str] = []
    handler = VisualModeInputHandler(ui, on_copy=copied.append)
    return ui, handler, copied


def _press(handler: VisualModeInputHandler, key: str | Keys) -> None:
    binding = next(
        binding
        for binding in handler._create_key_bindings().bindings
        if binding.keys == (key,)
    )
    handler._ui.render = lambda: None
    binding.handler(SimpleNamespace(data=key if isinstance(key, str) else ""))


def test_v_selects_entire_line_and_y_copies_it() -> None:
    ui, handler, copied = _viewer()

    _press(handler, "V")

    assert ui._linewise_selection
    assert ui.get_selected_text() == "second line"
    assert ui._is_position_selected(3, 0)
    assert ui._is_position_selected(3, len("second line") - 1)
    assert not ui._is_position_selected(2, 0)
    _press(handler, "y")
    assert copied == ["second line"]
    assert not ui._visual_mode
    assert not ui._linewise_selection


@pytest.mark.parametrize("down,up", [("j", "k"), (Keys.Down, Keys.Up)])
def test_v_extends_and_reverses_across_whole_lines(
    down: str | Keys,
    up: str | Keys,
) -> None:
    ui, handler, _ = _viewer()

    _press(handler, "V")
    _press(handler, down)
    assert ui.get_selected_text() == "second line\nlast"
    assert ui._is_position_selected(4, 0)
    _press(handler, up)
    assert ui.get_selected_text() == "second line"
    _press(handler, up)
    assert ui.get_selected_text() == "\nsecond line"
    assert ui._is_position_selected(2, 0)
    _press(handler, down)
    assert ui.get_selected_text() == "second line"


def test_v_switches_between_linewise_and_characterwise_without_losing_anchor() -> None:
    ui, handler, _ = _viewer()

    _press(handler, "v")
    _press(handler, "j")
    assert ui.get_selected_text() == "nd line\nlast"
    _press(handler, "V")
    assert ui.get_selected_text() == "second line\nlast"
    _press(handler, "v")
    assert ui.get_selected_text() == "nd line\nlast"
    _press(handler, "v")
    assert not ui._visual_mode
    _press(handler, "V")
    _press(handler, "V")
    assert not ui._visual_mode


def test_escape_clears_linewise_selection_and_messages_reset_mode() -> None:
    ui, handler, _ = _viewer()
    _press(handler, "V")
    _press(handler, Keys.Escape)

    assert not ui._visual_mode
    assert not ui._linewise_selection
    assert ui._selection_start is None
    ui.toggle_visual_mode(linewise=True)
    ui.set_messages([{"role": "user", "content": "new text"}])
    assert not ui._linewise_selection
    assert not ui._visual_mode


def test_v_is_search_text_in_search_mode() -> None:
    ui, handler, _ = _viewer()
    ui.start_search_mode()

    _press(handler, "V")

    assert ui._search_query == "V"
    assert not ui._visual_mode


def test_linewise_highlight_covers_line_from_first_to_last_character() -> None:
    ui, handler, _ = _viewer()
    _press(handler, "V")
    content = ui._create_content_panel().renderable

    assert any(
        content.plain[start:end] == "seco" and style == "on blue"
        for start, end, style in (
            (span.start, span.end, span.style) for span in content.spans
        )
    )
