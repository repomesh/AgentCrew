from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console, Group
from rich.text import Text

from AgentCrew.modules.agents.agent_runner import run_agent_loop
from AgentCrew.modules.console.console_ui import ConsoleUI
from AgentCrew.modules.console.constants import RICH_STYLE_GRAY, RICH_STYLE_WHITE
from AgentCrew.modules.console.ui_effects import UIEffects
from AgentCrew.modules.llm.token_usage import TokenUsage


class FakeStream:
    def __init__(self, events, tools, callback):
        self.events = events
        self.tools = tools
        self.callback = callback

    async def __aiter__(self):
        for event in self.events:
            yield event
        self.callback(self.tools, TokenUsage())

    async def aclose(self):
        pass

    def format_assistant_message(self, response, **kwargs):
        return {"role": "assistant", "content": response}


class FakeAgent:
    def __init__(self):
        self.turn = 0

    def process_messages(self, history, callback):
        self.turn += 1
        if self.turn == 1:
            return FakeStream(
                [("", "", ("thinking ", None)), ("", "", None)], [], callback
            )
        if self.turn == 2:
            return FakeStream(
                [("using tool", "using tool", None)], [{"name": "read_file"}], callback
            )
        return FakeStream([("done", "done", None)], [], callback)

    def extract_last_user_message_for_memory(self, history):
        return ""

    def store_memory_if_available(self, *args):
        pass

    def validate_tool_use(self, tool):
        return None

    async def execute_tool_call(self, tool):
        return "ok"

    def format_message(self, *args):
        return {"role": "tool", "content": "ok"}


@pytest.mark.asyncio
async def test_runner_progress_survives_empty_response_and_tool_recursion():
    events = []

    async def progress(kind, text):
        events.append((kind, text))

    agent = FakeAgent()
    response, _ = await run_agent_loop(agent, [], progress_callback=progress)
    assert response == "done"
    assert events == [
        ("thinking", "thinking "),
        ("response", "using tool"),
        ("tool", "read_file"),
        ("response", "done"),
    ]
    assert agent.turn == 3


def test_delegate_preview_shows_latest_two_wrapped_rows_and_parallel_isolation():
    console = Console(width=40, force_terminal=False, file=StringIO())
    effects = UIEffects(SimpleNamespace(console=console, message_handler=None))
    effects.update_delegate_progress("one", "ResearchAgent", "start", "")
    effects.update_delegate_progress("two", "WriterAgent", "start", "")
    effects.update_delegate_progress("one", "ResearchAgent", "thinking", "a" * 65)
    effects.update_delegate_progress("two", "WriterAgent", "response", "b" * 65)

    heading, *rows = effects._render_delegate_progress().renderables
    assert heading.plain == "📋 Delegating → #1 ResearchAgent:"
    assert [row.plain.strip() for row in rows[:2]] == ["a" * 24, "a" * 17]
    assert all(row.spans[-1].style == RICH_STYLE_GRAY for row in rows[:2])
    assert rows[2].plain.strip() == "#2 WriterAgent:"
    assert [row.plain.strip() for row in rows[3:]] == ["b" * 24, "b" * 17]
    assert all(row.spans[-1].style == RICH_STYLE_WHITE for row in rows[3:])

    effects.update_delegate_progress("one", "ResearchAgent", "tool", "read_file")
    renderable = effects._render_delegate_progress()
    assert isinstance(renderable, Group)
    assert all(isinstance(row, Text) for row in renderable.renderables)
    first, thinking, tool, second, *responses = renderable.renderables
    assert first.plain.count("Delegating →") == 1
    assert thinking.plain.strip() == "a" * 17
    assert thinking.spans[-1].style == RICH_STYLE_GRAY
    assert tool.plain.strip() == "Tool: read_file"
    assert tool.spans[-1].style == RICH_STYLE_GRAY
    assert second.plain.strip() == "#2 WriterAgent:"
    assert [row.plain.strip() for row in responses] == ["b" * 24, "b" * 17]
    assert all(row.spans[-1].style == RICH_STYLE_WHITE for row in responses)
    assert all("read_file" not in row.plain for row in (second, *responses))
    rendered = StringIO()
    Console(width=40, force_terminal=False, file=rendered).print(renderable)
    assert rendered.getvalue().count("Delegating →") == 1
    assert rendered.getvalue().splitlines()[0].startswith("📋 Delegating → #1")
    assert all(
        line.startswith(" " * Text("📋 Delegating → ").cell_len)
        for line in rendered.getvalue().splitlines()[1:]
    )

    effects.update_delegate_progress("one", "ResearchAgent", "finish", "")
    assert set(effects._delegate_agents) == {"two"}
    remaining = effects._render_delegate_progress().renderables
    assert [row.plain.strip() for row in remaining] == [
        "📋 Delegating → #1 WriterAgent:",
        "b" * 24,
        "b" * 17,
    ]
    effects.update_delegate_progress("two", "WriterAgent", "finish", "")
    assert not effects._delegate_agents
    assert effects._delegate_live is None


def test_delegate_start_stops_loading_without_static_output():
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    stopped = []
    ui = SimpleNamespace(
        console=console,
        ui_effects=SimpleNamespace(stop_loading_animation=lambda: stopped.append(True)),
    )
    ConsoleUI._on_delegate_started(
        ui,
        tool_use={
            "input": {"target_agent": "ResearchAgent", "task_description": "task"}
        },
    )
    assert stopped == [True]
    assert output.getvalue() == ""


def test_large_delegate_chunk_preserves_latest_text_and_cleanup():
    console = Console(width=30, force_terminal=False)
    effects = UIEffects(SimpleNamespace(console=console, message_handler=None))
    effects.update_delegate_progress("run", "Agent", "start", "")
    effects.update_delegate_progress("run", "Agent", "response", "a" * 9000 + "tail")
    assert "tail" in effects._delegate_agents["run"][1][-1][1]
    effects.cleanup()
    assert not effects._delegate_agents
    assert effects._delegate_live is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_delegate_handler_finishes_run_on_success_and_error(monkeypatch, fails):
    from importlib import import_module

    from AgentCrew.modules.agents.tools.delegate import get_delegate_tool_handler
    from AgentCrew.modules.events import AppEvents, EventBus

    module = import_module("AgentCrew.modules.agents.tools.delegate")
    local_agent_module = import_module("AgentCrew.modules.agents.local_agent")
    llm_module = import_module("AgentCrew.modules.llm.service_manager")
    registry_module = import_module("AgentCrew.modules.llm.model_registry")
    events = []
    bus = EventBus()
    bus.on(AppEvents.DELEGATE_PROGRESS, lambda **data: events.append(data))
    monkeypatch.setattr(module.EventBus, "get_instance", lambda: bus)

    class Clone:
        def __init__(self, **kwargs):
            pass

        def set_system_prompt(self, prompt):
            pass

        async def activate_async(self):
            if fails:
                raise RuntimeError("activation failed")

        async def deactivate_async(self):
            pass

    class FreshLLM:
        model = "model"

        async def close(self):
            pass

    llm = FreshLLM()
    manager = SimpleNamespace(
        agents={"Target": object()},
        current_agent=SimpleNamespace(name="Source"),
        get_local_agent=lambda name: SimpleNamespace(
            name=name,
            description="test",
            llm=SimpleNamespace(model="model", provider_name="provider"),
            services={},
            tools=[],
            temperature=0,
            system_prompt="",
            custom_system_prompt="",
            get_model=lambda: "model",
            record_conversation_usage=lambda usage: None,
        ),
    )
    monkeypatch.setattr(local_agent_module, "LocalAgent", Clone)
    monkeypatch.setattr(
        llm_module.ServiceManager,
        "get_instance",
        lambda: SimpleNamespace(initialize_standalone_service=lambda name: llm),
    )
    monkeypatch.setattr(
        registry_module.ModelRegistry,
        "get_instance",
        lambda: SimpleNamespace(get_model=lambda name: None),
    )

    async def fake_run_agent_loop(**kwargs):
        await kwargs["progress_callback"]("response", "answer")
        return "answer", TokenUsage()

    monkeypatch.setattr(module, "run_agent_loop", fake_run_agent_loop)
    handler = get_delegate_tool_handler(manager)
    if fails:
        with pytest.raises(RuntimeError, match="activation failed"):
            await handler(target_agent="Target", task_description="test")
    else:
        assert "answer" in await handler(target_agent="Target", task_description="test")
    assert [e["kind"] for e in events] == (
        ["start", "finish"] if fails else ["start", "response", "finish"]
    )
    assert len({e["run_id"] for e in events}) == 1
