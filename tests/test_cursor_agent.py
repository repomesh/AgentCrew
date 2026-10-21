"""Focused CursorAgent tests. The Cursor SDK is faked; no API key is used."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from AgentCrew.modules.agents.cursor_agent import (
    CursorAgent,
    _claim_cwd,
    _cwd_owners,
    _release_cwd,
    upsert_cursor_agent,
)
from AgentCrew.modules.agents.manager import AgentManager
from AgentCrew.modules.config.agents_config import (
    AgentsConfig,
    AgentsFileConfig,
    CursorAgentConfig,
    LocalAgentConfig,
)
from AgentCrew.modules.llm.token_usage import TokenUsage


class _FakeRun:
    def __init__(self, agent, script):
        self.agent = agent
        self.script = script
        self.status = "running"
        self.cancelled = False
        self.started = __import__("threading").Event()
        self.release = __import__("threading").Event()
        self.release.set()

    def stream(self):
        self.started.set()
        self.release.wait(timeout=5)
        tools = self.agent.custom_tools
        for step in self.script:
            if self.cancelled:
                break
            kind = step[0]
            if kind == "text":
                yield SimpleNamespace(type="assistant", text=step[1])
            elif kind == "call":
                name, args = step[1], step[2]
                step_result = tools[name].execute(args, None)
                self.agent.call_results.append(step_result)
                yield SimpleNamespace(type="tool_call", name=name, status="completed")
        self.status = "cancelled" if self.cancelled else "finished"

    def cancel(self):
        self.cancelled = True
        self.status = "cancelled"
        self.release.set()

    def wait(self):
        return SimpleNamespace(status=self.status, result=self.agent.wait_result)


class _FakeSdkAgent:
    def __init__(self, script):
        self.agent_id = "sdk-123"
        self.script = script
        self.custom_tools = {}
        self.create_kwargs = {}
        self.closed = False
        self.send_count = 0
        self.last_prompt = ""
        self.last_run = None
        self.call_results = []
        self.wait_result = ""

    def send(self, prompt, options=None):
        self.send_count += 1
        self.last_prompt = prompt
        self.last_run = _FakeRun(self, self.script)
        return self.last_run

    def close(self):
        self.closed = True


def _install_sdk(monkeypatch, script, wait_result=""):
    created = []

    def create(**kwargs):
        agent = _FakeSdkAgent(script)
        agent.create_kwargs = kwargs
        agent.custom_tools = kwargs["local"].custom_tools
        agent.wait_result = wait_result
        created.append(agent)
        return agent

    sdk = SimpleNamespace(
        Agent=SimpleNamespace(create=create),
        LocalAgentOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        CustomTool=lambda **kwargs: SimpleNamespace(**kwargs),
        AgentBusyError=type("AgentBusyError", (Exception,), {}),
    )
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(
        "AgentCrew.modules.agents.cursor_agent._load_cursor_sdk",
        lambda: sdk,
    )
    return created


@contextmanager
def _isolated_manager():
    manager = AgentManager.get_instance()
    saved_agents = dict(manager.agents)
    saved_current = manager.current_agent
    manager.agents = {}
    manager.current_agent = None
    try:
        yield manager
    finally:
        for agent in list(manager.agents.values()):
            dispose = getattr(agent, "dispose_sdk_agent", None)
            if dispose is not None:
                dispose()
        manager.agents = saved_agents
        manager.current_agent = saved_current


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


async def _collect(agent, messages, callback=None):
    chunks = []
    stream = agent.process_messages(messages, callback=callback)
    async for item in stream:
        chunks.append(item)
    return chunks


def test_config_load_registers_cursor_agent_without_llm(tmp_path, monkeypatch):
    config_path = tmp_path / "agents.toml"
    config_path.write_text(
        """
[[agents]]
name = "Engineer"
description = "local"
system_prompt = "keep"

[[cursor_agents]]
name = "Engineer"
description = "duplicate"
enabled = true

[[cursor_agents]]
name = "CursorCoder"
description = "Edits the repo"
model_id = "composer-2.5"
cwd = "."

[[cursor_agents]]
name = "DisabledCursor"
enabled = false
""",
        encoding="utf-8",
    )
    defs = AgentManager.load_agents_from_config(str(config_path))
    cursor_defs = [item for item in defs if item.get("agent_type") == "cursor"]
    assert [item["name"] for item in cursor_defs] == ["CursorCoder"]
    assert os.path.isabs(cursor_defs[0]["cwd"])

    calls = []
    monkeypatch.setattr(
        AgentManager,
        "resolve_llm_service_from_config",
        staticmethod(lambda cfg: calls.append(cfg)),
    )
    with _isolated_manager() as manager:
        assert upsert_cursor_agent(manager, cursor_defs[0])
        agent = manager.agents["CursorCoder"]
        assert isinstance(agent, CursorAgent)
        assert agent.get_provider() == "cursor"
        assert agent.get_model() == "composer-2.5"
        assert agent._sdk_agent is None
        assert calls == []


def test_config_write_preserves_cursor_agents(tmp_path, monkeypatch):
    config_path = tmp_path / "agents.toml"
    monkeypatch.setenv("SW_AGENTS_CONFIG", str(config_path))
    config = AgentsConfig()
    with patch.object(AgentsConfig, "reload"):
        config.write(
            AgentsFileConfig(
                agents=[
                    LocalAgentConfig(
                        name="Engineer",
                        description="local",
                        system_prompt="keep",
                    )
                ],
                cursor_agents=[
                    CursorAgentConfig(
                        name="CursorCoder",
                        description="Edits the repo",
                        model_id="composer-2.5",
                        cwd=".",
                    )
                ],
            )
        )

    import tomllib

    written = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert written["cursor_agents"][0]["name"] == "CursorCoder"
    assert written["cursor_agents"][0]["model_id"] == "composer-2.5"
    assert "system_prompt" not in written["cursor_agents"][0]
    assert "api_key" not in written["cursor_agents"][0]
    reread = AgentsFileConfig.from_dict(written)
    assert reread.cursor_agents[0].enabled is True
    assert reread.agents[0].system_prompt == "keep"


@pytest.mark.asyncio
async def test_transfer_records_intent_and_emits_one_tool_use(tmp_path, monkeypatch):
    created = _install_sdk(
        monkeypatch,
        [
            ("text", "working"),
            (
                "call",
                "transfer",
                {
                    "target_agent": "Writer",
                    "task_description": "Review the patch",
                    "post_action": "summarize",
                },
            ),
            ("text", "kept editing"),
        ],
    )
    tool_uses = []
    with _isolated_manager() as manager:
        cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        writer = SimpleNamespace(name="Writer")
        manager.agents = {"CursorCoder": cursor, "Writer": writer}
        manager.current_agent = cursor
        chunks = await _collect(
            cursor,
            [
                _user("older request"),
                {"role": "assistant", "content": [{"type": "text", "text": "old"}]},
                _user("latest please"),
            ],
            callback=lambda uses, _usage: tool_uses.append(uses),
        )
        assert manager.current_agent is cursor
        assert cursor._pending_transfer["target_agent"] == "Writer"
        assert len(tool_uses) == 1
        assert len(tool_uses[0]) == 1
        tool_use = tool_uses[0][0]
        assert tool_use["name"] == "transfer"
        assert tool_use["id"].startswith("cursor-transfer-")
        assert tool_use["input"] == {
            "target_agent": "Writer",
            "task_description": "Review the patch",
            "post_action": "summarize",
        }
        assert "latest please" in created[0].last_prompt
        assert "older request" not in created[0].last_prompt
        assert "stop and do not continue editing" in created[0].last_prompt
        assert "tools" not in created[0].create_kwargs
        local = created[0].create_kwargs["local"]
        assert local.setting_sources == ["project"]
        assert set(local.custom_tools) == {"transfer", "delegate"}
        assert created[0].last_run.cancelled
        assert not any(
            isinstance(item, tuple) and item[1] == "kept editing" for item in chunks
        )


def test_second_transfer_in_same_run_is_ignored(tmp_path):
    with _isolated_manager() as manager:
        cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        manager.agents = {
            "CursorCoder": cursor,
            "Writer": SimpleNamespace(name="Writer"),
            "Editor": SimpleNamespace(name="Editor"),
        }
        manager.current_agent = cursor
        first = cursor._execute_transfer(
            {"target_agent": "Writer", "task_description": "first"},
            None,
        )
        second = cursor._execute_transfer(
            {"target_agent": "Editor", "task_description": "second"},
            None,
        )
        assert first == second
        assert cursor._pending_transfer["target_agent"] == "Writer"
        assert manager.current_agent is cursor


@pytest.mark.asyncio
async def test_unknown_transfer_target_emits_no_tool_use(tmp_path, monkeypatch):
    created = _install_sdk(
        monkeypatch,
        [
            (
                "call",
                "transfer",
                {"target_agent": "Missing", "task_description": "nope"},
            ),
            ("text", "still here"),
        ],
    )
    tool_uses = []
    with _isolated_manager() as manager:
        cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        manager.agents = {
            "CursorCoder": cursor,
            "Writer": SimpleNamespace(name="Writer"),
        }
        manager.current_agent = cursor
        chunks = await _collect(
            cursor,
            [_user("go")],
            callback=lambda uses, _usage: tool_uses.append(uses),
        )
        assert tool_uses == []
        assert cursor._pending_transfer is None
        assert "not found" in created[0].call_results[0]
        assert "Writer" in created[0].call_results[0]
        assert any(
            isinstance(item, tuple) and item[1] == "still here" for item in chunks
        )
        assert manager.current_agent is cursor


def test_delegate_to_local_returns_text_without_switching(tmp_path, monkeypatch):
    from AgentCrew.modules.agents.local_agent import LocalAgent

    captured = {}

    async def fake_loop(agent, history, **_kwargs):
        captured["history"] = history
        captured["excluded"] = set(agent.excluded_tools)
        captured["ask"] = agent.ask_responder
        return "finished the summary", TokenUsage()

    fresh = SimpleNamespace(model=None, close_calls=0)

    async def close():
        fresh.close_calls += 1

    fresh.close = close
    monkeypatch.setattr(LocalAgent, "activate", lambda self: True)
    monkeypatch.setattr(LocalAgent, "deactivate", lambda self: None)
    monkeypatch.setattr(
        "AgentCrew.modules.agents.agent_runner.run_agent_loop",
        fake_loop,
    )
    monkeypatch.setattr(
        "AgentCrew.modules.llm.model_registry.ModelRegistry.get_instance",
        staticmethod(lambda: SimpleNamespace(get_model=lambda _model_id: None)),
    )
    monkeypatch.setattr(
        "AgentCrew.modules.llm.service_manager.ServiceManager.get_instance",
        staticmethod(
            lambda: SimpleNamespace(
                initialize_standalone_service=lambda _name: fresh,
            )
        ),
    )

    with _isolated_manager() as manager:
        cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        writer = LocalAgent(
            name="Writer",
            description="writes",
            llm_service=SimpleNamespace(provider_name="openai", model="gpt-test"),
            services={},
            tools=["web_search"],
            temperature=0.2,
        )
        manager.agents = {"CursorCoder": cursor, "Writer": writer}
        manager.current_agent = cursor
        result = cursor._execute_delegate(
            {
                "target_agent": "Writer",
                "task_description": "Summarize the diff",
                "context": "notes",
            }
        )
        assert result.startswith("## Result from Writer:")
        assert "finished the summary" in result
        assert manager.current_agent is cursor
        assert captured["excluded"] == {"delegate", "transfer"}
        assert "Summarize the diff" in captured["history"][0]["content"][0]["text"]
        refusal = asyncio.run(captured["ask"]([{"question": "what?"}]))
        assert "Continue from the given task" in refusal


def test_delegate_to_self_returns_error(tmp_path):
    with _isolated_manager() as manager:
        cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        manager.agents = {"CursorCoder": cursor}
        manager.current_agent = cursor
        result = cursor._execute_delegate(
            {"target_agent": "CursorCoder", "task_description": "again"}
        )
        assert "Cannot delegate to yourself" in result
        assert manager.current_agent is cursor


def test_delegate_to_cursor_refuses_same_cwd(tmp_path):
    with _isolated_manager() as manager:
        parent = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
        other = CursorAgent("CursorReviewer", "reviews", cwd=str(tmp_path))
        assert _claim_cwd(parent.cwd, parent._cwd_owner) is None
        parent._cwd_claimed = True
        manager.agents = {"CursorCoder": parent, "CursorReviewer": other}
        manager.current_agent = parent
        try:
            result = parent._execute_delegate(
                {
                    "target_agent": "CursorReviewer",
                    "task_description": "Check the patch",
                }
            )
        finally:
            _release_cwd(parent.cwd, parent._cwd_owner)
            parent._cwd_claimed = False
        assert "already active" in result
        assert manager.current_agent is parent
        assert (
            parent.cwd not in _cwd_owners
            or _cwd_owners[parent.cwd] != parent._cwd_owner
        )


@pytest.mark.asyncio
async def test_missing_sdk_or_key_yields_stream_error(tmp_path, monkeypatch):
    cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    chunks = await _collect(cursor, [_user("hello")])
    assert chunks
    assert "CURSOR_API_KEY" in chunks[0][0]

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")

    def missing():
        raise ImportError(
            "cursor-sdk is not installed. Install it with: uv sync --extra cursor"
        )

    monkeypatch.setattr(
        "AgentCrew.modules.agents.cursor_agent._load_cursor_sdk",
        missing,
    )
    chunks = await _collect(cursor, [_user("hello")])
    assert "cursor-sdk is not installed" in chunks[0][0]
    assert cursor._sdk_agent is None


def test_new_conversation_disposes_sdk_agent(tmp_path):
    from AgentCrew.modules.chat.message.conversation import ConversationManager

    cursor = CursorAgent("CursorCoder", "edits", cwd=str(tmp_path))
    fake = _FakeSdkAgent([])
    cursor._sdk_agent = fake
    cursor._sdk_agent_id = fake.agent_id
    assert _claim_cwd(cursor.cwd, cursor._cwd_owner) is None
    cursor._cwd_claimed = True

    handler = MagicMock()
    handler.persistent_service.start_conversation.return_value = "conv-new"
    handler.memory_service = None
    handler.agent_manager.agents = {}
    handler.agent = cursor
    ConversationManager(handler).start_new_conversation()

    assert fake.closed
    assert cursor._sdk_agent is None
    assert cursor._sdk_agent_id is None
    assert cursor.cwd not in _cwd_owners
