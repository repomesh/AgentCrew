import tomllib
from types import SimpleNamespace

import pytest
import tomli_w

from AgentCrew.modules.agents import cursor_agent
from AgentCrew.modules.config.agents_config import AgentsFileConfig, CursorAgentConfig


@pytest.fixture
def sdk(monkeypatch):
    created = []

    class FakeAgent:
        agent_id = "fake-id"

        def close(self):
            pass

        def send(self, _prompt):
            return SimpleNamespace(
                stream=lambda: iter(()),
                wait=lambda: SimpleNamespace(result="done"),
                cancel=lambda: None,
            )

    def create(**kwargs):
        created.append(kwargs)
        return FakeAgent()

    fake_sdk = SimpleNamespace(
        Agent=SimpleNamespace(create=create),
        ModelSelection=lambda **kwargs: SimpleNamespace(**kwargs),
        ModelParameterValue=lambda **kwargs: SimpleNamespace(**kwargs),
        LocalAgentOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        CustomTool=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(cursor_agent, "_load_cursor_sdk", lambda: fake_sdk)
    return created


def test_legacy_config_defaults_and_round_trips():
    raw = tomllib.loads('[[cursor_agents]]\nname = "Legacy"\nmodel_id = "auto-smart"\n')
    config = AgentsFileConfig.from_dict(raw)
    assert config.cursor_agents[0].optimize_for == "balanced"
    assert config.to_dict()["cursor_agents"][0]["optimize_for"] == "balanced"
    persisted = tomllib.loads(tomli_w.dumps(config.to_dict()))
    assert persisted["cursor_agents"][0]["optimize_for"] == "balanced"
    assert (
        cursor_agent.create_cursor_agent_from_definition(
            raw["cursor_agents"][0]
        ).optimize_for
        == "balanced"
    )


@pytest.mark.parametrize("value", ["cost", "balanced", "intelligence"])
def test_supported_values_round_trip(value):
    config = CursorAgentConfig.from_dict({"name": "Cursor", "optimize_for": value})
    assert config.to_dict()["optimize_for"] == value


@pytest.mark.parametrize("value", ["default", "Balance", "Balanced", "", None])
def test_unsupported_values_fail_closed(value):
    with pytest.raises(ValueError, match="Invalid Cursor optimize_for"):
        CursorAgentConfig.from_dict({"name": "Cursor", "optimize_for": value})
    with pytest.raises(ValueError, match="Invalid Cursor optimize_for"):
        cursor_agent.create_cursor_agent_from_definition(
            {"name": "Cursor", "optimize_for": value}
        )


@pytest.mark.parametrize("value", ["cost", "balanced", "intelligence"])
def test_auto_smart_always_passes_explicit_selection(tmp_path, sdk, value):
    agent = cursor_agent.CursorAgent(
        "Cursor", "", "auto-smart", cwd=str(tmp_path), optimize_for=value
    )
    try:
        agent._ensure_sdk_agent()
        model = sdk[-1]["model"]
        assert model.id == "auto-smart"
        assert [(param.id, param.value) for param in model.params] == [
            ("optimize_for", value)
        ]
    finally:
        agent.dispose_sdk_agent()


@pytest.mark.parametrize("model_id", ["auto", "composer-2.5"])
def test_other_models_still_pass_string_model(tmp_path, sdk, model_id):
    agent = cursor_agent.CursorAgent("Cursor", "", model_id, cwd=str(tmp_path))
    try:
        agent._ensure_sdk_agent()
        assert sdk[-1]["model"] == model_id
    finally:
        agent.dispose_sdk_agent()


def test_config_reload_rebuilds_auto_smart_sdk_agent(tmp_path, sdk):
    agent = cursor_agent.CursorAgent("Cursor", "", "auto-smart", cwd=str(tmp_path))
    try:
        agent._ensure_sdk_agent()
        agent.apply_config(
            {"model_id": "auto-smart", "cwd": str(tmp_path), "optimize_for": "cost"}
        )
        agent._ensure_sdk_agent()
        assert len(sdk) == 2
        assert sdk[-1]["model"].params[0].value == "cost"
        agent.apply_config({"model_id": "auto-smart", "cwd": str(tmp_path)})
        agent._ensure_sdk_agent()
        assert len(sdk) == 3
        assert sdk[-1]["model"].params[0].value == "balanced"
    finally:
        agent.dispose_sdk_agent()


def test_invalid_reload_preserves_current_selection(tmp_path):
    agent = cursor_agent.CursorAgent("Cursor", "", "auto-smart", cwd=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid Cursor optimize_for"):
        agent.apply_config({"model_id": "auto-smart", "optimize_for": "default"})
    assert agent.optimize_for == "balanced"
    assert agent.model_id == "auto-smart"


def test_other_model_ignores_optimize_for_changes(tmp_path, sdk):
    agent = cursor_agent.CursorAgent("Cursor", "", "auto", cwd=str(tmp_path))
    try:
        original = agent._ensure_sdk_agent()
        agent.apply_config(
            {"model_id": "auto", "cwd": str(tmp_path), "optimize_for": "cost"}
        )
        assert agent._ensure_sdk_agent() is original
        assert len(sdk) == 1
        assert sdk[0]["model"] == "auto"
    finally:
        agent.dispose_sdk_agent()


def test_one_shot_delegation_uses_same_selection(tmp_path, sdk):
    agent = cursor_agent.CursorAgent(
        "Cursor", "", "auto-smart", cwd=str(tmp_path), optimize_for="intelligence"
    )
    assert "done" in cursor_agent._delegate_to_cursor("Other", agent, "task", "context")
    model = sdk[-1]["model"]
    assert model.id == "auto-smart"
    assert [(param.id, param.value) for param in model.params] == [
        ("optimize_for", "intelligence")
    ]


@pytest.mark.parametrize("model_id", ["auto", "composer-2.5"])
def test_one_shot_other_models_keep_string_selection(tmp_path, sdk, model_id):
    agent = cursor_agent.CursorAgent("Cursor", "", model_id, cwd=str(tmp_path))
    assert "done" in cursor_agent._delegate_to_cursor("Other", agent, "task", "context")
    assert sdk[-1]["model"] == model_id
