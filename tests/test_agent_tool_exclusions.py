from types import SimpleNamespace

import pytest

from AgentCrew.modules.agents.local_agent import LocalAgent
from AgentCrew.modules.agents.manager import AgentMode
from AgentCrew.modules.agents.tools.delegate import get_delegate_tool_definition


def make_agent(mode, excluded_tools=None, is_remoting_mode=False):
    manager = SimpleNamespace(
        agent_mode=mode,
        agents={"source": object(), "target": object()},
        one_turn_process=True,
        get_agents_list_prompt=lambda: "agents-list-marker",
        get_transfer_system_prompt=lambda: "transfer-prompt-marker",
        get_delegate_system_prompt=lambda: "delegate-prompt-marker",
    )
    agent = LocalAgent(
        "source",
        "test agent",
        None,
        {"agent_manager": manager},
        [],
        is_remoting_mode=is_remoting_mode,
        excluded_tools=excluded_tools,
    )
    agent.register_tools()
    return agent


@pytest.mark.parametrize("mode", [AgentMode.TRANSFER, AgentMode.DELEGATE])
def test_excluded_collaboration_tools_have_no_definitions_or_prompts(mode):
    exclusions = ["delegate", "transfer"]
    agent = make_agent(mode, exclusions)
    exclusions.append("ask")

    assert agent.excluded_tools == frozenset({"delegate", "transfer"})
    assert not {"delegate", "transfer"} & agent.tool_definitions.keys()
    assert "agents-list-marker" not in "\n".join(agent.tool_prompts)
    assert "transfer-prompt-marker" not in "\n".join(agent.tool_prompts)
    assert "delegate-prompt-marker" not in "\n".join(agent.tool_prompts)


def test_excluded_transfer_does_not_fall_back_to_delegate():
    agent = make_agent(AgentMode.TRANSFER, {"transfer"})

    assert "transfer" not in agent.tool_definitions
    assert "delegate" not in agent.tool_definitions
    assert agent.tool_prompts == []
    assert agent._colaboration_mode == AgentMode.NONE


@pytest.mark.parametrize(
    ("mode", "expected", "prompt"),
    [
        (AgentMode.TRANSFER, "transfer", "transfer-prompt-marker"),
        (AgentMode.DELEGATE, "delegate", "delegate-prompt-marker"),
    ],
)
def test_non_excluded_collaboration_registration_unchanged(mode, expected, prompt):
    agent = make_agent(mode)

    assert expected in agent.tool_definitions
    assert "agents-list-marker" in agent.tool_prompts
    assert prompt in agent.tool_prompts


def test_transfer_mode_remoting_still_registers_delegate():
    agent = make_agent(AgentMode.TRANSFER, is_remoting_mode=True)

    assert "delegate" in agent.tool_definitions
    assert "transfer" not in agent.tool_definitions
    assert "delegate-prompt-marker" in agent.tool_prompts


def test_register_tool_guard_rejects_excluded_name():
    agent = make_agent(AgentMode.NONE, {"delegate"})

    agent.register_tool(get_delegate_tool_definition, lambda: None)

    assert "delegate" not in agent.tool_definitions
