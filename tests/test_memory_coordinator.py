"""Focused tests for the AgentMemoryCoordinator collaborator.

Covers the conversation-memory parsing/storage behavior extracted from
``LocalAgent``:

- last non-empty user message extraction (string / list / malformed / empty)
- assistant message extraction (last-user boundary, ask-tool pairing,
  malformed JSON arguments, rejected tools, current-response dedup/order)
- memory storage (absent service, wrong type, payload/session_id,
  fail-open logging)
- LocalAgent delegation boundary for the three wrappers
- Cursor opt-in memory storage through completed chat-turn finalization
"""

from __future__ import annotations

from unittest.mock import MagicMock

from AgentCrew.modules.agents.cursor_agent import CursorAgent
from AgentCrew.modules.agents.local_agent import LocalAgent
from AgentCrew.modules.agents.memory_coordinator import AgentMemoryCoordinator
from AgentCrew.modules.chat.message.handler import MessageHandler
from AgentCrew.modules.llm.token_usage import TokenUsage
from AgentCrew.modules.memory.base_service import BaseMemoryService


class _FakeMemory(BaseMemoryService):
    """Concrete in-memory fake satisfying BaseMemoryService's abstract API."""

    def __init__(self):
        self.stored: list[tuple] = []

    def store_conversation(
        self, user_msg, assistant_messages, agent_name, session_id=None
    ):
        self.stored.append((user_msg, assistant_messages, agent_name, session_id))

    def clear_conversation_context(self):
        return None

    def load_conversation_context(self, session_id, agent_name="None"):
        return None

    def retrieve_memory(self, *args, **kwargs):
        return []

    def build_system_prompt(self, *args, **kwargs):
        return ""

    def list_memory_headers(self, *args, **kwargs):
        return []

    def cleanup_old_memories(self, months=1):
        return 0

    def forget_topic(self, *args, **kwargs):
        return {"success": True}

    def forget_ids(self, ids, agent_name="None"):
        return {"success": True}

    def delete_by_conversation_id(self, conversation_id):
        return {"success": True, "count": 0}

    def get_agent_memory_corpus(self, *args, **kwargs):
        return []

    def mark_memories_evolved(self, *args, **kwargs):
        return None


def _make_agent(memory=None, name="spec-agent"):
    return LocalAgent(
        name=name,
        description="desc",
        llm_service=None,
        services={"memory": memory} if memory is not None else {},
        tools=[],
    )


# ---------------------------------------------------------------------------
# user message extraction
# ---------------------------------------------------------------------------


class TestExtractLastUserMessage:
    def test_string_content(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "  hello world  "},
            {"role": "assistant", "content": "hi"},
        ]
        assert agent.extract_last_user_message_for_memory(messages) == "hello world"

    def test_list_string_parts(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": ["  a  ", " ", "b  "]},
            {"role": "assistant", "content": "hi"},
        ]
        assert agent.extract_last_user_message_for_memory(messages) == "a b"

    def test_text_dict_parts(self):
        agent = _make_agent([])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "  first "},
                    {"type": "image", "text": "skip"},
                    {"type": "text", "text": "second"},
                ],
            },
            {"role": "assistant", "content": "hi"},
        ]
        assert agent.extract_last_user_message_for_memory(messages) == "first second"

    def test_malformed_skipped(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": [123, None, {"type": "text", "text": ""}]},
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "hi"},
        ]
        assert agent.extract_last_user_message_for_memory(messages) == ""

    def test_empty_returns_empty_string(self):
        agent = _make_agent([])
        assert agent.extract_last_user_message_for_memory([]) == ""


# ---------------------------------------------------------------------------
# assistant message extraction
# ---------------------------------------------------------------------------


class TestExtractAssistantMessages:
    def test_last_user_boundary(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "before boundary"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "after boundary"},
        ]
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        ) == ["after boundary"]

    def test_assistant_text_blocks_preserve_text_without_other_blocks(self):
        agent = _make_agent([])
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": " first "},
                    {"type": "image", "url": "image.png"},
                    "second",
                    {"type": "text", "text": "  "},
                ],
            }
        ]
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        ) == ["first second"]

    def test_multiple_assistants_preserved_in_order(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "  one  "},
            {"role": "assistant", "content": " two "},
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["one", "two"]

    def test_ask_pairing_with_dict_arguments(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "ask",
                        "arguments": {"question": "What color?"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "tool_name": "ask",
                "content": "blue",
            },
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["[User answered: blue | Question was: What color?]"]

    def test_ask_pairing_with_json_string_arguments(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "name": "ask",
                        "arguments": '{"question": "How many?"}',
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "tool_name": "ask",
                "content": "42",
            },
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["[User answered: 42 | Question was: How many?]"]

    def test_ask_pairing_malformed_json_degrades(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_3", "name": "ask", "arguments": "{not json"}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "tool_name": "ask",
                "content": "ok",
            },
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        # No question available -> falls back to the unpaired formatter.
        assert result == ["[User answered: ok]"]

    def test_ask_answer_without_question(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "tool",
                "tool_call_id": "call_4",
                "tool_name": "ask",
                "content": "sure",
            },
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["[User answered: sure]"]

    def test_rejected_tool_feedback(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "tool",
                "tool_name": "write_file",
                "tool_call_id": "t1",
                "content": "   denied  ",
                "is_rejected": True,
            },
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["[Tool rejected: write_file] denied"]

    def test_rejected_tool_unknown_name(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "denied", "is_rejected": True},
        ]
        result = agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages
        )
        assert result == ["[Tool rejected: unknown] denied"]

    def test_current_response_dedup_and_append(self):
        agent = _make_agent([])
        messages = [{"role": "assistant", "content": " final "}]
        # Dedup: already the final extracted message.
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages, current_response="final"
        ) == ["final"]
        # Append: different non-empty current response.
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages, current_response="other"
        ) == ["final", "other"]
        # Skip whitespace-only current response.
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages, current_response="   "
        ) == ["final"]

    def test_return_never_duplicates_order(self):
        agent = _make_agent([])
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert agent._memory_coordinator.extract_assistant_messages_for_memory(
            messages, current_response="b"
        ) == ["a", "b"]


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


class TestStoreMemory:
    def test_absent_service_is_noop(self):
        agent = _make_agent(None)
        agent.store_memory_if_available("u", [], "")
        # No exception raised, nothing stored.

    def test_wrong_service_type_is_noop(self):
        class NotMemory:
            def store_conversation(self, *a, **k):
                raise AssertionError("should not be called")

        agent = _make_agent(NotMemory())
        agent.store_memory_if_available("u", [], "")
        # Wrong type (not BaseMemoryService) is ignored.

    def test_correct_payload_and_session_id(self):
        memory = _FakeMemory()
        agent = _make_agent(memory)
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "resp"},
        ]
        agent.store_memory_if_available(
            "user-msg", messages, "resp", session_id="sess-1"
        )
        assert len(memory.stored) == 1
        user_msg, assistant_msgs, agent_name, session_id = memory.stored[0]
        assert user_msg == "user-msg"
        assert assistant_msgs == ["resp"]
        assert agent_name == "spec-agent"
        assert session_id == "sess-1"

    def test_service_exception_fail_open(self):
        class FailingMemory(_FakeMemory):
            def store_conversation(self, *a, **k):
                raise RuntimeError("boom")

        agent = _make_agent(FailingMemory())
        # Exception is swallowed (logged) so the turn is not broken.
        agent.store_memory_if_available(
            "u", [{"role": "assistant", "content": "r"}], "r"
        )


def _finalize_chat_turn(agent, store_memory=True):
    handler = MessageHandler.__new__(MessageHandler)
    handler.agent = agent
    handler.current_conversation_id = "conv-1"
    handler.current_user_input = {
        "role": "user",
        "content": [{"type": "text", "text": "Remember the fix"}],
    }
    handler.current_user_input_idx = 0
    handler.last_assisstant_response_idx = 0
    handler.streamline_messages = [
        handler.current_user_input,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The fix is ready"}],
        },
    ]
    handler.conversation_manager = MagicMock()
    handler.persistent_service = MagicMock()
    handler.bus = MagicMock()
    handler._turn_usage_ledger = {}
    handler._turn_usage_committed = {}
    messages = handler._finalize_current_turn(TokenUsage(), store_memory)
    assert messages == handler.streamline_messages
    assert handler.current_user_input is None
    assert handler.current_user_input_idx == -1
    handler.persistent_service.append_conversation_messages.assert_called_once_with(
        "conv-1", messages
    )


class TestChatMemoryFinalization:
    def test_cursor_with_memory_stores_user_and_assistant_under_agent_name(self):
        memory = _FakeMemory()
        cursor = CursorAgent(
            "CursorCoder", "edits", tools=["memory"], services={"memory": memory}
        )
        assert cursor._memory_coordinator.agent is cursor
        _finalize_chat_turn(cursor)
        assert memory.stored == [
            ("Remember the fix", ["The fix is ready"], "CursorCoder", None)
        ]

    def test_cursor_without_opt_in_does_not_store(self):
        memory = _FakeMemory()
        cursor = CursorAgent("CursorCoder", "edits", services={"memory": memory})
        _finalize_chat_turn(cursor)
        assert memory.stored == []

    def test_missing_and_failing_memory_services_do_not_break_turn(self):
        missing = CursorAgent("CursorCoder", "edits", tools=["memory"])
        _finalize_chat_turn(missing)

        class FailingMemory(_FakeMemory):
            def store_conversation(self, *args, **kwargs):
                raise RuntimeError("memory unavailable")

        failing = CursorAgent(
            "CursorCoder",
            "edits",
            tools=["memory"],
            services={"memory": FailingMemory()},
        )
        _finalize_chat_turn(failing)

    def test_local_storage_and_non_capable_agent_remain_unchanged(self):
        memory = _FakeMemory()
        local = _make_agent(memory, name="Engineer")
        _finalize_chat_turn(local)
        assert memory.stored == [
            ("Remember the fix", ["The fix is ready"], "Engineer", None)
        ]
        _finalize_chat_turn(object())

    def test_turn_memory_flag_remains_a_gate(self):
        memory = _FakeMemory()
        cursor = CursorAgent(
            "CursorCoder", "edits", tools=["memory"], services={"memory": memory}
        )
        _finalize_chat_turn(cursor, store_memory=False)
        assert memory.stored == []


# ---------------------------------------------------------------------------
# delegation boundary
# ---------------------------------------------------------------------------


class TestDelegationBoundary:
    def test_collaborator_wired_in_init(self):
        agent = _make_agent(None)
        assert isinstance(agent._memory_coordinator, AgentMemoryCoordinator)
        assert agent._memory_coordinator.agent is agent

    def test_wrappers_route_to_collaborator(self):
        agent = _make_agent(None)
        messages = [{"role": "user", "content": "hello"}]
        assert agent.extract_last_user_message_for_memory(messages) == "hello"
        assert (
            agent._memory_coordinator.extract_last_user_message_for_memory(messages)
            == "hello"
        )

    def test_collaborator_does_not_own_state(self):
        agent = _make_agent(None)
        coord = agent._memory_coordinator
        coord = agent._memory_coordinator
        # The collaborator never stores state-bearing attributes of its own.
        assert not hasattr(coord, "name")
        assert not hasattr(coord, "services")

    def test_removed_assistant_wrapper_absent(self):
        agent = _make_agent(None)
        assert not hasattr(agent, "_extract_assistant_messages_for_memory")
