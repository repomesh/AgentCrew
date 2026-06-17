from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from .client_communication import ClientCommunication
from .message_extraction import (
    extract_assistant_text,
    extract_thinking_text,
    extract_tool_calls,
    message_content_to_text,
)
from .session_state import AcpSessionState

if TYPE_CHECKING:
    from .session_store import AcpSessionStore


class SessionLifecycle:
    def __init__(
        self, session_store: AcpSessionStore, client_comm: ClientCommunication
    ):
        self._session_store = session_store
        self._client_comm = client_comm

    async def _persist_session(self, session_id: str, state: AcpSessionState):
        from dataclasses import asdict

        stored = await self._session_store.save_session(
            session_id=session_id,
            cwd=state.cwd,
            agent_name=state.agent_name,
            history=state.history,
            title=state.title,
            model_id=state.model_id,
            thought_level=state.thought_level,
            token_usage=asdict(state.token_usage),
        )
        state.updated_at = stored.updated_at

    async def _replay_history_to_client(
        self, session_id: str, history: list[dict[str, Any]]
    ):
        if self._client_comm.conn is None:
            return

        known_tool_calls: dict[str, dict[str, Any]] = {}
        for message in history:
            if not isinstance(message, dict):
                logger.warning(f"Skipping malformed ACP replay history item: {message}")
                continue
            role = message.get("role", "")
            if role == "user":
                await self._replay_user_message(session_id, message)
            elif role in ("assistant", "thinking", "consolidated"):
                await self._replay_assistant_message(
                    session_id, message, known_tool_calls
                )
            elif role == "tool":
                await self._replay_tool_message(session_id, message, known_tool_calls)

    async def _replay_user_message(self, session_id: str, message: dict[str, Any]):
        from acp import text_block
        from acp.schema import UserMessageChunk

        content = message_content_to_text(message.get("content", ""))
        if not content.strip() or self._client_comm.conn is None:
            return
        await self._client_comm.conn.session_update(
            session_id,
            UserMessageChunk(
                content=text_block(content),
                session_update="user_message_chunk",
            ),
        )

    async def _replay_assistant_message(
        self,
        session_id: str,
        message: dict[str, Any],
        known_tool_calls: dict[str, dict[str, Any]],
    ):
        if self._client_comm.conn is None:
            return
        thought_text = extract_thinking_text(message)
        if thought_text.strip():
            await self._client_comm.send_thought_chunk(session_id, thought_text)

        assistant_text = extract_assistant_text(message)
        if assistant_text.strip():
            await self._client_comm.send_agent_message(session_id, assistant_text)

        for tool_use in extract_tool_calls(message):
            known_tool_calls[tool_use["id"]] = tool_use
            await self._client_comm.send_tool_started(session_id, tool_use)

    async def _replay_tool_message(
        self,
        session_id: str,
        message: dict[str, Any],
        known_tool_calls: dict[str, dict[str, Any]],
    ):
        tool_call_id = message.get("tool_call_id")
        tool_name = message.get("tool_name")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            logger.warning(f"Skipping ACP replay tool result without id: {message}")
            return
        tool_use = known_tool_calls.get(tool_call_id)
        if tool_use is None:
            if not isinstance(tool_name, str) or not tool_name:
                logger.warning(
                    f"Skipping ACP replay tool result without known tool call: {message}"
                )
                return
            tool_use = {
                "id": tool_call_id,
                "name": tool_name,
                "input": {},
            }
            known_tool_calls[tool_call_id] = tool_use
            await self._client_comm.send_tool_started(session_id, tool_use)
        content = message_content_to_text(message.get("content", ""))
        is_error = bool(message.get("is_error") or message.get("is_rejected"))
        if isinstance(content, str) and content.startswith("ERROR:"):
            is_error = True
        await self._client_comm.send_tool_completed(
            session_id, tool_use, content, is_error
        )
