from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from a2a.types import (
    Task,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from AgentCrew.modules.agents.base import MessageType
from AgentCrew.modules.tools.parallel_executor import (
    execute_tools_in_parallel,
    is_sequential_tool,
)
from .adapters import convert_agent_response_to_a2a_artifact
from .exceptions import TaskCanceledException
from AgentCrew.modules.memory import BaseMemoryService
from AgentCrew.modules.llm.token_usage import TokenUsage


class ToolCallResult(Enum):
    CONTINUE = "continue"
    INPUT_REQUIRED = "input_required"


if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from AgentCrew.modules.agents import LocalAgent
    from .task_store import TaskStore
    from .task_streaming import TaskStreamingManager
    from .task_cancellation import TaskCancellationManager
    from .task_interaction import TaskInteractionHandler


class TaskExecutionEngine:
    def _extract_assistant_messages_for_memory(
        self, task_history: list[dict], current_response: str
    ) -> list[str]:
        assistant_messages: list[str] = []
        last_user_idx = -1
        for index, message in enumerate(task_history):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_idx = index

        for message in task_history[last_user_idx + 1 :]:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    assistant_messages.append(normalized)
        normalized_current_response = current_response.strip()
        if normalized_current_response and (
            not assistant_messages
            or assistant_messages[-1] != normalized_current_response
        ):
            assistant_messages.append(normalized_current_response)
        return assistant_messages

    def _extract_last_user_message_for_memory(self, task_history: list[dict]) -> str:
        for message in reversed(task_history):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", [])
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    return normalized
                continue
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    normalized = part.strip()
                    if normalized:
                        text_parts.append(normalized)
                elif isinstance(part, dict) and part.get("type") == "text":
                    normalized = str(part.get("text", "")).strip()
                    if normalized:
                        text_parts.append(normalized)
            if text_parts:
                return " ".join(text_parts)
        return ""

    def __init__(
        self,
        store: TaskStore,
        streaming: TaskStreamingManager,
        cancellation: TaskCancellationManager,
        interaction: TaskInteractionHandler,
        memory_service: Optional[BaseMemoryService] = None,
    ) -> None:
        self.store = store
        self.streaming = streaming
        self.cancellation = cancellation
        self.interaction = interaction
        self.memory_service = memory_service

    async def run(self, agent: LocalAgent, task: Task) -> None:
        if task.status.state in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
        }:
            logger.warning(
                f"Attempted to process task {task.id} in terminal state {task.status.state}"
            )
            return

        try:
            artifacts: List[Any] = []
            if not await self.store.has_task_history(task.context_id):
                raise ValueError("Task history is not existed")

            task_history = await self.store.get_task_history(task.context_id)
            retried_count = [0]

            (
                current_response,
                token_usage,
            ) = await self._process_task(
                agent, task, task_history, artifacts, retried_count
            )

            if self.cancellation.is_canceled(task.id):
                return

            if task.status.state == TaskState.input_required:
                logger.info(f"Task {task.id} paused for user input")
                return

            await self._finalize_task(
                agent,
                task,
                current_response,
                artifacts,
                task_history,
                input_tokens=token_usage.total_input_tokens,
                output_tokens=token_usage.output_tokens,
            )

        except TaskCanceledException:
            logger.info(f"Task {task.id} canceled during processing")
            if self.streaming.is_streaming_enabled(task.id):
                self.streaming.drain_nowait(task.id)
        except asyncio.CancelledError:
            logger.info(f"Task {task.id} asyncio task cancelled externally")
            raise
        except Exception as e:
            logger.error(str(e))
            task_history = await self.store.get_task_history(task.context_id)
            logger.debug(task_history)
            task.status.state = TaskState.failed
            task.status.timestamp = datetime.now().isoformat()
            await self.store.save_task(task)
            if self.streaming.is_streaming_enabled(task.id):
                await self.streaming.flush_task_events(task.id)
                await self.streaming.record_and_emit_event(
                    task.id,
                    TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    ),
                )
                await self.streaming.signal_end(task.id)
        finally:
            self.cancellation.cleanup(task.id)
            await self.streaming.cleanup(task.id)
            self.interaction.cleanup(task.id)

    async def _process_task(
        self,
        agent: LocalAgent,
        task: Task,
        task_history: List[Dict[str, Any]],
        artifacts: List[Any],
        retried_count: List[int],
        token_usage: Optional[TokenUsage] = None,
    ) -> tuple[str, TokenUsage]:
        if token_usage is None:
            token_usage = TokenUsage()
        try:
            current_response = ""
            response_message = ""
            thinking_content = ""
            thinking_signature = ""
            tool_uses: List[Dict[str, Any]] = []

            def process_result(_tool_uses, _token_usage):
                nonlocal tool_uses, token_usage
                tool_uses = _tool_uses
                if token_usage:
                    token_usage = token_usage.merge(_token_usage)
                else:
                    token_usage = _token_usage

            async for (
                response_message,
                chunk_text,
                thinking_chunk,
            ) in agent.process_messages(task_history, callback=process_result):
                if response_message:
                    current_response = response_message

                task.status.state = TaskState.working
                task.status.timestamp = datetime.now().isoformat()

                if thinking_chunk:
                    think_text_chunk, signature = thinking_chunk
                    if think_text_chunk:
                        thinking_content += think_text_chunk
                    if signature:
                        thinking_signature += signature

                if self.streaming.is_streaming_enabled(task.id):
                    await self._handle_streaming_chunk(
                        task, chunk_text, thinking_chunk, artifacts
                    )

                if self.cancellation.is_canceled(task.id):
                    raise TaskCanceledException(
                        f"Task {task.id} was canceled during streaming"
                    )

            if self.cancellation.is_canceled(task.id):
                raise TaskCanceledException(
                    f"Task {task.id} was canceled after streaming"
                )

            if tool_uses:
                if self.streaming.is_streaming_enabled(task.id):
                    tool_artifact = convert_agent_response_to_a2a_artifact(
                        "",
                        artifact_id=f"artifact_{task.id}_{len(artifacts)}",
                        tool_uses=tool_uses,
                    )
                    await self.streaming.record_and_emit_event(
                        task.id,
                        TaskArtifactUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            artifact=tool_artifact,
                        ),
                    )
                    await asyncio.sleep(0.7)

                thinking_data: Optional[Tuple[str, str]] = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = agent.format_message(
                    MessageType.Thinking, {"thinking": thinking_data}
                )
                if thinking_message:
                    await self._append_history_message(
                        task.context_id, thinking_message, task_history
                    )

                assistant_message = agent.format_message(
                    MessageType.Assistant,
                    {
                        "message": response_message,
                        "tool_uses": [
                            t for t in tool_uses if t.get("name", "") != "transfer"
                        ],
                    },
                )
                if assistant_message:
                    await self._append_history_message(
                        task.context_id, assistant_message, task_history
                    )

                tool_call_result = await self._execute_tool_calls(
                    agent, task, tool_uses, task_history
                )

                if tool_call_result == ToolCallResult.INPUT_REQUIRED:
                    return "", token_usage

                return await self._process_task(
                    agent,
                    task,
                    task_history,
                    artifacts,
                    retried_count,
                    token_usage,
                )

            return current_response, token_usage

        except Exception as e:
            if isinstance(e, TaskCanceledException):
                raise
            from openai import BadRequestError, APIError

            if isinstance(e, BadRequestError):
                if (
                    e.code == "model_max_prompt_tokens_exceeded"
                    or e.message.find("This endpoint's maximum context length is") >= 0
                ) and retried_count[0] < 5:
                    from AgentCrew.modules.agents import LocalAgent as _LocalAgent
                    from AgentCrew.modules.llm.model_registry import ModelRegistry

                    if isinstance(agent, _LocalAgent):
                        max_token = ModelRegistry.get_model_limit(agent.get_model())
                        agent.input_tokens_usage = max_token
                        retried_count[0] += 1
                        return await self._process_task(
                            agent,
                            task,
                            task_history,
                            artifacts,
                            retried_count,
                            token_usage,
                        )
            elif (
                isinstance(e, APIError)
                and str(e) == "InternalServerError"
                and retried_count[0] < 5
            ):
                return await self._process_task(
                    agent,
                    task,
                    task_history,
                    artifacts,
                    retried_count,
                    token_usage,
                )

            raise

    async def _handle_streaming_chunk(
        self,
        task: Task,
        chunk_text: Optional[str],
        thinking_chunk: Any,
        artifacts: List[Any],
    ) -> None:
        if thinking_chunk:
            think_text_chunk, _ = thinking_chunk
            if think_text_chunk:
                thinking_artifact = convert_agent_response_to_a2a_artifact(
                    think_text_chunk,
                    artifact_id=f"thinking_{task.id}_{datetime.now()}",
                )
                await self.streaming.record_and_emit_event(
                    task.id,
                    TaskArtifactUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        artifact=thinking_artifact,
                    ),
                )

        if chunk_text:
            artifact = convert_agent_response_to_a2a_artifact(
                chunk_text,
                artifact_id=f"artifact_{task.id}_{len(artifacts)}",
            )
            await self.streaming.record_and_emit_event(
                task.id,
                TaskArtifactUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    artifact=artifact,
                ),
            )

    async def _execute_tool_calls(
        self,
        agent: LocalAgent,
        task: Task,
        tool_uses: List[Dict[str, Any]],
        task_history: List[Dict[str, Any]],
    ) -> ToolCallResult:
        parallel_buffer: List[Dict[str, Any]] = []

        for i, tool_use in enumerate(tool_uses):
            if self.cancellation.is_canceled(task.id):
                raise TaskCanceledException(
                    f"Task {task.id} was canceled during tool execution"
                )
            tool_name = tool_use.get("name")
            if not tool_name:
                logger.error(f"Malformed tool_use missing name: {tool_use}")
                continue

            if is_sequential_tool(tool_use.get("name", "")):
                if parallel_buffer:
                    await self._flush_parallel(
                        agent, task, parallel_buffer, task_history
                    )
                    parallel_buffer = []
                result = await self._execute_single_tool(
                    agent, task, tool_use, task_history
                )
                if result == ToolCallResult.INPUT_REQUIRED:
                    remaining = tool_uses[i + 1 :]
                    await self._save_pending_tools(task.id, tool_use, remaining)
                    return ToolCallResult.INPUT_REQUIRED
            else:
                parallel_buffer.append(tool_use)

        if parallel_buffer:
            await self._flush_parallel(agent, task, parallel_buffer, task_history)

        return ToolCallResult.CONTINUE

    async def _execute_single_tool(
        self,
        agent: LocalAgent,
        task: Task,
        tool_use: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> ToolCallResult:
        tool_name = tool_use["name"]
        if tool_name == "ask":
            return await self._handle_ask_tool(agent, task, tool_use, task_history)
        else:
            try:
                tool_result = await agent.execute_tool_call(
                    tool_name, tool_use["input"]
                )
                tool_result_message = agent.format_message(
                    MessageType.ToolResult,
                    {"tool_use": tool_use, "tool_result": tool_result},
                )
                if tool_result_message:
                    await self._append_history_message(
                        task.context_id, tool_result_message, task_history
                    )
            except Exception as e:
                error_message = agent.format_message(
                    MessageType.ToolResult,
                    {
                        "tool_use": tool_use,
                        "tool_result": str(e),
                        "is_error": True,
                    },
                )
                if error_message:
                    await self._append_history_message(
                        task.context_id, error_message, task_history
                    )
            return ToolCallResult.CONTINUE

    async def _flush_parallel(
        self,
        agent: LocalAgent,
        task: Task,
        tool_uses: List[Dict[str, Any]],
        task_history: List[Dict[str, Any]],
    ) -> None:
        results = await execute_tools_in_parallel(tool_uses, agent.execute_tool_call)
        for r in results:
            msg = agent.format_message(
                MessageType.ToolResult,
                {
                    "tool_use": r.tool_use,
                    "tool_result": r.result,
                    "is_error": r.is_error,
                },
            )
            if msg:
                await self._append_history_message(task.context_id, msg, task_history)

    async def _handle_ask_tool(
        self,
        agent: LocalAgent,
        task: Task,
        tool_use: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> ToolCallResult:
        question = tool_use["input"].get("question", "")
        guided_answers = tool_use["input"].get("guided_answers", [])

        task.status.state = TaskState.input_required
        task.status.timestamp = datetime.now().isoformat()
        task.status.message = self.interaction.create_ask_message(
            question, guided_answers
        )

        await self.store.save_task(task)
        await self.streaming.flush_task_events(task.id)
        await self.streaming.record_and_emit_event(
            task.id,
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=task.status,
                final=False,
            ),
        )

        await self.streaming.signal_end(task.id)
        return ToolCallResult.INPUT_REQUIRED

    async def _finalize_task(
        self,
        agent: LocalAgent,
        task: Task,
        current_response: str,
        artifacts: List[Any],
        task_history: List[Dict[str, Any]],
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        if current_response.strip():
            assistant_message = agent.format_message(
                MessageType.Assistant, {"message": current_response}
            )
            if assistant_message:
                await self.store.append_task_history_message(
                    task.context_id, assistant_message
                )
            if self.memory_service:
                user_message = self._extract_last_user_message_for_memory(task_history)
                assistant_messages = self._extract_assistant_messages_for_memory(
                    task_history, current_response
                )
                self.memory_service.store_conversation(
                    user_message,
                    assistant_messages,
                    agent.name,
                    session_id=task.context_id,
                )

        final_artifact = convert_agent_response_to_a2a_artifact(
            current_response, artifact_id=f"artifact_{task.id}_final"
        )
        artifacts.append(final_artifact)

        task.status.state = TaskState.completed
        task.status.timestamp = datetime.now().isoformat()
        task.artifacts = artifacts
        await self.store.save_task(task)

        if self.streaming.is_streaming_enabled(task.id):
            await self.streaming.flush_task_events(task.id)
            await self.streaming.record_and_emit_event(
                task.id,
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    status=task.status,
                    final=True,
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                ),
            )
            await self.streaming.signal_end(task.id)

    async def _save_pending_tools(
        self,
        task_id: str,
        ask_tool_use: Dict[str, Any],
        remaining_tools: List[Dict[str, Any]],
    ) -> None:
        await self.store.save_pending_tools(task_id, ask_tool_use, remaining_tools)

    async def _append_history_message(
        self,
        context_id: str,
        message: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> None:
        await self.store.append_task_history_message(context_id, message)
        task_history.append(message)
