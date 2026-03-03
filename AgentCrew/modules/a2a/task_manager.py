from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING
from AgentCrew.modules.agents.base import MessageType
from loguru import logger
import tempfile
import os

from a2a.types import (
    CancelTaskResponse,
    JSONRPCError,
    GetTaskResponse,
    GetTaskSuccessResponse,
    JSONRPCErrorResponse,
    SendMessageResponse,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    CancelTaskSuccessResponse,
    SetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigResponse,
    SendMessageSuccessResponse,
    Task,
    TaskStatus,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Part,
    TextPart,
    DataPart,
    Role,
    Message,
)

from AgentCrew.modules.agents import LocalAgent
from .adapters import (
    convert_a2a_message_to_agent,
    convert_agent_response_to_a2a_artifact,
)
from .common.server.task_manager import TaskManager
from .errors import A2AError
from .task_store import TaskStore

if TYPE_CHECKING:
    from typing import Any, AsyncIterable, Dict, Optional, Union
    from AgentCrew.modules.agents import AgentManager
    from a2a.types import (
        CancelTaskRequest,
        TaskNotCancelableError,
        GetTaskPushNotificationConfigRequest,
        GetTaskRequest,
        SendMessageRequest,
        SendStreamingMessageRequest,
        SetTaskPushNotificationConfigRequest,
        TaskResubscriptionRequest,
        JSONRPCResponse,
    )


class AgentTaskManager(TaskManager):
    """Manages tasks for a specific agent"""

    TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed}
    INPUT_REQUIRED_STATES = {TaskState.input_required}

    def __init__(self, agent_name: str, agent_manager: AgentManager, store: TaskStore):
        self.agent_name = agent_name
        self.agent_manager = agent_manager
        self.store = store
        self.streaming_tasks: Dict[str, asyncio.Queue] = {}
        self.file_handler = None

        self.streaming_enabled_tasks: set[str] = set()

        self.pending_ask_responses: Dict[str, asyncio.Event] = {}
        self.ask_responses: Dict[str, str] = {}

        self.agent = self.agent_manager.get_agent(self.agent_name)
        if self.agent is None or not isinstance(self.agent, LocalAgent):
            raise ValueError(f"Agent {agent_name} not found or is not a LocalAgent")

        self.memory_service = self.agent.services.get("memory", None)

    def _is_terminal_state(self, state: TaskState) -> bool:
        """Check if a state is terminal."""
        return state in self.TERMINAL_STATES

    def _extract_text_from_message(self, message: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)

    def _validate_task_not_terminal(
        self, task: Task, operation: str
    ) -> Optional[TaskNotCancelableError]:
        """
        Validate that task is not in terminal state.

        Args:
            task: Task to check
            operation: Operation being attempted

        Returns:
            JSONRPCError if invalid, None if valid
        """
        if self._is_terminal_state(task.status.state):
            return A2AError.task_not_cancelable(task.id, task.status.state.value)
        return None

    async def on_send_message(
        self, request: SendMessageRequest | SendStreamingMessageRequest
    ) -> SendMessageResponse:
        """
        Handle message/send request for this agent.

        Args:
            request: The message request

        Returns:
            JSON-RPC response with task result
        """
        if not self.agent or not isinstance(self.agent, LocalAgent):
            return SendMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=-32001, message=f"Agent {self.agent_name} not found"
                    ),
                )
            )

        task_id = (
            request.params.message.task_id
            or f"task_{request.params.message.message_id}"
        )

        existing_task = await self.store.get_task(task_id)
        if existing_task:
            error = self._validate_task_not_terminal(existing_task, "send message")
            if error:
                return SendMessageResponse(
                    root=JSONRPCErrorResponse(id=request.id, error=error)
                )

            if existing_task.status.state in self.INPUT_REQUIRED_STATES:
                message = convert_a2a_message_to_agent(request.params.message)
                user_response = self._extract_text_from_message(message)

                if task_id in self.pending_ask_responses:
                    self.ask_responses[task_id] = user_response
                    self.pending_ask_responses[task_id].set()

                return SendMessageResponse(
                    root=SendMessageSuccessResponse(id=request.id, result=existing_task)
                )

        task = existing_task
        if not task:
            task = Task(
                id=task_id,
                context_id=request.params.message.context_id or f"ctx_{task_id}",
                status=TaskStatus(
                    state=TaskState.working, timestamp=datetime.now().isoformat()
                ),
            )
            await self.store.save_task(task)

        if not await self.store.has_task_history(task.context_id):
            await self.store.save_task_history(task.context_id, [])

        message = convert_a2a_message_to_agent(request.params.message)
        if next(
            (m for m in message.get("content", []) if m.get("type", "text") == "file"),
            None,
        ):
            from AgentCrew.modules.chat.file_handler import FileHandler

            new_parts = []
            if self.file_handler is None:
                self.file_handler = FileHandler()
            for part in message.get("content", []):
                if part.get("type") == "file":
                    temp_file = os.path.join(tempfile.gettempdir(), part["file_name"])
                    with open(temp_file, "wb") as f:
                        f.write(part["file_data"])
                    file_part = self.file_handler.process_file(temp_file)
                    if not file_part:
                        file_part = self.agent.format_message(
                            MessageType.FileContent, {"file_uri": temp_file}
                        )
                    if file_part:
                        new_parts.append(file_part)
                    else:
                        new_parts.append(
                            {
                                "type": "text",
                                "text": f"[Unsupported file: {part['file_name']}]",
                            }
                        )
                else:
                    new_parts.append(part)

            message["content"] = new_parts

        await self.store.append_task_history_message(task.context_id, message)

        asyncio.create_task(self._process_agent_task(self.agent, task))

        return SendMessageResponse(
            root=SendMessageSuccessResponse(id=request.id, result=task)
        )

    async def on_send_message_streaming(
        self, request: SendStreamingMessageRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        """
        Handle message/stream request for this agent.

        Args:
            request: The message request

        Yields:
            JSON-RPC responses with task updates
        """
        # Generate task ID from message
        task_id = (
            request.params.message.task_id
            or f"task_{request.params.message.message_id}"
        )

        self.streaming_enabled_tasks.add(task_id)

        # Create streaming queue
        queue = asyncio.Queue()
        self.streaming_tasks[task_id] = queue

        try:
            # Start the task
            response = await self.on_send_message(request)

            # If there was an error, yield it and stop
            if isinstance(response.root, JSONRPCErrorResponse):
                yield SendStreamingMessageResponse(root=response.root)
                return

            # Yield events from the queue
            while True:
                event = await queue.get()
                if event is None:  # End of stream
                    await self.store.delete_task(task_id)
                    break
                yield SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id=request.id, result=event
                    )
                )

        finally:
            self.streaming_tasks.pop(task_id, None)

    def _create_ask_tool_message(
        self, question: str, guided_answers: list[str]
    ) -> Message:
        """
        Create an A2A message for the ask tool's input-required state.

        Args:
            question: The question to ask the user
            guided_answers: List of suggested answers

        Returns:
            A2A Message with the question and guided answers
        """
        ask_data = {
            "type": "ask",
            "question": question,
            "guided_answers": guided_answers,
            "instruction": "Please respond with one of the guided answers or provide a custom response.",
        }

        return Message(
            message_id=f"ask_{hash(question)}",
            role=Role.agent,
            parts=[
                Part(root=TextPart(text=f"❓ {question}")),
                Part(root=DataPart(data=ask_data)),
            ],
        )

    async def _record_and_emit_event(
        self, task_id: str, event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]
    ):
        """
        Record event for replay and broadcast to all active subscribers.

        Args:
            task_id: Task ID
            event: Event to record and emit
        """
        await self.store.append_task_event(task_id, event)

        for key, queue in list(self.streaming_tasks.items()):
            if key.startswith(task_id):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for {key}")
                except Exception as e:
                    logger.error(f"Error emitting event to {key}: {e}")

    async def _process_agent_task(self, agent: LocalAgent, task: Task):
        """
        Process a task with the agent (background task).

        Args:
            agent: The agent to process the task
            message: The message to process
            task: The task object to update
        """
        if self._is_terminal_state(task.status.state):
            logger.warning(
                f"Attempted to process task {task.id} in terminal state {task.status.state}"
            )
            return

        try:
            artifacts = []
            if not await self.store.has_task_history(task.context_id):
                raise ValueError("Task history is not existed")

            input_tokens = 0
            output_tokens = 0

            async def _process_task():
                try:
                    current_response = ""
                    response_message = ""
                    thinking_content = ""
                    thinking_signature = ""
                    tool_uses = []

                    def process_result(_tool_uses, _input_tokens, _output_tokens):
                        nonlocal tool_uses, input_tokens, output_tokens
                        tool_uses = _tool_uses
                        input_tokens += _input_tokens
                        output_tokens += _output_tokens

                    task_history = await self.store.get_task_history(task.context_id)
                    async for (
                        response_message,
                        chunk_text,
                        thinking_chunk,
                    ) in agent.process_messages(task_history, callback=process_result):
                        if response_message:
                            current_response = response_message

                        task.status.state = TaskState.working
                        task.status.timestamp = datetime.now().isoformat()

                        if task.id in self.streaming_enabled_tasks:
                            if thinking_chunk:
                                think_text_chunk, signature = thinking_chunk
                                if think_text_chunk:
                                    thinking_content += think_text_chunk

                                    thinking_artifact = convert_agent_response_to_a2a_artifact(
                                        think_text_chunk,
                                        artifact_id=f"thinking_{task.id}_{datetime.now()}",
                                    )
                                    await self._record_and_emit_event(
                                        task.id,
                                        TaskArtifactUpdateEvent(
                                            task_id=task.id,
                                            context_id=task.context_id,
                                            artifact=thinking_artifact,
                                        ),
                                    )
                                if signature:
                                    thinking_signature += signature

                            if chunk_text:
                                artifact = convert_agent_response_to_a2a_artifact(
                                    chunk_text,
                                    artifact_id=f"artifact_{task.id}_{len(artifacts)}",
                                )
                                await self._record_and_emit_event(
                                    task.id,
                                    TaskArtifactUpdateEvent(
                                        task_id=task.id,
                                        context_id=task.context_id,
                                        artifact=artifact,
                                    ),
                                )

                    if tool_uses and len(tool_uses) > 0:
                        if task.id in self.streaming_enabled_tasks:
                            artifact = convert_agent_response_to_a2a_artifact(
                                "",
                                artifact_id=f"artifact_{task.id}_{len(artifacts)}",
                                tool_uses=tool_uses,
                            )
                            await self._record_and_emit_event(
                                task.id,
                                TaskArtifactUpdateEvent(
                                    task_id=task.id,
                                    context_id=task.context_id,
                                    artifact=artifact,
                                ),
                            )
                            await asyncio.sleep(0.7)

                        thinking_data = (
                            (thinking_content, thinking_signature)
                            if thinking_content
                            else None
                        )
                        thinking_message = agent.format_message(
                            MessageType.Thinking, {"thinking": thinking_data}
                        )
                        if thinking_message:
                            await self.store.append_task_history_message(
                                task.context_id, thinking_message
                            )

                        assistant_message = agent.format_message(
                            MessageType.Assistant,
                            {
                                "message": response_message,
                                "tool_uses": [
                                    t for t in tool_uses if t["name"] != "transfer"
                                ],
                            },
                        )
                        if assistant_message:
                            await self.store.append_task_history_message(
                                task.context_id, assistant_message
                            )

                        for tool_use in tool_uses:
                            tool_name = tool_use["name"]

                            if tool_name == "ask":
                                question = tool_use["input"].get("question", "")
                                guided_answers = tool_use["input"].get(
                                    "guided_answers", []
                                )

                                task.status.state = TaskState.input_required
                                task.status.timestamp = datetime.now().isoformat()
                                task.status.message = self._create_ask_tool_message(
                                    question, guided_answers
                                )

                                await self.store.save_task(task)

                                await self._record_and_emit_event(
                                    task.id,
                                    TaskStatusUpdateEvent(
                                        task_id=task.id,
                                        context_id=task.context_id,
                                        status=task.status,
                                        final=False,
                                    ),
                                )

                                wait_event = asyncio.Event()
                                self.pending_ask_responses[task.id] = wait_event

                                try:
                                    await asyncio.wait_for(
                                        wait_event.wait(), timeout=300
                                    )
                                    user_answer = self.ask_responses.get(
                                        task.id, "No response received"
                                    )
                                except asyncio.TimeoutError:
                                    user_answer = "User did not respond in time."
                                finally:
                                    self.pending_ask_responses.pop(task.id, None)
                                    self.ask_responses.pop(task.id, None)

                                tool_result = f"User's answer: {user_answer}"

                                task.status.state = TaskState.working
                                task.status.timestamp = datetime.now().isoformat()
                                task.status.message = None

                                tool_result_message = agent.format_message(
                                    MessageType.ToolResult,
                                    {"tool_use": tool_use, "tool_result": tool_result},
                                )
                                if tool_result_message:
                                    await self.store.append_task_history_message(
                                        task.context_id, tool_result_message
                                    )

                                await self._record_and_emit_event(
                                    task.id,
                                    TaskStatusUpdateEvent(
                                        task_id=task.id,
                                        context_id=task.context_id,
                                        status=task.status,
                                        final=False,
                                    ),
                                )

                            else:
                                try:
                                    tool_result = await agent.execute_tool_call(
                                        tool_name,
                                        tool_use["input"],
                                    )

                                    tool_result_message = agent.format_message(
                                        MessageType.ToolResult,
                                        {
                                            "tool_use": tool_use,
                                            "tool_result": tool_result,
                                        },
                                    )
                                    if tool_result_message:
                                        await self.store.append_task_history_message(
                                            task.context_id, tool_result_message
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
                                        await self.store.append_task_history_message(
                                            task.context_id, error_message
                                        )

                        return await _process_task()
                    return current_response
                except Exception as e:
                    from openai import BadRequestError

                    if isinstance(e, BadRequestError):
                        if e.code == "model_max_prompt_tokens_exceeded":
                            from AgentCrew.modules.agents import LocalAgent
                            from AgentCrew.modules.llm.model_registry import (
                                ModelRegistry,
                            )

                            if isinstance(agent, LocalAgent):
                                max_token = ModelRegistry.get_model_limit(
                                    agent.get_model()
                                )
                                agent.input_tokens_usage = max_token
                                return await _process_task()
                    raise e

            current_response = await _process_task()
            if current_response.strip():
                assistant_message = agent.format_message(
                    MessageType.Assistant,
                    {
                        "message": current_response,
                    },
                )
                if assistant_message:
                    await self.store.append_task_history_message(
                        task.context_id, assistant_message
                    )
                task_history = await self.store.get_task_history(task.context_id)
                user_message = task_history[0].get("content", [{}])[0].get("text", "")
                if self.memory_service:
                    self.memory_service.store_conversation(
                        user_message, current_response, self.agent_name
                    )

            artifact = convert_agent_response_to_a2a_artifact(
                current_response, artifact_id=f"artifact_{task.id}_final"
            )
            artifacts.append(artifact)

            task.status.state = TaskState.completed
            task.status.timestamp = datetime.now().isoformat()
            task.artifacts = artifacts
            await self.store.save_task(task)

            if task.id in self.streaming_enabled_tasks:
                await self._record_and_emit_event(
                    task.id,
                    TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    ),
                )

                for key in list(self.streaming_tasks.keys()):
                    if key.startswith(task.id):
                        queue = self.streaming_tasks[key]
                        await queue.put(None)

        except Exception as e:
            logger.error(str(e))
            task_history = await self.store.get_task_history(task.context_id)
            logger.debug(task_history)
            task.status.state = TaskState.failed
            task.status.timestamp = datetime.now().isoformat()
            await self.store.save_task(task)

            if task.id in self.streaming_enabled_tasks:
                await self._record_and_emit_event(
                    task.id,
                    TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    ),
                )

                for key in list(self.streaming_tasks.keys()):
                    if key.startswith(task.id):
                        queue = self.streaming_tasks[key]
                        await queue.put(None)

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """
        Handle tasks/get request for this agent.

        Args:
            request: The task request

        Returns:
            JSON-RPC response with task result
        """
        task_id = request.params.id
        task = await self.store.get_task(task_id)
        if not task:
            return GetTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=A2AError.task_not_found(task_id)
                )
            )

        return GetTaskResponse(root=GetTaskSuccessResponse(id=request.id, result=task))

    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """
        Handle tasks/cancel request for this agent.

        Args:
            request: The task request

        Returns:
            JSON-RPC response with task result
        """
        task_id = request.params.id
        task = await self.store.get_task(task_id)
        if not task:
            return CancelTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=A2AError.task_not_found(task_id)
                )
            )

        error = self._validate_task_not_terminal(task, "cancel")
        if error:
            return CancelTaskResponse(
                root=JSONRPCErrorResponse(id=request.id, error=error)
            )

        task.status.state = TaskState.canceled
        task.status.timestamp = datetime.now().isoformat()
        await self.store.save_task(task)

        if task_id in self.streaming_tasks:
            queue = self.streaming_tasks[task_id]
            await queue.put(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=task.context_id,
                    status=task.status,
                    final=True,
                )
            )
            await queue.put(None)

        return CancelTaskResponse(
            root=CancelTaskSuccessResponse(id=request.id, result=task)
        )

    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationConfigRequest
    ) -> SetTaskPushNotificationConfigResponse:
        return SetTaskPushNotificationConfigResponse(
            root=JSONRPCErrorResponse(
                id=request.id, error=A2AError.push_notification_not_supported()
            )
        )

    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationConfigRequest
    ) -> GetTaskPushNotificationConfigResponse:
        return GetTaskPushNotificationConfigResponse(
            root=JSONRPCErrorResponse(
                id=request.id, error=A2AError.push_notification_not_supported()
            )
        )

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        """
        Handle tasks/resubscribe request.

        Replays all events from task creation and continues with live updates.

        Args:
            request: The resubscription request

        Yields:
            Streaming responses with task updates
        """
        task_id = request.params.id

        task = await self.store.get_task(task_id)
        if not task:
            error = A2AError.task_not_found(task_id)
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(id=request.id, error=error)
            )
            return

        if task_id not in self.streaming_enabled_tasks:
            error = A2AError.unsupported_operation(
                "Task was not created with streaming enabled"
            )
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(id=request.id, error=error)
            )
            return

        stored_events = await self.store.get_task_events(task_id)
        for event in stored_events:
            yield SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(id=request.id, result=event)
            )

        if self._is_terminal_state(task.status.state):
            return

        queue = asyncio.Queue()
        resubscribe_key = f"{task_id}_resub_{request.id}"
        self.streaming_tasks[resubscribe_key] = queue

        try:
            while True:
                event = await queue.get()
                if event is None:
                    await self.store.delete_task(task_id)
                    break

                yield SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id=request.id, result=event
                    )
                )

                if isinstance(event, TaskStatusUpdateEvent):
                    if self._is_terminal_state(event.status.state):
                        break

        finally:
            if resubscribe_key in self.streaming_tasks:
                del self.streaming_tasks[resubscribe_key]

    # Legacy methods for backward compatibility
    async def on_send_task(self, request: SendMessageRequest) -> SendMessageResponse:
        """Legacy method - delegates to on_send_message"""
        return await self.on_send_message(request)

    async def on_send_task_subscribe(
        self, request: SendStreamingMessageRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        """Legacy method - delegates to on_send_message_streaming"""
        return await self.on_send_message_streaming(request)


class MultiAgentTaskManager:
    """Manages tasks for multiple agents"""

    def __init__(
        self,
        agent_manager: AgentManager,
        store_type: str = "memory",
        store_options: Optional[Dict[str, Any]] = None,
    ):
        from .task_store import create_task_store

        self.agent_manager = agent_manager
        self.agent_task_managers: Dict[str, AgentTaskManager] = {}

        for agent_name in agent_manager.agents:
            store = create_task_store(store_type, **(store_options or {}))
            self.agent_task_managers[agent_name] = AgentTaskManager(
                agent_name, agent_manager, store
            )

    def get_task_manager(self, agent_name: str) -> Optional[AgentTaskManager]:
        return self.agent_task_managers.get(agent_name)
