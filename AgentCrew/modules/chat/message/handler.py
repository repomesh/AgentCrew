from typing import Tuple, Optional, List
import asyncio
import os
import re
import shlex
import traceback

from loguru import logger
from AgentCrew.modules.agents.base import MessageType
from AgentCrew.modules.chat.history import ChatHistoryManager
from AgentCrew.modules.agents import AgentManager
from AgentCrew.modules.utils.file_handler import FileHandler

from AgentCrew.modules.memory import (
    BaseMemoryService,
    ContextPersistenceService,
)
from AgentCrew.modules.mcpclient import MCPSessionManager
from .command_processor import CommandProcessor
from .tool_manager import ToolManager
from .conversation import ConversationManager
from .base import Observable
from .prompt_evolution_coordinator import PromptEvolutionCoordinator
from AgentCrew.modules.chat.stream_session import StreamSession


_AT_AGENT_RE = re.compile(r"@([\.\w-]+)")


def _resolve_at_mention(user_input: str, agent_manager) -> tuple:
    match = _AT_AGENT_RE.search(user_input)
    if match:
        target = match.group(1)
        if target in agent_manager.agents:
            llm_content = (
                f"<Tag_Action>Transfer to {target} with the user request: "
                f"{user_input}</Tag_Action>"
            )
            return user_input, llm_content
    return user_input, user_input


class MessageHandler(Observable):
    """
    Handles message processing, interaction with the LLM service, and manages
    conversation history. Uses the Observer pattern to notify UI components
    about relevant events.
    """

    def __init__(
        self,
        memory_service: Optional[BaseMemoryService] = None,
        context_persistent_service: Optional[ContextPersistenceService] = None,
        with_voice: bool = False,
        voice_service=None,
    ):
        """
        Initializes the MessageHandler.

        Args:
            memory_service: Memory service for storing conversations.
            context_persistent_service: Service for persistent conversation storage.
        """
        super().__init__()
        self.agent_manager = AgentManager.get_instance()
        self.mcp_manager = MCPSessionManager.get_instance()
        self.agent = self.agent_manager.get_current_agent()
        self.memory_service = memory_service
        self.persistent_service = context_persistent_service
        self.history_manager = ChatHistoryManager()
        self.latest_assistant_response = ""
        self.conversation_turns = []
        self.current_user_input = None
        self.current_user_input_idx = -1
        self.last_assisstant_response_idx = -1
        self.file_handler: Optional[FileHandler] = None
        self._queued_attached_files = []
        self.stream_generator = None
        self.streamline_messages = []
        self._stream_session_counter = 0
        self._active_stream_session: Optional[StreamSession] = None
        self.current_conversation_id: Optional[str] = None  # ID for persistence
        self.prompt_evolution_coordinator = PromptEvolutionCoordinator(
            agent_getter=lambda: self.agent,
            notify=self._notify,
            memory_service=self.memory_service,
            persistence_service=self.persistent_service,
        )

        # Initialize components
        self.command_processor = CommandProcessor(self)
        self.tool_manager = ToolManager(self)
        self.conversation_manager = ConversationManager(self)

        self.conversation_manager.start_new_conversation()  # Initialize first conversation
        self._yolo_mode_check()

        self.voice_service = voice_service if with_voice else None

    def _yolo_mode_check(self):
        from AgentCrew.modules.config.global_config import GlobalConfig

        global_config = GlobalConfig().read()
        self.tool_manager.yolo_mode = global_config.get("global_settings", {}).get(
            "yolo_mode", False
        )

    def _messages_append(self, message):
        """Append a message to the agent history and streamline messages."""
        self.streamline_messages.append(message)

        self.agent.append_message(message)

    def _prepare_files_processing(self, file_command):
        file_paths_str: str = file_command[6:].strip()
        file_paths: List[str] = [
            os.path.expanduser(path.strip())
            for path in shlex.split(file_paths_str)
            if path.strip()
        ]

        for file_path in file_paths:
            self._queued_attached_files.append(file_path)
            self._notify("file_processing", {"file_path": file_path})

    async def process_user_input(
        self,
        user_input: str,
    ) -> Tuple[bool, bool]:
        """
        Processes user input, handles commands, and updates message history.

        Args:
            user_input: The input string from the user.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        self.history_manager.add_entry(user_input)

        if user_input.startswith("/file "):
            self._prepare_files_processing(user_input)
            return False, True
        if user_input.startswith("/retry"):
            return False, False

        # Process commands first
        command_result = await self.command_processor.process_command(user_input)
        if command_result.handled:
            return command_result.exit_flag, command_result.clear_flag

        # Delays file processing until user send message

        while len(self._queued_attached_files) > 0:
            file_command = self._queued_attached_files.pop(0)
            await self.command_processor.process_command(
                f"/file {shlex.quote(file_command)}"
            )

        # Add regular text message
        display_text, llm_content = _resolve_at_mention(user_input, self.agent_manager)

        self._messages_append(
            {
                "role": "user",
                "agent": self.agent.name,
                "content": [{"type": "text", "text": llm_content}],
            }
        )
        self.current_user_input = self.agent.history[-1]
        self.current_user_input_idx = len(self.streamline_messages) - 1
        self._notify(
            "user_message_created",
            {
                "message": self.agent.history[-1],
                "display_text": display_text,
                "with_files": False,
            },
        )

        return False, False

    def start_new_conversation(self):
        """Starts a new persistent conversation."""
        # Reset approved tools for the new conversation
        self.tool_manager.reset_approved_tools()
        self.conversation_manager.start_new_conversation()

    def resolve_tool_confirmation(self, confirmation_id, result):
        """
        Resolve a pending tool confirmation future with the user's decision.
        """
        self.tool_manager.resolve_tool_confirmation(confirmation_id, result)

    async def start_evolution_review(self) -> bool:
        return await self.prompt_evolution_coordinator.start_review()

    async def submit_pending_evolution_review(
        self, action: str, approved_summary: Optional[str] = None
    ) -> bool:
        return await self.prompt_evolution_coordinator.submit_review(
            action, approved_summary
        )

    def _create_stream_session(self) -> StreamSession:
        self._stream_session_counter += 1
        session = StreamSession(session_id=self._stream_session_counter)
        self._active_stream_session = session
        return session

    def _clear_stream_session(self, session: Optional[StreamSession]) -> None:
        if session and self._active_stream_session is session:
            self._active_stream_session = None
            self.stream_generator = None

    def has_active_stream(self) -> bool:
        session = self._active_stream_session
        return bool(session and not session.finished.is_set())

    def request_stop_stream(self) -> bool:
        session = self._active_stream_session
        if not session:
            return False
        if not session.mark_cancel_requested():
            return False

        self._notify("stream_cancel_requested", {"session_id": session.session_id})

        if session.loop and session.task:
            session.loop.call_soon_threadsafe(session.task.cancel)
        return True

    def _get_messages_for_current_turn(self) -> List[dict]:
        if self.last_assisstant_response_idx >= 0:
            return self.get_recent_agent_responses()
        if self.current_user_input_idx >= 0:
            return self.streamline_messages[self.current_user_input_idx + 1 :]
        return []

    def _extract_user_text(self, user_message: dict) -> str:
        user_input = ""
        content = user_message.get("content", "")
        if isinstance(content, list):
            for content_item in content:
                if content_item.get("type") == "text":
                    user_input += content_item.get("text", "")
        elif isinstance(content, str):
            user_input = content
        return user_input

    def _finalize_current_turn(
        self,
        assistant_response: str,
        input_tokens: int,
        output_tokens: int,
        *,
        store_memory: bool,
        emit_response_completed: bool,
    ) -> List[dict]:
        if assistant_response.strip():
            self._messages_append(
                self.agent.format_message(
                    MessageType.Assistant, {"message": assistant_response}
                )
            )

        if emit_response_completed:
            self._notify("response_completed", assistant_response)

        messages_for_this_turn = self._get_messages_for_current_turn()

        if self.current_conversation_id and messages_for_this_turn:
            try:
                if self.persistent_service:
                    if input_tokens > 0 or output_tokens > 0:
                        metadata = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                        self.persistent_service.store_conversation_metadata(
                            self.current_conversation_id, metadata
                        )

                    self.persistent_service.append_conversation_messages(
                        self.current_conversation_id,
                        messages_for_this_turn,
                    )
                    self._notify(
                        "conversation_saved", {"id": self.current_conversation_id}
                    )
            except Exception as e:
                error_message = f"Failed to save conversation turn to {self.current_conversation_id}: {str(e)}"
                logger.error(f"ERROR: {error_message}")
                self._notify("error", {"message": error_message})

        if self.current_user_input and self.current_user_input_idx >= 0:
            self.conversation_manager.store_conversation_turn(
                self.current_user_input, self.current_user_input_idx
            )
            if store_memory and self.memory_service:
                assistant_messages = self._extract_assistant_messages_for_memory(
                    messages_for_this_turn
                )

                try:
                    self.memory_service.store_conversation(
                        self._extract_user_text(self.current_user_input),
                        assistant_messages,
                        self.agent.name,
                    )
                except Exception as e:
                    self._notify(
                        "error", f"Failed to store conversation in memory: {str(e)}"
                    )
            self.current_user_input = None
            self.current_user_input_idx = -1

        self.last_assisstant_response_idx = len(self.streamline_messages)
        return messages_for_this_turn

    async def _run_stream_response(
        self,
        session: StreamSession,
        input_tokens: int,
        output_tokens: int,
    ) -> Tuple[Optional[str], int, int]:
        """
        Stream the assistant's response and return the response and token usage.

        Returns:
            Tuple of (assistant_response, input_tokens, output_tokens)
        """
        assistant_response = ""
        tool_uses = []
        thinking_content = ""  # Reset thinking content for new response
        thinking_signature = ""  # Store the signature
        start_thinking = False
        end_thinking = False
        has_stop_interupted = False

        if len(self.agent.history) == 0:
            return None, 0, 0

        # Create a reference to the streaming generator
        self.stream_generator = None

        def process_result(_tool_uses, _input_tokens, _output_tokens):
            nonlocal tool_uses, input_tokens, output_tokens
            tool_uses = _tool_uses
            input_tokens += _input_tokens
            output_tokens += _output_tokens

        try:
            self.stream_generator = self.agent.process_messages(callback=process_result)
            stream_iter = self.stream_generator.__aiter__()

            async def get_next_stream_item():
                if session.first_chunk_received:
                    return await stream_iter.__anext__()
                try:
                    next_item = await asyncio.wait_for(
                        stream_iter.__anext__(), timeout=session.first_chunk_timeout
                    )
                except asyncio.TimeoutError:
                    session.finalize("timed_out")
                    self._notify(
                        "stream_open_timeout",
                        {
                            "session_id": session.session_id,
                            "timeout": session.first_chunk_timeout,
                        },
                    )
                    raise TimeoutError(
                        f"Timed out waiting {session.first_chunk_timeout}s for the model stream to open"
                    )
                session.mark_streaming()
                return next_item

            while True:
                try:
                    next_item = await get_next_stream_item()
                except StopAsyncIteration:
                    break

                (
                    assistant_response,
                    chunk_text,
                    thinking_chunk,
                ) = next_item
                if session.cancel_requested:
                    has_stop_interupted = True
                    self._notify("streaming_stopped", assistant_response)
                    session.finalize("canceled")
                    await self.stream_generator.aclose()
                    self._finalize_current_turn(
                        assistant_response,
                        input_tokens,
                        output_tokens,
                        store_memory=False,
                        emit_response_completed=bool(assistant_response.strip()),
                    )
                    self._notify(
                        "stream_canceled",
                        {
                            "session_id": session.session_id,
                            "assistant_response": assistant_response,
                        },
                    )
                    return assistant_response, input_tokens, output_tokens

                # Accumulate thinking content if available
                if thinking_chunk:
                    think_text_chunk, signature = thinking_chunk

                    if not start_thinking:
                        # Notify about thinking process
                        self._notify("thinking_started", self.agent.name)
                        if not self.agent.is_streaming():
                            # Delays it a bit when using without stream
                            await asyncio.sleep(0.5)
                        start_thinking = True
                    if think_text_chunk:
                        thinking_content += think_text_chunk
                        self._notify("thinking_chunk", think_text_chunk)
                    if signature:
                        thinking_signature += signature
                if chunk_text:
                    # End thinking when chunk_text start
                    if not end_thinking and start_thinking:
                        self._notify("thinking_completed", thinking_content)
                        end_thinking = True
                    # Notify about response progress
                    if not self.agent.is_streaming():
                        # Delays it a bit when using without stream
                        await asyncio.sleep(0.3)
                    self._notify("response_chunk", (chunk_text, assistant_response))

            if not session.finished.is_set():
                session.finalize("completed")
            self.stream_generator = None

            # End thinking when break the response stream
            if not end_thinking and start_thinking:
                self._notify("thinking_completed", thinking_content)
                end_thinking = True

            # Handle tool use if needed
            if not has_stop_interupted and tool_uses and len(tool_uses) > 0:
                # Add thinking content as a separate message if available
                thinking_data = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = self.agent.format_message(
                    MessageType.Thinking, {"thinking": thinking_data}
                )
                if thinking_message:
                    self._messages_append(thinking_message)
                    self._notify("thinking_message_added", thinking_message)

                # Format assistant message with the response and tool uses
                tool_uses_without_transfer = [
                    t for t in tool_uses if t["name"] != "transfer"
                ]
                # only append message if there are tool uses other than transfer
                if len(tool_uses_without_transfer) > 0:
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                            "tool_uses": tool_uses_without_transfer,
                        },
                    )
                    self._messages_append(assistant_message)
                # ignore if message is empty
                elif assistant_response.strip():
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                        },
                    )
                    self._messages_append(assistant_message)
                self._notify("assistant_message_added", assistant_response)

                self._yolo_mode_check()

                # Process each tool use
                await self.tool_manager.execute_tools_batch(tool_uses)

                if input_tokens > 0 or output_tokens > 0:
                    self._notify(
                        "update_token_usage",
                        {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    )

                if has_stop_interupted:
                    # return as soon as possible
                    self._notify("response_completed", assistant_response)
                    return assistant_response, input_tokens, output_tokens

                return await self.get_assistant_response()

            self._finalize_current_turn(
                assistant_response,
                input_tokens,
                output_tokens,
                store_memory=True,
                emit_response_completed=True,
            )

            if self.agent_manager.defered_transfer:
                self.agent.history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""<Transfer_Post_Action_Reminder>{self.agent_manager.defered_transfer}. If action related to other agent, use `transfer` tool to chaining the work</Transfer_Post_Action_Reminder>""",
                            }
                        ],
                    }
                )
                self.agent_manager.defered_transfer = ""
                return await self.get_assistant_response()

            return assistant_response, input_tokens, output_tokens

        except asyncio.CancelledError:
            has_stop_interupted = True
            if self.stream_generator:
                try:
                    await self.stream_generator.aclose()
                except Exception:
                    pass
            if not session.finished.is_set():
                session.finalize("canceled")
            self._finalize_current_turn(
                assistant_response,
                input_tokens,
                output_tokens,
                store_memory=False,
                emit_response_completed=bool(assistant_response.strip()),
            )
            self._notify(
                "stream_canceled",
                {
                    "session_id": session.session_id,
                    "assistant_response": assistant_response,
                },
            )
            return assistant_response, input_tokens, output_tokens
        except GeneratorExit:
            return assistant_response, input_tokens, output_tokens
        except Exception as e:
            from openai import BadRequestError

            if isinstance(e, BadRequestError):
                if (
                    e.code == "model_max_prompt_tokens_exceeded"
                    or e.message.find("This endpoint's maximum context length is") >= 0
                ):
                    from AgentCrew.modules.agents import LocalAgent
                    from AgentCrew.modules.llm.model_registry import ModelRegistry

                    if isinstance(self.agent, LocalAgent):
                        max_token = ModelRegistry.get_model_limit(
                            self.agent.get_model()
                        )
                        self.agent.input_tokens_usage = max_token
                        return await self.get_assistant_response()
            if self.current_user_input:
                self.conversation_manager.store_conversation_turn(
                    self.current_user_input, self.current_user_input_idx
                )
                self.current_user_input = None
                self.current_user_input_idx = -1
            if self.current_conversation_id and self.last_assisstant_response_idx >= 0:
                messages_for_this_turn = self.get_recent_agent_responses()
                if messages_for_this_turn and self.persistent_service:
                    metadata = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                    self.persistent_service.store_conversation_metadata(
                        self.current_conversation_id, metadata
                    )

                    self.persistent_service.append_conversation_messages(
                        self.current_conversation_id,
                        messages_for_this_turn,
                    )
                    self._notify(
                        "conversation_saved", {"id": self.current_conversation_id}
                    )
            self.last_assisstant_response_idx = len(self.streamline_messages)

            error_message = str(e)
            traceback_str = traceback.format_exc()
            logger.error(f"{error_message} \n {traceback_str}")
            self._notify(
                "error",
                {
                    "message": error_message,
                    "messages": self.agent.history,
                },
            )
            if not session.finished.is_set():
                session.finalize("failed")
            return None, 0, 0

    async def get_assistant_response(
        self, input_tokens=0, output_tokens=0
    ) -> Tuple[Optional[str], int, int]:
        loop = asyncio.get_running_loop()
        session = self._create_stream_session()
        task = loop.create_task(
            self._run_stream_response(session, input_tokens, output_tokens)
        )
        session.bind(loop, task)

        audio_handler = None
        if self.voice_service is not None:
            audio_handler = getattr(self.voice_service, "audio_handler", None)

        if audio_handler is not None:
            audio_handler.is_processing = True
            clear_buffered_audio = getattr(audio_handler, "clear_buffered_audio", None)
            if callable(clear_buffered_audio):
                clear_buffered_audio()

        try:
            return await task
        finally:
            if audio_handler is not None:
                audio_handler.is_processing = False
            if not session.finished.is_set() and task.cancelled():
                session.finalize("canceled")
            self._clear_stream_session(session)

    def get_recent_agent_responses(self) -> List:
        return self.streamline_messages[self.last_assisstant_response_idx :]

    def _extract_assistant_messages_for_memory(self, messages: List[dict]) -> List[str]:
        assistant_messages: List[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    assistant_messages.append(normalized)
        return assistant_messages

    # Delegate conversation management methods
    def list_conversations(self):
        """Lists available conversations from the persistence service."""
        return self.conversation_manager.list_conversations()

    def list_conversations_with_forks(self):
        """Lists available conversations with fork relationship information."""
        return self.conversation_manager.list_conversations_with_forks()

    def load_conversation(self, conversation_id: str):
        """Loads a specific conversation history and sets it as active."""
        # Reset approved tools for the loaded conversation
        self.tool_manager.reset_approved_tools()
        return self.conversation_manager.load_conversation(conversation_id)

    def delete_conversation_by_id(self, conversation_id: str) -> bool:
        """Deletes a conversation by its ID."""
        return self.conversation_manager.delete_conversation_by_id(conversation_id)

    def _is_voice_enabled(self) -> bool:
        """Check if voice is enabled in current agent settings."""
        try:
            if self.voice_service is None:
                return False

            if hasattr(self.agent, "voice_enabled"):
                return getattr(self.agent, "voice_enabled") == "enabled"

            return False
        except Exception as e:
            logger.warning(f"Failed to read voice_enabled setting: {e}")
            return False

    def _get_configured_voice_id(self) -> Optional[str]:
        """Get the voice ID from current agent settings or return default."""
        try:
            if hasattr(self.agent, "voice_id"):
                return getattr(self.agent, "voice_id", None)

            return None

        except Exception as e:
            logger.warning(f"Failed to read voice_id from agent config: {e}")
            return None
