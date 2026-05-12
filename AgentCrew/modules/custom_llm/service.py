from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.openai import OpenAIService
from AgentCrew.modules.llm.base import AsyncIterator
from AgentCrew.modules.llm.token_usage import TokenUsage
from typing import Any, Tuple
import json
from loguru import logger


class CustomLLMService(OpenAIService):
    """Custom LLM service that can connect to any OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        provider_name: str,
        extra_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the CustomLLMService.

        Args:
            base_url (str): The base URL of the OpenAI-compatible API.
            api_key (str): The API key for the service.
            provider_name (str): The name of the custom provider.
            is_stream (bool): Whether to enable streaming responses by default.
            extra_headers (Optional[list[dict[str, str]]]): Custom HTTP headers to include in API requests.
        """
        super().__init__(api_key=api_key, base_url=base_url)
        self._provider_name = provider_name
        logger.info(
            f"Initialized Custom LLM Service for provider: {provider_name} at {base_url}"
        )
        self.extra_headers = extra_headers

    @staticmethod
    def _has_usable_tool_name(name: Any) -> bool:
        return isinstance(name, str) and bool(name.strip())

    @staticmethod
    def _safe_json_loads(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not raw_arguments:
            return {}
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
                return parsed_arguments if isinstance(parsed_arguments, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _normalize_tool_call_for_request(
        self, raw_tool_call: dict[str, Any]
    ) -> dict[str, Any] | None:
        tool_call = dict(raw_tool_call)
        tool_name = tool_call.get("name")
        if not self._has_usable_tool_name(tool_name):
            logger.warning(
                "Dropping malformed assistant tool call without a usable name before provider conversion"
            )
            return None

        return {
            "id": tool_call.get("id"),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_call.get("arguments", {})),
            },
        }

    def _append_non_stream_tool_call(
        self, tool_uses: list[dict[str, Any]], tool_call: Any
    ) -> bool:
        function = getattr(tool_call, "function", None)
        tool_name = getattr(function, "name", None) if function else None
        if not self._has_usable_tool_name(tool_name):
            logger.warning(
                "Dropping malformed provider tool call without a usable name"
            )
            return False

        tool_uses.append(
            {
                "id": getattr(tool_call, "id", None)
                or f"toolu_{tool_name}_{len(tool_uses)}",
                "name": tool_name,
                "input": self._safe_json_loads(
                    getattr(function, "arguments", "") if function else ""
                ),
                "type": getattr(tool_call, "type", "function"),
                "response": "",
            }
        )
        return True

    def _resolve_stream_tool_call_index(
        self, tool_uses: list[dict[str, Any]], tool_call_delta: Any
    ) -> int | None:
        tool_call_index = getattr(tool_call_delta, "index", None)
        if tool_call_index is not None:
            while len(tool_uses) <= tool_call_index:
                tool_uses.append(
                    {
                        "id": None,
                        "name": "",
                        "input": {},
                        "type": "function",
                        "response": "",
                    }
                )
            return tool_call_index

        tool_call_id = getattr(tool_call_delta, "id", None)
        if tool_call_id:
            for existing_index, tool_use in enumerate(tool_uses):
                if tool_use.get("id") == tool_call_id:
                    return existing_index

        function = getattr(tool_call_delta, "function", None)
        tool_name = getattr(function, "name", None) if function else None
        if self._has_usable_tool_name(tool_name):
            tool_uses.append(
                {
                    "id": tool_call_id or f"toolu_{tool_name}_{len(tool_uses)}",
                    "name": tool_name,
                    "input": {},
                    "type": "function",
                    "response": "",
                }
            )
            return len(tool_uses) - 1

        logger.debug(
            "Skipping malformed tool call delta without a usable name and without a matching existing tool call"
        )
        return None

    def _merge_stream_tool_call_delta(
        self, tool_uses: list[dict[str, Any]], tool_call_delta: Any
    ) -> int | None:
        tool_call_index = self._resolve_stream_tool_call_index(
            tool_uses, tool_call_delta
        )
        if tool_call_index is None:
            return None

        tool_use = tool_uses[tool_call_index]

        if hasattr(tool_call_delta, "id") and tool_call_delta.id:
            tool_use["id"] = tool_call_delta.id

        if hasattr(tool_call_delta, "function"):
            if hasattr(tool_call_delta.function, "name") and self._has_usable_tool_name(
                tool_call_delta.function.name
            ):
                tool_use["name"] = tool_call_delta.function.name

            if (
                hasattr(tool_call_delta.function, "arguments")
                and tool_call_delta.function.arguments
            ):
                current_args = tool_use.get("args_json", "")
                tool_use["args_json"] = (
                    current_args + tool_call_delta.function.arguments
                )

                try:
                    parsed_arguments = json.loads(tool_use["args_json"])
                    if isinstance(parsed_arguments, dict):
                        tool_use["input"] = parsed_arguments
                except json.JSONDecodeError:
                    pass

        return tool_call_index

    async def process_message(self, prompt: str, temperature: float = 0) -> str:
        result_text = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        stream = await self.client.chat.completions.create(
            model=self.model,
            timeout=60,
            max_tokens=3000,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            extra_headers=self.extra_headers,
        )

        async for chunk in stream:
            if (
                chunk.choices
                and hasattr(chunk.choices[0].delta, "content")
                and chunk.choices[0].delta.content is not None
            ):
                result_text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage:
                if hasattr(chunk.usage, "prompt_tokens"):
                    input_tokens = chunk.usage.prompt_tokens
                if hasattr(chunk.usage, "completion_tokens"):
                    output_tokens = chunk.usage.completion_tokens
                if (
                    hasattr(chunk.usage, "prompt_tokens_details")
                    and chunk.usage.prompt_tokens_details
                ):
                    if hasattr(chunk.usage.prompt_tokens_details, "cached_tokens"):
                        cached_tokens = (
                            chunk.usage.prompt_tokens_details.cached_tokens or 0
                        )

        if cached_tokens:
            input_tokens = input_tokens - cached_tokens
        total_cost = self.calculate_cost(input_tokens, output_tokens, cached_tokens)

        logger.info("\nToken Usage Statistics:")
        logger.info(f"Input tokens: {input_tokens:,}")
        logger.info(f"Output tokens: {output_tokens:,}")
        if cached_tokens:
            logger.info(f"Cached tokens: {cached_tokens:,}")
        logger.info(f"Total tokens: {input_tokens + output_tokens + cached_tokens:,}")
        logger.info(f"Estimated cost: ${total_cost:.4f}")

        if "thinking" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            THINK_STARTED = "<think>"
            THINK_STOPED = "</think>"

            if (
                result_text.find(THINK_STARTED) >= 0
                and result_text.find(THINK_STOPED) >= 0
            ):
                result_text = (
                    result_text[: result_text.find(THINK_STARTED)]
                    + result_text[
                        (result_text.find(THINK_STOPED) + len(THINK_STOPED)) :
                    ]
                )

        return result_text

    def _convert_internal_format(self, messages: list[dict[str, Any]]):
        for msg in messages:
            msg.pop("agent", None)
            if "tool_calls" in msg and msg.get("tool_calls", []):
                normalized_tool_calls = []
                for raw_tool_call in msg["tool_calls"]:
                    normalized_tool_call = self._normalize_tool_call_for_request(
                        raw_tool_call
                    )
                    if normalized_tool_call:
                        normalized_tool_calls.append(normalized_tool_call)
                msg["tool_calls"] = normalized_tool_calls
            if msg.get("role") == "consolidated":
                msg["role"] = "user"
                msg.pop("metadata", None)
            elif msg.get("role") == "tool":
                msg.pop("tool_name", None)
                msg.pop("is_rejected", None)
                if isinstance(msg.get("content", ""), list):
                    cleaned_tool_content = []
                    for tool_content in msg["content"]:
                        if isinstance(tool_content, dict):
                            if tool_content.get("type", "text") == "text":
                                cleaned_tool_content.append(
                                    tool_content.get("text", "")
                                )
                    msg["content"] = "\n".join(cleaned_tool_content)
            elif msg.get("role") == "assistant":
                if isinstance(msg.get("content", ""), list):
                    for assistant_content in msg["content"]:
                        if isinstance(assistant_content, dict):
                            if assistant_content.get("type", "text") == "thinking":
                                assistant_content["type"] = "text"
                                assistant_content["text"] = (
                                    f"<think>{assistant_content.get('thinking', '')}</think>"
                                )
                                assistant_content.pop("thinking", None)

        return messages

    def _build_stream_params(
        self,
    ) -> Tuple[dict[str, Any], bool]:
        """Build stream parameters for the API call.

        Override this in derived classes to customize parameters without
        re-implementing the entire streaming method.

        Returns:
            tuple: (stream_params_dict, is_streamable)
        """
        stream_params = {}
        stream_params["model"] = self.model
        stream_params["temperature"] = self.temperature
        stream_params["extra_body"] = {"min_p": 0.02}
        stream_params["stream_options"] = {"include_usage": True}

        full_model_id = f"{self._provider_name}/{self.model}"

        forced_sample_params = ModelRegistry.get_model_sample_params(full_model_id)
        if forced_sample_params:
            if forced_sample_params.temperature is not None:
                stream_params["temperature"] = forced_sample_params.temperature
            if forced_sample_params.top_p is not None:
                stream_params["top_p"] = forced_sample_params.top_p
            if forced_sample_params.top_k is not None:
                stream_params["extra_body"]["top_k"] = forced_sample_params.top_k
            if forced_sample_params.frequency_penalty is not None:
                stream_params["frequency_penalty"] = (
                    forced_sample_params.frequency_penalty
                )
            if forced_sample_params.presence_penalty is not None:
                stream_params["presence_penalty"] = (
                    forced_sample_params.presence_penalty
                )
            if forced_sample_params.repetition_penalty is not None:
                stream_params["extra_body"]["repetition_penalty"] = (
                    forced_sample_params.repetition_penalty
                )
            if forced_sample_params.min_p is not None:
                stream_params["extra_body"]["min_p"] = forced_sample_params.min_p

        model_capabilities = ModelRegistry.get_model_capabilities(full_model_id)

        if self.tools and "tool_use" in model_capabilities:
            stream_params["tools"] = self.tools

        if "thinking" in model_capabilities and self.reasoning_effort:
            stream_params["reasoning_effort"] = self.reasoning_effort

        if "structured_output" in model_capabilities and self.structured_output:
            stream_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "default",
                    "schema": self.structured_output,
                },
            }

        is_streamable = "stream" in model_capabilities
        return stream_params, is_streamable

    async def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""

        stream_params, is_streamable = self._build_stream_params()

        stream_params["messages"] = self._convert_internal_format(messages)

        if self.system_prompt:
            stream_params["messages"] = [
                {"role": "system", "content": self.system_prompt}
            ] + stream_params["messages"]

        if is_streamable:
            self._is_thinking = False
            return await self.client.chat.completions.create(
                **stream_params,
                stream=True,
                extra_headers=self.extra_headers,
            )

        else:
            response = await self.client.chat.completions.create(
                **stream_params,
                stream=False,
                extra_headers=self.extra_headers,
            )

            if response.usage:
                self.current_input_tokens = response.usage.prompt_tokens
                self.current_output_tokens = response.usage.completion_tokens
            else:
                self.current_input_tokens = 0
                self.current_output_tokens = 0

            return AsyncIterator(response.choices)

    def process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: list[dict]
    ) -> Tuple[str, list[dict], TokenUsage, str | None, tuple | None]:
        if "stream" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            return self._process_stream_chunk(chunk, assistant_response, tool_uses)
        else:
            return self._process_non_stream_chunk(chunk, assistant_response, tool_uses)

    def _process_non_stream_chunk(
        self, chunk, assistant_response, tool_uses
    ) -> Tuple[str, list[dict], TokenUsage, str | None, tuple | None]:
        """
        Process a single chunk from the streaming response.

        Args:
            chunk: The chunk from the stream
            assistant_response: Current accumulated assistant response
            tool_uses: Current tool use information

        Returns:
            tuple: (
                updated_assistant_response,
                updated_tool_uses,
                input_tokens,
                output_tokens,
                chunk_text,
                thinking_data
            )
        """
        # Check if this is a non-streaming response (for tool use)
        thinking_content = None

        input_tokens = self.current_input_tokens
        self.current_input_tokens = 0
        output_tokens = self.current_output_tokens
        self.current_output_tokens = 0
        if hasattr(chunk, "message"):
            # This is a complete response, not a streaming chunk
            message = chunk.message
            content = message.content or " "
            if hasattr(message, "reasoning") and message.reasoning:
                thinking_content = (message.reasoning, None)
            if "thinking" in ModelRegistry.get_model_capabilities(
                f"{self._provider_name}/{self.model}"
            ):
                THINK_STARTED = "<think>"
                THINK_STOPED = "</think>"
                think_start_idx = content.find(THINK_STARTED)
                think_stop_idx = content.find(THINK_STOPED)
                if think_start_idx >= 0 and think_stop_idx >= 0:
                    thinking_content = (content[think_start_idx:think_stop_idx], None)
                    content = (
                        content[:think_start_idx]
                        + content[think_stop_idx + len(THINK_STOPED) :]
                    )
            # Check for tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    self._append_non_stream_tool_call(tool_uses, tool_call)

                # Return with tool use information and the full content
                return (
                    content,
                    tool_uses,
                    TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    content,  # Return the full content to be printed
                    thinking_content,
                )

            # Check for tool call format in the response
            tool_call_start = "<tool_call>"
            tool_call_end = "<｜tool▁calls▁end｜>"

            if tool_call_start in content and tool_call_end in content:
                start_idx = content.find(tool_call_start)
                end_idx = content.find(tool_call_end) + len(tool_call_end)

                tool_call_content = content[
                    start_idx + len(tool_call_start) : end_idx - len(tool_call_end)
                ]

                try:
                    tool_data = json.loads(tool_call_content)
                    tool_uses.append(
                        {
                            "id": f"toolu_{len(tool_uses)}",  # Generate an ID
                            "name": tool_data.get("name", ""),
                            "input": tool_data.get("arguments", {}),
                            "type": "function",
                            "response": "",
                        }
                    )

                    # Remove the tool call from the response
                    content = content[:start_idx] + content[end_idx:]
                except json.JSONDecodeError:
                    # If we can't parse the JSON, just continue
                    pass

            # Regular response without tool calls
            return (
                content,
                tool_uses,
                TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                content,  # Return the full content to be printed
                thinking_content,
            )

        # Handle regular streaming chunk
        chunk_text = chunk.choices[0].delta.content or ""
        updated_assistant_response = assistant_response + chunk_text

        return (
            updated_assistant_response,
            tool_uses,
            TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            chunk_text,
            thinking_content,
        )

    def _process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: list[dict]
    ) -> Tuple[str, list[dict], TokenUsage, str | None, tuple | None]:
        """
        Process a single chunk from the streaming response.

        Args:
            chunk: The chunk from the stream
            assistant_response: Current accumulated assistant response
            tool_uses: Current tool use information

        Returns:
            tuple: (
                updated_assistant_response,
                updated_tool_uses,
                input_tokens,
                output_tokens,
                chunk_text,
                thinking_data
            )
        """
        chunk_text = ""
        input_tokens = 0
        output_tokens = 0
        thinking_content = None  # OpenAI doesn't support thinking mode

        cached_tokens = 0
        if hasattr(chunk, "usage"):
            if hasattr(chunk.usage, "prompt_tokens"):
                input_tokens = chunk.usage.prompt_tokens
            if hasattr(chunk.usage, "completion_tokens"):
                output_tokens = chunk.usage.completion_tokens
            if (
                hasattr(chunk.usage, "prompt_tokens_details")
                and chunk.usage.prompt_tokens_details
            ):
                if hasattr(chunk.usage.prompt_tokens_details, "cached_tokens"):
                    cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens or 0

        # Handle regular content chunks
        if (
            chunk.choices
            and len(chunk.choices) > 0
            and hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            chunk_text = chunk.choices[0].delta.content
            if "<think>" in chunk_text:
                self._is_thinking = True

            if self._is_thinking:
                thinking_content = chunk_text
                if "<think>" in thinking_content:
                    thinking_content = thinking_content.replace("<think>", "")
                if "</think>" in thinking_content:
                    # Remove thinking end tag
                    thinking_content = thinking_content.replace("</think>", "")
            else:
                assistant_response += chunk_text

            if "</think>" in chunk_text:
                self._is_thinking = False
                chunk_text = None

            if self._is_thinking:
                chunk_text = None
            # Remove chunk_text if still in thinking mode

        # Handle final chunk with usage information

        # Handle tool call chunks
        if (
            chunk.choices
            and len(chunk.choices) > 0
            and hasattr(chunk.choices[0].delta, "tool_calls")
        ):
            delta_tool_calls = chunk.choices[0].delta.tool_calls
            if delta_tool_calls:
                for tool_call_delta in delta_tool_calls:
                    self._merge_stream_tool_call_delta(tool_uses, tool_call_delta)

        return (
            assistant_response or " ",
            tool_uses,
            TokenUsage(
                input_tokens=input_tokens - cached_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            ),
            chunk_text,
            (thinking_content, None) if thinking_content else None,
        )
