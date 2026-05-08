from typing import Any, Dict, List, Optional, Tuple

from AgentCrew.modules.custom_llm.service import CustomLLMService
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.token_usage import TokenUsage


class OpenCodeService(CustomLLMService):
    def _normalize_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized_tool_calls = []
        for raw_tool_call in tool_calls:
            normalized_tool_call = self._normalize_tool_call_for_request(raw_tool_call)
            if normalized_tool_call:
                normalized_tool_calls.append(normalized_tool_call)
        return normalized_tool_calls

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, list):
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content_parts.append(item.get("text", ""))
                    elif "text" in item:
                        content_parts.append(item["text"])
                    elif "content" in item:
                        content_parts.append(str(item["content"]))
                    else:
                        content_parts.append(str(item))
                elif item is not None:
                    content_parts.append(str(item))
            return "\n".join(part for part in content_parts if part)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    def _extract_assistant_content_and_reasoning(
        self, content: Any
    ) -> Tuple[str, Optional[str], bool]:
        if not isinstance(content, list):
            return self._stringify_content(content), None, False

        text_parts = []
        reasoning_parts = []
        has_non_reasoning_content = False

        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "thinking":
                    reasoning_text = item.get("thinking") or item.get("text")
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                elif item_type == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
                        has_non_reasoning_content = True
                elif "text" in item:
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
                        has_non_reasoning_content = True
                elif item.get("content") is not None:
                    text_parts.append(str(item["content"]))
                    has_non_reasoning_content = True
                else:
                    text_parts.append(str(item))
                    has_non_reasoning_content = True
            elif item is not None:
                text_parts.append(str(item))
                has_non_reasoning_content = True

        return (
            "\n".join(part for part in text_parts if part),
            "\n".join(part for part in reasoning_parts if part) or None,
            bool(reasoning_parts) and not has_non_reasoning_content,
        )

    def _convert_internal_format(self, messages: List[Dict[str, Any]]):
        converted_messages = []
        pending_reasoning_content = None

        for raw_msg in messages:
            msg = dict(raw_msg)
            msg.pop("agent", None)
            role = msg.get("role", "")

            if role == "assistant":
                content_text, extracted_reasoning, thinking_only = (
                    self._extract_assistant_content_and_reasoning(
                        msg.get("content", "")
                    )
                )
                reasoning_content = (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or extracted_reasoning
                )

                if thinking_only and not msg.get("tool_calls"):
                    if reasoning_content:
                        if pending_reasoning_content:
                            pending_reasoning_content = (
                                f"{pending_reasoning_content}\n{reasoning_content}"
                            )
                        else:
                            pending_reasoning_content = reasoning_content
                    continue

                if "tool_calls" in msg and msg.get("tool_calls", []):
                    msg["tool_calls"] = self._normalize_tool_calls(msg["tool_calls"])

                if pending_reasoning_content and not reasoning_content:
                    reasoning_content = pending_reasoning_content

                pending_reasoning_content = None
                msg["content"] = content_text
                if reasoning_content:
                    msg["reasoning_content"] = reasoning_content
                converted_messages.append(msg)
                continue

            if role == "tool":
                msg.pop("tool_name", None)
                msg.pop("is_rejected", None)
                msg["content"] = self._stringify_content(msg.get("content", ""))
                converted_messages.append(msg)
                continue

            msg["content"] = self._stringify_content(msg.get("content", ""))
            converted_messages.append(msg)

        return converted_messages

    def process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: List[Dict]
    ) -> Tuple[str, List[Dict], TokenUsage, Optional[str], Optional[tuple]]:
        if "stream" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            return self._process_stream_chunk(chunk, assistant_response, tool_uses)
        return self._process_non_stream_chunk(chunk, assistant_response, tool_uses)

    def _process_non_stream_chunk(
        self, chunk, assistant_response, tool_uses
    ) -> Tuple[str, List[Dict], TokenUsage, Optional[str], Optional[tuple]]:
        input_tokens = self.current_input_tokens
        self.current_input_tokens = 0
        output_tokens = self.current_output_tokens
        self.current_output_tokens = 0
        thinking_data = (" ", None) if self.model.startswith("deepseek-v4") else None

        if hasattr(chunk, "message"):
            message = chunk.message
            content = self._stringify_content(getattr(message, "content", " ") or " ")
            reasoning_content = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )
            if reasoning_content:
                thinking_data = (reasoning_content, None)

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    self._append_non_stream_tool_call(tool_uses, tool_call)
                return (
                    content,
                    tool_uses,
                    TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    content,
                    thinking_data,
                )

            return (
                content,
                tool_uses,
                TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                content,
                thinking_data,
            )

        return (
            assistant_response or " ",
            tool_uses,
            TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            None,
            thinking_data,
        )

    def _process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: List[Dict]
    ) -> Tuple[str, List[Dict], TokenUsage, Optional[str], Optional[tuple]]:
        chunk_text = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        thinking_content = " " if self.model.startswith("deepseek-v4") else None

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

        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if (
                hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None
            ):
                thinking_content = delta.reasoning_content
            elif hasattr(delta, "reasoning") and delta.reasoning is not None:
                thinking_content = delta.reasoning

            if hasattr(delta, "content") and delta.content is not None:
                chunk_text = delta.content
                assistant_response += chunk_text

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    self._merge_stream_tool_call_delta(tool_uses, tool_call_delta)

        return (
            assistant_response or " ",
            tool_uses,
            TokenUsage(
                input_tokens=input_tokens - cached_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            ),
            chunk_text or None,
            (thinking_content, None) if thinking_content else None,
        )
