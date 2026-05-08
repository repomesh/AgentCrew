from AgentCrew.modules.llm.model_registry import ModelRegistry
from .service import CustomLLMService
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from uuid import uuid4
from AgentCrew.modules.llm.token_usage import TokenUsage


class GithubCopilotService(CustomLLMService):
    def __init__(
        self, api_key: Optional[str] = None, provider_name: str = "github_copilot"
    ):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GITHUB_COPILOT_API_KEY")
            if not api_key:
                raise ValueError(
                    "GITHUB_COPILOT_API_KEY not found in environment variables"
                )
        super().__init__(
            api_key=api_key,
            base_url="https://api.githubcopilot.com",
            provider_name=provider_name,
            extra_headers={
                "Copilot-Integration-Id": "vscode-chat",
                "Editor-Plugin-Version": "CopilotChat.nvim/*",
                "Editor-Version": "Neovim/0.9.0",
            },
        )
        self.model = "gpt-4.1"
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        self._is_thinking = False
        # self._interaction_id = None
        logger.info("Initialized Github Copilot Service")

    def set_think(self, budget_tokens) -> bool:
        """
        Enable or disable thinking mode with the specified token budget.

        Args:
            budget_tokens (int): Token budget for thinking. 0 to disable thinking mode.

        Returns:
            bool: True if thinking mode is supported and successfully set, False otherwise.
        """
        if "thinking" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            if budget_tokens == "0" or budget_tokens == "none":
                self.reasoning_effort = None
                return True
            elif budget_tokens not in ["low", "medium", "high", "max"]:
                raise ValueError("budget_tokens must be low, medium or high")

            self.reasoning_effort = budget_tokens
            return True
        logger.info("Thinking mode is not supported for OpenAI models.")
        return False

    def _github_copilot_token_to_open_ai_key(self, copilot_api_key):
        """
        Convert GitHub Copilot token to OpenAI key format.

        Args:
            copilot_api_key: The GitHub Copilot token

        Returns:
            Updated OpenAI compatible token
        """
        openai_api_key = self.client.api_key

        if openai_api_key.startswith("ghu") or int(
            dict(x.split("=") for x in openai_api_key.split(";"))["exp"]
        ) < int(datetime.now().timestamp()):
            import requests

            headers = {
                "Authorization": f"Bearer {copilot_api_key}",
                "Content-Type": "application/json",
            }
            if self.extra_headers:
                headers.update(self.extra_headers)
            res = requests.get(
                "https://api.github.com/copilot_internal/v2/token", headers=headers
            )
            self.client.api_key = res.json()["token"]

    def _is_github_provider(self):
        if self.base_url:
            from urllib.parse import urlparse

            parsed_url = urlparse(self.base_url)
            host = parsed_url.hostname
            if host and host.endswith(".githubcopilot.com"):
                return True
        return False

    def _convert_internal_format(self, messages: List[Dict[str, Any]]):
        thinking_block = None
        for i, msg in enumerate(messages):
            msg.pop("agent", None)
            if msg.get("role") == "assistant":
                if thinking_block:
                    msg["reasoning_text"] = thinking_block.get("thinking", "")
                    msg["reasoning_opaque"] = thinking_block.get("signature", "")
                    thinking_block = None
                    del messages[i - 1]
                if isinstance(msg.get("content", ""), List):
                    thinking_block = next(
                        (
                            block
                            for block in msg.get("content", [])
                            if block.get("type", "text") == "thinking"
                        ),
                        None,
                    )
                    msg["content"] = []

            if "tool_calls" in msg and msg.get("tool_calls", []):
                normalized_tool_calls = []
                for raw_tool_call in msg["tool_calls"]:
                    normalized_tool_call = self._normalize_tool_call_for_request(
                        raw_tool_call
                    )
                    if normalized_tool_call:
                        normalized_tool_calls.append(normalized_tool_call)
                msg["tool_calls"] = normalized_tool_calls

            if msg.get("role") == "tool":
                # Special treatment for GitHub Copilot GPT-4.1 model
                # At the the time of writing, GitHub Copilot GPT-4.1 model cannot read tool results with array content
                msg.pop("tool_name", None)
                msg.pop("is_rejected", None)
                if isinstance(msg.get("content", ""), List):
                    if self._is_github_provider() and self.model != "gpt-4.1":
                        # OpenAI format for tool responses
                        parsed_tool_result = []
                        for tool_content in msg["content"]:
                            if tool_content.get("type", "text") == "image_url":
                                if "vision" in ModelRegistry.get_model_capabilities(
                                    f"{self._provider_name}/{self.model}"
                                ):
                                    parsed_tool_result.append(tool_content)
                            else:
                                parsed_tool_result.append(tool_content)
                        msg["content"] = parsed_tool_result
                    else:
                        parsed_tool_result = []
                        for tool_content in msg["content"]:
                            # Skipping non-text tool results when vision is unavailable
                            # if res.get("type", "text") == "image_url":
                            #     if "vision" in ModelRegistry.get_model_capabilities(self.model):
                            #         parsed_tool_result.append(res)
                            # else:
                            if tool_content.get("type", "text") == "text":
                                parsed_tool_result.append(tool_content.get("text", ""))
                        msg["content"] = (
                            "\n".join(parsed_tool_result) if parsed_tool_result else ""
                        )
                elif isinstance(msg.get("content", ""), str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]

        return messages

    async def process_message(self, prompt: str, temperature: float = 0) -> str:
        if self._is_github_provider():
            self.base_url = self.base_url.rstrip("/")
            self._github_copilot_token_to_open_ai_key(self.api_key)
            if self.extra_headers:
                self.extra_headers["X-Initiator"] = "user"
                self.extra_headers["X-Request-Id"] = str(uuid4())
        return await super().process_message(prompt, temperature)

    def _process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: List[Dict]
    ) -> Tuple[str, List[Dict], TokenUsage, Optional[str], Optional[tuple]]:
        chunk_text = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        thinking_content = None
        thinking_signature = None

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

        if (not chunk.choices) or (len(chunk.choices) == 0):
            return (
                assistant_response or " ",
                tool_uses,
                TokenUsage(
                    input_tokens=input_tokens - cached_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                ),
                "",
                (thinking_content, None) if thinking_content else None,
            )

        delta_chunk = chunk.choices[0].delta

        # Handle thinking content
        if (
            hasattr(delta_chunk, "reasoning_text")
            and delta_chunk.reasoning_text is not None
        ):
            thinking_content = delta_chunk.reasoning_text

        if (
            hasattr(delta_chunk, "reasoning_opaque")
            and delta_chunk.reasoning_opaque is not None
        ):
            thinking_signature = delta_chunk.reasoning_opaque
        # Handle regular content chunks
        if hasattr(delta_chunk, "content") and delta_chunk.content is not None:
            chunk_text = chunk.choices[0].delta.content
            assistant_response += chunk_text

        # Handle tool call chunks
        if hasattr(delta_chunk, "tool_calls"):
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
            (thinking_content, thinking_signature)
            if thinking_content or thinking_signature
            else None,
        )

    async def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""

        if self._is_github_provider():
            self.base_url = self.base_url.rstrip("/")
            self._github_copilot_token_to_open_ai_key(self.api_key)
            # if len([m for m in messages if m.get("role") == "assistant"]) == 0:
            #     self._interaction_id = str(uuid4())
            if self.extra_headers:
                self.extra_headers["X-Initiator"] = (
                    "user"
                    if messages[-1].get("role", "assistant") == "user"
                    else "agent"
                )
                self.extra_headers["X-Request-Id"] = str(uuid4())
                if (
                    len(
                        [
                            m
                            for m in messages
                            if isinstance(m.get("content", ""), list)
                            and len(
                                [
                                    n
                                    for n in m.get("content", [])
                                    if n.get("type", "text") == "image_url"
                                ]
                            )
                            > 0
                        ]
                    )
                    > 0
                ):
                    if "vision" in ModelRegistry.get_model_capabilities(
                        f"{self._provider_name}/{self.model}"
                    ):
                        self.extra_headers["Copilot-Vision-Request"] = "true"

                # if self._interaction_id:
                #     self.extra_headers["X-Interaction-Id"] = self._interaction_id
            # Special handling for GitHub Copilot GPT-4.1 model
            # TODO: Find a better way to handle this
            if self.model == "gpt-4.1":
                for m in messages:
                    if m.get("role") == "tool" and isinstance(m.get("content"), list):
                        parsed_content = []
                        for content in m.get("content", []):
                            if content.get("type", "text") == "text":
                                parsed_content.append(content.get("text", ""))
                        m["content"] = "\n".join(parsed_content)
        return await super().stream_assistant_response(messages)
