"""Experimental direct xAI subscription OAuth Responses adapter."""

import asyncio
from typing import Any

from AgentCrew.modules.llm.token_usage import TokenUsage
from AgentCrew.modules.openai.response_service import OpenAIResponseService
from AgentCrew.modules.xai.oauth import XAIOAuth

XAI_BASE_URL = "https://api.x.ai/v1"


class XAISubscriptionResponseService(OpenAIResponseService):
    """Reuse Responses history/event orchestration, without OpenAI-only options."""

    def __init__(self, token_path: str | None = None):
        self._oauth = XAIOAuth(token_path)
        token = self._oauth.get_valid_access_token()
        super().__init__(api_key=token, base_url=XAI_BASE_URL)
        self.client = self.client.with_options(max_retries=0)
        self._provider_name = "xai_subscription"
        self.model = "grok-4.7"
        self.reasoning_effort = "high"

    async def _ensure_valid_token(self) -> None:
        self.client.api_key = await asyncio.to_thread(
            self._oauth.get_valid_access_token
        )

    def set_think(self, budget_tokens: str) -> bool:
        if budget_tokens in ("none", "0"):
            raise ValueError("Grok 4.7 reasoning cannot be disabled")
        if budget_tokens not in ("low", "medium", "high"):
            raise ValueError("Grok 4.7 reasoning effort must be low, medium or high")
        self.reasoning_effort = budget_tokens
        return True

    async def stream_assistant_response(self, messages: list[dict[str, Any]]) -> Any:
        await self._ensure_valid_token()
        params = {
            "model": self.model,
            "input": self._convert_internal_format(messages),
            "instructions": self.system_prompt or "You are a helpful assistant.",
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": self.reasoning_effort},
        }
        if self.tools:
            params["tools"] = [
                tool for tool in self.tools if tool.get("type") == "function"
            ]
        return await self.client.responses.create(**params)

    @staticmethod
    def _check_stream_event(event_type: str | None) -> None:
        if event_type in ("response.failed", "response.incomplete", "error"):
            raise RuntimeError(
                "xAI Responses stream failed or incomplete; request was not replayed"
            )

    def process_stream_chunk(
        self,
        chunk: Any,
        assistant_response: str,
        tool_uses: list[dict],
        stream_state: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict], TokenUsage, str | None, tuple | None]:
        event_type = getattr(chunk, "type", None)
        self._check_stream_event(event_type)
        if event_type == "response.completed" and stream_state is not None:
            response = getattr(chunk, "response", None)
            for item in getattr(response, "output", None) or []:
                if getattr(item, "type", None) == "reasoning" and not any(
                    existing.get("id") == getattr(item, "id", None)
                    for existing in stream_state.get("reasoning_items", [])
                ):
                    self._capture_reasoning_item(item, stream_state)
        return super().process_stream_chunk(
            chunk, assistant_response, tool_uses, stream_state
        )

    async def process_message(
        self, prompt: str | list, temperature: float = 0, model_id: str | None = None
    ) -> str:
        if model_id is not None and model_id != self.model:
            raise ValueError("xAI Subscription currently supports only grok-4.7")
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        stream = await self.stream_assistant_response(messages)
        text = ""
        try:
            async for event in stream:
                event_type = getattr(event, "type", None)
                self._check_stream_event(event_type)
                if event_type == "response.output_text.delta":
                    text += getattr(event, "delta", "") or ""
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        return text

    async def validate_spec(self, prompt: str) -> str:
        raise NotImplementedError("xAI Subscription structured output is not validated")

    async def get_response(self, response_id: str) -> dict[str, Any]:
        raise NotImplementedError("xAI Subscription does not store Responses")

    async def cancel_response(self, response_id: str) -> bool:
        raise NotImplementedError("xAI Subscription does not use background Responses")

    async def delete_response(self, response_id: str) -> bool:
        raise NotImplementedError("xAI Subscription does not store Responses")
