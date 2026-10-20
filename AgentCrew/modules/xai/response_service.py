"""Experimental direct xAI subscription OAuth Responses adapter."""

import asyncio
from dataclasses import dataclass
from typing import Any

from AgentCrew.modules.agents.message_metadata import (
    get_responses_provider_state,
    strip_internal_metadata,
)
from AgentCrew.modules.llm.base import BaseLLMService
from AgentCrew.modules.llm.token_usage import TokenUsage
from AgentCrew.modules.openai.response_service import OpenAIResponseService
from AgentCrew.modules.xai.oauth import XAIOAuth

XAI_BASE_URL = "https://api.x.ai/v1"


@dataclass(frozen=True)
class _ModelProfile:
    default_effort: str | None
    efforts: frozenset[str]
    encrypted_reasoning: bool = False


_MODEL_PROFILES = {
    "grok-4.7": _ModelProfile(
        "high", frozenset({"low", "medium", "high", "xhigh"}), True
    ),
    "grok-4.3": _ModelProfile("low", frozenset({"none", "low", "medium", "high"})),
    "grok-build-0.1": _ModelProfile(None, frozenset()),
}


class _CompletedResponsesStream:
    """Require terminal completion only when a Responses stream ends naturally."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._completed = False
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            if not self._completed:
                raise RuntimeError(
                    "xAI Responses stream ended without completion; request was not replayed"
                ) from None
            raise
        if getattr(event, "type", None) == "response.completed":
            self._completed = True
        return event

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        exit_stream = getattr(self._stream, "__aexit__", None)
        if exit_stream is not None:
            return await exit_stream(exc_type, exc_val, exc_tb)
        await self.aclose()
        return False

    async def aclose(self):
        self._closed = True
        close = getattr(self._stream, "aclose", None) or getattr(
            self._stream, "close", None
        )
        if close is not None:
            await close()


class XAISubscriptionResponseService(OpenAIResponseService):
    """Reuse Responses history/event orchestration, without OpenAI-only options."""

    strict_reasoning_selection = True

    def __init__(self, token_path: str | None = None):
        self._oauth = XAIOAuth(token_path)
        token = self._oauth.get_valid_access_token()
        super().__init__(api_key=token, base_url=XAI_BASE_URL)
        self.client = self.client.with_options(max_retries=0)
        self._provider_name = "xai"
        self._profiles_ready = True
        self.model = "grok-4.7"

    @property
    def model(self) -> str:
        return BaseLLMService.model.fget(self)

    @model.setter
    def model(self, value: str) -> None:
        if not getattr(self, "_profiles_ready", False):
            self._model = value
            return
        profile = _MODEL_PROFILES.get(value)
        if profile is None:
            raise ValueError(f"Unsupported xAI subscription Responses model: {value}")
        self._model = value
        self.reasoning_effort = profile.default_effort

    @property
    def _profile(self) -> _ModelProfile:
        return _MODEL_PROFILES[self.model]

    @property
    def allow_unconfigured_reasoning(self) -> bool:
        return not self._profile.efforts

    async def _ensure_valid_token(self) -> None:
        self.client.api_key = await asyncio.to_thread(
            self._oauth.get_valid_access_token
        )

    def set_think(self, budget_tokens: str) -> bool:
        profile = self._profile
        if budget_tokens not in profile.efforts:
            if self.model == "grok-4.7" and budget_tokens in ("none", "0"):
                raise ValueError("Grok 4.7 reasoning cannot be disabled")
            if not profile.efforts:
                raise ValueError(f"{self.model} reasoning effort is not configurable")
            allowed = ", ".join(sorted(profile.efforts))
            raise ValueError(f"{self.model} reasoning effort must be {allowed}")
        self.reasoning_effort = budget_tokens
        return True

    def _convert_internal_format(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self._profile.encrypted_reasoning:
            return super()._convert_internal_format(strip_internal_metadata(messages))
        filtered = []
        for message in messages:
            state = get_responses_provider_state(message)
            if state and (
                state.get("provider") != self.provider_name
                or state.get("model") != self.model
            ):
                filtered.extend(strip_internal_metadata([message]))
            else:
                filtered.append(message)
        return super()._convert_internal_format(filtered)

    def _capture_reasoning_item(
        self, item: Any, stream_state: dict[str, Any] | None
    ) -> None:
        if (
            self._profile.encrypted_reasoning
            and stream_state is not None
            and stream_state.get("model") == self.model
        ):
            super()._capture_reasoning_item(item, stream_state)

    async def stream_assistant_response(self, messages: list[dict[str, Any]]) -> Any:
        await self._ensure_valid_token()
        params = {
            "model": self.model,
            "input": self._convert_internal_format(messages),
            "instructions": self.system_prompt or "You are a helpful assistant.",
            "stream": True,
            "store": False,
        }
        if self._profile.encrypted_reasoning:
            params["include"] = ["reasoning.encrypted_content"]
        if self.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.reasoning_effort}
        if self.tools:
            params["tools"] = [
                tool for tool in self.tools if tool.get("type") == "function"
            ]
        return _CompletedResponsesStream(await self.client.responses.create(**params))

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
            self.model = model_id
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        text = ""
        async with await self.stream_assistant_response(messages) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                self._check_stream_event(event_type)
                if event_type == "response.output_text.delta":
                    text += getattr(event, "delta", "") or ""
        return text

    async def validate_spec(self, prompt: str) -> str:
        raise NotImplementedError("xAI Subscription structured output is not validated")

    async def get_response(self, response_id: str) -> dict[str, Any]:
        raise NotImplementedError("xAI Subscription does not store Responses")

    async def cancel_response(self, response_id: str) -> bool:
        raise NotImplementedError("xAI Subscription does not use background Responses")

    async def delete_response(self, response_id: str) -> bool:
        raise NotImplementedError("xAI Subscription does not store Responses")
