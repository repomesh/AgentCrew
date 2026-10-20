"""Experimental direct xAI subscription OAuth Responses adapter."""

import asyncio
from typing import Any

from AgentCrew.modules.agents.message_metadata import (
    get_responses_provider_state,
    strip_internal_metadata,
)
from AgentCrew.modules.llm.base import BaseLLMService
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.reasoning_selection import REASONING_LEVELS
from AgentCrew.modules.llm.token_usage import TokenUsage
from AgentCrew.modules.llm.types import Model
from AgentCrew.modules.openai.response_service import OpenAIResponseService
from AgentCrew.modules.xai.oauth import XAIOAuth

XAI_BASE_URL = "https://api.x.ai/v1"


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
        self._models_ready = True
        self.model = "grok-4.7"

    @property
    def model(self) -> str:
        return BaseLLMService.model.fget(self)

    @model.setter
    def model(self, value: str) -> None:
        if not getattr(self, "_models_ready", False):
            self._model = value
            return
        model = self._model_definition(value)
        self._model = value
        self.reasoning_effort = model.default_reasoning

    @staticmethod
    def _model_definition(model_id: str) -> Model:
        model = ModelRegistry.get_instance().get_model(f"xai/{model_id}")
        if model is None or model.provider != "xai" or model.id != model_id:
            raise ValueError(
                f"Unsupported xAI subscription Responses model: {model_id}"
            )
        return model

    @property
    def allow_unconfigured_reasoning(self) -> bool:
        return "thinking" not in self._model_definition(self.model).capabilities

    async def _ensure_valid_token(self) -> None:
        self.client.api_key = await asyncio.to_thread(
            self._oauth.get_valid_access_token
        )

    def set_think(self, budget_tokens: str) -> bool:
        if budget_tokens not in REASONING_LEVELS:
            raise ValueError(
                f"Invalid reasoning effort '{budget_tokens}'. "
                f"Must be one of: {', '.join(REASONING_LEVELS)}"
            )
        if self.allow_unconfigured_reasoning:
            return False
        self.reasoning_effort = budget_tokens
        return True

    def _convert_internal_format(
        self, messages: list[dict[str, Any]], *, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        target_model = self.model if model_id is None else model_id
        model = self._model_definition(target_model)
        if "thinking" not in model.capabilities:
            return super()._convert_internal_format(
                strip_internal_metadata(messages), model_id=target_model
            )
        filtered = []
        for message in messages:
            state = get_responses_provider_state(message)
            if state and (
                state.get("provider") != self.provider_name
                or state.get("model") != target_model
            ):
                filtered.extend(strip_internal_metadata([message]))
            else:
                filtered.append(message)
        return super()._convert_internal_format(filtered, model_id=target_model)

    def _capture_reasoning_item(
        self, item: Any, stream_state: dict[str, Any] | None
    ) -> None:
        if (
            "thinking" in self._model_definition(self.model).capabilities
            and stream_state is not None
            and stream_state.get("provider") == self.provider_name
            and stream_state.get("model") == self.model
        ):
            super()._capture_reasoning_item(item, stream_state)

    async def _create_responses_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        reasoning_effort: str | None,
    ) -> _CompletedResponsesStream:
        model = self._model_definition(model_id)
        await self._ensure_valid_token()
        params = {
            "model": model_id,
            "input": self._convert_internal_format(messages, model_id=model_id),
            "instructions": self.system_prompt or "You are a helpful assistant.",
            "stream": True,
            "store": False,
        }
        if "thinking" in model.capabilities:
            params["include"] = ["reasoning.encrypted_content"]
            if reasoning_effort is not None:
                params["reasoning"] = {"effort": reasoning_effort}
        if self.tools:
            params["tools"] = [
                tool for tool in self.tools if tool.get("type") == "function"
            ]
        return _CompletedResponsesStream(await self.client.responses.create(**params))

    async def stream_assistant_response(self, messages: list[dict[str, Any]]) -> Any:
        return await self._create_responses_stream(
            messages, model_id=self.model, reasoning_effort=self.reasoning_effort
        )

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
        target_model = self.model if model_id is None else model_id
        model = self._model_definition(target_model)
        reasoning_effort = (
            self.reasoning_effort
            if target_model == self.model
            else model.default_reasoning
        )
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        text = ""
        async with await self._create_responses_stream(
            messages, model_id=target_model, reasoning_effort=reasoning_effort
        ) as stream:
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
