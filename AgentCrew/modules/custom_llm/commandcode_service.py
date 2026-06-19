from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from .service import CustomLLMService


class CommandCodeService(CustomLLMService):
    """CommandCode-specific LLM service that enforces function-type tool calls.

    CommandCode's API requires tool_calls to use type="function" (the OpenAI
    standard) rather than type="tool_call" (AgentCrew's internal format).
    This service overrides the tool call normalization to always set type to
    "function" regardless of the internal format's type value.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("COMMAND_CODE_API_KEY")
        base_url = os.getenv(
            "COMMAND_CODE_BASE_URL", "https://api.commandcode.ai/provider/v1"
        )
        if not api_key:
            logger.error("COMMAND_CODE_API_KEY not found in environment variables")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            provider_name="commandcode",
            extra_headers={"x-cmd-zdr": "1"},
        )
        self.model = "deepseek/deepseek-v4-pro"
        logger.info("Initialized Command Code Service")

    def _normalize_tool_call_for_request(
        self, raw_tool_call: dict[str, Any]
    ) -> dict[str, Any] | None:
        normalized = super()._normalize_tool_call_for_request(raw_tool_call)
        if normalized is not None:
            normalized["type"] = "function"
        return normalized
