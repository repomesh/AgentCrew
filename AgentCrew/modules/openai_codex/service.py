from typing import Optional, Any

from loguru import logger

from AgentCrew.modules.openai import OpenAIResponseService
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.openai_codex.oauth import OpenAICodexOAuth

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."


class OpenAICodexService(OpenAIResponseService):
    def __init__(self, token_path: Optional[str] = None):
        self._oauth = OpenAICodexOAuth(token_path=token_path)

        access_token = self._oauth.get_valid_access_token()
        if not access_token:
            raise ValueError(
                "No valid OpenAI Codex OAuth token found. "
                "Run 'agentcrew chatgpt-auth' to authenticate with your ChatGPT subscription."
            )

        super().__init__(
            api_key=access_token,
            base_url=CODEX_BASE_URL,
        )
        self._provider_name = "openai_codex"
        self.model = "gpt-5-codex"
        logger.info("Initialized OpenAI Codex Service (ChatGPT subscription)")

    def _ensure_valid_token(self):
        new_token = self._oauth.get_valid_access_token()
        if new_token and new_token != self.client.api_key:
            self.client.api_key = new_token
            logger.debug("Refreshed OAuth access token for Codex service")
        elif not new_token:
            logger.warning(
                "OAuth token expired and could not be refreshed. "
                "Re-run 'agentcrew chatgpt-auth' to re-authenticate."
            )

    async def process_message(self, prompt: str, temperature: float = 0) -> str:
        self._ensure_valid_token()
        request_params = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "stream": True,
            "store": False,
            "instructions": self.system_prompt or DEFAULT_CODEX_INSTRUCTIONS,
        }
        if self._extra_headers:
            request_params["extra_headers"] = self._extra_headers

        if self.reasoning_effort and "thinking" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        result_text = ""
        input_tokens = 0
        output_tokens = 0

        async for event in await self.client.responses.create(**request_params):
            if event.type == "response.output_text.delta":
                result_text += event.delta
            elif event.type == "response.completed":
                usage = getattr(event.response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)

        total_cost = self.calculate_cost(input_tokens, output_tokens)
        logger.info("\nCodex Response API Token Usage Statistics:")
        logger.info(f"Input tokens: {input_tokens:,}")
        logger.info(f"Output tokens: {output_tokens:,}")
        logger.info(f"Total tokens: {input_tokens + output_tokens:,}")
        logger.info(f"Estimated cost: ${total_cost:.4f}")

        return result_text

    async def stream_assistant_response(self, messages) -> Any:
        self._ensure_valid_token()

        input_data = self._convert_internal_format(messages)
        full_model_id = f"{self._provider_name}/{self.model}"

        stream_params = {
            "model": self.model,
            "input": input_data,
            "stream": True,
            "instructions": self.system_prompt or DEFAULT_CODEX_INSTRUCTIONS,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }

        forced_sample_params = ModelRegistry.get_model_sample_params(full_model_id)
        if forced_sample_params:
            if forced_sample_params.temperature is not None:
                stream_params["temperature"] = forced_sample_params.temperature
            if forced_sample_params.top_p is not None:
                stream_params["top_p"] = forced_sample_params.top_p

        if "thinking" in ModelRegistry.get_model_capabilities(full_model_id):
            if self.reasoning_effort:
                stream_params["reasoning"] = {"effort": self.reasoning_effort}

        if self._extra_headers:
            stream_params["extra_headers"] = self._extra_headers

        if self.tools and "tool_use" in ModelRegistry.get_model_capabilities(
            full_model_id
        ):
            stream_params["tools"] = self.tools.copy()

        if (
            "structured_output" in ModelRegistry.get_model_capabilities(full_model_id)
            and self.structured_output
        ):
            stream_params["text"] = {
                "format": {
                    "name": "default",
                    "type": "json_schema",
                    "json_schema": self.structured_output,
                }
            }

        return await self.client.responses.create(**stream_params)
