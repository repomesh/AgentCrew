"""Curated experimental xAI Subscription Responses models."""

from AgentCrew.modules.llm.types import Model

XAI_SUBSCRIPTION_MODELS = [
    Model(
        id="grok-4.7",
        provider="xai",
        name="Grok 4.7 (xAI Subscription)",
        description="Experimental xAI subscription OAuth via direct Responses API",
        capabilities=["tool_use", "thinking", "stream"],
        default_reasoning="high",
        max_context_token=256_000,
        input_token_price_1m=0.0,
        cached_token_price_1m=0.0,
        output_token_price_1m=0.0,
    ),
]
