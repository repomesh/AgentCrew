import pytest

from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.types import Model
from AgentCrew.modules.utils.vision_preprocessing import (
    VisionDescriptionCache,
    VisionPreprocessingUtils,
    build_vision_cache_key,
    fingerprint_image_url,
    normalize_remote_url,
)


class StubLLM:
    provider_name = "test_provider"
    model = "text-model"

    def __init__(self):
        self.describe_calls = 0
        self.seen_model_ids = []

    async def process_message(self, prompt, temperature=0, model_id=None):
        self.describe_calls += 1
        self.seen_model_ids.append(model_id)
        return "A detailed generated description."


@pytest.fixture(autouse=True)
def model_registry_fixture():
    registry = ModelRegistry.get_instance()
    old_models = registry.models.copy()
    old_current = registry.current_model
    registry.register_model(
        Model(
            id="text-model",
            provider="test_provider",
            name="Text Model",
            description="Text model",
            capabilities=["stream"],
            vision_model="vision-model",
        )
    )
    registry.register_model(
        Model(
            id="vision-model",
            provider="test_provider",
            name="Vision Model",
            description="Vision model",
            capabilities=["stream", "vision"],
        )
    )
    yield
    registry.models = old_models
    registry.current_model = old_current


def test_fingerprint_data_url_uses_decoded_bytes():
    first = fingerprint_image_url("data:image/png;base64,aGVsbG8=")
    second = fingerprint_image_url("data:image/png;base64,aGVsbG8=")

    assert first == second
    assert first["image_source_type"] == "data_url"
    assert first["image_mime_type"] == "image/png"


def test_remote_url_normalization_removes_fragment_and_sorts_query():
    normalized = normalize_remote_url("HTTPS://Example.com/image.png?b=2&a=1#frag")

    assert normalized == "https://example.com/image.png?a=1&b=2"


def test_cache_key_changes_when_vision_model_changes():
    key_a = build_vision_cache_key(
        image_fingerprint="sha256:image",
        provider="provider",
        vision_model="vision-a",
    )
    key_b = build_vision_cache_key(
        image_fingerprint="sha256:image",
        provider="provider",
        vision_model="vision-b",
    )

    assert key_a != key_b


def test_cache_round_trip(tmp_path):
    cache = VisionDescriptionCache(base_path=tmp_path)
    cache.set("sha256:test", {"description": "cached"})

    assert cache.get("sha256:test")["description"] == "cached"


@pytest.mark.asyncio
async def test_preprocess_replaces_image_and_uses_current_llm_service_cache(tmp_path):
    cache = VisionDescriptionCache(base_path=tmp_path)
    llm = StubLLM()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                },
            ],
        }
    ]

    await VisionPreprocessingUtils.preprocess_messages(messages, llm, cache)

    assert messages[0]["content"][1]["type"] == "text"
    assert "A detailed generated description." in messages[0]["content"][1]["text"]
    assert llm.describe_calls == 1
    assert llm.seen_model_ids == ["vision-model"]
