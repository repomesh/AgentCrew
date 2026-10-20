"""Offline contract tests for experimental xAI subscription integration."""

import json
import multiprocessing
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from AgentCrew.modules.agents.message_metadata import build_responses_metadata
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.service_manager import ServiceManager
from AgentCrew.modules.xai.oauth import XAIAuthenticationError, XAIOAuth
from AgentCrew.modules.xai.response_service import XAISubscriptionResponseService


class Response:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class Client:
    def __init__(self, replies, requests):
        self.replies = replies
        self.requests = requests

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def post(self, url, data):
        self.requests.append((url, data))
        return self.replies.pop(0)


def cross_process_token(path, ready, results):
    ready.set()
    try:
        results.put(XAIOAuth(path).get_valid_access_token())
    except Exception as exc:
        results.put(type(exc).__name__)


@pytest.fixture
def oauth(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_OAUTH_CLIENT_ID", "agentcrew-authorized-test-client")
    return XAIOAuth(tmp_path / "auth.json")


def set_client(monkeypatch, replies, requests=None):
    requests = requests if requests is not None else []
    monkeypatch.setattr(
        "AgentCrew.modules.xai.oauth.httpx2.Client",
        lambda **kwargs: Client(replies, requests),
    )
    return requests


def test_client_id_required(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_OAUTH_CLIENT_ID", raising=False)
    with pytest.raises(XAIAuthenticationError, match="XAI_OAUTH_CLIENT_ID"):
        XAIOAuth(tmp_path / "auth").get_valid_access_token()
    with pytest.raises(XAIAuthenticationError, match="XAI_OAUTH_CLIENT_ID"):
        XAIOAuth(tmp_path / "auth").login()


def test_device_pending_slowdown_success(oauth, monkeypatch):
    responses = [
        Response(
            200,
            {
                "device_code": "private-device",
                "user_code": "public-code",
                "verification_uri": "https://auth.x.ai/verify",
                "expires_in": 400,
                "interval": 1,
            },
        ),
        Response(400, {"error": "authorization_pending"}),
        Response(400, {"error": "slow_down"}),
        Response(
            200,
            {
                "access_token": "private-access",
                "refresh_token": "private-refresh",
                "expires_in": 3600,
            },
        ),
    ]
    requests = set_client(monkeypatch, responses)
    sleeps = []
    monkeypatch.setattr("AgentCrew.modules.xai.oauth.time.sleep", sleeps.append)
    displays = []
    assert oauth.login(display=displays.append)
    assert sleeps == [1, 1, 6]
    assert len(displays) == 1 and "public-code" in displays[0]
    assert oauth.get_valid_access_token() == "private-access"
    assert requests[0][1]["client_id"] == "agentcrew-authorized-test-client"
    assert requests[-1][1]["device_code"] == "private-device"
    assert (
        json.loads(oauth.token_path.read_text())["refresh_token"] == "private-refresh"
    )
    if os.name == "posix":
        assert stat.S_IMODE(oauth.token_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(oauth.token_path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "error,message",
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
        ("unexpected", "failed"),
    ],
)
def test_device_errors(oauth, monkeypatch, error, message):
    set_client(
        monkeypatch,
        [
            Response(
                200,
                {
                    "device_code": "device",
                    "user_code": "code",
                    "verification_uri": "https://auth.x.ai/verify",
                    "expires_in": 60,
                    "interval": 1,
                },
            ),
            Response(400, {"error": error, "refresh_token": "secret-response"}),
        ],
    )
    monkeypatch.setattr("AgentCrew.modules.xai.oauth.time.sleep", lambda _: None)
    with pytest.raises(XAIAuthenticationError, match=message) as exc:
        oauth.login(display=lambda _: None)
    assert "secret-response" not in str(exc.value)


def test_device_cancellation_and_malformed_response(oauth, monkeypatch):
    set_client(monkeypatch, [Response(200, {"device_code": "a"})])
    with pytest.raises(XAIAuthenticationError, match="device authorization failed"):
        oauth.login(display=lambda _: None)
    set_client(
        monkeypatch,
        [
            Response(
                200,
                {
                    "device_code": "a",
                    "user_code": "b",
                    "verification_uri": "https://auth.x.ai/verify",
                    "expires_in": 60,
                },
            )
        ],
    )
    monkeypatch.setattr(
        "AgentCrew.modules.xai.oauth.time.sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        oauth.login(display=lambda _: None)
    assert not oauth.token_path.exists()


def test_device_rejects_malformed_token_without_persisting(oauth, monkeypatch):
    set_client(
        monkeypatch,
        [
            Response(
                200,
                {
                    "device_code": "device",
                    "user_code": "code",
                    "verification_uri": "https://auth.x.ai/verify",
                    "expires_in": 60,
                },
            ),
            Response(200, {"access_token": "secret", "expires_in": 3600}),
        ],
    )
    monkeypatch.setattr("AgentCrew.modules.xai.oauth.time.sleep", lambda _: None)
    with pytest.raises(XAIAuthenticationError, match="invalid OAuth token response"):
        oauth.login(display=lambda _: None)
    assert not oauth.token_path.exists()


def test_refresh_rotation_single_flight(oauth, monkeypatch):
    oauth._save_tokens(
        {
            "access_token": "old",
            "refresh_token": "old-refresh",
            "expires_at": time.time() - 100,
        }
    )
    requests = set_client(
        monkeypatch,
        [
            Response(
                200,
                {"access_token": "new", "refresh_token": "rotated", "expires_in": 3600},
            )
        ],
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(
            pool.map(
                lambda _: XAIOAuth(oauth.token_path).get_valid_access_token(), range(8)
            )
        )
    assert tokens == ["new"] * 8
    assert len(requests) == 1
    assert json.loads(oauth.token_path.read_text())["refresh_token"] == "rotated"


def test_cross_process_refresh_reloads_rotated_file(oauth):
    oauth._save_tokens(
        {"access_token": "old", "refresh_token": "old-refresh", "expires_at": 1}
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    with oauth._refresh_lock():
        process = context.Process(
            target=cross_process_token, args=(oauth.token_path, ready, results)
        )
        process.start()
        assert ready.wait(10)
        oauth._save_tokens(
            {
                "access_token": "new",
                "refresh_token": "rotated-refresh",
                "expires_at": time.time() + 3600,
            }
        )
    process.join(10)
    assert process.exitcode == 0
    assert results.get(timeout=5) == "new"


def test_refresh_failure_does_not_disclose_secrets(oauth, monkeypatch):
    oauth._save_tokens(
        {
            "access_token": "secret-access",
            "refresh_token": "secret-refresh",
            "expires_at": 1,
        }
    )
    set_client(
        monkeypatch,
        [Response(403, {"error": "invalid_grant", "refresh_token": "secret-response"})],
    )
    logs = []
    sink = logger.add(logs.append)
    try:
        with pytest.raises(XAIAuthenticationError, match="xai-auth") as exc:
            oauth.get_valid_access_token()
    finally:
        logger.remove(sink)
    assert not any(
        secret in str(entry)
        for entry in logs
        for secret in ("secret-access", "secret-refresh", "secret-response")
    )
    assert not any(
        secret in str(exc.value)
        for secret in ("secret-access", "secret-refresh", "secret-response")
    )


@pytest.mark.asyncio
async def test_routing_request_and_stream_events(oauth, monkeypatch):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    monkeypatch.setattr(
        "AgentCrew.modules.xai.oauth.DEFAULT_TOKEN_PATH", oauth.token_path
    )
    manager = ServiceManager.get_instance()
    service = manager.get_service("xai_subscription")
    assert manager.get_service("xai_subscription") is service
    standalone = manager.initialize_standalone_service("xai_subscription")
    assert standalone is not service
    assert standalone.provider_name == "xai_subscription"
    model = ModelRegistry.get_instance().get_model("xai_subscription/grok-4.7")
    assert model and model.resolved_service_name() == "xai_subscription"
    assert manager.get_service_for_model(model) is service
    assert model.capabilities == ["tool_use", "thinking", "stream"]
    with pytest.raises(ValueError, match="cannot be disabled"):
        service.set_think("none")
    service.register_tool(
        {"type": "function", "name": "calc", "parameters": {"type": "object"}},
        lambda _: None,
    )
    create = AsyncMock(return_value="stream")
    service.client.responses.create = create
    assert (
        await service.stream_assistant_response([{"role": "user", "content": "sum"}])
        == "stream"
    )
    params = create.call_args.kwargs
    assert params["reasoning"] == {"effort": "high"}
    assert params["store"] is False and params["include"] == [
        "reasoning.encrypted_content"
    ]
    assert params["tools"][0]["name"] == "calc"
    assert "service_tier" not in params and "temperature" not in params
    assert service.client.max_retries == 0
    assert str(service.client.base_url).rstrip("/") == "https://api.x.ai/v1"
    state = service.create_stream_state()
    tools = []
    text = ""
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_text.delta", delta="hi"),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(type="function_call", call_id="call_1", name="calc"),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=1, delta='{"x":'
        ),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            output_index=1,
            arguments='{"x":1}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call", call_id="call_1", name="calc", arguments='{"x":1}'
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=2,
            item=SimpleNamespace(
                type="reasoning",
                id="reason_1",
                encrypted_content="opaque",
                summary=[{"type": "summary_text", "text": "reason"}],
                content=None,
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=5,
                    input_tokens_details=SimpleNamespace(cached_tokens=2),
                    output_tokens_details=None,
                )
            ),
        ),
    ]
    for event in events:
        text, tools, usage, _, _ = service.process_stream_chunk(
            event, text, tools, state
        )
    assert text == "hi" and tools[0]["id"] == "call_1" and tools[0]["input"] == {"x": 1}
    assert (
        usage.cached_tokens == 2
        and state["reasoning_items"][0]["encrypted_content"] == "opaque"
    )
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "_metadata": build_responses_metadata(
            "xai_subscription", "grok-4.7", state["reasoning_items"]
        ),
    }
    assert (
        service._convert_internal_format([message])[0]["encrypted_content"] == "opaque"
    )
    message["_metadata"] = build_responses_metadata(
        "openai_codex", "grok-4.7", state["reasoning_items"]
    )
    assert service._convert_internal_format([message])[0]["role"] == "assistant"
    await service.close()
    await standalone.close()
    manager.services.pop("xai_subscription", None)


@pytest.mark.asyncio
async def test_tool_only_multiple_calls_null_usage_and_mismatched_model(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    text, tools = "", []
    state = service.create_stream_state()
    events = [
        SimpleNamespace(type="response.output_item.added", output_index=0, item=None),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=0,
            item=SimpleNamespace(
                type="function_call", call_id="id-a", name="one", arguments='{"a":1}'
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call", call_id="id-b", name="two", arguments='{"b":2}'
            ),
        ),
        SimpleNamespace(
            type="response.completed", response=SimpleNamespace(usage=None)
        ),
    ]
    for event in events:
        text, tools, usage, _, _ = service.process_stream_chunk(
            event, text, tools, state
        )
    assert not text and usage.input_tokens == 0
    assert [(tool["id"], tool["input"]) for tool in tools] == [
        ("id-a", {"a": 1}),
        ("id-b", {"b": 2}),
    ]
    msg = {
        "role": "assistant",
        "content": "hello",
        "_metadata": build_responses_metadata(
            "xai_subscription",
            "other-model",
            [{"type": "reasoning", "encrypted_content": "opaque"}],
        ),
    }
    assert service._convert_internal_format([msg]) == [
        {"role": "assistant", "content": "hello"}
    ]
    await service.close()


@pytest.mark.asyncio
async def test_incomplete_stream_and_terminal_only_reasoning(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    state = service.create_stream_state()
    reason = SimpleNamespace(
        type="reasoning",
        id="reason",
        encrypted_content="opaque",
        summary=[],
        content=None,
    )
    event = SimpleNamespace(
        type="response.completed", response=SimpleNamespace(usage=None, output=[reason])
    )
    service.process_stream_chunk(event, "", [], state)
    service.process_stream_chunk(event, "", [], state)
    assert len(state["reasoning_items"]) == 1
    for kind in ("response.incomplete", "response.failed", "error"):
        with pytest.raises(RuntimeError, match="not replayed"):
            service.process_stream_chunk(SimpleNamespace(type=kind), "", [], state)
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 503])
async def test_http_failure_never_replays_request(oauth, status):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    create = AsyncMock(side_effect=RuntimeError(f"HTTP {status}"))
    service.client.responses.create = create
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        await service.stream_assistant_response([{"role": "user", "content": "hi"}])
    create.assert_awaited_once()
    assert service.client.max_retries == 0
    await service.close()


@pytest.mark.asyncio
async def test_one_shot_stream_closes_on_error(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    closed = []

    async def events():
        try:
            yield SimpleNamespace(type="response.output_text.delta", delta="hi")
            raise RuntimeError("stream broken")
        finally:
            closed.append(True)

    service.client.responses.create = AsyncMock(return_value=events())
    with pytest.raises(RuntimeError, match="stream broken"):
        await service.process_message("hello")
    assert closed == [True]
    with pytest.raises(ValueError, match="only grok-4.7"):
        await service.process_message("hello", model_id="other")
    await service.close()


def test_no_key_detection_and_cli_choices(oauth, monkeypatch):
    from AgentCrew.modules.config.global_config import GlobalConfig
    from AgentCrew.setup import PROVIDER_LIST, ApplicationSetup, common_options

    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    monkeypatch.setattr(
        "AgentCrew.modules.xai.oauth.DEFAULT_TOKEN_PATH", oauth.token_path
    )
    monkeypatch.setattr(
        GlobalConfig, "get_last_used_provider", lambda _: "xai_subscription"
    )
    assert "xai_subscription" in PROVIDER_LIST
    assert XAIOAuth.is_available()
    assert ApplicationSetup.detect_provider(object()) == "xai_subscription"
    assert any(
        "xai_subscription" in str(getattr(option.type, "choices", ()))
        for option in common_options(lambda: None).__click_params__
    )
    monkeypatch.setenv("AGENTCREW_PROVIDER", "openai")
    assert ApplicationSetup.detect_provider(object()) == "openai"
