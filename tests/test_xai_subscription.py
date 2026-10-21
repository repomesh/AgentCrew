"""Offline contract tests for experimental xAI subscription integration."""

import asyncio
import json
import multiprocessing
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from loguru import logger

from AgentCrew.modules.agents.agent_response_stream import AgentResponseStream
from AgentCrew.modules.agents.message_formatter import AgentMessageFormatter
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


class EventStream:
    """Fake SSE stream with a context manager and explicit close accounting."""

    def __init__(self, events):
        self.events = list(events)
        self.closes = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def close(self):
        self.closes += 1


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
    service = manager.get_service("xai")
    assert manager.get_service("xai") is service
    assert [name for name in manager.service_factories if name.startswith("xai")] == [
        "xai"
    ]
    standalone = manager.initialize_standalone_service("xai")
    assert standalone is not service
    assert standalone.provider_name == "xai"
    model = ModelRegistry.get_instance().get_model("xai/grok-4.7")
    assert model and model.resolved_service_name() == "xai"
    assert manager.get_service_for_model(model) is service
    assert model.capabilities == ["tool_use", "thinking", "stream"]
    assert model.max_context_token == 256_000
    assert (
        model.input_token_price_1m,
        model.output_token_price_1m,
        model.cached_token_price_1m,
    ) == (0.0, 0.0, 0.0)
    assert service.set_think("none")
    assert service.reasoning_effort == "none"
    assert service.set_think(model.default_reasoning)
    service.register_tool(
        {"type": "function", "name": "calc", "parameters": {"type": "object"}},
        lambda _: None,
    )
    raw_stream = EventStream(
        [SimpleNamespace(type="response.completed", response=None)]
    )
    create = AsyncMock(return_value=raw_stream)
    service.client.responses.create = create
    async with await service.stream_assistant_response(
        [{"role": "user", "content": "sum"}]
    ) as stream:
        assert [event.type async for event in stream] == ["response.completed"]
    assert raw_stream.closes == 1
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
    assert state["provider"] == "xai"
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
            "xai", "grok-4.7", state["reasoning_items"]
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
    manager.services.pop("xai", None)


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
            "xai",
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
    with pytest.raises(
        ValueError, match="Unsupported xAI subscription Responses model: other"
    ):
        await service.process_message("hello", model_id="other")
    await service.close()


def test_no_key_detection_and_cli_choices(oauth, monkeypatch):
    from AgentCrew.main import cli
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
    monkeypatch.setattr(GlobalConfig, "get_last_used_provider", lambda _: "xai")
    assert "xai" in PROVIDER_LIST
    assert XAIOAuth.is_available()
    assert ApplicationSetup.detect_provider(object()) == "xai"
    cli_options = common_options(lambda: None).__click_params__
    for name in ("provider", "memory_llm"):
        option = next(option for option in cli_options if option.name == name)
        assert "xai" in option.type.choices
    create_agent_provider = next(
        option
        for option in cli.commands["create-agent"].params
        if option.name == "provider"
    )
    assert "xai" in create_agent_provider.type.choices
    monkeypatch.setenv("AGENTCREW_PROVIDER", "openai")
    assert ApplicationSetup.detect_provider(object()) == "openai"


@pytest.mark.parametrize("level", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_xai_reasoning_levels_accept(oauth, level):
    from AgentCrew.modules.llm.reasoning_selection import apply_reasoning_to_service

    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    assert apply_reasoning_to_service(service, level, explicit=False)
    assert service.reasoning_effort == level


@pytest.mark.parametrize("level", ["0", "ultra"])
def test_xai_reasoning_levels_reject(oauth, level):
    from AgentCrew.modules.llm.reasoning_selection import apply_reasoning_to_service

    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    with pytest.raises(ValueError, match="Invalid reasoning effort"):
        apply_reasoning_to_service(service, level, explicit=False)
    assert service.reasoning_effort == "high"


def test_logout_corrupt_cache_and_cli_recovery(oauth, monkeypatch):
    from AgentCrew.main import cli

    oauth.token_path.write_text("{broken-json-secret")
    with pytest.raises(XAIAuthenticationError, match="--logout"):
        XAIOAuth(oauth.token_path)
    monkeypatch.setattr(
        "AgentCrew.modules.xai.oauth.DEFAULT_TOKEN_PATH", oauth.token_path
    )
    monkeypatch.delenv("XAI_OAUTH_CLIENT_ID")
    runner = CliRunner()
    assert "xai-auth" in cli.commands
    result = runner.invoke(cli.commands["xai-auth"], ["--logout"])
    assert result.exit_code == 0
    assert "Remote revocation is not supported" in result.output
    assert "broken-json-secret" not in result.output
    assert not oauth.token_path.exists()
    assert runner.invoke(cli.commands["xai-auth"], ["--logout"]).exit_code == 0
    monkeypatch.setenv("XAI_OAUTH_CLIENT_ID", "agentcrew-authorized-test-client")
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
            Response(
                200,
                {
                    "access_token": "access-secret",
                    "refresh_token": "refresh-secret",
                    "expires_in": 3600,
                },
            ),
        ],
    )
    monkeypatch.setattr("AgentCrew.modules.xai.oauth.time.sleep", lambda _: None)
    assert runner.invoke(cli.commands["xai-auth"], []).exit_code == 0
    assert XAIOAuth(oauth.token_path).has_valid_tokens
    assert "access-secret" not in result.output
    if os.name == "posix":
        assert stat.S_IMODE(oauth.token_path.stat().st_mode) == 0o600
        assert (
            stat.S_IMODE(
                oauth.token_path.with_name(oauth.token_path.name + ".lock")
                .stat()
                .st_mode
            )
            == 0o600
        )


@pytest.mark.skipif(
    os.name == "nt", reason="symlinks may require Windows developer mode"
)
@pytest.mark.parametrize("kind", ["parent", "token", "lock"])
def test_oauth_rejects_symlinked_paths(tmp_path, kind):
    real = tmp_path / "real"
    real.mkdir()
    destination = real / "auth.json"
    destination.write_text("unchanged")
    parent = tmp_path / "credentials"
    if kind == "parent":
        parent.symlink_to(real, target_is_directory=True)
        path = parent / "auth.json"
    else:
        parent.mkdir()
        path = parent / "auth.json"
        if kind == "token":
            path.symlink_to(destination)
        else:
            path.with_name(path.name + ".lock").symlink_to(destination)
    with pytest.raises(XAIAuthenticationError, match="Unsafe"):
        XAIOAuth(path, load_tokens=False).logout()
    with pytest.raises(XAIAuthenticationError, match="Unsafe"):
        XAIOAuth(path)
    assert destination.read_text() == "unchanged"


@pytest.mark.parametrize("kind", ["parent", "token", "lock"])
def test_oauth_rejects_non_regular_paths(tmp_path, kind):
    parent = tmp_path / "credentials"
    if kind == "parent":
        parent.write_text("unchanged")
    else:
        parent.mkdir()
    path = parent / "auth.json"
    if kind == "token":
        path.mkdir()
    elif kind == "lock":
        path.with_name(path.name + ".lock").mkdir()
    with pytest.raises(XAIAuthenticationError, match="Unsafe"):
        XAIOAuth(path, load_tokens=False).logout()
    with pytest.raises(XAIAuthenticationError, match="Unsafe"):
        XAIOAuth(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events,should_fail",
    [
        ([], True),
        (["response.output_text.delta"], True),
        (["response.completed"], False),
    ],
)
async def test_stream_requires_natural_terminal(oauth, events, should_fail):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    raw = EventStream(
        [SimpleNamespace(type=kind, delta="partial", response=None) for kind in events]
    )
    service.client.responses.create = AsyncMock(return_value=raw)

    async def consume():
        async with await service.stream_assistant_response(
            [{"role": "user", "content": "hi"}]
        ) as stream:
            return [event.type async for event in stream]

    if should_fail:
        with pytest.raises(
            RuntimeError, match="without completion; request was not replayed"
        ):
            await consume()
    else:
        assert await consume() == events
    assert raw.closes == 1
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
async def test_explicit_terminal_error_closes_without_replay(oauth, kind):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    raw = EventStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            SimpleNamespace(type=kind),
        ]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    with pytest.raises(
        RuntimeError, match="failed or incomplete; request was not replayed"
    ):
        async with await service.stream_assistant_response([]) as stream:
            async for event in stream:
                service.process_stream_chunk(
                    event, "", [], service.create_stream_state()
                )
    assert raw.closes == 1
    raw = EventStream([SimpleNamespace(type=kind)])
    service.client.responses.create = AsyncMock(return_value=raw)
    with pytest.raises(
        RuntimeError, match="failed or incomplete; request was not replayed"
    ):
        await service.process_message("hi")
    assert raw.closes == 1
    await service.close()


@pytest.mark.asyncio
async def test_intentional_early_close_and_process_message_eof(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    raw = EventStream(
        [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    async with await service.stream_assistant_response([]) as stream:
        assert (await anext(stream)).delta == "partial"
    assert raw.closes == 1
    raw = EventStream(
        [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    stream = await service.stream_assistant_response([])
    await stream.aclose()
    assert raw.closes == 1
    raw = EventStream(
        [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    with pytest.raises(RuntimeError, match="without completion"):
        await service.process_message("hi")
    assert raw.closes == 1
    raw = EventStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="done"),
            SimpleNamespace(type="response.completed", response=None),
        ]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    assert await service.process_message("hi") == "done"
    assert raw.closes == 1
    await service.close()


@pytest.mark.asyncio
async def test_two_turn_local_tools_replay_encrypted_reasoning(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    service = XAISubscriptionResponseService(token_path=str(oauth.token_path))
    service.register_tool(
        {"type": "function", "name": "calc", "parameters": {"type": "object"}},
        lambda _: None,
    )
    encrypted = "opaque-continuation-payload"
    reason = SimpleNamespace(
        type="reasoning",
        id="rs_first",
        encrypted_content=encrypted,
        status="completed",
        summary=[],
        content=None,
    )
    first_events = [
        SimpleNamespace(type="response.output_item.done", output_index=0, item=reason),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call", call_id="call_a", name="calc", arguments='{"a":1}'
            ),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=2,
            item=SimpleNamespace(type="function_call", call_id="call_b", name="calc"),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=2, delta='{"b":'
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=2, delta="2}"
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=2,
            item=SimpleNamespace(
                type="function_call", call_id="call_b", name="calc", arguments='{"b":2}'
            ),
        ),
        SimpleNamespace(
            type="response.completed", response=SimpleNamespace(usage=None)
        ),
    ]
    create = AsyncMock(
        side_effect=[
            EventStream(first_events),
            EventStream(
                [
                    SimpleNamespace(
                        type="response.completed", response=SimpleNamespace(usage=None)
                    )
                ]
            ),
        ]
    )
    service.client.responses.create = create
    user = {"role": "user", "content": "calculate two values"}
    fake_agent = SimpleNamespace(name="fixture", llm=service)
    formatter = AgentMessageFormatter(fake_agent)
    fake_agent.format_message = formatter.format_message
    tool_uses = []

    async def iterator(state):
        text = ""
        async with await service.stream_assistant_response([user]) as raw:
            async for event in raw:
                text, tools, _, _, _ = service.process_stream_chunk(
                    event, text, tool_uses, state
                )
                tool_uses[:] = tools
                yield text, None, None

    stream = AgentResponseStream(fake_agent, service.create_stream_state, iterator)
    async for _ in stream:
        pass
    assert [tool["id"] for tool in tool_uses] == ["call_a", "call_b"]
    assistant = stream.format_assistant_message("", tool_uses=tool_uses)
    assert assistant is not None
    saved_reason = assistant["_metadata"]["provider_state"]["openai_responses"][
        "output_items"
    ][0]
    assert saved_reason["encrypted_content"] == encrypted
    results = [
        formatter.format_tool_result(tool, {"answer": value})
        for tool, value in zip(tool_uses, (1, 2), strict=True)
    ]
    async with await service.stream_assistant_response(
        [user, assistant, *results]
    ) as second:
        assert [event.type async for event in second] == ["response.completed"]
    request = create.call_args_list[1].kwargs
    converted = request["input"]
    assert [item.get("type") for item in converted] == [
        None,
        "reasoning",
        None,
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert converted[1] == saved_reason
    assert [item["call_id"] for item in converted if item.get("call_id")] == [
        "call_a",
        "call_b",
        "call_a",
        "call_b",
    ]
    assert [
        json.loads(item["arguments"])
        for item in converted
        if item.get("type") == "function_call"
    ] == [{"a": 1}, {"b": 2}]
    assert [
        json.loads(item["output"])
        for item in converted
        if item.get("type") == "function_call_output"
    ] == [{"answer": 1}, {"answer": 2}]
    assert request["store"] is False and request["include"] == [
        "reasoning.encrypted_content"
    ]
    assert (
        assistant["_metadata"]["provider_state"]["openai_responses"]["output_items"][0][
            "encrypted_content"
        ]
        == encrypted
    )
    await service.close()


@pytest.fixture
def curated_service(oauth):
    oauth._save_tokens(
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    return XAISubscriptionResponseService(token_path=str(oauth.token_path))


@pytest.mark.asyncio
async def test_curated_models_resolve_and_reapply_defaults(curated_service):
    from AgentCrew.modules.llm.model_selection import (
        ModelSelection,
        ModelSelectionSource,
    )
    from AgentCrew.modules.llm.reasoning_selection import apply_reasoning_to_service

    manager = ServiceManager.get_instance()
    registry = ModelRegistry.get_instance()
    expected = {
        "grok-4.7": ("high", ["tool_use", "thinking", "stream"]),
        "grok-build-0.1": (None, ["tool_use", "stream"]),
        "grok-4.3": ("low", ["tool_use", "thinking", "stream"]),
    }
    assert {model.id for model in registry.get_models_by_provider("xai")} == set(
        expected
    )
    service = curated_service
    try:
        for model_id in ("grok-4.7", "grok-build-0.1", "grok-4.3"):
            model = registry.get_model(f"xai/{model_id}")
            assert model is not None and model.provider == "xai"
            assert model.resolved_service_name() == "xai"
            assert model.capabilities == expected[model_id][1]
            assert model.default_reasoning == expected[model_id][0]
            assert model.max_context_token == 256_000
            assert (
                model.input_token_price_1m,
                model.cached_token_price_1m,
                model.output_token_price_1m,
            ) == (0.0, 0.0, 0.0)
            manager.apply_model_defaults(service, model)
            assert service.model == model_id
            assert service.reasoning_effort == expected[model_id][0]
            assert apply_reasoning_to_service(service, model.default_reasoning)
            selection = ModelSelection.from_model_id(
                f"xai/{model_id}", ModelSelectionSource.RUNTIME_ARGS
            )
            dedicated = manager.get_service_for_selection(selection, standalone=True)
            assert dedicated.model == model_id
            assert dedicated.reasoning_effort == expected[model_id][0]
            await dedicated.close()
        service.set_think("medium")
        manager.apply_model_defaults(service, registry.get_model("xai/grok-4.7"))
        assert service.reasoning_effort == "high"
        assert registry.get_model("xai/grok-4.7").default
        assert not registry.get_model("xai/grok-4.3").default
    finally:
        await service.close()


@pytest.mark.parametrize(
    ("model_id", "thinking"),
    [("grok-4.7", True), ("grok-4.3", True), ("grok-build-0.1", False)],
)
@pytest.mark.asyncio
async def test_curated_model_reasoning_contract(curated_service, model_id, thinking):
    from AgentCrew.modules.llm.reasoning_selection import (
        REASONING_LEVELS,
        apply_reasoning_to_service,
    )

    service = curated_service
    service.model = model_id
    assert service.allow_unconfigured_reasoning is not thinking
    if thinking:
        for level in REASONING_LEVELS:
            assert apply_reasoning_to_service(service, level, explicit=True)
            assert service.reasoning_effort == level
    else:
        assert apply_reasoning_to_service(service, None)
        for level in REASONING_LEVELS:
            assert service.set_think(level) is False
            with pytest.raises(ValueError, match="not supported"):
                apply_reasoning_to_service(service, level, explicit=True)
        assert service.reasoning_effort is None
    for level in ("0", "ultra"):
        with pytest.raises(ValueError, match="Invalid reasoning effort"):
            service.set_think(level)
    await service.close()


@pytest.mark.asyncio
async def test_xai_registry_definitions_control_defaults_and_capabilities(
    curated_service, monkeypatch
):
    import importlib

    from AgentCrew.modules.llm.types import Model

    xai_module = importlib.import_module("AgentCrew.modules.xai.response_service")
    assert not hasattr(xai_module, "_MODEL_PROFILES")
    assert not hasattr(xai_module, "_ModelProfile")
    assert "reasoning_efforts" not in Model.model_fields
    assert "encrypted_reasoning" not in Model.model_fields

    service = curated_service
    model = ModelRegistry.get_instance().get_model("xai/grok-4.3")
    monkeypatch.setattr(model, "default_reasoning", "xhigh")
    service.model = "grok-4.3"
    assert service.reasoning_effort == "xhigh"
    monkeypatch.setattr(model, "capabilities", ["tool_use", "stream"])
    assert service.allow_unconfigured_reasoning
    assert service.set_think("low") is False
    state = service.create_stream_state()
    service.process_stream_chunk(
        SimpleNamespace(
            type="response.output_item.done",
            output_index=0,
            item=SimpleNamespace(
                type="reasoning", id="reason-1", encrypted_content="synthetic"
            ),
        ),
        "",
        [],
        state,
    )
    assert state["reasoning_items"] == []
    service.client.responses.create = AsyncMock(
        return_value=EventStream([SimpleNamespace(type="response.completed")])
    )
    async with await service.stream_assistant_response(
        [{"role": "user", "content": "hello"}]
    ) as stream:
        assert [event.type async for event in stream] == ["response.completed"]
    request = service.client.responses.create.call_args.kwargs
    assert "include" not in request and "reasoning" not in request
    with pytest.raises(
        ValueError, match="Unsupported xAI subscription Responses model"
    ):
        service.model = "grok-build"
    assert service.model == "grok-4.3"
    monkeypatch.setattr(model, "provider", "openai")
    with pytest.raises(
        ValueError, match="Unsupported xAI subscription Responses model"
    ):
        service.model = "grok-4.3"
    await service.close()


@pytest.mark.parametrize("model_id", ["grok-4.7", "grok-4.3"])
@pytest.mark.asyncio
async def test_thinking_models_forward_none_effort(curated_service, model_id):
    service = curated_service
    service.model = model_id
    assert service.set_think("none")
    service.client.responses.create = AsyncMock(
        return_value=EventStream([SimpleNamespace(type="response.completed")])
    )
    async with await service.stream_assistant_response(
        [{"role": "user", "content": "hello"}]
    ) as stream:
        assert [event.type async for event in stream] == ["response.completed"]
    request = service.client.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort": "none"}
    assert request["include"] == ["reasoning.encrypted_content"]
    await service.close()


@pytest.mark.parametrize(
    ("model_id", "level"), [("grok-4.7", "none"), ("grok-4.3", "xhigh")]
)
@pytest.mark.asyncio
async def test_model_specific_effort_error_propagates_without_replay(
    curated_service, model_id, level
):
    service = curated_service
    service.model = model_id
    assert service.set_think(level)
    create = AsyncMock(side_effect=RuntimeError("xAI rejected effort"))
    service.client.responses.create = create
    with pytest.raises(RuntimeError, match="xAI rejected effort"):
        await service.process_message("hello", model_id=model_id)
    assert create.await_count == 1
    assert create.call_args.kwargs["reasoning"] == {"effort": level}
    assert service.model == model_id and service.reasoning_effort == level
    assert service.client.max_retries == 0
    await service.close()


@pytest.mark.parametrize(
    ("model_id", "default_effort", "include"),
    [
        ("grok-4.7", "high", True),
        ("grok-4.3", "low", True),
        ("grok-build-0.1", None, False),
    ],
)
@pytest.mark.asyncio
async def test_curated_model_request_and_local_tools(
    curated_service, model_id, default_effort, include
):
    service = curated_service
    service.model = model_id
    service.register_tool(
        {"type": "function", "name": "calc", "parameters": {"type": "object"}},
        lambda _: None,
    )
    service.tools.append({"type": "web_search"})
    raw = EventStream(
        [
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="function_call",
                    call_id="call_a",
                    name="calc",
                    arguments='{"x":1}',
                ),
            ),
            SimpleNamespace(
                type="response.completed", response=SimpleNamespace(usage=None)
            ),
        ]
    )
    service.client.responses.create = AsyncMock(return_value=raw)
    tools = []
    state = service.create_stream_state()
    async with await service.stream_assistant_response(
        [{"role": "user", "content": "sum"}]
    ) as stream:
        async for event in stream:
            _, tools, _, _, _ = service.process_stream_chunk(event, "", tools, state)
    request = service.client.responses.create.call_args.kwargs
    assert request["model"] == model_id and request["store"] is False
    assert request.get("reasoning") == (
        {"effort": default_effort} if default_effort is not None else None
    )
    assert (request.get("include") == ["reasoning.encrypted_content"]) is include
    assert request["tools"] == [service.tools[0]]
    assert tools[0]["id"] == "call_a" and tools[0]["input"] == {"x": 1}
    assert raw.closes == 1 and service.client.max_retries == 0
    await service.close()


@pytest.mark.asyncio
async def test_process_message_selects_curated_model_ids(curated_service):
    service = curated_service
    created = []
    for model_id, default_effort in (
        ("grok-4.3", "low"),
        ("grok-build-0.1", None),
        ("grok-4.7", "high"),
    ):
        raw = EventStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="ok"),
                SimpleNamespace(type="response.completed", response=None),
            ]
        )
        service.client.responses.create = AsyncMock(return_value=raw)
        assert await service.process_message("hello", model_id=model_id) == "ok"
        request = service.client.responses.create.call_args.kwargs
        assert request["model"] == model_id
        assert request.get("reasoning") == (
            {"effort": default_effort} if default_effort is not None else None
        )
        assert service.model == "grok-4.7"
        assert service.reasoning_effort == "high"
        assert raw.closes == 1
        created.append(request)
    assert created[0]["include"] == ["reasoning.encrypted_content"]
    assert "include" not in created[1]
    assert created[2]["include"] == ["reasoning.encrypted_content"]
    service.set_think("xhigh")
    service.client.responses.create = AsyncMock(
        return_value=EventStream([SimpleNamespace(type="response.completed")])
    )
    assert await service.process_message("hello", model_id="grok-4.7") == ""
    assert service.client.responses.create.call_args.kwargs["reasoning"] == {
        "effort": "xhigh"
    }
    with pytest.raises(
        ValueError, match="Unsupported xAI subscription Responses model"
    ):
        await service.process_message("hello", model_id="grok-build")
    assert service.model == "grok-4.7"
    assert service.reasoning_effort == "xhigh"
    assert service.client.responses.create.await_count == 1
    await service.close()


@pytest.mark.asyncio
async def test_process_message_override_keeps_state_and_uses_target_metadata(
    curated_service,
):
    service = curated_service
    service.model = "grok-4.3"
    service.set_think("medium")
    service.register_tool(
        {"type": "function", "name": "calc", "parameters": {"type": "object"}},
        lambda _: None,
    )
    from AgentCrew.modules.agents.message_metadata import (
        attach_stream_metadata,
        get_responses_provider_state,
    )

    reason = {"type": "reasoning", "id": "reason-1", "encrypted_content": "synthetic"}
    assistant = {"role": "assistant", "content": "previous answer"}
    attach_stream_metadata(
        assistant,
        {"provider": "xai", "model": "grok-4.7", "reasoning_items": [reason]},
    )
    messages = [{"role": "user", "content": "follow up"}, assistant]
    requests = []

    async def create(**params):
        requests.append(params)
        return EventStream([SimpleNamespace(type="response.completed")])

    service.client.responses.create = AsyncMock(side_effect=create)
    for model_id in ("grok-4.7", "grok-build-0.1", "grok-4.3"):
        assert await service.process_message(messages, model_id=model_id) == ""
        assert service.model == "grok-4.3"
        assert service.reasoning_effort == "medium"
    grok_47, build, grok_43 = requests
    assert grok_47["model"] == "grok-4.7"
    assert grok_47["reasoning"] == {"effort": "high"}
    assert grok_47["include"] == ["reasoning.encrypted_content"]
    assert grok_47["input"][1] == reason
    assert grok_47["tools"] == [service.tools[0]]
    assert build["model"] == "grok-build-0.1"
    assert "reasoning" not in build and "include" not in build
    assert not any(item.get("type") == "reasoning" for item in build["input"])
    assert grok_43["model"] == "grok-4.3"
    assert grok_43["reasoning"] == {"effort": "medium"}
    assert grok_43["include"] == ["reasoning.encrypted_content"]
    assert not any(item.get("type") == "reasoning" for item in grok_43["input"])
    assert get_responses_provider_state(assistant) is not None
    await service.close()


@pytest.mark.asyncio
async def test_process_message_override_failures_leave_service_unchanged(
    curated_service,
):
    service = curated_service
    service.set_think("xhigh")
    create = AsyncMock(side_effect=RuntimeError("request failed"))
    service.client.responses.create = create
    with pytest.raises(RuntimeError, match="request failed"):
        await service.process_message("hello", model_id="grok-4.3")
    assert service.model == "grok-4.7"
    assert service.reasoning_effort == "xhigh"
    assert create.await_count == 1

    service.client.responses.create = AsyncMock(
        return_value=EventStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
    )
    with pytest.raises(RuntimeError, match="ended without completion"):
        await service.process_message("hello", model_id="grok-build-0.1")
    assert service.model == "grok-4.7"
    assert service.reasoning_effort == "xhigh"

    with pytest.raises(
        ValueError, match="Unsupported xAI subscription Responses model"
    ):
        await service.process_message("hello", model_id="grok-build")
    assert service.model == "grok-4.7"
    assert service.reasoning_effort == "xhigh"
    assert service.client.responses.create.await_count == 1
    await service.close()


@pytest.mark.asyncio
async def test_process_message_concurrent_model_overrides_are_isolated(curated_service):
    service = curated_service
    service.set_think("xhigh")
    ready = asyncio.Event()
    calls = []

    async def create(**params):
        calls.append(params)
        if len(calls) == 2:
            ready.set()
        await ready.wait()
        assert service.model == "grok-4.7"
        assert service.reasoning_effort == "xhigh"
        return EventStream([SimpleNamespace(type="response.completed")])

    service.client.responses.create = AsyncMock(side_effect=create)
    await asyncio.wait_for(
        asyncio.gather(
            service.process_message("first", model_id="grok-4.3"),
            service.process_message("second", model_id="grok-build-0.1"),
        ),
        timeout=5,
    )
    by_model = {request["model"]: request for request in calls}
    assert set(by_model) == {"grok-4.3", "grok-build-0.1"}
    assert by_model["grok-4.3"]["reasoning"] == {"effort": "low"}
    assert "reasoning" not in by_model["grok-build-0.1"]
    assert by_model["grok-4.3"]["include"] == ["reasoning.encrypted_content"]
    assert "include" not in by_model["grok-build-0.1"]
    assert service.model == "grok-4.7"
    assert service.reasoning_effort == "xhigh"
    await service.close()


@pytest.mark.asyncio
async def test_curated_model_reasoning_state_requires_exact_provider_and_model(
    curated_service,
):
    from AgentCrew.modules.agents.message_metadata import attach_stream_metadata

    service = curated_service
    item = {"type": "reasoning", "id": "opaque-1", "encrypted_content": "synthetic"}
    user = {"role": "user", "content": "next"}
    for model_id in ("grok-4.7", "grok-4.3", "grok-build-0.1"):
        service.model = model_id
        state = service.create_stream_state()
        service.process_stream_chunk(
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(**item),
                output_index=0,
            ),
            "",
            [],
            state,
        )
        assert bool(state["reasoning_items"]) is (model_id != "grok-build-0.1")
        state["reasoning_items"] = [item]
        for stored_provider, stored_model in (
            ("xai", "grok-4.7"),
            ("xai", "grok-4.7-pro"),
            ("openai_codex", model_id),
            ("xai", "grok-4.3"),
            ("xai", "grok-build-0.1"),
        ):
            state["provider"], state["model"] = stored_provider, stored_model
            assistant = {"role": "assistant", "content": "done"}
            attach_stream_metadata(assistant, state)
            converted = service._convert_internal_format([user, assistant])
            replayed = any(entry.get("type") == "reasoning" for entry in converted)
            assert replayed is (
                model_id != "grok-build-0.1"
                and stored_provider == "xai"
                and stored_model == model_id
            )
    await service.close()


@pytest.mark.parametrize("model_id", ["grok-4.7", "grok-4.3"])
@pytest.mark.asyncio
async def test_wrong_provider_stream_state_never_captures_reasoning(
    curated_service, model_id
):
    from AgentCrew.modules.agents.message_metadata import (
        attach_stream_metadata,
        get_responses_provider_state,
    )

    service = curated_service
    service.model = model_id
    item = SimpleNamespace(
        type="reasoning", id="reason-1", encrypted_content="synthetic"
    )
    for event in (
        SimpleNamespace(type="response.output_item.done", item=item, output_index=0),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output=[item], usage=None),
        ),
    ):
        state = service.create_stream_state()
        state["provider"] = "openai_codex"
        service._capture_reasoning_item(item, state)
        service.process_stream_chunk(event, "", [], state)
        assert state["reasoning_items"] == []
        assistant = {"role": "assistant", "content": "done"}
        attach_stream_metadata(assistant, state)
        assert get_responses_provider_state(assistant) is None
    await service.close()


@pytest.mark.parametrize("model_id", ["grok-4.7", "grok-4.3", "grok-build-0.1"])
@pytest.mark.asyncio
async def test_curated_model_terminal_cancellation_and_no_replay(
    curated_service, model_id
):
    service = curated_service
    service.model = model_id
    partial = SimpleNamespace(type="response.output_text.delta", delta="partial")
    raw = EventStream([partial])
    create = AsyncMock(return_value=raw)
    service.client.responses.create = create
    seen = []
    with pytest.raises(RuntimeError, match="ended without completion"):
        async with await service.stream_assistant_response(
            [{"role": "user", "content": "x"}]
        ) as stream:
            async for event in stream:
                seen.append(event.type)
    assert seen == ["response.output_text.delta"]
    assert raw.closes == 1 and create.await_count == 1

    cancelled = EventStream([partial, SimpleNamespace(type="response.completed")])
    service.client.responses.create = AsyncMock(return_value=cancelled)
    async with await service.stream_assistant_response(
        [{"role": "user", "content": "x"}]
    ) as stream:
        assert (await anext(stream)).type == "response.output_text.delta"
        await stream.aclose()
    assert cancelled.closes >= 1
    await service.close()


USER_URL = "https://cli-chat-proxy.grok.com/v1/user?include=subscription"
BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
PERIOD_END = "2026-10-28T12:34:56Z"


def install_usage_client(monkeypatch, handler):
    calls = []

    class UsageClient:
        def __init__(self, *args, **kwargs):
            calls.append({"event": "init", "args": args, "kwargs": kwargs})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None, timeout=None):
            call = {
                "url": url,
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
            calls.append(call)
            result = handler(url)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(
        "AgentCrew.modules.xai.response_service.httpx2.AsyncClient",
        UsageClient,
    )
    return calls


def _usage_call(calls, url):
    matches = [call for call in calls if call.get("url") == url]
    assert matches, url
    return matches[-1]


def _expire_token(oauth):
    oauth._save_tokens(
        {
            "access_token": "stale-token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 10,
        }
    )


@pytest.mark.asyncio
async def test_xai_usage_weekly_credits(curated_service, oauth, monkeypatch):
    _expire_token(oauth)
    set_client(
        monkeypatch,
        [Response(200, {"access_token": "refreshed-token", "expires_in": 3600})],
    )

    def handler(url):
        if url == USER_URL:
            return Response(200, {"userId": "user-1", "subscriptionTier": "SuperGrok"})
        if url == BILLING_URL:
            return Response(
                200,
                {
                    "config": {
                        "creditUsagePercent": 42.5,
                        "currentPeriod": {
                            "type": "USAGE_PERIOD_TYPE_WEEKLY",
                            "end": PERIOD_END,
                        },
                        "prepaidBalance": {"val": 1250},
                        "productUsage": [{"product": "GrokBuild", "usagePercent": 9}],
                    }
                },
            )
        raise AssertionError(url)

    calls = install_usage_client(monkeypatch, handler)
    try:
        usage = await curated_service.get_usage()
    finally:
        await curated_service.close()

    assert usage["supported"] is True
    assert usage["provider"] == "xai"
    assert usage["plan_type"] == "SuperGrok"
    assert usage["message"] is None
    assert usage["credits"] == {"balance": "$12.50"}
    assert len(usage["limits"]) == 1
    limit = usage["limits"][0]
    assert limit["name"] == "weekly"
    assert limit["used_percent"] == 42.5
    assert limit["remaining_percent"] == 57.5
    assert limit["window_seconds"] == 604800
    assert (
        limit["reset_at"]
        == datetime.fromisoformat("2026-10-28T12:34:56+00:00").timestamp()
    )
    billing = _usage_call(calls, BILLING_URL)
    assert billing["headers"]["Authorization"] == "Bearer refreshed-token"
    assert billing["headers"]["X-XAI-Token-Auth"] == "xai-grok-cli"
    assert billing["headers"]["x-userid"] == "user-1"
    assert billing["timeout"] == 15
    assert "x-grok-client-version" not in {key.lower() for key in billing["headers"]}
    user = _usage_call(calls, USER_URL)
    assert "x-userid" not in {key.lower() for key in user["headers"]}
    assert user["headers"]["Authorization"] == "Bearer refreshed-token"
    assert user["timeout"] == 10


@pytest.mark.asyncio
async def test_xai_usage_omitted_percent_is_zero(curated_service, monkeypatch):
    def handler(url):
        if url == USER_URL:
            return Response(200, {})
        return Response(
            200,
            {
                "config": {
                    "currentPeriod": {
                        "type": "USAGE_PERIOD_TYPE_WEEKLY",
                        "end": PERIOD_END,
                    }
                }
            },
        )

    install_usage_client(monkeypatch, handler)
    try:
        usage = await curated_service.get_usage()
    finally:
        await curated_service.close()

    assert usage["supported"] is True
    assert usage["limits"] == [
        {
            "name": "weekly",
            "used_percent": 0.0,
            "remaining_percent": 100.0,
            "window_seconds": 604800,
            "reset_at": datetime.fromisoformat("2026-10-28T12:34:56+00:00").timestamp(),
            "reset_after_seconds": None,
        }
    ]
    assert usage["credits"] is None
    assert usage["message"] is None


@pytest.mark.asyncio
async def test_xai_usage_legacy_monthly_fallback(curated_service, monkeypatch):
    def handler(url):
        if url == USER_URL:
            return Response(200, {})
        return Response(
            200,
            {
                "config": {
                    "monthlyLimit": {"val": 2000},
                    "used": {"val": 500},
                }
            },
        )

    install_usage_client(monkeypatch, handler)
    try:
        usage = await curated_service.get_usage()
    finally:
        await curated_service.close()

    assert usage["limits"] == [
        {
            "name": "monthly",
            "used_percent": 25.0,
            "remaining_percent": 75.0,
            "window_seconds": None,
            "reset_at": None,
            "reset_after_seconds": None,
        }
    ]
    assert usage["message"] is None


@pytest.mark.asyncio
async def test_xai_usage_on_demand_only_when_cap_positive(curated_service, monkeypatch):
    payloads = [
        {
            "config": {
                "creditUsagePercent": 10,
                "currentPeriod": {"type": "USAGE_PERIOD_TYPE_MONTHLY"},
                "onDemandCap": {"val": 0},
                "onDemandUsed": {"val": 50},
            }
        },
        {
            "config": {
                "creditUsagePercent": 10,
                "onDemandCap": {"val": 400},
                "onDemandUsed": {"val": 500},
            }
        },
        {"config": {"creditUsagePercent": 10, "onDemandCap": {}}},
    ]
    index = {"n": 0}

    def handler(url):
        if url == USER_URL:
            return Response(200, {})
        payload = payloads[index["n"]]
        index["n"] += 1
        return Response(200, payload)

    install_usage_client(monkeypatch, handler)
    try:
        capped = await curated_service.get_usage()
        active = await curated_service.get_usage()
        empty_cap = await curated_service.get_usage()
    finally:
        await curated_service.close()

    assert [limit["name"] for limit in capped["limits"]] == ["monthly"]
    assert [limit["name"] for limit in active["limits"]] == ["allowance", "on-demand"]
    on_demand = active["limits"][1]
    assert on_demand["used_percent"] == 125.0
    assert on_demand["remaining_percent"] == 0.0
    assert on_demand["reset_at"] is None
    assert [limit["name"] for limit in empty_cap["limits"]] == ["allowance"]


@pytest.mark.asyncio
async def test_xai_usage_missing_token_does_not_call_network(
    curated_service, oauth, monkeypatch
):
    constructed = []

    def boom(*_args, **_kwargs):
        constructed.append(True)
        raise AssertionError("network client should not be constructed")

    monkeypatch.setattr("AgentCrew.modules.xai.oauth.httpx2.Client", boom)
    monkeypatch.setattr(
        "AgentCrew.modules.xai.response_service.httpx2.AsyncClient", boom
    )
    oauth.token_path.unlink()
    try:
        with pytest.raises(XAIAuthenticationError, match="agentcrew xai-auth"):
            await curated_service.get_usage()
    finally:
        await curated_service.close()
    assert constructed == []


@pytest.mark.asyncio
async def test_xai_usage_billing_http_error(curated_service, monkeypatch):
    secret = "secret-account-data"

    def handler(url):
        if url == USER_URL:
            return Response(200, {"subscriptionTier": "SuperGrok"})
        return Response(
            401,
            {"config": {"creditUsagePercent": 0}, "leak": secret},
        )

    install_usage_client(monkeypatch, handler)
    try:
        with pytest.raises(RuntimeError, match="status 401") as exc_info:
            await curated_service.get_usage()
    finally:
        await curated_service.close()
    assert secret not in str(exc_info.value)
    assert "creditUsagePercent" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_xai_usage_user_lookup_failure_still_returns_billing(
    curated_service, monkeypatch
):
    def handler(url):
        if url == USER_URL:
            raise RuntimeError("plan lookup failed")
        return Response(
            200,
            {
                "config": {
                    "creditUsagePercent": 42.5,
                    "currentPeriod": {
                        "type": "USAGE_PERIOD_TYPE_WEEKLY",
                        "end": PERIOD_END,
                    },
                }
            },
        )

    calls = install_usage_client(monkeypatch, handler)
    try:
        usage = await curated_service.get_usage()
    finally:
        await curated_service.close()

    assert usage["plan_type"] is None
    assert usage["limits"][0]["name"] == "weekly"
    assert usage["limits"][0]["used_percent"] == 42.5
    assert usage["limits"][0]["remaining_percent"] == 57.5
    billing = _usage_call(calls, BILLING_URL)
    assert "x-userid" not in billing["headers"]
    assert all(key.lower() != "x-userid" for key in billing["headers"])
