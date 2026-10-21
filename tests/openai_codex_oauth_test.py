import base64
import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from AgentCrew.modules.openai_codex import oauth as oauth_module
from AgentCrew.modules.openai_codex.oauth import OpenAICodexOAuth


def _jwt_with_exp(exp_seconds: int) -> str:
    header = (
        base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}')
        .rstrip(b"=")
        .decode("ascii")
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp_seconds}).encode("utf-8"))
        .rstrip(b"=")
        .decode("ascii")
    )
    return f"{header}.{payload}.signature"


class TestOpenAICodexOAuth:
    def test_loads_legacy_nested_file(self, tmp_path):
        token_path = tmp_path / "auth.json"
        expires = int((time.time() + 3600) * 1000)
        token_path.write_text(
            json.dumps(
                {
                    "openai-codex": {
                        "type": "oauth",
                        "access": "legacy-access",
                        "refresh": "legacy-refresh",
                        "expires": expires,
                    }
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))

        assert oauth.access_token == "legacy-access"
        assert oauth._tokens["refresh"] == "legacy-refresh"
        assert oauth._tokens["expires"] == expires
        assert oauth.has_valid_tokens is True

    def test_loads_legacy_flat_file(self, tmp_path):
        token_path = tmp_path / "auth.json"
        expires = int((time.time() + 3600) * 1000)
        token_path.write_text(
            json.dumps(
                {
                    "type": "oauth",
                    "access": "flat-access",
                    "refresh": "flat-refresh",
                    "expires": expires,
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))

        assert oauth.access_token == "flat-access"
        assert oauth._tokens["refresh"] == "flat-refresh"
        assert oauth._tokens["expires"] == expires
        assert oauth.has_valid_tokens is True

    def test_loads_codex_cli_style_file(self, tmp_path):
        token_path = tmp_path / "auth.json"
        exp_seconds = int(time.time()) + 3600
        access_token = _jwt_with_exp(exp_seconds)
        token_path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "last_refresh": 1712505600000,
                    "tokens": {
                        "access_token": access_token,
                        "refresh_token": "cli-refresh",
                        "id_token": "cli-id-token",
                        "account_id": "account-123",
                    },
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))

        assert oauth.access_token == access_token
        assert oauth._tokens["refresh"] == "cli-refresh"
        assert oauth._tokens["id_token"] == "cli-id-token"
        assert oauth._tokens["account_id"] == "account-123"
        assert oauth._tokens["auth_mode"] == "chatgpt"
        assert oauth._tokens["last_refresh"] == "2024-04-07T16:00:00Z"
        assert oauth._tokens["expires"] == exp_seconds * 1000
        assert oauth.has_valid_tokens is True

    def test_save_writes_codex_cli_style_shape(self, tmp_path):
        token_path = tmp_path / "auth.json"
        oauth = OpenAICodexOAuth(token_path=str(token_path))
        oauth._tokens = {
            "type": "oauth",
            "access": "new-access",
            "refresh": "new-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "id_token": "new-id-token",
            "account_id": "account-456",
        }

        oauth._save_tokens()

        data = json.loads(token_path.read_text(encoding="utf-8"))
        assert data["auth_mode"] == "chatgpt"
        assert isinstance(data["last_refresh"], str)
        datetime.fromisoformat(data["last_refresh"])
        assert data["tokens"]["access_token"] == "new-access"
        assert data["tokens"]["refresh_token"] == "new-refresh"
        assert data["tokens"]["id_token"] == "new-id-token"
        assert data["tokens"]["account_id"] == "account-456"
        assert "openai-codex" not in data

    def test_save_migrates_legacy_nested_file_and_preserves_unrelated_keys(
        self, tmp_path
    ):
        token_path = tmp_path / "auth.json"
        token_path.write_text(
            json.dumps(
                {
                    "openai-codex": {
                        "type": "oauth",
                        "access": "legacy-access",
                        "refresh": "legacy-refresh",
                        "expires": int((time.time() + 3600) * 1000),
                    },
                    "custom": {"enabled": True},
                    "tokens": {
                        "custom_token_metadata": "keep-me",
                    },
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))
        oauth._tokens["access"] = "migrated-access"
        oauth._tokens["refresh"] = "migrated-refresh"
        oauth._save_tokens()

        data = json.loads(token_path.read_text(encoding="utf-8"))
        assert "openai-codex" not in data
        assert data["custom"] == {"enabled": True}
        assert data["auth_mode"] == "chatgpt"
        assert isinstance(data["last_refresh"], str)
        datetime.fromisoformat(data["last_refresh"])
        assert data["tokens"]["access_token"] == "migrated-access"
        assert data["tokens"]["refresh_token"] == "migrated-refresh"
        assert data["tokens"]["custom_token_metadata"] == "keep-me"

    def test_loads_codex_cli_file_with_legacy_numeric_last_refresh(self, tmp_path):
        token_path = tmp_path / "auth.json"
        exp_seconds = int(time.time()) + 3600
        access_token = _jwt_with_exp(exp_seconds)
        token_path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "last_refresh": 1712505600000,
                    "tokens": {
                        "access_token": access_token,
                        "refresh_token": "cli-refresh",
                    },
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))

        assert oauth._tokens["last_refresh"] == "2024-04-07T16:00:00Z"
        assert oauth._tokens["expires"] == exp_seconds * 1000

    def test_loads_codex_cli_file_with_expires_in_fallback(self, tmp_path):
        token_path = tmp_path / "auth.json"
        token_path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": "not-a-jwt",
                        "refresh_token": "cli-refresh",
                    },
                }
            ),
            encoding="utf-8",
        )

        oauth = OpenAICodexOAuth(token_path=str(token_path))

        assert oauth.access_token == "not-a-jwt"
        assert oauth._tokens["refresh"] == "cli-refresh"
        assert "expires" not in oauth._tokens
        assert oauth.has_valid_tokens is True


def _response(status, data=None):
    return SimpleNamespace(status_code=status, json=lambda: data)


def _mock_client(monkeypatch, responses):
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = responses
    monkeypatch.setattr(oauth_module.httpx2, "Client", lambda **kwargs: client)
    return client


def _id_token(account_id="account-from-jwt"):
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    return f"header.{payload}.signature"


def _device_response(interval="3"):
    return _response(
        200,
        {
            "device_auth_id": "secret-device-id",
            "user_code": "ABCD-EFGH",
            "interval": interval,
        },
    )


def _poll_success():
    return _response(
        200,
        {
            "authorization_code": "secret-authorization-code",
            "code_challenge": "challenge",
            "code_verifier": "secret-verifier",
        },
    )


def test_device_login_pending_exchange_and_persistence(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    displays = []
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(oauth_module.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(oauth_module.time, "sleep", sleep)
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: pytest.fail("device login must not open a browser"),
    )
    client = _mock_client(
        monkeypatch,
        [
            _device_response(),
            _response(403),
            _response(404),
            _poll_success(),
            _response(
                200,
                {
                    "access_token": "secret-access-token",
                    "refresh_token": "secret-refresh-token",
                    "id_token": _id_token(),
                    "expires_in": 3600,
                },
            ),
        ],
    )
    oauth = OpenAICodexOAuth(token_path=str(path))

    assert oauth.login(display=displays.append) is True
    assert sleeps == [3, 3]
    assert oauth.account_id == "account-from-jwt"
    assert OpenAICodexOAuth(token_path=str(path)).account_id == "account-from-jwt"
    saved = json.loads(path.read_text())
    assert saved["auth_mode"] == "chatgpt"
    assert saved["tokens"]["refresh_token"] == "secret-refresh-token"
    assert saved["tokens"]["account_id"] == "account-from-jwt"
    assert (path.stat().st_mode & 0o777) == 0o600
    assert "https://auth.openai.com/codex/device" in "\n".join(displays)
    assert "ABCD-EFGH" in "\n".join(displays)
    assert "secret-device-id" not in "\n".join(displays)
    posts = client.post.call_args_list
    assert posts[0].args == (oauth_module.DEVICE_CODE_ENDPOINT,)
    assert posts[0].kwargs["json"] == {"client_id": oauth_module.CLIENT_ID}
    assert posts[0].kwargs["headers"] == {"Content-Type": "application/json"}
    for post in posts[1:4]:
        assert post.args == (oauth_module.DEVICE_TOKEN_ENDPOINT,)
        assert post.kwargs["headers"] == {"Content-Type": "application/json"}
        assert post.kwargs["json"] == {
            "device_auth_id": "secret-device-id",
            "user_code": "ABCD-EFGH",
        }
    assert posts[4].args == (oauth_module.TOKEN_ENDPOINT,)
    assert posts[4].kwargs["data"] == {
        "grant_type": "authorization_code",
        "client_id": oauth_module.CLIENT_ID,
        "code": "secret-authorization-code",
        "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        "code_verifier": "secret-verifier",
    }
    assert posts[4].kwargs["headers"] == {
        "Content-Type": "application/x-www-form-urlencoded"
    }


@pytest.mark.parametrize(
    "device",
    [
        {},
        {"device_auth_id": "secret-device-id", "user_code": "ABCD", "interval": "0"},
        {"device_auth_id": "secret-device-id", "user_code": "ABCD", "interval": "no"},
        {"device_auth_id": "secret-device-id", "user_code": "ABCD"},
        {"device_auth_id": "secret-device-id", "interval": "5"},
        ["invalid"],
    ],
)
def test_device_login_rejects_invalid_user_code_response(tmp_path, monkeypatch, device):
    client = _mock_client(monkeypatch, [_response(200, device)])
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: pytest.fail(message)) is False
    assert client.post.call_count == 1
    assert not (tmp_path / "auth.json").exists()


@pytest.mark.parametrize("invalid_poll", [{}, {"authorization_code": "secret"}, []])
def test_device_login_rejects_invalid_poll_success(tmp_path, monkeypatch, invalid_poll):
    client = _mock_client(
        monkeypatch, [_device_response(), _response(200, invalid_poll)]
    )
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert client.post.call_count == 2


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_device_login_stops_on_terminal_status(tmp_path, monkeypatch, status):
    client = _mock_client(monkeypatch, [_device_response(), _response(status)])
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert client.post.call_count == 2


def test_device_login_times_out_at_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(oauth_module.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(oauth_module.time, "sleep", sleep)
    client = _mock_client(
        monkeypatch, [_device_response("700"), _response(404), _response(403)]
    )
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert sleeps == [700, 200]
    assert client.post.call_count == 3


def test_device_login_network_failure_does_not_log_response_or_secrets(
    tmp_path, monkeypatch
):
    import httpx2
    from loguru import logger

    client = _mock_client(
        monkeypatch,
        [_device_response(), httpx2.ConnectError("secret-device-id")],
    )
    log_lines = []
    sink_id = logger.add(log_lines.append, format="{message}")
    try:
        oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
        assert oauth.login(display=lambda message: None) is False
    finally:
        logger.remove(sink_id)
    assert client.post.call_count == 2
    assert "secret-device-id" not in "\n".join(log_lines)
    assert not (tmp_path / "auth.json").exists()


def test_device_login_cancelled(tmp_path, monkeypatch):
    client = _mock_client(monkeypatch, [_device_response(), KeyboardInterrupt()])
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert client.post.call_count == 2
    assert not (tmp_path / "auth.json").exists()


@pytest.mark.parametrize("status", [403, 404, 429])
def test_device_login_rejects_user_code_http_failure(tmp_path, monkeypatch, status):
    client = _mock_client(monkeypatch, [_response(status)])
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: pytest.fail(message)) is False
    assert client.post.call_count == 1


def test_device_login_rejects_malformed_json(tmp_path, monkeypatch):
    def malformed_json():
        raise ValueError("secret-device-id")

    client = _mock_client(
        monkeypatch,
        [_device_response(), SimpleNamespace(status_code=200, json=malformed_json)],
    )
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert client.post.call_count == 2
    assert not (tmp_path / "auth.json").exists()


def test_token_update_preserves_refresh_and_account_metadata(tmp_path):
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    oauth._tokens = {
        "refresh": "previous-refresh",
        "id_token": _id_token("previous-account"),
        "account_id": "previous-account",
    }
    oauth._update_tokens_from_response({"access_token": "new-access"})
    assert oauth._tokens["refresh"] == "previous-refresh"
    assert oauth._tokens["account_id"] == "previous-account"
    assert oauth._tokens["id_token"] == _id_token("previous-account")


def test_device_login_rejects_malformed_token_exchange(tmp_path, monkeypatch):
    _mock_client(
        monkeypatch,
        [
            _device_response(),
            _poll_success(),
            _response(200, {"access_token": "secret-access-token"}),
        ],
    )
    oauth = OpenAICodexOAuth(token_path=str(tmp_path / "auth.json"))
    assert oauth.login(display=lambda message: None) is False
    assert not (tmp_path / "auth.json").exists()
