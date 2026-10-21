import base64
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx2
from loguru import logger

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
DEVICE_CODE_ENDPOINT = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_ENDPOINT = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URL = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
TOKEN_ENDPOINT = f"{AUTH_BASE_URL}/oauth/token"
DEFAULT_TOKEN_PATH = os.path.expanduser("~/.codex/auth.json")
LEGACY_TOP_LEVEL_KEYS = {
    "openai-codex",
    "access",
    "refresh",
    "expires",
    "type",
    "id_token",
    "account_id",
}


def _current_time_ms() -> int:
    return int(time.time() * 1000)


def _current_time_rfc3339() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_refresh_time(value: Any) -> str | None:
    if isinstance(value, (int, float)) and value > 0:
        return (
            datetime.fromtimestamp(value / 1000, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            datetime.fromisoformat(stripped)
            return stripped.replace("+00:00", "Z")
        except ValueError:
            return None

    return None


def _decode_base64url_json(value: str) -> dict[str, Any] | None:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
        if isinstance(payload, dict):
            return payload
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


def _derive_expires_from_access_token(access_token: str) -> int | None:
    if not isinstance(access_token, str):
        return None

    token_parts = access_token.split(".")
    if len(token_parts) != 3:
        return None

    payload = _decode_base64url_json(token_parts[1])
    if not payload:
        return None

    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp > 0:
        return int(exp * 1000)

    return None


def _extract_account_id(data: dict[str, Any]) -> str | None:
    account_id = data.get("account_id")
    if isinstance(account_id, str) and account_id:
        return account_id

    account = data.get("account")
    if isinstance(account, dict):
        nested_account_id = account.get("id") or account.get("account_id")
        if isinstance(nested_account_id, str) and nested_account_id:
            return nested_account_id

    id_token = data.get("id_token")
    if isinstance(id_token, str):
        parts = id_token.split(".")
        claims = _decode_base64url_json(parts[1]) if len(parts) == 3 else None
        if claims:
            auth_claims = claims.get("https://api.openai.com/auth")
            if isinstance(auth_claims, dict):
                nested_id = auth_claims.get("chatgpt_account_id")
                if isinstance(nested_id, str) and nested_id:
                    return nested_id

    return None


class OpenAICodexOAuth:
    def __init__(self, token_path: str | None = None):
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self._tokens: dict[str, Any] | None = None
        self._load_tokens()

    def _normalize_tokens(
        self,
        raw_tokens: dict[str, Any],
        *,
        auth_mode: Any | None = None,
        last_refresh: Any | None = None,
    ) -> dict[str, Any] | None:
        access_token = raw_tokens.get("access") or raw_tokens.get("access_token")
        refresh_token = (
            raw_tokens.get("refresh") or raw_tokens.get("refresh_token") or ""
        )

        if not access_token and not refresh_token:
            return None

        expires = raw_tokens.get("expires")
        if not isinstance(expires, (int, float)) or expires <= 0:
            expires = raw_tokens.get("expires_at")
        if not isinstance(expires, (int, float)) or expires <= 0:
            expires_in = raw_tokens.get("expires_in")
            if isinstance(expires_in, (int, float)) and expires_in > 0:
                expires = _current_time_ms() + int(expires_in * 1000)
        if not isinstance(expires, (int, float)) or expires <= 0:
            expires = _derive_expires_from_access_token(access_token)

        normalized = {
            "type": raw_tokens.get("type") or "oauth",
            "access": access_token,
            "refresh": refresh_token,
        }

        if isinstance(expires, (int, float)) and expires > 0:
            normalized["expires"] = int(expires)

        id_token = raw_tokens.get("id_token")
        if isinstance(id_token, str) and id_token:
            normalized["id_token"] = id_token

        account_id = raw_tokens.get("account_id")
        if isinstance(account_id, str) and account_id:
            normalized["account_id"] = account_id

        if isinstance(auth_mode, str) and auth_mode:
            normalized["auth_mode"] = auth_mode

        normalized_last_refresh = _normalize_refresh_time(last_refresh)
        if normalized_last_refresh:
            normalized["last_refresh"] = normalized_last_refresh

        return normalized

    def _load_tokens(self):
        if not os.path.exists(self.token_path):
            return

        try:
            with open(self.token_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load OAuth tokens from {self.token_path}: {e}")
            self._tokens = None
            return

        if not isinstance(data, dict):
            self._tokens = None
            return

        if isinstance(data.get("tokens"), dict):
            self._tokens = self._normalize_tokens(
                data["tokens"],
                auth_mode=data.get("auth_mode"),
                last_refresh=data.get("last_refresh"),
            )
            if self._tokens:
                return

        legacy_nested = data.get("openai-codex")
        if isinstance(legacy_nested, dict):
            self._tokens = self._normalize_tokens(legacy_nested)
            if self._tokens:
                return

        if data.get("access") or data.get("refresh"):
            self._tokens = self._normalize_tokens(data)
            if self._tokens:
                return

        self._tokens = None

    def _save_tokens(self):
        if not self._tokens:
            return

        dir_path = os.path.dirname(self.token_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        existing: dict[str, Any] = {}
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}

        current_time_rfc3339 = _current_time_rfc3339()
        self._tokens["last_refresh"] = current_time_rfc3339
        self._tokens["auth_mode"] = "chatgpt"

        top_level = {
            key: value
            for key, value in existing.items()
            if key not in LEGACY_TOP_LEVEL_KEYS
            and key not in {"auth_mode", "last_refresh", "tokens"}
        }
        top_level["auth_mode"] = "chatgpt"
        top_level["last_refresh"] = current_time_rfc3339

        existing_tokens = existing.get("tokens")
        tokens_payload = (
            dict(existing_tokens) if isinstance(existing_tokens, dict) else {}
        )

        access_token = self._tokens.get("access")
        if isinstance(access_token, str) and access_token:
            tokens_payload["access_token"] = access_token

        refresh_token = self._tokens.get("refresh")
        if isinstance(refresh_token, str) and refresh_token:
            tokens_payload["refresh_token"] = refresh_token

        id_token = self._tokens.get("id_token")
        if isinstance(id_token, str) and id_token:
            tokens_payload["id_token"] = id_token

        account_id = self._tokens.get("account_id")
        if isinstance(account_id, str) and account_id:
            tokens_payload["account_id"] = account_id

        top_level["tokens"] = tokens_payload

        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(top_level, f, indent=2)
        os.chmod(self.token_path, 0o600)

    @property
    def has_valid_tokens(self) -> bool:
        if not self._tokens:
            return False
        expires = self._tokens.get("expires", 0)
        if isinstance(expires, (int, float)) and expires > 0:
            return time.time() * 1000 < expires - 60_000
        return bool(self._tokens.get("access"))

    @property
    def access_token(self) -> str | None:
        if self._tokens:
            return self._tokens.get("access")
        return None

    @property
    def account_id(self) -> str | None:
        """Return the ChatGPT account ID from stored tokens, if available."""
        if self._tokens:
            return self._tokens.get("account_id")
        return None

    def get_valid_access_token(self) -> str | None:
        if self.has_valid_tokens:
            return self.access_token
        if self._tokens and self._tokens.get("refresh") and self._refresh_token():
            return self.access_token
        return None

    def _update_tokens_from_response(self, data: dict[str, Any]):
        expires_in = data.get("expires_in", 3600)
        previous_tokens = self._tokens or {}
        self._tokens = {
            "type": "oauth",
            "access": data["access_token"],
            "refresh": data.get("refresh_token") or previous_tokens.get("refresh", ""),
            "expires": int((time.time() + expires_in) * 1000),
            "auth_mode": "chatgpt",
            "last_refresh": _current_time_rfc3339(),
        }

        id_token = data.get("id_token")
        if isinstance(id_token, str) and id_token:
            self._tokens["id_token"] = id_token
        elif previous_tokens.get("id_token"):
            self._tokens["id_token"] = previous_tokens["id_token"]

        account_id = _extract_account_id(data) or previous_tokens.get("account_id")
        if isinstance(account_id, str) and account_id:
            self._tokens["account_id"] = account_id

    def _refresh_token(self) -> bool:
        if not self._tokens or not self._tokens.get("refresh"):
            return False
        try:
            with httpx2.Client(timeout=30) as client:
                resp = client.post(
                    TOKEN_ENDPOINT,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": CLIENT_ID,
                        "refresh_token": self._tokens["refresh"],
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code != 200:
                    logger.error(
                        f"Token refresh failed: {resp.status_code} {resp.text}"
                    )
                    return False
                data = resp.json()
                previous_tokens = dict(self._tokens)
                self._update_tokens_from_response(data)
                if "id_token" not in self._tokens and previous_tokens.get("id_token"):
                    self._tokens["id_token"] = previous_tokens["id_token"]
                if "account_id" not in self._tokens and previous_tokens.get(
                    "account_id"
                ):
                    self._tokens["account_id"] = previous_tokens["account_id"]
                self._save_tokens()
                logger.info("OAuth token refreshed successfully")
                return True
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    @classmethod
    def is_available(cls) -> bool:
        """Check whether the user has valid Codex OAuth tokens on disk."""
        try:
            return cls().has_valid_tokens
        except Exception:
            return False

    @classmethod
    def get_auth(cls) -> dict[str, str] | None:
        """Return {access_token, account_id} dict if valid OAuth tokens exist."""
        try:
            oauth = cls()
            token = oauth.get_valid_access_token()
            if token:
                return {
                    "access_token": token,
                    "account_id": oauth.account_id or "",
                }
        except Exception:
            return None
        return None

    def login(self, display: Callable[[str], None] = print) -> bool:
        """Authorize through Codex device login without a local callback server."""
        try:
            with httpx2.Client(timeout=30) as client:
                response = client.post(
                    DEVICE_CODE_ENDPOINT,
                    json={"client_id": CLIENT_ID},
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    logger.error(
                        "Device code request failed with status {}",
                        response.status_code,
                    )
                    return False
                device = response.json()
                if not isinstance(device, dict):
                    raise TypeError("Invalid device code response")
                device_auth_id = device.get("device_auth_id")
                user_code = device.get("user_code") or device.get("usercode")
                raw_interval = device.get("interval")
                if (
                    not isinstance(device_auth_id, str)
                    or not device_auth_id
                    or not isinstance(user_code, str)
                    or not user_code
                    or isinstance(raw_interval, bool)
                    or not isinstance(raw_interval, (str, int))
                ):
                    raise ValueError("Invalid device code response")
                interval = int(raw_interval)
                if interval <= 0:
                    raise ValueError("Invalid device code response")

                display(f"Open {DEVICE_VERIFICATION_URL} and sign in to ChatGPT.")
                display(
                    f"Enter this one-time code (expires in 15 minutes): {user_code}"
                )
                display(
                    "Only enter this code if you initiated this login in AgentCrew."
                )

                deadline = time.monotonic() + 15 * 60
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error("Device authorization timed out after 15 minutes")
                        return False
                    response = client.post(
                        DEVICE_TOKEN_ENDPOINT,
                        json={
                            "device_auth_id": device_auth_id,
                            "user_code": user_code,
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=min(30, remaining),
                    )
                    if response.status_code == 200:
                        code_data = response.json()
                        if not isinstance(code_data, dict) or not all(
                            isinstance(code_data.get(field), str) and code_data[field]
                            for field in (
                                "authorization_code",
                                "code_challenge",
                                "code_verifier",
                            )
                        ):
                            raise ValueError("Invalid device authorization response")
                        break
                    if response.status_code not in (403, 404):
                        logger.error(
                            "Device authorization failed with status {}",
                            response.status_code,
                        )
                        return False
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error("Device authorization timed out after 15 minutes")
                        return False
                    time.sleep(min(interval, remaining))

                response = client.post(
                    TOKEN_ENDPOINT,
                    data={
                        "grant_type": "authorization_code",
                        "client_id": CLIENT_ID,
                        "code": code_data["authorization_code"],
                        "redirect_uri": DEVICE_REDIRECT_URI,
                        "code_verifier": code_data["code_verifier"],
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if response.status_code != 200:
                    logger.error(
                        "Token exchange failed with status {}", response.status_code
                    )
                    return False
                tokens = response.json()
                if not isinstance(tokens, dict) or not all(
                    isinstance(tokens.get(field), str) and tokens[field]
                    for field in ("access_token", "refresh_token", "id_token")
                ):
                    raise ValueError("Invalid token exchange response")
                self._update_tokens_from_response(tokens)
                self._save_tokens()
                logger.info("OpenAI Codex OAuth authentication successful")
                return True
        except KeyboardInterrupt:
            logger.info("Device authorization cancelled")
            return False
        except (httpx2.HTTPError, OSError, ValueError, TypeError) as exc:
            logger.error("Device authorization failed: {}", type(exc).__name__)
            return False
