"""AgentCrew-owned xAI OAuth device authorization and rotating token storage."""

import json
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import httpx2

DEVICE_ENDPOINT = "https://auth.x.ai/oauth2/device/code"
TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
SCOPES = "openid profile email offline_access grok-cli:access api:access"
DEFAULT_TOKEN_PATH = Path.home() / ".AgentCrew" / "xai_oauth.json"


class XAIAuthenticationError(RuntimeError):
    """Authentication failed without exposing provider response or credentials."""


class XAIOAuth:
    """Device-code OAuth with an AgentCrew-only, process-safe token cache."""

    _locks_guard = threading.Lock()
    _locks: ClassVar[dict[str, threading.Lock]] = {}

    def __init__(self, token_path: str | Path | None = None):
        self.token_path = Path(token_path or DEFAULT_TOKEN_PATH).expanduser()
        self._tokens = self._load_tokens()

    @staticmethod
    def _client_id() -> str:
        client_id = os.getenv("XAI_OAUTH_CLIENT_ID", "").strip()
        if not client_id:
            raise XAIAuthenticationError(
                "Set XAI_OAUTH_CLIENT_ID to an AgentCrew-authorized xAI OAuth client ID before using xAI Subscription."
            )
        return client_id

    @classmethod
    def is_available(cls) -> bool:
        """Return whether a configured client and refreshable AgentCrew tokens exist."""
        try:
            cls._client_id()
            return cls().has_valid_tokens
        except (XAIAuthenticationError, OSError):
            return False

    def _load_tokens(self) -> dict[str, Any] | None:
        if self.token_path.is_symlink():
            raise XAIAuthenticationError("Refusing symlinked xAI OAuth token file")
        try:
            fd = os.open(self.token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, encoding="utf-8") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise XAIAuthenticationError(
                        "Refusing non-regular xAI OAuth token file"
                    )
                if os.name == "posix":
                    os.fchmod(handle.fileno(), 0o600)
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise XAIAuthenticationError(
                "Cannot read AgentCrew xAI OAuth tokens; run 'agentcrew xai-auth' again"
            ) from exc
        if not isinstance(data, dict):
            return None
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        expiry = data.get("expires_at")
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(refresh, str)
            or not refresh
        ):
            return None
        if isinstance(expiry, bool) or not isinstance(expiry, (int, float)):
            return None
        return {"access_token": access, "refresh_token": refresh, "expires_at": expiry}

    @property
    def has_valid_tokens(self) -> bool:
        return bool(self._tokens and self._tokens["refresh_token"])

    def _save_tokens(self, data: dict[str, Any]) -> None:
        parent = self.token_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or self.token_path.is_symlink():
            raise XAIAuthenticationError("Refusing symlinked xAI OAuth token path")
        if os.name == "posix":
            parent.chmod(0o700)
        fd, name = tempfile.mkstemp(dir=parent, prefix=".xai-oauth-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(name, 0o600)
            os.replace(name, self.token_path)
            if os.name == "posix":
                dir_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self._tokens = data

    @contextmanager
    def _refresh_lock(self):
        path = str(self.token_path.resolve())
        with self._locks_guard:
            lock = self._locks.setdefault(path, threading.Lock())
        with lock:
            self.token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_path = self.token_path.with_name(self.token_path.name + ".lock")
            if lock_path.is_symlink():
                raise XAIAuthenticationError("Refusing symlinked xAI OAuth lock file")
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                if os.name == "posix":
                    os.fchmod(fd, 0o600)
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                elif os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    if os.name == "posix":
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    elif os.name == "nt":
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)

    def _token_data(
        self, data: Any, previous_refresh: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise XAIAuthenticationError("xAI returned an invalid OAuth token response")
        access = data.get("access_token")
        refresh = data.get("refresh_token") or previous_refresh
        expires = data.get("expires_in")
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(refresh, str)
            or not refresh
            or isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or expires <= 0
        ):
            raise XAIAuthenticationError("xAI returned an invalid OAuth token response")
        return {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": time.time() + expires,
        }

    def get_valid_access_token(self) -> str:
        client_id = self._client_id()
        with self._refresh_lock():
            self._tokens = self._load_tokens()
            if not self._tokens:
                raise XAIAuthenticationError(
                    "xAI OAuth login required; run 'agentcrew xai-auth'"
                )
            if self._tokens["expires_at"] > time.time() + 60:
                return self._tokens["access_token"]
            try:
                with httpx2.Client(timeout=30) as client:
                    response = client.post(
                        TOKEN_ENDPOINT,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": client_id,
                            "refresh_token": self._tokens["refresh_token"],
                        },
                    )
                    if response.status_code != 200:
                        raise XAIAuthenticationError(
                            "xAI OAuth refresh failed; run 'agentcrew xai-auth' to reauthenticate if the session expired"
                        )
                    tokens = self._token_data(
                        response.json(), self._tokens["refresh_token"]
                    )
                self._save_tokens(tokens)
                return tokens["access_token"]
            except (httpx2.HTTPError, ValueError) as exc:
                raise XAIAuthenticationError(
                    "xAI OAuth refresh failed; run 'agentcrew xai-auth' to reauthenticate"
                ) from exc

    def login(self, display: Callable[[str], None] = print) -> bool:
        """Poll RFC 8628 device authorization until completion, refusal or expiry."""
        client_id = self._client_id()
        try:
            with httpx2.Client(timeout=30) as client:
                response = client.post(
                    DEVICE_ENDPOINT,
                    data={"client_id": client_id, "scope": SCOPES},
                )
                if response.status_code != 200:
                    raise XAIAuthenticationError(
                        "xAI device authorization failed; check your OAuth client authorization"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("invalid device response")
                code = data.get("device_code")
                user_code = data.get("user_code")
                url = data.get("verification_uri_complete") or data.get(
                    "verification_uri"
                )
                expires = data.get("expires_in")
                interval = data.get("interval", 5)
                if (
                    not all(
                        isinstance(value, str) and value
                        for value in (code, user_code, url)
                    )
                    or not isinstance(expires, (int, float))
                    or expires <= 0
                    or not isinstance(interval, (int, float))
                    or interval < 0
                ):
                    raise ValueError("invalid device response")
                display(
                    f"Visit {url} and enter the displayed xAI authorization code: {user_code}"
                )
                deadline = time.monotonic() + expires
                interval = max(1, interval)
                while time.monotonic() < deadline:
                    time.sleep(min(interval, max(0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        break
                    result = client.post(
                        TOKEN_ENDPOINT,
                        data={
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "client_id": client_id,
                            "device_code": code,
                        },
                    )
                    if result.status_code == 200:
                        tokens = self._token_data(result.json())
                        with self._refresh_lock():
                            self._save_tokens(tokens)
                        return True
                    error_data = result.json()
                    error = (
                        error_data.get("error")
                        if isinstance(error_data, dict)
                        else None
                    )
                    if error == "authorization_pending":
                        continue
                    if error == "slow_down":
                        interval += 5
                        continue
                    if error == "access_denied":
                        raise XAIAuthenticationError("xAI authorization denied")
                    if error == "expired_token":
                        raise XAIAuthenticationError(
                            "xAI device authorization expired; run 'agentcrew xai-auth' again"
                        )
                    raise XAIAuthenticationError("xAI device token exchange failed")
        except (httpx2.HTTPError, ValueError, TypeError) as exc:
            raise XAIAuthenticationError(
                "xAI OAuth device authorization failed; try 'agentcrew xai-auth' again"
            ) from exc
        raise XAIAuthenticationError(
            "xAI device authorization expired; run 'agentcrew xai-auth' again"
        )

    def logout(self) -> None:
        """Remove AgentCrew's local tokens; remote revocation is not yet supported."""
        with self._refresh_lock():
            self.token_path.unlink(missing_ok=True)
            self._tokens = None
