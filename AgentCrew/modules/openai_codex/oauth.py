import hashlib
import base64
import secrets
import json
import os
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, Any
from threading import Thread, Event

import httpx
from loguru import logger

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_ENDPOINT = "https://auth.openai.com/oauth/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
SCOPES = "openid profile email offline_access"
DEFAULT_TOKEN_PATH = os.path.expanduser("~/.codex/auth.json")
CALLBACK_HOST = "127.0.0.1"
CALLBACK_REDIRECT_HOST = "localhost"
CALLBACK_PORT_RANGE = (1455, 1475)


def _generate_pkce_pair():
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path != "/auth/callback":
            self._send_response(
                "<html><body><h2>Waiting for authentication...</h2></body></html>"
            )
            return

        if "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            self._send_response(
                "<html><body><h2>Authentication successful!</h2>"
                "<p>You can close this window and return to AgentCrew.</p></body></html>"
            )
        elif "error" in params:
            _OAuthCallbackHandler.error = params.get(
                "error_description", params["error"]
            )[0]
            self._send_response(
                f"<html><body><h2>Authentication failed</h2>"
                f"<p>{_OAuthCallbackHandler.error}</p></body></html>"
            )
        else:
            self._send_response(
                "<html><body><h2>Unexpected response</h2></body></html>"
            )

        self.server.shutdown_event.set()

    def _send_response(self, html: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass


class _ShutdownHTTPServer(HTTPServer):
    def __init__(self, *args, **kwargs):
        self.shutdown_event = Event()
        super().__init__(*args, **kwargs)


class OpenAICodexOAuth:
    def __init__(self, token_path: Optional[str] = None):
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self._tokens: Optional[Dict[str, Any]] = None
        self._load_tokens()

    def _load_tokens(self):
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "openai-codex" in data:
                    self._tokens = data["openai-codex"]
                elif isinstance(data, dict) and "access" in data:
                    self._tokens = data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load OAuth tokens from {self.token_path}: {e}"
                )
                self._tokens = None

    def _save_tokens(self):
        if not self._tokens:
            return
        dir_path = os.path.dirname(self.token_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        existing = {}
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = {}

        existing["openai-codex"] = self._tokens
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
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
    def access_token(self) -> Optional[str]:
        if self._tokens:
            return self._tokens.get("access")
        return None

    def get_valid_access_token(self) -> Optional[str]:
        if self.has_valid_tokens:
            return self.access_token
        if self._tokens and self._tokens.get("refresh"):
            if self._refresh_token():
                return self.access_token
        return None

    def _refresh_token(self) -> bool:
        if not self._tokens or not self._tokens.get("refresh"):
            return False
        try:
            with httpx.Client(timeout=30) as client:
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
                self._tokens["access"] = data["access_token"]
                if "refresh_token" in data:
                    self._tokens["refresh"] = data["refresh_token"]
                expires_in = data.get("expires_in", 3600)
                self._tokens["expires"] = int((time.time() + expires_in) * 1000)
                self._save_tokens()
                logger.info("OAuth token refreshed successfully")
                return True
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def login(self) -> bool:
        code_verifier, code_challenge = _generate_pkce_pair()

        server = None
        callback_port = None
        for port in range(CALLBACK_PORT_RANGE[0], CALLBACK_PORT_RANGE[1]):
            try:
                server = _ShutdownHTTPServer(
                    (CALLBACK_HOST, port), _OAuthCallbackHandler
                )
                callback_port = port
                break
            except OSError:
                continue

        if not server or callback_port is None:
            logger.error("Could not find available port for OAuth callback")
            return False

        redirect_uri = f"http://{CALLBACK_REDIRECT_HOST}:{callback_port}/auth/callback"

        _OAuthCallbackHandler.auth_code = None
        _OAuthCallbackHandler.error = None

        state = secrets.token_urlsafe(32)

        auth_params = urlencode(
            {
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": SCOPES,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
            }
        )
        auth_url = f"{AUTH_ENDPOINT}?{auth_params}"

        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        logger.info("Opening browser for OpenAI authentication...")
        webbrowser.open(auth_url)

        server.shutdown_event.wait(timeout=300)
        server.shutdown()

        if _OAuthCallbackHandler.error:
            logger.error(f"OAuth error: {_OAuthCallbackHandler.error}")
            return False

        if not _OAuthCallbackHandler.auth_code:
            logger.error("OAuth flow timed out or no authorization code received")
            return False

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    TOKEN_ENDPOINT,
                    data={
                        "grant_type": "authorization_code",
                        "client_id": CLIENT_ID,
                        "code": _OAuthCallbackHandler.auth_code,
                        "redirect_uri": redirect_uri,
                        "code_verifier": code_verifier,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code != 200:
                    logger.error(
                        f"Token exchange failed: {resp.status_code} {resp.text}"
                    )
                    return False
                data = resp.json()
                expires_in = data.get("expires_in", 3600)
                self._tokens = {
                    "type": "oauth",
                    "access": data["access_token"],
                    "refresh": data.get("refresh_token", ""),
                    "expires": int((time.time() + expires_in) * 1000),
                }
                self._save_tokens()
                logger.info("OpenAI Codex OAuth authentication successful")
                return True
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False
