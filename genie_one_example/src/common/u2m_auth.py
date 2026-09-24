"""OAuth U2M (authorization code + PKCE) via REST only — no Databricks SDK.

Genie One MCP on Unity Gateway is an on-behalf-of-user API. This module
opens a browser, listens on DATABRICKS_REDIRECT_URI, and exchanges the
code for an access token.

Docs (endpoint + required ``ai-gateway`` scope):
    https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests

_DEFAULT_REDIRECT_URI = "http://localhost:8020/callback"
_PKCE_TIMEOUT_SEC = 180
_U2M_AUTH_TYPES = frozenset({"oauth_u2m", "u2m_custom_oauth_app"})
_TOKEN_ENV_AUTH_TYPES = frozenset({"u2m_token_env"})
# Unity Gateway MCP Services require the ai-gateway scope. The older Genie One MCP
# endpoint used `genie` and is deprecated.
_DEFAULT_SCOPE = "ai-gateway"


def normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith("https://") and not host.startswith("http://"):
        host = f"https://{host}"
    return host


def oauth_scope() -> str:
    """Scope requested at authorize time.

    ``ai-gateway`` is required for ``/ai-gateway/mcp-services/system.ai.genie_one_mcp``.
    ``genie`` applied only to the deprecated ``/api/2.0/mcp/genie`` URL.
    """
    requested = os.environ.get("DATABRICKS_OAUTH_SCOPE", "").strip()
    if not requested or requested == "genie":
        return _DEFAULT_SCOPE
    return requested


def _force_fresh_login() -> bool:
    return os.environ.get("DATABRICKS_OAUTH_FORCE_LOGIN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def generate_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def exchange_authorization_code(
    host: str,
    *,
    client_id: str,
    redirect_uri: str,
    code: str,
    code_verifier: str,
    client_secret: Optional[str] = None,
) -> str:
    token_url = f"{normalize_host(host)}/oidc/v1/token"
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        token_data["client_secret"] = client_secret

    response = requests.post(
        token_url,
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"Token exchange failed HTTP {response.status_code}: {response.text[:2000]}",
            response=response,
        )
    payload = response.json()
    granted = payload.get("scope")
    if granted:
        print(f"Granted OAuth scope: {granted}")
    return payload["access_token"]


def run_pkce_browser_flow(
    host: str,
    *,
    client_id: str,
    redirect_uri: str = _DEFAULT_REDIRECT_URI,
    client_secret: Optional[str] = None,
    scope: Optional[str] = None,
    timeout_sec: int = _PKCE_TIMEOUT_SEC,
) -> str:
    host = normalize_host(host)
    scope = scope or oauth_scope()
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if _force_fresh_login():
        auth_params["prompt"] = "login"

    auth_url = f"{host}/oidc/v1/authorize?{urlencode(auth_params)}"
    parsed = urlparse(redirect_uri)
    port = int(parsed.port or 8020)

    class CallbackHandler(BaseHTTPRequestHandler):
        auth_code: Optional[str] = None
        oauth_error: Optional[str] = None
        callback_received = threading.Event()

        def do_GET(self) -> None:  # type: ignore[override]
            query = parse_qs(urlparse(self.path).query)
            if "/callback" in self.path:
                if query.get("error"):
                    CallbackHandler.oauth_error = (
                        f"{query.get('error', [''])[0]}: "
                        f"{query.get('error_description', [''])[0]}"
                    )
                else:
                    CallbackHandler.auth_code = query.get("code", [None])[0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Authentication complete. You may close this tab.")
                CallbackHandler.callback_received.set()
            else:
                self.send_response(204)
                self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    CallbackHandler.auth_code = None
    CallbackHandler.oauth_error = None
    CallbackHandler.callback_received.clear()

    server = HTTPServer(("localhost", port), CallbackHandler)
    server.timeout = timeout_sec

    print(f"Opening browser for Databricks U2M sign-in ({host})")
    print(f"Requested scope: {scope}")
    print(f"Redirect URI:    {redirect_uri}")
    webbrowser.open(auth_url)

    while not CallbackHandler.callback_received.is_set():
        server.handle_request()
    server.server_close()

    if CallbackHandler.oauth_error:
        raise RuntimeError(
            "OAuth authorize returned an error in the callback: "
            f"{CallbackHandler.oauth_error}"
        )
    if not CallbackHandler.auth_code:
        raise RuntimeError(
            "PKCE timed out — no authorization code received. "
            "Finish sign-in in the browser and match DATABRICKS_REDIRECT_URI to the OAuth app."
        )

    return exchange_authorization_code(
        host,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code=CallbackHandler.auth_code,
        code_verifier=verifier,
        client_secret=client_secret,
    )


def acquire_access_token(*, scope_override: Optional[str] = None) -> str:
    """Acquire a U2M token, optionally for a mode-specific OAuth scope.

    Genie One MCP through Unity Gateway uses ``ai-gateway``. The separate
    Genie One Responses API uses ``genie`` according to its public API spec.
    Callers must choose the scope for the API surface they are testing.
    """
    auth_type = os.environ.get("APP_AUTH_TYPE", "oauth_u2m").strip().lower() or "oauth_u2m"
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if not host:
        raise EnvironmentError("DATABRICKS_HOST is required.")

    if auth_type in _TOKEN_ENV_AUTH_TYPES:
        token = os.environ.get("DATABRICKS_ACCESS_TOKEN", "").strip()
        if not token:
            raise EnvironmentError("u2m_token_env requires DATABRICKS_ACCESS_TOKEN.")
        print("Using DATABRICKS_ACCESS_TOKEN from the environment (value not printed).")
        return token

    if auth_type not in _U2M_AUTH_TYPES:
        raise ValueError(
            f"Unsupported APP_AUTH_TYPE '{auth_type}'. "
            "This example uses U2M: oauth_u2m or u2m_custom_oauth_app."
        )

    client_id = os.environ.get("DATABRICKS_U2M_CLIENT_ID", "").strip()
    if not client_id:
        raise EnvironmentError(
            "U2M requires DATABRICKS_U2M_CLIENT_ID from Account Console → App connections."
        )
    redirect_uri = os.environ.get("DATABRICKS_REDIRECT_URI", _DEFAULT_REDIRECT_URI).strip()
    client_secret = os.environ.get("DATABRICKS_U2M_CLIENT_SECRET", "").strip() or None
    return run_pkce_browser_flow(
        host,
        client_id=client_id,
        redirect_uri=redirect_uri,
        client_secret=client_secret,
        scope=scope_override or oauth_scope(),
    )
