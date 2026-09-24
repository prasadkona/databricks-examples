"""OAuth U2M authorization-code + PKCE for Genie Agent workspace REST APIs."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests

DEFAULT_REDIRECT_URI = "http://localhost:8020/callback"
DEFAULT_SCOPE = "genie"
PKCE_TIMEOUT_SECONDS = 180


def normalize_host(host: str) -> str:
    """Normalize a Databricks workspace host without changing its hostname."""
    host = host.strip().rstrip("/")
    if not host.startswith(("https://", "http://")):
        host = f"https://{host}"
    return host


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
    client_secret: Optional[str],
) -> str:
    """Exchange a U2M authorization code without printing credentials."""
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
        f"{normalize_host(host)}/oidc/v1/token",
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"Token exchange failed (HTTP {response.status_code}): "
            f"{response.text[:2000]}",
            response=response,
        )
    payload = response.json()
    if payload.get("scope"):
        print(f"Granted OAuth scope: {payload['scope']}")
    return str(payload["access_token"])


def run_pkce_browser_flow(
    host: str,
    *,
    client_id: str,
    redirect_uri: str,
    client_secret: Optional[str],
    scope: str = DEFAULT_SCOPE,
    timeout_seconds: int = PKCE_TIMEOUT_SECONDS,
) -> str:
    """Open browser login, receive the loopback callback, and return a token."""
    host = normalize_host(host)
    verifier, challenge = generate_pkce_pair()
    expected_state = secrets.token_urlsafe(16)
    callback = urlparse(redirect_uri)
    if callback.scheme != "http" or callback.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("DATABRICKS_REDIRECT_URI must be an HTTP localhost callback.")
    if not callback.port:
        raise ValueError("DATABRICKS_REDIRECT_URI must include a localhost port.")

    authorize_url = f"{host}/oidc/v1/authorize?{urlencode({
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
        'state': expected_state,
    })}"

    class CallbackHandler(BaseHTTPRequestHandler):
        authorization_code: Optional[str] = None
        oauth_error: Optional[str] = None
        received = threading.Event()

        def do_GET(self) -> None:  # type: ignore[override]
            parsed = urlparse(self.path)
            if parsed.path != callback.path:
                self.send_response(404)
                self.end_headers()
                return
            query = parse_qs(parsed.query)
            if query.get("state", [""])[0] != expected_state:
                CallbackHandler.oauth_error = "OAuth callback state did not match."
            elif query.get("error"):
                CallbackHandler.oauth_error = (
                    f"{query.get('error', [''])[0]}: "
                    f"{query.get('error_description', [''])[0]}"
                )
            else:
                CallbackHandler.authorization_code = query.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Authentication complete. You may close this tab.")
            CallbackHandler.received.set()

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer((callback.hostname, callback.port), CallbackHandler)
    server.timeout = 1
    print(f"Opening browser for Databricks U2M sign-in ({host})")
    print(f"Requested scope: {scope}")
    print(f"Redirect URI:    {redirect_uri}")
    webbrowser.open(authorize_url)

    deadline = time.monotonic() + timeout_seconds
    try:
        while not CallbackHandler.received.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if CallbackHandler.oauth_error:
        raise RuntimeError(CallbackHandler.oauth_error)
    if not CallbackHandler.authorization_code:
        raise TimeoutError(
            "OAuth callback timed out. Finish sign-in and confirm the registered "
            "redirect URL exactly matches DATABRICKS_REDIRECT_URI."
        )
    return exchange_authorization_code(
        host,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code=CallbackHandler.authorization_code,
        code_verifier=verifier,
        client_secret=client_secret,
    )


def acquire_access_token(*, scope: str = DEFAULT_SCOPE) -> str:
    """Acquire a token for the signed-in user; never print the token."""
    auth_type = os.environ.get("APP_AUTH_TYPE", "oauth_u2m").strip().lower()
    if auth_type == "u2m_token_env":
        token = os.environ.get("DATABRICKS_ACCESS_TOKEN", "").strip()
        if not token:
            raise EnvironmentError("u2m_token_env requires DATABRICKS_ACCESS_TOKEN.")
        print("Using DATABRICKS_ACCESS_TOKEN (value not printed).")
        return token
    if auth_type not in {"oauth_u2m", "u2m_custom_oauth_app"}:
        raise ValueError(
            f"Unsupported APP_AUTH_TYPE '{auth_type}'. Use oauth_u2m."
        )

    host = os.environ.get("DATABRICKS_HOST", "").strip()
    client_id = os.environ.get("DATABRICKS_U2M_CLIENT_ID", "").strip()
    if not host:
        raise EnvironmentError("DATABRICKS_HOST is required.")
    if not client_id:
        raise EnvironmentError("DATABRICKS_U2M_CLIENT_ID is required.")
    return run_pkce_browser_flow(
        host,
        client_id=client_id,
        client_secret=(
            os.environ.get("DATABRICKS_U2M_CLIENT_SECRET", "").strip() or None
        ),
        redirect_uri=(
            os.environ.get("DATABRICKS_REDIRECT_URI", DEFAULT_REDIRECT_URI).strip()
            or DEFAULT_REDIRECT_URI
        ),
        scope=scope,
    )
