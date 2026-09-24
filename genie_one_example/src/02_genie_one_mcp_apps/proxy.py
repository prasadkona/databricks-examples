"""Server-side U2M-authenticated Streamable HTTP proxy for Genie One MCP."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from env_load import load_connection_env  # noqa: E402
from mcp_client import MCP_SERVICE_PATH  # noqa: E402
from u2m_auth import acquire_access_token, normalize_host  # noqa: E402

UPSTREAM_HEADERS = {
    "accept",
    "content-type",
    "last-event-id",
    "mcp-protocol-version",
    "mcp-session-id",
}
DOWNSTREAM_HEADERS = {
    "cache-control",
    "content-type",
    "mcp-session-id",
    "retry-after",
}
traces: deque[dict[str, object]] = deque(maxlen=200)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    env_path = load_connection_env()
    # Unity Gateway's Genie One service requires this exact scope. Override the
    # legacy `genie` value in the shared env without exposing any credentials.
    os.environ["DATABRICKS_OAUTH_SCOPE"] = "ai-gateway"
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required")

    print(f"[proxy] Loaded connection settings from {env_path}")
    print("[proxy] Authenticating with OAuth U2M scope: ai-gateway")
    token = acquire_access_token()
    app.state.endpoint = f"{normalize_host(host)}{MCP_SERVICE_PATH}"
    app.state.token = token
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=30))
    print("[proxy] Authentication complete; bearer token remains server-side")
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Genie One MCP local proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id", "mcp-session-id"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready", "scope": "ai-gateway"}


@app.get("/api/traces")
async def get_traces() -> dict[str, list[dict[str, object]]]:
    return {"traces": list(traces)}


def safe_request_summary(body: bytes) -> tuple[str, str | None]:
    """Extract operation names only; never retain arguments or credentials."""
    try:
        import json

        payload = json.loads(body) if body else {}
        method = str(payload.get("method") or "")
        params = payload.get("params") or {}
        tool = str(params.get("name") or "") or None
        return method, tool
    except (ValueError, TypeError, AttributeError):
        return "", None


@app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def proxy_mcp(request: Request) -> Response:
    body = await request.body()
    method, tool = safe_request_summary(body)
    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in UPSTREAM_HEADERS
    }
    request_headers["Authorization"] = f"Bearer {request.app.state.token}"
    request_headers["User-Agent"] = "genie-one-mcp-apps-local-host/1.0"
    request_headers["Accept-Encoding"] = "identity"

    started = time.perf_counter()
    upstream_request = request.app.state.client.build_request(
        request.method,
        request.app.state.endpoint,
        headers=request_headers,
        content=body,
    )
    upstream = await request.app.state.client.send(upstream_request, stream=True)
    duration_ms = round((time.perf_counter() - started) * 1000)
    traces.append(
        {
            "sequence": len(traces) + 1,
            "method": method or request.method,
            "tool": tool,
            "status": upstream.status_code,
            "durationMs": duration_ms,
        }
    )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() in DOWNSTREAM_HEADERS
    }

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            # Yield decoded bytes because hop-by-hop compression headers are
            # intentionally not forwarded to the browser.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=None,
    )


@app.exception_handler(httpx.HTTPError)
async def upstream_error(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
    # Do not include request headers because they contain the bearer token.
    return JSONResponse(
        status_code=502,
        content={"error": "Databricks MCP upstream request failed", "detail": str(exc)},
    )
