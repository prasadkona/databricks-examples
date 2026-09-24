# Mode 3: Genie One MCP Apps UI with a connection screen

This is Mode 2 plus a **connection configuration screen**. The user does not
need a `.env` file. They open the app, enter Databricks connection details,
complete OAuth, then ask a question.

1. **Connection tab** — workspace host, OAuth client ID, client secret, and
   redirect URL (pre-filled). On-screen instructions: create a Databricks
   OAuth app and set the redirect URL to the value shown in the UI.
2. **Ask Genie One tab** (unlocked only after a successful connection) —
   sample question is pre-filled; edit or clear it. Nothing runs until
   **Execute**.

No Genie One request starts on page load or immediately after connection. Like
Mode 2, the result is Genie One’s official **Interactive View** (`ui://` HTML),
hosted with AppBridge in an origin-isolated double iframe.

> This is a **single-user local example**, not a production multi-user auth
> service. It keeps one access token in process memory. A production service
> must map encrypted per-user tokens to secure server sessions (typically via
> an `HttpOnly` cookie), isolate MCP sessions/traces, and handle token expiry,
> logout, CSRF, rate limits, and multiple backend instances.

Product overview: [../../README.md](../../README.md). Mode 2 baseline:
[`../02_genie_one_mcp_apps/`](../02_genie_one_mcp_apps/).

## When to use this

When you want the same Genie One View as Mode 2, but connection settings should
be entered in the product UI (host, client ID, secret, redirect URL) instead
of a pre-loaded `.env`. If the client cannot implement
[MCP Apps](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx),
use Mode 1 (text tools) instead.

## Connection step

The UI requires:

- Databricks workspace host
- OAuth app client ID
- OAuth app client secret
- Redirect URL (pre-filled with the local callback shown in the UI)

Before connecting, create or open a Databricks OAuth app, add the exact
redirect URL shown in the UI, and save the app. It must be allowed to request
the **`ai-gateway`** scope.

The browser sends these values only to the local proxy. The proxy runs OAuth
U2M authorization code + PKCE, opens Databricks sign-in, and exchanges the
callback code server-side. Values are not written to files or browser storage.
The UI clears its client-secret field after success. The proxy retains only
the access token, MCP endpoint, and workspace hostname in process memory.

Blank fields, an invalid host/redirect URL, OAuth rejection, or MCP discovery
failure keeps the user on the Connection tab with an actionable error. Only
successful OAuth plus MCP `initialize` / `tools/list` enables the **Ask Genie One** tab.

## Query and MCP Apps flow

The Ask Genie One tab starts with the sample question used by the other examples. The
user can edit it or select **Clear**. **Execute** is disabled while it is blank.
Nothing executes until the user selects Execute.

**Same Unity Gateway MCP Service** as Mode 1. OAuth U2M with scope
**`ai-gateway`**. The bearer token stays on the **server-side proxy**; it
never enters browser JavaScript, query strings, host logs, or git.

**Initialize** must include:

```json
{
  "capabilities": {
    "extensions": {
      "io.modelcontextprotocol/ui": {
        "mimeTypes": ["text/html;profile=mcp-app"]
      }
    }
  }
}
```

On Execute:

1. `tools/list` — expect `view_ask` (preferred over `genie_ask`).
2. Call **`view_ask` once** with the question. Response `_meta.ui.resourceUri`
   is a `ui://…` resource.
3. `resources/read` the HTML (`text/html;profile=mcp-app`).
4. Load it with AppBridge + PostMessageTransport in the sandboxed iframe.
5. The View itself polls **`view_poll_response`**. The host must not treat
   extra `view_ask` calls as status checks.

## Architecture

```text
Browser host (this test UI)
  ├── Streamable HTTP → local authenticated reverse proxy → Unity Gateway MCP
  └── outer sandbox origin
        └── inner iframe (Genie One View HTML)
```

The proxy forwards MCP bodies and session headers (`Mcp-Session-Id`,
`Mcp-Protocol-Version`, `Last-Event-ID`) without exposing the token to the
page.

## Layout

```text
run_web.sh           # venv, npm build, proxy + host processes
proxy.py             # in-memory configuration, U2M, authenticated MCP proxy
src/index.tsx        # Connection and Query tabs, validation, traces
src/implementation.ts
src/sandbox.ts
```

## Run

From this directory:

```bash
./run_web.sh
```

The launcher installs missing Python/npm dependencies, builds the TypeScript
host, and starts the proxy, browser host, isolated sandbox origin, and a
temporary OAuth loopback listener. **Listen addresses are printed at
startup.** Fill in the connection screen, complete Databricks sign-in, then
use **Ask Genie One**. Do not commit host, client, or secret values.

## Docs

- [Genie One MCP server](https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp) (MCP App View)
- [MCP Apps specification](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
- [Official basic host](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/basic-host)
- [Managed MCP servers](https://docs.databricks.com/aws/en/agents/mcp-tools/managed-mcp)
