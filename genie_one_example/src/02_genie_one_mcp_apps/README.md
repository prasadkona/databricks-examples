# Mode 2: Genie One MCP Apps client (interactive View)

Local **MCP Apps host** for the same Genie One MCP Service as Mode 1. This
client advertises the MCP Apps UI extension so Genie One prefers **`view_ask`**
and returns its official **Interactive View** (`ui://` HTML). The page chrome
(question box, traces, collapsible overview) is this test app. The iframe
content is **Genie One’s MCP App**, not a custom chart.

Adapted from the official
[MCP Apps basic host](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/basic-host)
(`AppBridge` + `PostMessageTransport`, origin-isolated double iframe).

Product overview: [../../README.md](../../README.md).

## When to use this

Product UIs that can host an iframe (Claude Desktop–style hosts, custom web
clients). If the client cannot implement
[MCP Apps](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx),
use Mode 1 (text tools) instead.

## How the API works in this mode

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

Then:

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

## Rich experience (examples)

The Interactive View can show progress, charts, tables, citations, and
Explore in Genie One — the same Genie One UI shipped as an MCP App:

- [Chart + table](images/02_example1.png)
- [Narrative answer with sources](images/02_example2.png)

## Layout

```text
run_web.sh           # venv, npm build, proxy + host processes
proxy.py             # U2M + Streamable HTTP reverse proxy
src/index.tsx        # test-app chrome, traces, overview
src/implementation.ts
src/sandbox.ts
images/              # screenshots of Genie One’s View in this host
```

## Run

From this directory:

```bash
./run_web.sh
```

The launcher installs missing Python/npm dependencies, builds the TypeScript
host, and starts the proxy, browser host, isolated sandbox origin, and a
temporary OAuth loopback listener. **Listen addresses are printed at
startup.** Complete Databricks sign-in if prompted. Expand **Test app
overview** on the page for the spec walkthrough.

Override the sample question with the env vars in
[`.env.template`](../../.env.template). Do not commit host, client, or
secret values.

## Docs

- [Genie One MCP server](https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp) (MCP App View)
- [MCP Apps specification](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
- [Official basic host](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/basic-host)
- [Managed MCP servers](https://docs.databricks.com/aws/en/agents/mcp-tools/managed-mcp)

To enter host / client ID / secret in a **connection screen** instead of
`.env`, see [Mode 3](../03_genie_one_mcp_apps_improved_ui/).
