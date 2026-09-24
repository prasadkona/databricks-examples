# Integrating with Genie Agents

Examples for calling one curated Databricks **Genie Agent** through its public
REST APIs, including a custom visualization UI. Genie Agents were previously
called Genie Spaces, which is why the Conversation API retains `/spaces/{id}`
URLs.

All examples target the same agent identifier, use OAuth U2M as the signed-in
user, and honor that user’s Unity Catalog access.

## Choose an API

| Example | API | Response model | Best for |
|---|---|---|---|
| **01** | Genie Agent Mode API, `enable_viz=false` | Streaming Server-Sent Events CLI | Inspecting research, SQL, tables, reports, and citations without chart generation |
| **02** | Genie Agent Mode API, `enable_viz=true` | Streaming Server-Sent Events UI | Rendering Genie Agent Mode progress, SQL, answers, citations, and generated charts |
| **03** | Genie Agent Conversation API | Start + poll REST CLI | Simpler chat/Q&A integrations and follow-up conversations |

### 01 — Genie Agent Mode Responses API, visualization disabled

```text
POST /api/2.0/genie/agents/{agent_id}/responses
Accept: text/event-stream
```

Agent Mode creates and refines a research plan, runs SQL, iterates on results,
and streams typed output items. It is currently **Beta** and requires the
workspace preview plus `CAN QUERY` on the agent. This CLI always sends
`enable_viz: false`.

Events:

```text
response.created
response.output_item.added / updated / done
response.completed | response.failed
```

Details:
[`src/01_genie_agents_agent_mode_api/README.md`](src/01_genie_agents_agent_mode_api/README.md)

### 02 — Genie Agent Mode visualization UI

```text
POST /api/2.0/genie/agents/{agent_id}/responses
Accept: text/event-stream
enable_viz: true
```

This local React UI uses the same Agent Mode endpoint as Example 01, but
forces visualization generation on. It displays the SSE lifecycle, narrative,
SQL, table data, citations, and the rendered visualization when the Genie
Agent determines that a chart is appropriate.

Details:
[`src/02_genie_agents_agent_mode_visualization_ui/README.md`](src/02_genie_agents_agent_mode_visualization_ui/README.md)

### 03 — Genie Agent Conversation REST API

```text
POST /api/2.0/genie/spaces/{agent_id}/start-conversation
GET  /api/2.0/genie/spaces/{agent_id}/conversations/{conversation_id}/messages/{message_id}
```

This API starts a thread, returns IDs, and requires polling until the message
is terminal. It can then return text, generated SQL, suggested questions, and
query results. It also supports follow-ups using the same conversation.

Details:
[`src/03_genie_agents_conversation_api/README.md`](src/03_genie_agents_conversation_api/README.md)

## Authentication and configuration

The APIs require the granular OAuth scope **`genie`**. For a browser-present
user, these examples use OAuth U2M authorization code + PKCE. The token is
kept in process memory and never printed.

Examples 01 and 03 load configuration from local environment files. Example
02 collects connection settings in its browser connection screen and keeps
them only in the local proxy process.

Env search order (first existing file wins; exported variables are not overwritten):

1. `GENIE_AGENTS_ENV_FILE`, if set
2. `[folder]/_local/.env` — `[folder]` is the parent of this git checkout
3. `[folder]/genie_agents_*/.env` — this example directory
4. `[folder]/.env` — the git checkout root (the parent of `genie_agents_*`)

Required keys:

```text
DATABRICKS_HOST
APP_AUTH_TYPE=oauth_u2m
DATABRICKS_U2M_CLIENT_ID
DATABRICKS_U2M_CLIENT_SECRET
DATABRICKS_REDIRECT_URI
GENIE_AGENT_ID
```

The same `GENIE_AGENT_ID` goes into both API path families:

```text
/api/2.0/genie/spaces/{GENIE_AGENT_ID}/...
/api/2.0/genie/agents/{GENIE_AGENT_ID}/...
```

Copy [`.env.template`](.env.template) for placeholders. Do not commit real
workspace hosts, agent IDs, OAuth client values, or tokens.

## Run

Agent Mode SSE:

```bash
cd src/01_genie_agents_agent_mode_api
./run.sh
```

Agent Mode visualization UI:

```bash
cd src/02_genie_agents_agent_mode_visualization_ui
./run_web.sh
```

Conversation REST:

```bash
cd src/03_genie_agents_conversation_api
./run.sh
```

Examples 01 and 03 accept `GENIE_QUESTION`. Example 02 starts with a
chart-friendly question that can be edited in the UI.

## Layout

```text
genie_agents_example/
  .env.template
  README.md
  src/
    common/                              # env discovery + OAuth U2M
    01_genie_agents_agent_mode_api/             # SSE CLI, enable_viz=false
    02_genie_agents_agent_mode_visualization_ui/# SSE UI, enable_viz=true
    03_genie_agents_conversation_api/    # start + poll REST
```

## Official documentation

- [Agent mode APIs in Genie Agents](https://docs.databricks.com/aws/en/genie-agents/api)
- [Agent Mode create response](https://docs.databricks.com/api/genie/v1/agent-mode-create-response)
- [Use the Genie Agents Conversation API](https://docs.databricks.com/aws/en/genie-agents/conversation-api)
- [Start conversation](https://docs.databricks.com/api/genie/v1/genie-start-conversation)
- [Get conversation message](https://docs.databricks.com/api/genie/v1/genie-get-conversation-message)
- [Databricks API OAuth scopes](https://docs.databricks.com/api/workspace/scopes)

Management APIs (`POST` / `PATCH /api/2.0/genie/spaces`) and Genie Agent MCP
are intentionally outside this example suite.
