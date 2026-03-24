# Agent & Deployment Deep Dive

This document covers building the multi-agent orchestrator, testing it locally, deploying it as a Databricks App, and running comprehensive tests against the deployed endpoint.

## Agent Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                               │
│                    (agent_server/start_server.py)                     │
│                                                                       │
│   POST /invocations ──► MLflow ResponsesAgent ──► agent.py           │
│   POST /chat         ──► (chat proxy)                                 │
│   GET  /health       ──► health check                                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│               SEC Financial Analyst Orchestrator                      │
│                        (agent.py)                                     │
│                                                                       │
│   Model: Claude 3.7 Sonnet (databricks-claude-3-7-sonnet)           │
│   Framework: OpenAI Agents SDK                                        │
│   Tracing: MLflow auto-instrumented                                   │
│   Max turns: 25                                                       │
│                                                                       │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    Routing Instructions                       │   │
│   │                                                               │   │
│   │  Quantitative ──► Genie Space (revenue, margins, stock)     │   │
│   │  Qualitative  ──► Knowledge Assistant (risk factors, MD&A)  │   │
│   │  Analytical   ──► UC Functions (valuation, comparison)      │   │
│   │  Comprehensive ──► Multiple sources combined                │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   Tools registered:                                                   │
│                                                                       │
│   ┌─────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│   │ MCP Server  │  │ Serving Endpoint    │  │ Function Tools     │  │
│   │ (Genie)     │  │ (query_knowledge_   │  │                    │  │
│   │             │  │  assistant)         │  │ get_valuation_score│  │
│   │ Natural     │  │                     │  │ compare_peers      │  │
│   │ language    │  │ RAG over SEC 10-K   │  │ get_growth_        │  │
│   │ SQL via MCP │  │ PDF documents       │  │   trajectory       │  │
│   │ protocol    │  │                     │  │ get_risk_summary   │  │
│   └─────────────┘  └─────────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## How the Agent Works

### Tool Types

The orchestrator registers three categories of tools:

**1. Genie Space (MCP Server)**

Connected via `databricks_openai.agents.McpServer`, the Genie Space is exposed as an MCP tool. The agent sends natural language questions and the Genie backend translates them to SQL, executes against the gold tables, and returns structured results.

```python
McpServer(
    url=build_mcp_url(f"/api/2.0/mcp/genie/{GENIE_SPACE_ID}"),
    name="Query structured financial data..."
)
```

**2. Knowledge Assistant (Serving Endpoint)**

The KA is a Databricks serving endpoint that provides RAG over the indexed SEC PDF documents. Wrapped as a function tool via `_make_subagent_tool()`, it sends questions to the endpoint and returns the response text.

```python
async def _call(question: str) -> str:
    response = await _tool_client.responses.create(
        model=endpoint,
        input=[{"role": "user", "content": question}],
    )
    return response.output_text
```

**3. UC Functions (Function Tools)**

Four `@function_tool`-decorated async functions that execute SQL against Unity Catalog functions via the Statement Execution API:

| Function | UC Function Called | Returns |
|----------|-------------------|---------|
| `get_valuation_score(ticker)` | `sec_fin_valuation_score` | Score 1-100, recommendation, component breakdown |
| `compare_peers(ticker)` | `sec_fin_compare_peers` | Metric, company value, peer average, above/below |
| `get_growth_trajectory(ticker)` | `sec_fin_growth_trajectory` | Growth metrics and classification |
| `get_risk_summary(ticker)` | `sec_fin_risk_summary` | Risk type, severity (HIGH/MEDIUM/LOW), description |

### Routing Logic

The orchestrator's system prompt instructs Claude to route queries:

- **Quantitative** ("What is NVIDIA's revenue?") → Genie Space
- **Qualitative** ("What are Apple's risk factors?") → Knowledge Assistant
- **Analytical** ("Is NVIDIA overvalued?") → UC Functions
- **Comprehensive** ("Should I invest in NVIDIA?") → Combines multiple tools

The agent can make up to 25 tool-calling turns per request (`max_turns=25`), allowing it to gather data from multiple sources and synthesize a coherent answer.

### MLflow Integration

The agent is fully instrumented via MLflow:
- `mlflow.openai.autolog()` auto-traces all LLM calls
- Session tracking via `mlflow.trace.session` metadata
- Both `invoke` and `stream` handlers are decorated with MLflow's `@invoke()` / `@stream()`
- All tool calls, inputs, and outputs are captured in traces

## Test Agent Locally

**Script:** `notebooks/agent_src/test_agent.py`
**Command:** `uv run test-agent`

### What It Does

1. Starts the FastAPI server as a subprocess (unless `--no-start`)
2. Waits for health check to pass
3. Sends test queries to `POST /invocations`
4. Validates response content (keyword matching)
5. Optionally validates MLflow traces for expected tool calls

### Test Modes

```bash
uv run test-agent                        # default: 1 smoke test
uv run test-agent --full                 # all 8 test scenarios
uv run test-agent --skip-trace-validation # skip MLflow trace checks
uv run test-agent --no-start             # test against running server
```

### Smoke Test (Default)

A single comprehensive query that exercises all three tool types:

> "Provide a comprehensive investment analysis of NVIDIA (NVDA). Include their financial performance metrics from the database, key risk factors and strategic insights from their SEC 10-K filing, and compute their valuation score and peer comparison. Synthesize everything into a clear investment recommendation."

This validates that the agent can orchestrate across Genie, KA, and UC functions in a single request.

### Full Test Suite (--full)

8 test scenarios covering each tool type individually and in combination:

| Test | Tool(s) | Query |
|------|---------|-------|
| Genie: Financial Query | Genie | NVIDIA total revenue FY2024 |
| Genie: Stock Query | Genie | Apple stock price performance |
| KA: Risk Factors | KA | Apple's key risk factors from 10-K |
| KA: Business Strategy | KA | NVIDIA's AI strategy from 10-K |
| UC: Valuation Score | UC | NVIDIA's valuation score |
| UC: Peer Comparison | UC | Compare NVIDIA to peers |
| UC: Risk Summary | UC | Apple's risk assessment |
| Multi-Tool: Investment | All | Full investment analysis of NVIDIA |

### Request/Response Flow

```
Test Runner                    Agent Server
    │                              │
    │  POST /invocations           │
    │  {"input": [{...}]}          │
    │──────────────────────────────►│
    │                              │──► Claude routes query
    │                              │──► Tool call(s)
    │                              │──► Claude synthesizes
    │  {"output": [{...}]}         │
    │◄──────────────────────────────│
    │                              │
    │  Validate response keywords  │
    │  Validate MLflow traces      │
```

## Deploy to Databricks Apps

**Script:** `notebooks/agent_src/deploy_agent_app.py`
**Command:** `uv run deploy-agent-app`

### Deployment Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Bundle   │     │  Source   │     │  Save    │     │  Grant   │     │  Verify  │
│  Deploy   │────►│  Code    │────►│  App     │────►│  SP      │────►│  1 Query │
│           │     │  Deploy  │     │  Endpoint│     │  Perms   │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

1. **Bundle Deploy** - `databricks bundle deploy -t dev` from `app/` directory
2. **Source Code Deploy** - `databricks apps deploy` pushes the app source code
3. **Save App Endpoint** - Captures `APP_URL`, `APP_SP_CLIENT_ID`, `APP_SP_ID` to central config
4. **Grant SP Permissions** - Grants `SELECT` and `EXECUTE` on UC schema to the app's Service Principal
5. **Verify Deployment** - Sends 1 test query with verbose output (full request/response printed)

### Usage

```bash
uv run deploy-agent-app                    # deploy + verify (default)
uv run deploy-agent-app --skip-test        # deploy without verification query
uv run deploy-agent-app --skip-validation  # skip local agent test before deploy
uv run deploy-agent-app -t prod            # deploy to prod target
```

### App DAB Configuration

The `app/databricks.yml` configures:

**Environment Variables** - Passed to the deployed app container:

```yaml
env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: UC_CATALOG
    value: "your_catalog"
  - name: GENIE_SPACE_ID
    value: ""
  # ... all config values
```

**Resources** - Grants the app's Service Principal access:

```yaml
resources:
  - name: knowledge_assistant
    serving_endpoint:
      name: 
      permission: CAN_QUERY
  - name: sql_warehouse
    sql_warehouse:
      id: your-warehouse-id
      permission: CAN_USE
```

### Service Principal Authentication

When deployed, the app runs as a Service Principal (SP). The DAB automatically:
- Creates/assigns an SP to the app
- Grants `CAN_QUERY` on the KA serving endpoint
- Grants `CAN_USE` on the SQL warehouse

The `deploy-agent-app` script additionally grants:
- `SELECT` on all tables in the UC schema (for UC function queries)
- `EXECUTE` on all functions in the UC schema

These permissions are granted via the Unity Catalog Permissions REST API:

```
PUT /api/2.1/unity-catalog/permissions/schema/{catalog}.{schema}
```

### Deployed App Endpoint

After deployment, the app URL and SP details are saved to the central config:

```bash
APP_URL=https://agent-sec-financial-analyst-v2-{workspace-id}.your-workspace.cloud.databricks.com
APP_SP_CLIENT_ID=5027b6b5-de5a-4ac4-b3c4-9ed76c795b16
APP_SP_ID=8395348716507
```

### Verification Query

After deployment, the deploy script sends one test query with full verbose output:

```
▶ Verification query against deployed app

  Request:
    URL: https://...
    Auth: OAuth token (first 20 chars shown)
    Body: {"input": [{"role": "user", "content": "..."}]}

  Response:
    Status: 200
    Tool calls detected: genie, knowledge_assistant, valuation_score
    Agent response: (full text)
```

## Test Deployed App

**Script:** `notebooks/agent_src/test_deployed_agent_app.py`
**Command:** `uv run test-agent-app`

### Test Modes

```bash
uv run test-agent-app                            # 1 smoke test
uv run test-agent-app --full                     # all 8 agent test scenarios
uv run test-agent-app --test-services            # test backend services individually
uv run test-agent-app --test-services --full     # services + all agent tests
uv run test-agent-app --app-url https://...      # override app URL
```

### Testing Flow

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  OAuth     │     │  Health      │     │  Service     │     │  Agent   │
│  Token     │────►│  Check       │────►│  Tests       │────►│  Tests   │
│            │     │              │     │  (optional)  │     │          │
└────────────┘     └──────────────┘     └──────────────┘     └──────────┘
                                               │                    │
                                               ▼                    ▼
                                        ┌──────────────┐     ┌──────────┐
                                        │  UC funcs    │     │  Smoke   │
                                        │  Genie API   │     │  or Full │
                                        │  KA endpoint │     │  Suite   │
                                        └──────────────┘     └──────────┘
                                               │                    │
                                               └────────┬───────────┘
                                                        ▼
                                                 ┌──────────────┐
                                                 │  Final       │
                                                 │  Report      │
                                                 └──────────────┘
```

### Backend Service Tests (`--test-services`)

Tests each backend service independently, bypassing the agent:

| Service | Method | What It Tests |
|---------|--------|---------------|
| UC: valuation_score | SQL Statement API | `SELECT * FROM sec_fin_valuation_score('NVDA')` |
| UC: compare_peers | SQL Statement API | `SELECT * FROM sec_fin_compare_peers('NVDA')` |
| Genie Space | Genie Conversation API | `POST /api/2.0/genie/spaces/{id}/start-conversation` |
| Knowledge Assistant | Serving Endpoint API | `POST /serving-endpoints/{endpoint}/invocations` |

This helps isolate whether failures come from the agent logic or the underlying services.

### Agent Tests

Same test cases as the local agent tests, but sent to the deployed app URL with OAuth authentication instead of the local server with no auth.

The deployed app requires an OAuth token, obtained via:

```python
databricks auth token --host $DATABRICKS_HOST --profile $PROFILE
```

### Final Report

After all tests complete, the test script prints a comprehensive report:

```
═══════════════════════════════════════
         FINAL TEST REPORT
═══════════════════════════════════════

Backend Service Tests:
  ✓ UC: valuation_score       3.2s
  ✓ UC: compare_peers         2.8s
  ✓ Genie Space               5.1s
  ✓ Knowledge Assistant       4.5s

Agent Endpoint Tests:
  ✓ Smoke: Genie+KA+UC       45.2s

Overall: PASS (5/5 passed)
Total time: 60.8s
```

## App Structure

The deployed app is self-contained in the `app/` directory:

```
app/
├── agent_server/
│   ├── agent.py              # Multi-agent orchestrator
│   ├── start_server.py       # FastAPI server (MLflow AgentServer)
│   └── utils.py              # MCP URL builder, stream helpers
├── databricks.yml            # DAB config (env vars, SP resources)
├── pyproject.toml            # App dependencies
└── uv.lock                   # Locked dependencies
```

### Key Dependencies

From `app/pyproject.toml`:
- `openai-agents` - OpenAI Agents SDK
- `databricks-openai` - Databricks OpenAI client with MCP support
- `mlflow[databricks]` - MLflow with Databricks integration
- `databricks-sdk` - Databricks Python SDK
- `uvicorn` - ASGI server

## Configuration

All agent configuration is driven by environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `UC_CATALOG` | `your_catalog` | Unity Catalog for functions and tables |
| `UC_SCHEMA` | `your_schema` | Schema for functions and tables |
| `TABLE_PREFIX` | `sec_fin_` | Prefix for UC function names |
| `SQL_WAREHOUSE_ID` | `your-warehouse-id` | Warehouse for UC function execution |
| `GENIE_SPACE_ID` | (from config) | Genie Space for structured queries |
| `KA_ENDPOINT` | `` | Knowledge Assistant serving endpoint |

When running locally, these are loaded from `_local/config/databricks.env`.
When deployed, they are injected via `databricks.yml` environment configuration.

## Troubleshooting

### "Max turns (10) exceeded"

The agent's `max_turns` is set to 25. If you still hit this, the agent may be stuck in a tool-calling loop. Check MLflow traces to see which tools are being called repeatedly.

### HTTP 504 Gateway Timeout on deployed app

The Databricks Apps reverse proxy has a ~120 second timeout. Complex multi-tool queries that take longer will be cut off. Solutions:
- Use simpler queries that invoke fewer tools
- Test locally with `uv run test-agent` first (no proxy timeout)
- The smoke test is designed to stay within the timeout

### "INSUFFICIENT_PERMISSIONS" errors

The app's Service Principal needs:
- `CAN_QUERY` on the KA serving endpoint (granted by DAB)
- `CAN_USE` on the SQL warehouse (granted by DAB)
- `SELECT` on UC schema tables (granted by deploy script)
- `EXECUTE` on UC schema functions (granted by deploy script)

If you redeploy with a new SP, run `deploy-agent-app` again to re-grant permissions.

### Agent returns empty or irrelevant responses

Check MLflow traces in the Databricks UI:
1. Navigate to the MLflow experiment
2. Find the trace for your request
3. Inspect which tools were called and their return values
4. Verify the routing instructions in `agent.py` match your query type

### OAuth token errors when testing deployed app

Ensure your Databricks CLI profile has a valid token:

```bash
databricks auth login --profile sec_financial_analyst
```

The `test-agent-app` script obtains an OAuth token automatically using the CLI.

### App crashes after deploy

Check the app logs via the Databricks UI (Apps page). Common causes:
- Missing environment variables in `databricks.yml`
- Dependency version conflicts (check `uv.lock` exists in `app/`)
- MLflow experiment ID not accessible to the Service Principal
