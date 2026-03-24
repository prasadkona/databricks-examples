# SEC Financial Analyst - Multi-Agent AI System

An end-to-end AI agent system built on Databricks that analyzes SEC 10-K filings and financial data. Drop any company's SEC filings into a Unity Catalog volume — the pipeline automatically discovers companies, extracts financial metrics, and loads stock data. The agent intelligently routes user questions to three specialized data sources, combining structured data analytics, unstructured document analysis, and complex computations into a single conversational interface.

## The Use Case

Investment analysts spend hours manually cross-referencing financial filings, market data, and analytical models. This project demonstrates how a multi-agent AI orchestrator automates that workflow:

- **"What is the company's revenue and how does it compare to peers?"** — Queries structured financial tables via Genie Space, then runs a UC function for peer comparison
- **"What are the key risk factors from the 10-K?"** — Searches SEC filing PDFs via Knowledge Assistant
- **"Should I invest in this company?"** — Orchestrates across all three sources: valuation score (UC function), revenue data (Genie), and risk factors (Knowledge Assistant)

The agent decides which tools to use, calls them in the right order, and synthesizes a coherent answer — all traced via MLflow for observability.

## End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User / Browser                                │
│                     (Databricks App or Local Server)                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  HTTP /invocations
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SEC Financial Analyst Orchestrator                      │
│              (Claude 3.7 Sonnet · OpenAI Agents SDK · MLflow)              │
│                                                                             │
│   Routing Logic: quantitative → Genie, qualitative → KA, analytical → UC  │
└────────────┬──────────────────────┬───────────────────────┬─────────────────┘
             │                      │                       │
             ▼                      ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Genie Space     │   │ Knowledge Assistant  │   │   UC Functions       │
│  (MCP Server)    │   │ (Serving Endpoint)   │   │   (SQL Warehouse)    │
│                  │   │                      │   │                      │
│ Natural language │   │ RAG over SEC 10-K    │   │ valuation_score()    │
│ SQL queries over │   │ PDF filings:         │   │ compare_peers()      │
│ structured data: │   │                      │   │ growth_trajectory()  │
│                  │   │ • Risk factors       │   │ risk_summary()       │
│ • Revenue        │   │ • MD&A analysis      │   │                      │
│ • Margins        │   │ • Business strategy  │   │ Composite scoring,   │
│ • Stock prices   │   │ • Accounting policy  │   │ cross-company        │
│ • Segments       │   │ • Legal proceedings  │   │ comparisons          │
│ • Geography      │   │                      │   │                      │
└────────┬─────────┘   └──────────┬───────────┘   └──────────┬───────────┘
         │                        │                           │
         ▼                        ▼                           ▼
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Delta Tables     │   │ UC Volume            │   │ Delta Tables         │
│ (Gold layer)     │   │ (PDF documents)      │   │ (Gold layer)         │
│                  │   │                      │   │                      │
│ SDP Pipeline     │   │ SEC 10-K filings     │   │ Same tables as       │
│ output: company  │   │ (any company —       │   │ Genie, accessed      │
│ financials,      │   │ auto-discovered)     │   │ via SQL functions    │
│ segments, stock  │   │                      │   │                      │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
```

## Two Phases

The project has two distinct phases, each documented in its own deep-dive README:

### Phase 1: Data Foundation

Build the complete data platform: ingest SEC filings, transform raw data through a medallion architecture pipeline, create analytical views and UC functions, and set up a Genie Space for natural language SQL.

**[Data Pipeline Deep Dive (README-data-pipeline.md)](README-data-pipeline.md)**

**[AI Functions Deep Dive (README-sdp-ai-functions.md)](README-sdp-ai-functions.md)** — How `ai_parse_document`, `ai_classify`, and `ai_extract` work together.

```
SEC PDFs ──► UC Volume ──► Knowledge Assistant (RAG)
                  │
                  ▼
            SDP Pipeline (ai_parse → ai_classify → ai_extract)
                  │
                  ├──► company_tickers_registry (auto-discovered)
                  │              │
                  │              ▼
                  │        yfinance API ──► Stock data
                  │                              │
                  ▼                              ▼
            Bronze → Silver → Gold Delta Tables ─┬─► Views
                                                 ├─► UC Functions
                                                 └─► Genie Space
```

> **AI Functions:** The pipeline uses three Databricks AI functions to process SEC filings — each document is parsed only once. See **[README-sdp-ai-functions.md](README-sdp-ai-functions.md)** for the full deep dive.

### Phase 2: AI Agent

Build the multi-agent orchestrator, test it locally, deploy as a Databricks App with automated Service Principal permissions, and run comprehensive tests against the deployed endpoint.

**[Agent & Deployment Deep Dive (README-agent.md)](README-agent.md)**

**[Execution & Testing Workflow (README-execution.md)](README-execution.md)** - Complete guide to all execution options, flags, and workflows.

```
agent.py ──► local test ──► deploy to Databricks App ──► test deployed app
```

## Technology Stack

| Component         | Technology                                                |
|-------------------|-----------------------------------------------------------|
| Orchestrator LLM  | Claude 3.7 Sonnet (`databricks-claude-3-7-sonnet`)       |
| Agent Framework   | OpenAI Agents SDK + MLflow ResponsesAgent                 |
| Data Pipeline     | Spark Declarative Pipelines (Lakeflow), serverless        |
| Storage           | Delta Lake, Unity Catalog (medallion architecture)        |
| Structured Query  | Databricks Genie Space (MCP Server)                       |
| Document Q&A      | Databricks Knowledge Assistant (RAG serving endpoint)     |
| Analytics         | Unity Catalog SQL Functions                               |
| Observability     | MLflow Tracing (auto-instrumented)                        |
| Deployment        | Databricks Apps + Databricks Asset Bundles                |
| Package Manager   | `uv`                                                      |

## Prerequisites

| Requirement          | Detail                                                      |
|----------------------|-------------------------------------------------------------|
| Databricks workspace | With Unity Catalog, Genie, Knowledge Assistants, Apps       |
| Python               | 3.11+                                                       |
| `uv`                 | Python package manager (`pip install uv` or `brew install uv`) |
| Databricks CLI       | v0.200+ (`brew install databricks`)                         |
| PAT Token            | Personal Access Token for your workspace                    |

## Quick Start

### 1. Clone and configure

```bash
cd agentbricks_custom_agent
```

All configuration lives in a single central config file. Copy the example and fill in your values:

```bash
# Create your central config (not checked into git)
mkdir -p ../_local/config
cp .env.example ../_local/config/databricks.env
```

Edit `_local/config/databricks.env` with your workspace details:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...your-pat-token
UC_CATALOG=your_catalog
UC_SCHEMA=your_schema
SQL_WAREHOUSE_ID=your_warehouse_id
CLUSTER_ID=your_cluster_id
KA_ENDPOINT=your-ka-endpoint
WORKSPACE_PROJECT_ROOT=/Workspace/Users/you@company.com/projects
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Build the data foundation (Phase 1)

Run the full pipeline sequence - this creates tables, deploys the SDP pipeline, builds views, UC functions, and a Genie Space:

```bash
uv run run-sequence       # ~10-12 min
```

### 4. Validate backend services

Before testing the agent, verify all services are working:

```bash
uv run test-services      # test KA + Genie + UC functions independently
```

### 5. Test the agent locally (Phase 2)

```bash
uv run test-agent         # smoke test: 1 query hitting Genie + KA + UC
uv run test-agent --full  # all 8 test scenarios
```

### 6. Deploy to Databricks Apps

```bash
uv run deploy-agent-app   # deploy + auto-verify with 1 query
```

### 7. Test the deployed app

```bash
uv run test-agent-app                     # smoke test against deployed app
uv run test-agent-app --test-services     # test backend services individually
uv run test-agent-app --full              # all 8 test scenarios
```

## Project Structure

```
agentbricks_custom_agent/
├── app/                                    # Self-contained Databricks App
│   ├── agent_server/
│   │   ├── agent.py                        # Multi-agent orchestrator
│   │   ├── start_server.py                 # FastAPI server entry point
│   │   └── utils.py                        # MCP URL builder, stream helpers
│   ├── databricks.yml                      # App DAB config (env vars + SP resources)
│   └── pyproject.toml                      # App dependencies
│
├── notebooks/                              # Pipeline + orchestration scripts
│   ├── config.py                           # Shared config (reads from central env)
│   │
│   ├── data_engg_src/                      # Data Engineering modules
│   │   ├── setup/
│   │   │   └── setup_sec_documents.py      # Download SEC PDFs to UC Volume
│   │   ├── ingest/
│   │   │   ├── refresh_stock_prices.py     # Incremental stock refresh (daily job)
│   │   │   └── load_stock_data.py          # [DEPRECATED] Legacy standalone loader
│   │   ├── transform/
│   │   │   ├── deploy_sdp_pipeline.py      # Deploy and run SDP pipeline
│   │   │   └── sdp_pipeline_src/           # SDP pipeline (DAB)
│   │   │       ├── databricks.yml          # Bundle config + variables
│   │   │       ├── 00_company_registry.py  # DLT: company/ticker registry
│   │   │       ├── 00_bronze_stock_initial.py  # DLT: yfinance stock loader
│   │   │       └── 01-07_*.sql             # Bronze → Silver → Gold transforms
│   │   └── serve/
│   │       ├── create_stock_views.py       # Analytical views
│   │       ├── create_uc_functions.py      # UC analytical functions
│   │       └── create_genie_space.py       # Genie Space for SQL queries
│   │
│   ├── agent_src/                          # Agent Lifecycle modules
│   │   ├── test_agent.py                   # Local agent smoke tests
│   │   ├── deploy_agent_app.py             # Deploy to Databricks Apps
│   │   ├── test_deployed_agent_app.py      # Test deployed app endpoints
│   │   ├── test_services.py                # Test KA + Genie + UC independently
│   │   └── trace_validator.py              # MLflow trace validation
│   │
│   ├── agentbricks_ka_src/                 # Knowledge Assistant lifecycle
│   ├── demo_cleanup_src/                   # Cleanup utilities
│   ├── demo_shared/                        # Shared utilities (bootstrap, config, API)
│   │
│   ├── run_sequence.py                     # Main orchestrator (--ka, --data-eng, etc.)
│   ├── run_workspace_notebooks.py          # Run notebooks via Jobs API
│   └── sync_workspace.py                   # Sync files to workspace
│
├── pyproject.toml                          # Project config + uv script definitions
├── .env.example                            # Config template
├── README.md                               # This file
├── README-data-pipeline.md                 # Phase 1 deep dive
├── README-sdp-ai-functions.md              # AI functions deep dive
├── README-agent.md                         # Phase 2 deep dive
└── README-execution.md                     # Execution & testing workflow
```

## Command Reference

### run-sequence (Main Orchestrator)

| Command | Purpose |
|---------|---------|
| `uv run run-sequence` | Data engineering only (default) |
| `uv run run-sequence --ka` | KA lifecycle (build, deploy, test) |
| `uv run run-sequence --data-eng` | Data engineering (SDP pipeline + views + Genie) |
| `uv run run-sequence --refresh-stocks` | Run incremental stock price update |
| `uv run run-sequence --deploy-agent` | Data + services + agent lifecycle |
| `uv run run-sequence --all` | Full lifecycle (KA + data + agent) |
| `uv run run-sequence --all --quick` | Full lifecycle with quick tests |
| `uv run run-sequence --all --from services` | Resume from a specific phase |
| `uv run run-sequence --all --dry-run` | Preview execution plan without running |

### Individual Commands

| Command | Purpose |
|---------|---------|
| `uv run demo-cleanup all` | Delete everything (app, Genie, tables, KA) |
| `uv run demo-cleanup tables` | Drop tables and pipeline only |
| `uv run deploy-sdp-pipeline` | Deploy and run the SDP pipeline |
| `uv run refresh-stocks` | Incremental stock price update (daily job) |
| `uv run refresh-stocks --dry-run` | Preview what would be fetched |
| `uv run test-services` | Test KA + Genie + UC functions |
| `uv run test-agent` | Test agent locally (smoke test) |
| `uv run test-agent --full` | Run all 8 local test scenarios |
| `uv run deploy-agent-app` | Deploy to Databricks Apps |
| `uv run test-agent-app` | Test deployed app |

See **[README-execution.md](README-execution.md)** for complete workflow documentation.

## Demo Questions

**Structured data** (Genie Space):
- "What is NVIDIA's total revenue for FY2024?"
- "Compare gross margins across NVIDIA, Apple, and Samsung"

**Document analysis** (Knowledge Assistant):
- "What are Apple's key risk factors from their 10-K?"
- "What does NVIDIA say about AI demand in their MD&A?"

**Complex analytics** (UC Functions):
- "What's NVIDIA's valuation score?"
- "How does NVIDIA compare to its peers?"

**Multi-source** (orchestration):
- "Give me a complete investment overview of NVIDIA" (all three tools)
- "Should I invest in Apple?" (valuation + risks + financials)

## Configuration

All scripts read from a single central config file at `_local/config/databricks.env`. Key variables:

| Variable               | Purpose                                    |
|------------------------|--------------------------------------------|
| `DATABRICKS_HOST`      | Workspace URL                              |
| `DATABRICKS_TOKEN`     | Personal Access Token                      |
| `UC_CATALOG`           | Unity Catalog name                         |
| `UC_SCHEMA`            | Schema name                                |
| `UC_VOLUME`            | UC Volume name containing SEC PDFs         |
| `SEC_DOCS_SUBFOLDER`   | Subfolder within volume (e.g., `sec_2024`) |
| `SQL_WAREHOUSE_ID`     | SQL Warehouse for queries                  |
| `CLUSTER_ID`           | Cluster for workspace notebook jobs        |
| `KA_ENDPOINT`          | Knowledge Assistant serving endpoint       |
| `GENIE_SPACE_ID`       | Auto-populated by step 07                  |
| `APP_URL`              | Auto-populated by deploy-app               |
| `WORKSPACE_PROJECT_ROOT` | Workspace path for notebook sync         |

Dynamic IDs (like `GENIE_SPACE_ID` and `APP_URL`) are automatically written back to this file by the pipeline scripts.

## Author

**Prasad Kona** — [prasad.kona@gmail.com](mailto:prasad.kona@gmail.com)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
