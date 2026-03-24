# Data Pipeline Deep Dive

This document covers the data foundation: ingesting SEC filings, transforming them through a medallion architecture pipeline with automatic company discovery, loading stock data for discovered companies, creating analytical views and UC functions, and setting up a Genie Space for natural language SQL.

For a detailed explanation of the AI functions used in the pipeline, see **[README-sdp-ai-functions.md](README-sdp-ai-functions.md)**.

## Pipeline Architecture

```
                        ┌────────────────────────────────────┐
                        │  SEC 10-K PDFs (any company)       │
                        │  /Volumes/${catalog}/${schema}/    │
                        │  ${volume}/${docs_subfolder}/      │
                        └─────────────────┬──────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │                                                           │
            ▼                                                           ▼
┌───────────────────────┐                               ┌──────────────────────────────┐
│ Knowledge Assistant   │                               │        SDP Pipeline           │
│ (indexes PDFs for     │                               │                              │
│  RAG-based Q&A)       │                               │  ai_parse_document (1×)      │
└───────────────────────┘                               │  ai_classify (2×)            │
                                                        │  ai_extract (1×)             │
                                                        └──────────────┬───────────────┘
                                                                       │
┌──────────────────────────────────────────────────────────────────────┴──────────┐
│                                                                                  │
│   ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐ │
│   │     BRONZE       │     │      SILVER         │     │      REGISTRY        │ │
│   │ sec_parsed_      │ ──► │ sec_financial_      │ ──► │ company_tickers_     │ │
│   │ documents        │     │ metrics             │     │ registry             │ │
│   │                  │     │                     │     │                      │ │
│   │ ai_parse_document│     │ ai_extract:         │     │ NO AI calls —        │ │
│   │ ai_classify x2   │     │ ticker_symbol,      │     │ reads silver output  │ │
│   │                  │     │ company_name, etc.  │     │                      │ │
│   └──────────────────┘     └─────────────────────┘     └──────────┬───────────┘ │
│                                                                    │            │
│   ┌────────────────────────────────────────────────────────────────┘            │
│   │                                                                              │
│   ▼                                                                              │
│   ┌──────────────────┐     ┌─────────────────────┐                              │
│   │ BRONZE (DLT)     │     │      SILVER         │      ┌──────────────────┐    │
│   │ stock_initial    │ ──► │ stock_daily_prices  │ ◄─── │ EXTERNAL Delta   │    │
│   │                  │     │                     │      │ stock_daily_     │    │
│   │ yfinance 2y      │     │ UNION + dedup       │      │ refresh          │    │
│   │ history (auto)   │     │                     │      │ (incremental)    │    │
│   └──────────────────┘     └─────────┬───────────┘      └──────────────────┘    │
│                                      │                   ▲                       │
│                                      │                   │ uv run refresh-stocks │
│                                      │                   └───────────────────────│
└──────────────────────────────────────┼───────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌──────────────┐            ┌──────────────────┐            ┌──────────────┐
│    GOLD      │            │      GOLD        │            │    GOLD      │
│ company_     │            │ revenue_by_      │            │ stock_       │
│ financials   │            │ segment /        │            │ summary      │
│              │            │ geography        │            │              │
│ JOIN registry│            │ JOIN registry    │            │ JOIN registry│
│ (no CASE)    │            │ (no CASE)        │            │ (no CASE)    │
└──────┬───────┘            └──────┬───────────┘            └──────┬───────┘
       │                           │                                │
       ▼                           ▼                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           Analytical Views                                │
│  sec_fin_company_overview    sec_fin_peer_comparison                     │
│  sec_fin_stock_performance   sec_fin_revenue_breakdown                   │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────┐
          ▼                          ▼                      ▼
┌───────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│   UC Functions    │   │   Genie Space     │   │ Agent (Phase 2)  │
│                   │   │                   │   │                  │
│ valuation_score   │   │ Natural language  │   │ Queries views,   │
│ compare_peers     │   │ SQL over all      │   │ functions, and   │
│ growth_trajectory │   │ Gold tables and   │   │ Genie via MCP    │
│ risk_summary      │   │ views             │   │                  │
└───────────────────┘   └───────────────────┘   └──────────────────┘
```

## Folder Structure

```
notebooks/data_engg_src/
├── setup/
│   └── setup_sec_documents.py      # Download SEC PDFs to UC Volume
├── ingest/
│   ├── refresh_stock_prices.py     # Incremental stock refresh (daily job)
│   └── load_stock_data.py          # [DEPRECATED] Legacy standalone loader
├── transform/
│   ├── deploy_sdp_pipeline.py      # Deploy and run SDP pipeline
│   └── sdp_pipeline_src/           # SDP pipeline (2 Python + 7 SQL)
│       ├── databricks.yml          # Bundle config + variables
│       ├── 00_company_registry.py  # DLT MV: company/ticker registry
│       ├── 00_bronze_stock_initial.py  # DLT table: yfinance 2y history
│       ├── 01_bronze_parsed_documents.sql
│       ├── 02_silver_financial_metrics.sql
│       ├── 03_gold_company_financials.sql
│       ├── 04_gold_revenue_segments.sql
│       ├── 05_gold_revenue_geography.sql
│       ├── 06_silver_stock_daily.sql
│       └── 07_gold_stock_summary.sql
└── serve/
    ├── create_stock_views.py       # Analytical views
    ├── create_uc_functions.py      # UC analytical functions
    └── create_genie_space.py       # Genie Space for SQL queries
```

## Step-by-Step Breakdown

### Setup: Download SEC Documents (Optional)

**Script:** `notebooks/data_engg_src/setup/setup_sec_documents.py` (runs on Databricks)

Downloads SEC 10-K annual report PDFs and stores them in a Unity Catalog Volume:

```
/Volumes/{catalog}/{schema}/{volume}/{docs_subfolder}/
├── COMPANY1_10K.pdf       # Any company's annual report
├── COMPANY2_10K.pdf       # Companies are auto-discovered
└── ...
```

This step is optional — the PDFs only need to be downloaded once. The pipeline automatically discovers company names and ticker symbols from the PDF content using `ai_extract`. The Knowledge Assistant indexes these files for RAG-based document Q&A.

### Cleanup Tables

**Script:** `notebooks/demo_cleanup_src/cleanup_tables.py`
**Command:** `uv run demo-cleanup tables`

Drops all SDP pipeline tables and deletes the pipeline for a clean start. This ensures idempotent re-runs.

```bash
uv run demo-cleanup tables           # drop all pipeline tables, delete pipeline
uv run demo-cleanup all              # full cleanup (app, Genie, tables, KA)
```

**Tables dropped:**
- `bronze_sec_parsed_documents` — DLT streaming table
- `silver_sec_financial_metrics` — DLT streaming table
- `company_tickers_registry` — DLT materialized view
- `bronze_stock_initial` — DLT table (yfinance history)
- `bronze_stock_daily_refresh` — external Delta table (incremental)
- `silver_stock_daily_prices` — DLT materialized view
- `gold_company_financials`, `gold_revenue_by_segment`, `gold_revenue_by_geography`, `gold_stock_summary` — DLT materialized views
- `bronze_stock_daily_prices` — legacy table (backward compat)

**Pipeline deleted:** `sec_financial_analyst_pipeline` (found via paginated list API).

### Transform: Deploy & Run SDP Pipeline

**Script:** `notebooks/data_engg_src/transform/deploy_sdp_pipeline.py` (runs locally)
**Command:** `uv run deploy-sdp-pipeline`

This is the most complex step. It deploys a Spark Declarative Pipeline (SDP) as a Databricks Asset Bundle, runs it, and verifies the output.

```bash
uv run deploy-sdp-pipeline             # deploy + run + verify
uv run deploy-sdp-pipeline --no-verify # skip table count verification
uv run deploy-sdp-pipeline -t prod     # deploy to prod target
```

#### How the SDP Pipeline Works

The pipeline is defined in `notebooks/data_engg_src/transform/sdp_pipeline_src/` as a Databricks Asset Bundle with 2 Python files and 7 SQL files:

```
sdp_pipeline_src/
├── databricks.yml                          # Bundle config + variables
├── 00_company_registry.py                  # DLT MV: company/ticker registry
├── 00_bronze_stock_initial.py              # DLT table: yfinance 2y history
├── 01_bronze_parsed_documents.sql          # Parse PDFs → text + classify
├── 02_silver_financial_metrics.sql         # Extract metrics with AI
├── 03_gold_company_financials.sql          # Consolidated financials
├── 04_gold_revenue_segments.sql            # Revenue by segment
├── 05_gold_revenue_geography.sql           # Revenue by geography
├── 06_silver_stock_daily.sql               # UNION stock sources + dedup
└── 07_gold_stock_summary.sql               # Stock summary
```

The pipeline DAG executes in dependency order: bronze → silver → registry → stock → gold.

#### Bronze Layer

**`01_bronze_parsed_documents.sql`** — Reads raw PDF files from the UC Volume, parses them using `ai_parse_document()`, and classifies them using `ai_classify()`. Uses a CTE pattern to ensure each document is parsed only once.

```sql
WITH raw_parsed AS (
  SELECT
    path, _metadata.file_name AS file_name,
    regexp_extract(_metadata.file_name, '^([^_\\.]+)', 1) AS company_key,
    ai_parse_document(content, map('version', '2.0', ...)) AS parsed_content
  FROM STREAM(read_files('/Volumes/${catalog}/${schema}/${volume}/${docs_subfolder}', format => 'binaryFile'))
)
SELECT *,
  ai_classify(parsed_content, ARRAY('SEC_10K_Annual_Report', ...)) AS document_type,
  ai_classify(parsed_content, ARRAY('Semiconductor', ...)) AS industry_sector
FROM raw_parsed;
```

**Key design:** The CTE parses each PDF once; the outer SELECT calls `ai_classify` twice on the already-parsed `parsed_content` (no re-parsing). See **[README-sdp-ai-functions.md](README-sdp-ai-functions.md)** for the full deep dive.

#### Silver Layer

**`02_silver_financial_metrics.sql`** — Extracts structured financial data from parsed text using `ai_extract()` with a JSON schema. Includes `ticker_symbol` and `exchange` for automatic company discovery.

```sql
SELECT
  path, file_name, company_key, parsed_content,
  document_type,      -- passed through from bronze (no re-classify)
  industry_sector,    -- passed through from bronze (no re-classify)
  ai_extract(parsed_content, '{
    "ticker_symbol": {"type": "string", ...},
    "exchange": {"type": "string", ...},
    "company_name": {"type": "string", ...},
    "total_revenue_billions": {"type": "number", ...},
    ...
  }') AS extracted_metrics
FROM STREAM(bronze_sec_parsed_documents) WHERE parsed_content IS NOT NULL;
```

**Key technology:** `ai_extract()` takes unstructured text and a JSON schema, then uses an LLM to extract structured data matching that schema. The `ticker_symbol` field enables the pipeline to automatically discover and load stock data for any company.

**`06_silver_stock_daily.sql`** — Combines stock data from two sources:
- `bronze_stock_initial` (DLT-managed, full 2-year history from pipeline)
- `bronze_stock_daily_refresh` (external Delta table, incremental updates)

Uses `UNION ALL` with `ROW_NUMBER()` deduplication to select the latest record per `(ticker, trade_date)` pair.

#### Registry Layer (New)

**`00_company_registry.py`** — DLT Python Materialized View that derives a clean company/ticker registry from the silver layer. No additional AI calls — simply reads the `ticker_symbol`, `company_name`, and `exchange` fields already extracted by `ai_extract`.

```python
@dlt.table(name="company_tickers_registry")
def company_tickers_registry():
    silver = dlt.read("silver_sec_financial_metrics")
    return (
        silver.select(
            col("extracted_metrics")["response"]["company_name"].alias("company_name"),
            col("extracted_metrics")["response"]["ticker_symbol"].alias("ticker"),
            col("extracted_metrics")["response"]["exchange"].alias("exchange"),
            ...
        )
        .filter(col("ticker").isNotNull())
        .dropDuplicates(["ticker"])
    )
```

**Downstream consumers:**
- `00_bronze_stock_initial.py` — reads tickers to fetch stock history via yfinance
- Gold tables (03, 04, 05, 07) — join to get company name/ticker without hardcoded CASE statements

#### Stock Loading (Integrated)

**`00_bronze_stock_initial.py`** — DLT Python Table that reads all tickers from `company_tickers_registry` and fetches 2-year historical stock data via yfinance. Runs as part of the main pipeline.

```python
@dlt.table(name="bronze_stock_initial")
def bronze_stock_initial():
    registry = dlt.read("company_tickers_registry")
    tickers = [r.ticker for r in registry.select("ticker").collect()]
    # Fetch yfinance data for each ticker, return unioned DataFrame
    ...
```

For incremental daily updates after initial load, use `uv run refresh-stocks` (see [Stock Refresh Job](#stock-refresh-job) below).

#### Gold Layer

Four materialized views create the final analytical tables:

| SQL File | Output Table | Description |
|----------|-------------|-------------|
| `03_gold_company_financials.sql` | `gold_company_financials` | Revenue, margins, EPS, assets per company |
| `04_gold_revenue_segments.sql` | `gold_revenue_by_segment` | Revenue by business segment |
| `05_gold_revenue_geography.sql` | `gold_revenue_by_geography` | Revenue by geographic region |
| `07_gold_stock_summary.sql` | `gold_stock_summary` | Stock price summary per company |

**Key design:** Gold tables JOIN `company_tickers_registry` to get company names and tickers dynamically — no hardcoded CASE statements. This enables the pipeline to work with any set of SEC filings.

The gold tables compute derived metrics like gross margin %, operating margin %, and net margin % from the AI-extracted financial data.

#### Deploy Script Internals

The `deploy_sdp_pipeline.py` script handles the full lifecycle:

1. **Build variable overrides** — Reads `UC_CATALOG`, `UC_SCHEMA`, `UC_VOLUME`, `SEC_DOCS_SUBFOLDER` from env vars and maps them to bundle variables via `BUNDLE_VAR_MAP`
2. **Validate** — `databricks bundle validate -t dev --var catalog=... --var volume=...`
3. **Deploy** — `databricks bundle deploy -t dev --auto-approve --var ...`
4. **Find pipeline** — Paginated `GET /api/2.0/pipelines` with `max_results=100`
5. **Run** — `POST /api/2.0/pipelines/{id}/updates` (full refresh)
6. **Wait** — 90s initial wait (serverless resource provisioning), then poll every 15s
7. **Retry** — Auto-retries once on FAILED (transient resource issues)
8. **Verify** — `SELECT COUNT(*)` on all gold/silver tables
9. **Report** — Prints pipeline URL and writes to `_local/last_pipeline_url.txt`

Typical runtime: ~5-8 minutes (mostly pipeline execution after ~90s resource wait).

### Serve: Create Analytical Views

**Script:** `notebooks/data_engg_src/serve/create_stock_views.py` (runs on Databricks)
**Command:** `uv run run-workspace-notebooks 05`

Creates SQL views on top of the gold tables for richer analytics:

| View | Purpose |
|------|---------|
| `sec_fin_company_overview` | Combined company snapshot with all financial metrics |
| `sec_fin_peer_comparison` | Side-by-side metrics for cross-company analysis |
| `sec_fin_stock_performance` | Stock price performance summary |
| `sec_fin_revenue_breakdown` | Revenue segment and geography detail |

These views are used by the Genie Space and UC functions.

### Serve: Create UC Functions

**Script:** `notebooks/data_engg_src/serve/create_uc_functions.py` (runs on Databricks)
**Command:** `uv run run-workspace-notebooks 06`

Creates four SQL table-valued functions in Unity Catalog:

#### `sec_fin_valuation_score(ticker)`

Computes a composite valuation score (1-100) based on:
- P/E ratio score (compares to peer average)
- Revenue growth score
- Profitability score (net margin analysis)

Returns: ticker, company, overall score, PE score, growth score, profitability score

#### `sec_fin_compare_peers(ticker)`

Compares a company against peers on key metrics:
- Revenue growth YoY %
- Gross margin %
- Operating margin %
- Net margin %

Returns: metric name, company value, peer average

#### `sec_fin_growth_trajectory(ticker)`

Analyzes revenue growth trends with contextual interpretation:
- Current FY revenue
- YoY growth rate
- Growth classification (hyper-growth / strong / moderate / stable / declining)

#### `sec_fin_risk_summary(ticker)`

Assesses financial risks with severity levels:
- Valuation risk (based on P/E ratio)
- Growth sustainability risk
- Debt level risk (based on debt-to-equity ratio)

Returns: risk type, severity (HIGH/MEDIUM/LOW), description

### Serve: Create Genie Space

**Script:** `notebooks/data_engg_src/serve/create_genie_space.py` (runs on Databricks)
**Command:** `uv run run-workspace-notebooks 07`

Creates a Databricks Genie Space named `SEC_Financial_Data_Explorer` that provides natural language SQL access to:
- `gold_company_financials` - Company financial metrics
- `gold_revenue_by_segment` - Segment breakdowns
- `gold_revenue_by_geography` - Geographic breakdowns
- `gold_stock_summary` - Stock price summaries
- `silver_stock_daily_prices` - Daily stock price history
- All four analytical views

The Genie Space ID is automatically captured by `run_sequence.py` and written to the central config file for use by the agent.

### Stock Refresh Job

**Script:** `notebooks/data_engg_src/ingest/refresh_stock_prices.py` (runs locally or as Databricks Job)
**Command:** `uv run refresh-stocks`

For incremental stock price updates after the initial pipeline run, use the refresh job:

```bash
uv run refresh-stocks                     # Fetch new data for all tickers
uv run refresh-stocks --dry-run           # Preview what would be fetched
uv run refresh-stocks --ticker NVDA       # Refresh a single ticker only
```

The script:
1. Reads all tickers from `company_tickers_registry`
2. Finds the latest known `trade_date` per ticker in `bronze_stock_daily_refresh`
3. Fetches only new rows from yfinance (since latest date + 1 day)
4. Appends to `bronze_stock_daily_refresh` (external Delta table)

The `silver_stock_daily_prices` view UNIONs `bronze_stock_initial` and `bronze_stock_daily_refresh`, so new data is immediately visible after refresh.

Schedule this as a daily Databricks Job for automatic updates.

## Running the Full Sequence

The `run_sequence.py` script orchestrates all steps in order:

```bash
uv run run-sequence              # data engineering only (~8-10 min)
uv run run-sequence --all        # full lifecycle including agent
uv run run-sequence --data-eng --refresh-stocks  # pipeline + incremental stock update
```

What happens with `--data-eng` (default):

1. **sync-workspace** — Uploads all `notebooks/` files and `sdp_pipeline_src/` to workspace
2. **cleanup** — Drops all tables and deletes the pipeline
3. **transform** — Deploys and runs the SDP pipeline (discovers companies, loads stock history, builds gold tables)
4. **serve (05, 06, 07)** — Creates views, UC functions, and Genie Space
5. **capture IDs** — Fetches the Genie Space ID and writes to central config

Stock loading is now fully integrated into the SDP pipeline via `00_bronze_stock_initial.py` — no separate ingest step is required.

If duplicate Genie Spaces are detected (Databricks auto-appends ` (2)`, ` (3)` suffixes), the script keeps the most recently created one and deletes the older duplicates.

## SDP Pipeline DAB Configuration

```yaml
bundle:
  name: sec_financial_analyst_pipeline

variables:
  catalog:
    description: Unity Catalog name
    default: your_catalog
  schema:
    description: Schema name
    default: your_schema
  volume:
    description: UC Volume name containing the PDF documents
    default: ka_demo
  docs_subfolder:
    description: Subfolder within the volume containing PDFs
    default: sec_2024

resources:
  pipelines:
    sec_financial_analyst_pipeline:
      name: sec_financial_analyst_pipeline
      catalog: ${var.catalog}
      target: ${var.schema}
      serverless: true
      continuous: false
      development: false
      channel: CURRENT
      libraries:
        - file:
            path: 00_company_registry.py    # DLT Python files
        - file:
            path: 00_bronze_stock_initial.py
        - file:
            path: 01_bronze_parsed_documents.sql
        # ... other SQL files

targets:
  dev:
    mode: production    # prevents [dev user] prefix on pipeline name
    default: true
```

The `mode: production` setting on the dev target prevents the Databricks CLI from auto-prefixing the pipeline name with `[dev <username>]`, ensuring consistent create/delete operations.

The `volume` and `docs_subfolder` variables are injected by `deploy_sdp_pipeline.py` via `--var` overrides when `UC_VOLUME` and `SEC_DOCS_SUBFOLDER` env vars are set.

## Data Summary

| Layer    | Table                           | Description                                      |
|----------|---------------------------------|--------------------------------------------------|
| Bronze   | `bronze_sec_parsed_documents`   | Parsed SEC PDFs with `ai_parse_document` output  |
| Silver   | `silver_sec_financial_metrics`  | AI-extracted financial data                      |
| Registry | `company_tickers_registry`      | Auto-discovered company/ticker mapping           |
| Bronze   | `bronze_stock_initial`          | 2-year yfinance history (DLT-managed)            |
| External | `bronze_stock_daily_refresh`    | Incremental stock updates (refresh job)          |
| Silver   | `silver_stock_daily_prices`     | UNION of initial + refresh, deduplicated         |
| Gold     | `gold_company_financials`       | Consolidated financials per company              |
| Gold     | `gold_revenue_by_segment`       | Revenue by business segment                      |
| Gold     | `gold_revenue_by_geography`     | Revenue by geographic region                     |
| Gold     | `gold_stock_summary`            | Latest stock price summary                       |

Row counts depend on the number of SEC filings in the configured UC Volume. Companies are automatically discovered from the PDF content via `ai_extract`.

## Required Environment Variables

| Variable              | Used By                            |
|-----------------------|------------------------------------|
| `DATABRICKS_HOST`     | All steps                          |
| `DATABRICKS_TOKEN`    | All steps (PAT-only auth)          |
| `UC_CATALOG`          | All data engineering steps         |
| `UC_SCHEMA`           | All data engineering steps         |
| `UC_VOLUME`           | SDP pipeline (PDF source volume)   |
| `SEC_DOCS_SUBFOLDER`  | SDP pipeline (subfolder in volume) |
| `SQL_WAREHOUSE_ID`    | Cleanup, transform verification    |
| `CLUSTER_ID`          | Workspace notebook jobs            |
| `WORKSPACE_PROJECT_ROOT` | sync-workspace, all workspace jobs |

## Troubleshooting

### Pipeline fails with WAITING_FOR_RESOURCES

The deploy script waits 90 seconds before first poll to allow serverless resources to provision. If it still fails, it auto-retries once. Check the printed error events from the pipeline events API.

### Tables have 0 rows after pipeline

Check that:
1. SEC PDFs exist in the configured UC Volume path (`/Volumes/{catalog}/{schema}/{volume}/{docs_subfolder}/`)
2. The `UC_VOLUME` and `SEC_DOCS_SUBFOLDER` env vars are set correctly
3. The pipeline variables were injected (look for "Pipeline variable overrides" in deploy output)

### Stock data missing

Stock loading is now integrated into the SDP pipeline. If `bronze_stock_initial` is empty, check that `company_tickers_registry` has rows — this means `ai_extract` successfully extracted ticker symbols from the PDFs.

For incremental updates after initial load, run `uv run refresh-stocks`.

### Pipeline name conflict

The cleanup step (`demo-cleanup tables`) deletes the pipeline before each run. If cleanup was skipped, run `uv run demo-cleanup tables` manually. The deploy script also handles this by deleting and redeploying automatically.

### Genie Space not created or ID not captured

Check that the Genie creation step ran successfully. The `run_sequence.py` script automatically captures the Genie Space ID and writes it to the central config. If running steps individually, update `GENIE_SPACE_ID` in the config manually.

### Authentication errors

All local scripts use PAT-only auth. Never set `DATABRICKS_CONFIG_PROFILE`, `DATABRICKS_CLIENT_ID`, or `DATABRICKS_CLIENT_SECRET` — these are explicitly unset by every script to prevent conflicts.
