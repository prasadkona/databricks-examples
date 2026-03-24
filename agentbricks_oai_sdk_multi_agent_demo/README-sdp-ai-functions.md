# SDP Pipeline: Databricks AI Functions Deep Dive

This document explains how the SDP (Spark Declarative Pipeline) uses Databricks AI functions to transform raw SEC PDF filings into structured, queryable data — with a key design principle: **each document is parsed only once**.

---

## Level 1: High-Level Summary

The pipeline uses **three Databricks AI functions** to process SEC filings:

| Function | Purpose | Called In | Call Count |
|----------|---------|-----------|------------|
| `ai_parse_document` | Convert PDF binary to structured text | Bronze layer (CTE) | 1× per document |
| `ai_classify` | Classify document type and industry | Bronze layer (outer SELECT) | 2× per document |
| `ai_extract` | Extract financial metrics to JSON | Silver layer | 1× per document |

**Total AI calls per document: 4** (1 parse + 2 classify + 1 extract)

The registry and gold layers perform no AI calls — they derive data from the silver layer output.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         AI Function Call Flow                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PDF Binary ──► ai_parse_document ──► parsed_content (text)            │
│                        │                                                 │
│                        ├──► ai_classify (document_type)                 │
│                        │                                                 │
│                        ├──► ai_classify (industry_sector)               │
│                        │                                                 │
│                        └──► ai_extract (ticker, financials, segments)   │
│                                        │                                 │
│                                        └──► company_tickers_registry    │
│                                             (no AI calls — just SELECT) │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Level 2: End-to-End Pipeline Chain

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SDP Pipeline Data Flow                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

  UC Volume: /Volumes/${catalog}/${schema}/${volume}/${docs_subfolder}/*.pdf
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  01_bronze_parsed_documents.sql                                                 │
│  ────────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│  WITH raw_parsed AS (                                                           │
│    SELECT ... ai_parse_document(content, ...) AS parsed_content ...            │
│  )  ◄── CTE: parse PDF once, store result                                      │
│  SELECT                                                                         │
│    *,                                                                           │
│    ai_classify(parsed_content, ['SEC_10K_Annual_Report', ...]) AS document_type,│
│    ai_classify(parsed_content, ['Semiconductor', ...]) AS industry_sector       │
│  FROM raw_parsed;  ◄── Reuses parsed_content from CTE (no re-parse)            │
│                                                                                 │
│  Output: bronze_sec_parsed_documents (streaming table)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  02_silver_financial_metrics.sql                                                │
│  ────────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│  SELECT                                                                         │
│    ...,                                                                         │
│    document_type,        ◄── Passed through from bronze (no re-classify)       │
│    industry_sector,      ◄── Passed through from bronze (no re-classify)       │
│    ai_extract(parsed_content, '{schema...}') AS extracted_metrics               │
│  FROM STREAM(bronze_sec_parsed_documents);                                      │
│                                                                                 │
│  Extracts: ticker_symbol, exchange, company_name, fiscal_year, revenue,         │
│            margins, EPS, business_segments[], geographic_revenue[]              │
│                                                                                 │
│  Output: silver_sec_financial_metrics (streaming table)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  00_company_registry.py                                                         │
│  ────────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│  @dlt.table(name="company_tickers_registry")                                    │
│  def company_tickers_registry():                                                │
│      silver = dlt.read("silver_sec_financial_metrics")                          │
│      return silver.select(                                                      │
│          extracted_metrics["response"]["company_name"],                         │
│          extracted_metrics["response"]["ticker_symbol"],                        │
│          extracted_metrics["response"]["exchange"], ...                         │
│      ).dropDuplicates(["ticker"])                                               │
│                                                                                 │
│  NO AI CALLS — just reading already-extracted fields from silver               │
│                                                                                 │
│  Output: company_tickers_registry (materialized view)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│  00_bronze_stock_initial.py       │   │  Gold Tables (03, 04, 05, 07)     │
│  ────────────────────────────────  │   │  ────────────────────────────────  │
│                                   │   │                                   │
│  Reads tickers from registry,     │   │  JOIN company_tickers_registry    │
│  fetches 2-year stock history     │   │  to get company name and ticker   │
│  via yfinance API                 │   │  without hardcoded CASE           │
│                                   │   │  statements                       │
│  NO AI CALLS                      │   │  NO AI CALLS                      │
└───────────────────────────────────┘   └───────────────────────────────────┘
```

---

## Level 3: Per-Function Deep Dives

### `ai_parse_document` — PDF to Structured Text

**Location:** `01_bronze_parsed_documents.sql` (CTE inner query)

**Purpose:** Converts raw PDF binary content into structured text that can be processed by downstream AI functions.

**Key design:** Called exactly **once per document** inside a CTE. The `parsed_content` result is stored and reused by `ai_classify` calls in the outer SELECT.

```sql
WITH raw_parsed AS (
  SELECT
    path,
    _metadata.file_name AS file_name,
    _metadata.file_size AS file_size,
    regexp_extract(_metadata.file_name, '^([^_\\.]+)', 1) AS company_key,
    -- ai_parse_document: called ONCE per document here
    ai_parse_document(
      content,
      map(
        'version',                  '2.0',
        'descriptionElementTypes',  '*'
      )
    ) AS parsed_content,
    current_timestamp() AS _ingested_at
  FROM STREAM(read_files(
    '/Volumes/${catalog}/${schema}/${volume}/${docs_subfolder}',
    format => 'binaryFile'
  ))
)
-- parsed_content is now available for downstream ai_classify calls
SELECT * FROM raw_parsed;
```

**Options:**
- `version: '2.0'` — Uses the latest parsing model
- `descriptionElementTypes: '*'` — Extracts all element types (tables, headers, paragraphs)

---

### `ai_classify` — Document Classification

**Location:** `01_bronze_parsed_documents.sql` (outer SELECT)

**Purpose:** Classifies each document into predefined categories. Called **twice** per document:
1. **Document type** — SEC_10K_Annual_Report, SEC_10Q_Quarterly_Report, etc.
2. **Industry sector** — Semiconductor, Consumer_Electronics, Software, etc.

**Key design:** Operates on `parsed_content` from the CTE — the PDF has already been parsed, so no re-parsing occurs.

```sql
SELECT
  *,
  -- Classification 1: Document type
  ai_classify(
    parsed_content,
    ARRAY(
      'SEC_10K_Annual_Report',
      'SEC_10Q_Quarterly_Report',
      'Annual_Report',
      'Earnings_Release',
      'Other_Financial_Document'
    )
  ) AS document_type,
  
  -- Classification 2: Industry sector
  ai_classify(
    parsed_content,
    ARRAY(
      'Semiconductor',
      'Consumer_Electronics',
      'Software',
      'Financial_Services',
      'Healthcare',
      'Energy',
      'Retail',
      'Other'
    )
  ) AS industry_sector
FROM raw_parsed;
```

---

### `ai_extract` — Structured Data Extraction

**Location:** `02_silver_financial_metrics.sql`

**Purpose:** Extracts structured financial data from the parsed text using a JSON schema. This enables generic company discovery — the pipeline does not need hardcoded company names or tickers.

**Key design:**
- Operates on `parsed_content` from bronze (already parsed text, not raw bytes)
- `document_type` and `industry_sector` are passed through from bronze (no re-classification)
- Schema includes `ticker_symbol` and `exchange` for company discovery

```sql
SELECT
  path, file_name, company_key, parsed_content,
  -- Pass through bronze ai_classify results (no re-classify)
  document_type,
  industry_sector,
  -- ai_extract: operates on parsed_content from bronze
  ai_extract(
    parsed_content,
    '{
      "company_name": {
        "type": "string",
        "description": "Full legal name of the company as it appears in the filing"
      },
      "ticker_symbol": {
        "type": "string",
        "description": "Primary stock exchange ticker symbol (e.g. NVDA, AAPL, 005930.KS)"
      },
      "exchange": {
        "type": "string",
        "description": "Stock exchange where the company is primarily listed"
      },
      "fiscal_year": { "type": "integer", ... },
      "total_revenue_billions": { "type": "number", ... },
      "business_segments": { "type": "array", ... },
      "geographic_revenue": { "type": "array", ... }
    }',
    options => map('version', '2.0')
  ) AS extracted_metrics
FROM STREAM(bronze_sec_parsed_documents)
WHERE parsed_content IS NOT NULL;
```

**Output schema fields:**
| Field | Type | Purpose |
|-------|------|---------|
| `ticker_symbol` | string | Drives `company_tickers_registry` and stock data loading |
| `exchange` | string | Stock exchange (NASDAQ, NYSE, KRX) |
| `company_name` | string | Full legal name for display |
| `fiscal_year` | integer | Report fiscal year |
| `total_revenue_billions` | number | Total annual revenue |
| `business_segments[]` | array | Revenue by business segment |
| `geographic_revenue[]` | array | Revenue by geographic region |

---

### Registry Derivation — No AI Calls

**Location:** `00_company_registry.py`

**Purpose:** Creates a clean company/ticker registry from the silver layer. This table drives:
1. Stock data loading (`00_bronze_stock_initial.py`)
2. Gold table joins (replacing hardcoded CASE statements)

**Key design:** No AI function calls — simply reads already-extracted fields from silver.

```python
@dlt.table(
    name="company_tickers_registry",
    comment="Company name and ticker registry — derived from silver ai_extract output. "
            "No additional AI calls. Drives stock loading and gold table joins.",
)
def company_tickers_registry():
    silver = dlt.read("silver_sec_financial_metrics")
    return (
        silver
        .select(
            coalesce(
                col("extracted_metrics")["response"]["company_name"].cast("string"),
                col("company_key"),
            ).alias("company_name"),
            col("extracted_metrics")["response"]["ticker_symbol"].cast("string").alias("ticker"),
            col("extracted_metrics")["response"]["exchange"].cast("string").alias("exchange"),
            col("file_name").alias("source_file"),
            col("document_type"),
            col("industry_sector"),
        )
        .filter(col("ticker").isNotNull() & (col("ticker") != ""))
        .dropDuplicates(["ticker"])
    )
```

---

## No-Reparse Guarantee

> **Why does this matter?**
>
> `ai_parse_document` is the most expensive AI function call — it processes raw PDF bytes through an LLM to extract structured text. By using a CTE pattern in the bronze layer, we ensure:
>
> 1. **Cost efficiency** — Each PDF is parsed exactly once, not once per downstream consumer
> 2. **Latency reduction** — Downstream layers (silver, gold) operate on pre-parsed text
> 3. **Consistency** — All downstream AI functions (`ai_classify`, `ai_extract`) operate on the same parsed representation
>
> The CTE pattern stores `parsed_content` in memory during the streaming microbatch, allowing the outer SELECT to call `ai_classify` twice without re-invoking `ai_parse_document`.

---

## Pipeline DAG Execution Order

The DLT pipeline DAG ensures correct execution order:

```
01_bronze_parsed_documents.sql
         │
         ▼
02_silver_financial_metrics.sql
         │
         ▼
00_company_registry.py  ◄── Must run after silver
         │
         ▼
00_bronze_stock_initial.py  ◄── Must run after registry
         │
         ▼
06_silver_stock_daily.sql
         │
         ├──► 03_gold_company_financials.sql
         ├──► 04_gold_revenue_segments.sql
         ├──► 05_gold_revenue_geography.sql
         └──► 07_gold_stock_summary.sql
```

All Python DLT files are registered in `databricks.yml` as `file` libraries alongside the SQL files. DLT infers dependencies from `dlt.read()` / `STREAM()` calls.

---

## Configuration

Pipeline variables (set via `databricks bundle deploy --var key=value` or in `databricks.yml`):

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `your_catalog` |
| `schema` | Schema name | `your_schema` |
| `volume` | UC Volume name containing PDFs | `ka_demo` |
| `docs_subfolder` | Subfolder within volume | `sec_2024` |

These variables make the pipeline generic — drop any company's SEC filings into the configured volume path and the pipeline will:
1. Parse and classify the documents
2. Extract the company name and ticker
3. Automatically load stock data for discovered companies
4. Build gold tables with correct company/ticker joins
