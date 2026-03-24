-- ============================================================================
-- Silver Layer: Extract Financial Metrics from Parsed Documents
-- ============================================================================
--
-- Extracts structured financial data from the already-parsed SEC documents
-- using ai_extract() with JSON schema.
--
-- IMPORTANT: parsed_content comes from bronze — ai_parse_document was already
-- called there. This layer calls ai_extract on the text result (no re-parse).
--
-- Schema additions vs. original:
--   ticker_symbol  — enables generic company discovery (drives company_tickers_registry)
--   exchange       — stock exchange name (NASDAQ, NYSE, KRX, etc.)
--
-- document_type and industry_sector are passed through from bronze (ai_classify output).
-- ============================================================================

CREATE OR REFRESH STREAMING TABLE silver_sec_financial_metrics
TBLPROPERTIES (
  'delta.feature.variantType-preview' = 'supported'
)
COMMENT 'Extracted financial metrics from SEC documents using ai_extract v2.0 — includes ticker_symbol for generic company discovery'
AS
SELECT
  path,
  file_name,
  company_key,
  parsed_content,
  -- Pass through bronze ai_classify results (no re-parse needed)
  document_type,
  industry_sector,
  -- ai_extract: operates on parsed_content from bronze (already parsed text, not raw bytes)
  -- Cast VARIANT to STRING since ai_extract expects STRING input
  ai_extract(
    CAST(parsed_content AS STRING),
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
        "description": "Stock exchange where the company is primarily listed (e.g. NASDAQ, NYSE, KRX)"
      },
      "fiscal_year": {
        "type": "integer",
        "description": "Fiscal year the report covers"
      },
      "fiscal_year_end_date": {
        "type": "string",
        "description": "Fiscal year end date in YYYY-MM-DD format"
      },
      "total_revenue_billions": {
        "type": "number",
        "description": "Total annual revenue in billions USD"
      },
      "cost_of_revenue_billions": {
        "type": "number",
        "description": "Cost of revenue or cost of goods sold in billions USD"
      },
      "gross_profit_billions": {
        "type": "number",
        "description": "Gross profit in billions USD"
      },
      "operating_income_billions": {
        "type": "number",
        "description": "Operating income in billions USD"
      },
      "net_income_billions": {
        "type": "number",
        "description": "Net income in billions USD"
      },
      "total_assets_billions": {
        "type": "number",
        "description": "Total assets in billions USD"
      },
      "total_liabilities_billions": {
        "type": "number",
        "description": "Total liabilities in billions USD"
      },
      "total_equity_billions": {
        "type": "number",
        "description": "Total stockholders equity in billions USD"
      },
      "cash_and_equivalents_billions": {
        "type": "number",
        "description": "Cash and cash equivalents in billions USD"
      },
      "eps_diluted": {
        "type": "number",
        "description": "Diluted earnings per share"
      },
      "revenue_growth_yoy_pct": {
        "type": "number",
        "description": "Year over year revenue growth percentage"
      },
      "business_segments": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "segment_name": {
              "type": "string",
              "description": "Name of the business segment"
            },
            "segment_revenue_billions": {
              "type": "number",
              "description": "Revenue for this segment in billions USD"
            }
          }
        },
        "description": "List of business segments with their revenues"
      },
      "geographic_revenue": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "region": {
              "type": "string",
              "description": "Geographic region name"
            },
            "region_revenue_billions": {
              "type": "number",
              "description": "Revenue for this region in billions USD"
            }
          }
        },
        "description": "Revenue breakdown by geographic region"
      }
    }',
    options => map(
      'version', '2.0'
    )
  ) AS extracted_metrics,
  current_timestamp() AS _extracted_at
FROM STREAM(bronze_sec_parsed_documents)
WHERE parsed_content IS NOT NULL;
