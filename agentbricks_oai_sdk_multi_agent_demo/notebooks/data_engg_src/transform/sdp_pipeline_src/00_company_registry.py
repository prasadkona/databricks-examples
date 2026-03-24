"""
Company Tickers Registry — DLT Python Materialized View

Derives the company/ticker registry from the silver layer — no additional AI function calls.

`ticker_symbol` and `company_name` were already extracted by `ai_extract`
in `02_silver_financial_metrics`. This view simply selects and deduplicates
that data into a clean registry table.

Downstream consumers:
  - `00_bronze_stock_initial.py` — reads tickers to fetch stock history via yfinance
  - Gold tables (03, 04, 05, 07) — join to get company name and ticker without CASE statements

Execution order in pipeline DAG:
  bronze → silver → company_tickers_registry → bronze_stock_initial → gold
"""

import dlt
from pyspark.sql.functions import col, coalesce, expr


@dlt.table(
    name="company_tickers_registry",
    comment=(
        "Company name and ticker registry — derived from silver ai_extract output. "
        "No additional AI calls. Drives stock loading and gold table joins."
    ),
    table_properties={
        "delta.enableChangeDataFeed": "true",
    },
)
def company_tickers_registry():
    """
    Derives the company/ticker registry from silver_sec_financial_metrics.

    Uses ai_extract fields already computed in silver:
      - extracted_metrics:response:ticker_symbol  — primary stock ticker
      - extracted_metrics:response:company_name   — full legal name
      - extracted_metrics:response:exchange        — stock exchange

    Falls back to company_key (file stem) for company_name if ai_extract
    did not return a value.

    NOTE: extracted_metrics is VARIANT type, so we use variant_get() to extract fields.
    """
    silver = dlt.read("silver_sec_financial_metrics")

    return (
        silver
        .select(
            coalesce(
                expr("variant_get(extracted_metrics, '$.response.company_name', 'STRING')"),
                col("company_key"),
            ).alias("company_name"),
            expr("variant_get(extracted_metrics, '$.response.ticker_symbol', 'STRING')").alias("ticker"),
            expr("variant_get(extracted_metrics, '$.response.exchange', 'STRING')").alias("exchange"),
            col("file_name").alias("source_file"),
            col("document_type"),
            col("industry_sector"),
        )
        .filter(col("ticker").isNotNull() & (col("ticker") != ""))
        .dropDuplicates(["ticker"])
    )
