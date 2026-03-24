"""
Bronze: Initial Stock Price History — DLT Python Table

Fetches full 2-year stock price history for all companies discovered
in `company_tickers_registry`.

Design notes:
  - Uses mapInPandas to avoid collect() which is not supported in serverless DLT.
  - Reads tickers from `company_tickers_registry` (derived from `ai_extract` — no hardcoding).
  - Fetches via `yfinance` — one API call per ticker.
  - Per-ticker errors are caught and logged; pipeline continues with available data.
  - Delta deduplication in `silver_stock_daily_prices` handles idempotent re-runs.

Incremental refresh (latest prices for existing companies) is handled separately
by `refresh_stock_prices.py` Databricks Job, which writes to `bronze_stock_daily_refresh`.
`silver_stock_daily_prices` UNIONs both tables.

Execution order in pipeline DAG:
  company_tickers_registry → bronze_stock_initial → silver_stock_daily_prices → gold_stock_summary
"""

import dlt
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Iterator

from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# Schema for stock price data (output schema for mapInPandas)
STOCK_SCHEMA = StructType([
    StructField("ticker",              StringType(),    False),
    StructField("trade_date",          DateType(),      True),
    StructField("open_price",          DoubleType(),    True),
    StructField("high_price",          DoubleType(),    True),
    StructField("low_price",           DoubleType(),    True),
    StructField("close_price",         DoubleType(),    True),
    StructField("volume",              LongType(),      True),
    StructField("dividends",           DoubleType(),    True),
    StructField("stock_splits",        DoubleType(),    True),
    StructField("ingestion_timestamp", TimestampType(), True),
])


def _fetch_stock_data(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Pandas UDF (mapInPandas) that fetches 2-year stock history for each ticker.
    Each input batch contains rows with a 'ticker' column.
    Returns a DataFrame with full stock price history.
    """
    for batch in batches:
        results = []
        for ticker in batch["ticker"].unique():
            try:
                hist = yf.Ticker(ticker).history(period="2y")
                if hist.empty:
                    print(f"  WARNING: No data returned for {ticker}")
                    continue

                hist = hist.reset_index()
                # Build output DataFrame
                df = pd.DataFrame({
                    "ticker": ticker,
                    "trade_date": pd.to_datetime(hist["Date"]).dt.date,
                    "open_price": hist["Open"].astype(float),
                    "high_price": hist["High"].astype(float),
                    "low_price": hist["Low"].astype(float),
                    "close_price": hist["Close"].astype(float),
                    "volume": hist["Volume"].astype("int64"),
                    "dividends": hist["Dividends"].astype(float),
                    "stock_splits": hist["Stock Splits"].astype(float),
                    "ingestion_timestamp": pd.Timestamp.now(),
                })
                results.append(df)
                print(f"  OK: {ticker} — {len(df)} rows fetched")

            except Exception as exc:
                print(f"  WARNING: Could not fetch data for {ticker}: {exc}")
                continue

        if results:
            yield pd.concat(results, ignore_index=True)
        else:
            # Return empty DataFrame with correct schema
            yield pd.DataFrame(columns=[
                "ticker", "trade_date", "open_price", "high_price", "low_price",
                "close_price", "volume", "dividends", "stock_splits", "ingestion_timestamp"
            ])


@dlt.table(
    name="bronze_stock_initial",
    comment=(
        "Full 2-year stock price history for all companies discovered in "
        "company_tickers_registry. Fetched via yfinance using mapInPandas. "
        "Delta deduplication in silver_stock_daily_prices handles idempotency. "
        "For incremental refresh use refresh_stock_prices.py (bronze_stock_daily_refresh)."
    ),
    table_properties={
        "delta.enableChangeDataFeed": "true",
    },
)
def bronze_stock_initial():
    """
    Reads tickers from company_tickers_registry and fetches 2-year history
    for each one via yfinance using mapInPandas (avoids collect()).

    On re-run: re-fetches all tickers; silver_stock_daily_prices deduplicates
    by (ticker, trade_date), keeping the latest ingestion_timestamp.
    """
    registry = dlt.read("company_tickers_registry")

    # Use mapInPandas to fetch stock data for each ticker
    # This avoids collect() which is not supported in serverless DLT
    stock_data = (
        registry
        .select("ticker")
        .distinct()
        .repartition(1)  # Process all tickers in single partition for sequential API calls
        .mapInPandas(_fetch_stock_data, schema=STOCK_SCHEMA)
    )

    return stock_data
