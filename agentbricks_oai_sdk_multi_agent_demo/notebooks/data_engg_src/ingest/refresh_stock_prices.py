#!/usr/bin/env python3
# Databricks notebook source
# MAGIC %md
# MAGIC # Refresh Stock Prices (Incremental)
# MAGIC
# MAGIC Databricks Job script for **incremental** stock price updates.
# MAGIC
# MAGIC Reads all tickers from `company_tickers_registry` (populated by the SDP pipeline),
# MAGIC finds the latest known trade date per ticker in `bronze_stock_daily_refresh`, and
# MAGIC fetches only the new data since that date via yfinance.
# MAGIC
# MAGIC ## Architecture
# MAGIC ```
# MAGIC company_tickers_registry (SDP pipeline output)
# MAGIC         │
# MAGIC         ▼ read tickers
# MAGIC bronze_stock_daily_refresh (external Delta table)
# MAGIC         │ MAX(trade_date) per ticker
# MAGIC         ▼
# MAGIC yfinance API → new rows since latest date
# MAGIC         │
# MAGIC         ▼ APPEND
# MAGIC bronze_stock_daily_refresh
# MAGIC         │
# MAGIC         ▼ (picked up by silver_stock_daily_prices UNION on next pipeline refresh)
# MAGIC silver_stock_daily_prices → gold_stock_summary
# MAGIC ```
# MAGIC
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC uv run refresh-stocks                     # local (uses central config)
# MAGIC uv run refresh-stocks --dry-run           # print what would be fetched
# MAGIC uv run refresh-stocks --ticker NVDA       # refresh a single ticker
# MAGIC ```
# MAGIC
# MAGIC Run as a scheduled Databricks Job for automatic daily/weekly refresh.
# MAGIC
# MAGIC **Requires:** DATABRICKS_HOST, DATABRICKS_TOKEN, UC_CATALOG, UC_SCHEMA,
# MAGIC              SQL_WAREHOUSE_ID (or CLUSTER_ID for Databricks Connect)

# COMMAND ----------

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bootstrap

# COMMAND ----------

from notebooks.demo_shared.bootstrap import bootstrap

_project_root, _central_config = bootstrap(__file__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import yfinance as yf
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import (
    DateType, DoubleType, LongType, StringType,
    StructField, StructType, TimestampType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG     = os.environ.get("UC_CATALOG", "your_catalog").strip()
SCHEMA      = os.environ.get("UC_SCHEMA",  "your_schema").strip()
HOST        = os.environ.get("DATABRICKS_HOST", "").strip()
TOKEN       = os.environ.get("DATABRICKS_TOKEN", "").strip()

REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.company_tickers_registry"
REFRESH_TABLE  = f"{CATALOG}.{SCHEMA}.bronze_stock_daily_refresh"

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------


def get_spark() -> SparkSession:
    """Get or create a Spark session (local or Databricks Connect)."""
    return SparkSession.builder.getOrCreate()


def get_latest_dates(spark: SparkSession) -> dict[str, date]:
    """Return MAX(trade_date) per ticker from the refresh table.
    Returns empty dict if the table does not yet exist."""
    try:
        df = spark.sql(
            f"SELECT ticker, MAX(trade_date) AS latest "
            f"FROM {REFRESH_TABLE} GROUP BY ticker"
        )
        return {r.ticker: r.latest for r in df.collect()}
    except Exception:
        return {}


def get_registry_tickers(spark: SparkSession) -> list[str]:
    """Read all tickers from company_tickers_registry."""
    try:
        df = spark.table(REGISTRY_TABLE)
        tickers = [r.ticker for r in df.select("ticker").collect()]
        if not tickers:
            print(f"WARNING: {REGISTRY_TABLE} is empty — no tickers to refresh")
        return tickers
    except Exception as exc:
        print(f"ERROR: Could not read {REGISTRY_TABLE}: {exc}", file=sys.stderr)
        return []


def fetch_incremental(spark: SparkSession, ticker: str, since: date | None, dry_run: bool) -> int:
    """Fetch new rows for one ticker since `since` date. Returns row count appended."""
    start = (since + timedelta(days=1)) if since else (date.today() - timedelta(days=30))
    end   = date.today()

    if start > end:
        print(f"  {ticker}: already up-to-date (latest={since})")
        return 0

    print(f"  {ticker}: fetching {start} → {end} (since={since or 'none — last 30 days'})")

    if dry_run:
        return 0

    try:
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
        if hist.empty:
            print(f"    No data returned for {ticker}")
            return 0

        hist = hist.reset_index()
        hist = hist.rename(columns={
            "Date": "trade_date", "Open": "open_price", "High": "high_price",
            "Low": "low_price", "Close": "close_price", "Volume": "volume",
            "Dividends": "dividends", "Stock Splits": "stock_splits",
        })
        # Normalize trade_date — yfinance may return timezone-aware timestamps
        hist["trade_date"] = pd.to_datetime(hist["trade_date"]).dt.tz_localize(None).dt.date

        sdf = (
            spark.createDataFrame(
                hist[["trade_date", "open_price", "high_price", "low_price",
                      "close_price", "volume", "dividends", "stock_splits"]]
            )
            .withColumn("trade_date", col("trade_date").cast(DateType()))
            .withColumn("ticker", lit(ticker))
            .withColumn("ingestion_timestamp", current_timestamp())
            .select(
                "ticker", "trade_date", "open_price", "high_price", "low_price",
                "close_price", "volume", "dividends", "stock_splits", "ingestion_timestamp",
            )
        )
        row_count = sdf.count()
        (
            sdf.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(REFRESH_TABLE)
        )
        print(f"    Appended {row_count} rows to {REFRESH_TABLE}")
        return row_count

    except Exception as exc:
        print(f"    WARNING: Could not refresh {ticker}: {exc}")
        return 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incremental stock price refresh — appends new rows to bronze_stock_daily_refresh"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without writing any data")
    parser.add_argument("--ticker", metavar="TICKER",
                        help="Refresh a single ticker only (default: all from registry)")
    args = parser.parse_args()

    if not HOST or not TOKEN:
        print("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN required", file=sys.stderr)
        return 1

    spark = get_spark()

    print("=" * 60)
    print("Incremental Stock Price Refresh")
    print("=" * 60)
    print(f"  Registry table:  {REGISTRY_TABLE}")
    print(f"  Refresh table:   {REFRESH_TABLE}")
    if args.dry_run:
        print("  Mode: DRY RUN — no data will be written")
    print()

    # Get all tickers from the pipeline-managed registry
    if args.ticker:
        tickers = [args.ticker]
        print(f"Single-ticker mode: {args.ticker}")
    else:
        tickers = get_registry_tickers(spark)
        if not tickers:
            return 1

    print(f"Tickers to refresh ({len(tickers)}): {tickers}")

    # Find latest known dates per ticker
    latest_dates = get_latest_dates(spark)

    # Fetch incremental data
    total_rows = 0
    for ticker in tickers:
        latest = latest_dates.get(ticker)
        rows = fetch_incremental(spark, ticker, latest, dry_run=args.dry_run)
        total_rows += rows

    print()
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN complete — no rows written")
    else:
        print(f"Refresh complete — {total_rows} total rows appended to {REFRESH_TABLE}")
    print("=" * 60)
    return 0


# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
