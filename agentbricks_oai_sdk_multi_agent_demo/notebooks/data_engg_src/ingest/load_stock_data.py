# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Load Stock Market Data (Bronze) [DEPRECATED]
# MAGIC
# MAGIC > **DEPRECATED**: Stock loading is now handled automatically by the SDP pipeline.
# MAGIC > `00_bronze_stock_initial.py` (DLT Python table) fetches full history for all
# MAGIC > companies discovered via `ai_extract` in `company_tickers_registry`.
# MAGIC > For incremental price updates, use `uv run refresh-stocks` (refresh_stock_prices.py).
# MAGIC >
# MAGIC > This file is kept as a **standalone fallback only** — it is no longer called
# MAGIC > by `run_sequence.py` or `run-workspace-notebooks`.
# MAGIC
# MAGIC This notebook fetches historical stock price data from Yahoo Finance and writes it
# MAGIC to a **bronze table** for downstream processing by the Lakeflow SDP pipeline.
# MAGIC 
# MAGIC ## Architecture
# MAGIC ```
# MAGIC Yahoo Finance API
# MAGIC         │
# MAGIC         ▼ [yfinance Python]
# MAGIC ┌─────────────────────────────┐
# MAGIC │ bronze_stock_daily_prices   │  Raw stock data (this notebook)
# MAGIC └─────────────────────────────┘
# MAGIC         │
# MAGIC         ▼ [SDP Pipeline]
# MAGIC ┌─────────────────────────────┐
# MAGIC │ silver_stock_daily_prices   │  With technical indicators
# MAGIC └─────────────────────────────┘
# MAGIC         │
# MAGIC         ▼ [SDP Pipeline]
# MAGIC ┌─────────────────────────────┐
# MAGIC │ gold_stock_summary          │  Latest prices & metrics
# MAGIC └─────────────────────────────┘
# MAGIC ```
# MAGIC 
# MAGIC **Prerequisites:**
# MAGIC - Run `00_setup_sec_documents.py` first (downloads SEC PDFs to volume)

# COMMAND ----------

# MAGIC %pip install yfinance pandas --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType, TimestampType

print(f"yfinance version: {yf.__version__}")
print(f"pandas version: {pd.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# Bronze table for raw stock data
BRONZE_STOCK_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.bronze_stock_daily_prices"

# Tickers to fetch
TICKERS = ["NVDA", "AAPL", "005930.KS"]

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=STOCK_DATA_YEARS * 365)

print("=" * 60)
print("Stock Data Configuration")
print("=" * 60)
print(f"\nBronze Table: {BRONZE_STOCK_TABLE}")
print(f"Tickers: {TICKERS}")
print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fetch Raw Stock Data from Yahoo Finance

# COMMAND ----------

print("=" * 60)
print("Fetching Stock Data from Yahoo Finance")
print("=" * 60)

all_stock_data = []

for ticker in TICKERS:
    print(f"\n📈 Fetching {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist is not None and len(hist) > 0:
            # Reset index to get Date as column
            hist = hist.reset_index()
            
            # Convert timezone-aware datetime to timezone-naive
            if 'Date' in hist.columns:
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            
            # Add metadata
            hist['ticker'] = ticker
            hist['ingestion_timestamp'] = pd.Timestamp.now()
            
            # Rename columns to standardized names
            hist = hist.rename(columns={
                'Date': 'trade_date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            # Select only needed columns
            hist = hist[['ticker', 'trade_date', 'open_price', 'high_price', 'low_price', 
                        'close_price', 'volume', 'dividends', 'stock_splits', 'ingestion_timestamp']]
            
            all_stock_data.append(hist)
            print(f"   ✅ Fetched {len(hist)} rows")
        else:
            print(f"   ⚠️ No data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Bronze Table Schema

# COMMAND ----------

# Define bronze table schema
bronze_schema = StructType([
    StructField("ticker", StringType(), False),
    StructField("trade_date", DateType(), False),
    StructField("open_price", DoubleType(), True),
    StructField("high_price", DoubleType(), True),
    StructField("low_price", DoubleType(), True),
    StructField("close_price", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("dividends", DoubleType(), True),
    StructField("stock_splits", DoubleType(), True),
    StructField("ingestion_timestamp", TimestampType(), True)
])

# Create table if not exists
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {BRONZE_STOCK_TABLE} (
    ticker STRING NOT NULL,
    trade_date DATE NOT NULL,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE,
    close_price DOUBLE,
    volume BIGINT,
    dividends DOUBLE,
    stock_splits DOUBLE,
    ingestion_timestamp TIMESTAMP
)
USING DELTA
COMMENT 'Raw stock price data from Yahoo Finance - Bronze layer'
""")

print(f"✅ Bronze table ready: {BRONZE_STOCK_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write to Bronze Table

# COMMAND ----------

if all_stock_data:
    # Combine all data
    combined_df = pd.concat(all_stock_data, ignore_index=True)
    
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(combined_df)
    
    # Cast types properly
    spark_df = spark_df \
        .withColumn("trade_date", F.col("trade_date").cast("date")) \
        .withColumn("volume", F.col("volume").cast("long")) \
        .withColumn("ingestion_timestamp", F.col("ingestion_timestamp").cast("timestamp"))
    
    # Write to bronze table (overwrite for full refresh)
    spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(BRONZE_STOCK_TABLE)
    
    row_count = spark.sql(f"SELECT COUNT(*) FROM {BRONZE_STOCK_TABLE}").collect()[0][0]
    print(f"\n✅ Wrote {row_count} rows to {BRONZE_STOCK_TABLE}")
else:
    print("\n⚠️ No stock data to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Bronze Data

# COMMAND ----------

print("=" * 60)
print("Bronze Table Summary")
print("=" * 60)

# Count by ticker
print("\n📊 Rows per ticker:")
display(spark.sql(f"""
SELECT 
    ticker,
    COUNT(*) as row_count,
    MIN(trade_date) as earliest_date,
    MAX(trade_date) as latest_date,
    ROUND(AVG(close_price), 2) as avg_close_price
FROM {BRONZE_STOCK_TABLE}
GROUP BY ticker
ORDER BY ticker
"""))

# COMMAND ----------

# Sample data
print("\n📈 Recent prices (last 5 days per ticker):")
display(spark.sql(f"""
SELECT ticker, trade_date, close_price, volume
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) as rn
    FROM {BRONZE_STOCK_TABLE}
)
WHERE rn <= 5
ORDER BY ticker, trade_date DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

row_count = spark.sql(f"SELECT COUNT(*) FROM {BRONZE_STOCK_TABLE}").collect()[0][0]
ticker_count = spark.sql(f"SELECT COUNT(DISTINCT ticker) FROM {BRONZE_STOCK_TABLE}").collect()[0][0]

print(f"""
{'=' * 60}
Stock Data Bronze Load - Complete
{'=' * 60}

Bronze Table: {BRONZE_STOCK_TABLE}
Total Rows: {row_count:,}
Tickers: {ticker_count}

The raw stock data is now available for the SDP pipeline to transform
into silver/gold tables with technical indicators and summaries.

Next Steps:
  1. Run `uv run deploy-sdp-pipeline` (SDP pipeline handles stock loading automatically)
  2. Or run 05_create_stock_views.py for stock analysis views
""")
