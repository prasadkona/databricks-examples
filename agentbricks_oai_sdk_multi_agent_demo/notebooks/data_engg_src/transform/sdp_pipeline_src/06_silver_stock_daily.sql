-- ============================================================================
-- Silver Layer: Stock Daily Prices with Technical Indicators
-- ============================================================================
--
-- Transforms raw stock data from bronze into silver with technical
-- indicators (SMAs, daily returns, volatility).
--
-- Source:
--   bronze_stock_initial — DLT-managed table; full 2y history loaded by
--                          00_bronze_stock_initial.py on each pipeline run.
--
-- NOTE: For incremental stock updates, run `uv run refresh-stocks` after the
-- pipeline completes. That writes to bronze_stock_daily_refresh (external table)
-- which can be UNION'd in a separate view if needed.
--
-- Deduplication: ROW_NUMBER() keeps the most recent record per (ticker, trade_date).
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_stock_daily_prices
COMMENT 'Stock prices with technical indicators from bronze_stock_initial'
AS
WITH combined_raw AS (
    -- Full history loaded by the core pipeline (DLT-managed)
    SELECT
        ticker, trade_date, open_price, high_price, low_price,
        close_price, volume, dividends, stock_splits, ingestion_timestamp
    FROM bronze_stock_initial
),
deduplicated AS (
    -- Keep only the latest record for each (ticker, trade_date) pair
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date
            ORDER BY ingestion_timestamp DESC
        ) AS _rn
    FROM combined_raw
),
with_returns AS (
    SELECT
        ticker,
        trade_date,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        dividends,
        stock_splits,
        ingestion_timestamp,
        ROUND(
            (close_price - LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date)) /
            NULLIF(LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date), 0) * 100,
            4
        ) AS daily_return_pct
    FROM deduplicated
    WHERE _rn = 1
)
SELECT
    ticker,
    trade_date,
    open_price,
    high_price,
    low_price,
    close_price,
    close_price AS adj_close_price,
    volume,
    dividends,
    stock_splits,
    ingestion_timestamp,
    daily_return_pct,

    -- Simple Moving Averages
    ROUND(AVG(close_price) OVER (
        PARTITION BY ticker
        ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ), 2) AS sma_20,

    ROUND(AVG(close_price) OVER (
        PARTITION BY ticker
        ORDER BY trade_date
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ), 2) AS sma_50,

    ROUND(AVG(close_price) OVER (
        PARTITION BY ticker
        ORDER BY trade_date
        ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
    ), 2) AS sma_200,

    -- 20-day Volatility
    ROUND(STDDEV(daily_return_pct) OVER (
        PARTITION BY ticker
        ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ), 4) AS volatility_20d

FROM with_returns;
