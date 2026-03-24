-- ============================================================================
-- Gold Layer: Stock Summary
-- ============================================================================
--
-- This materialized view creates a summary table with latest prices,
-- valuation metrics, and technical signals for each stock.
--
-- Source: silver_stock_daily_prices
-- Target: gold_stock_summary (for Genie Space and agent queries)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_stock_summary
COMMENT 'Stock summary with latest prices and metrics - Gold layer'
AS
WITH latest_prices AS (
    -- Get the latest row for each ticker
    SELECT *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) as rn
        FROM silver_stock_daily_prices
    )
    WHERE rn = 1
),
yearly_data AS (
    -- One row per ticker with 52-week and return inputs
    SELECT ticker, fifty_two_week_low, fifty_two_week_high, year_start_price, one_year_ago_price
    FROM (
        SELECT 
            ticker,
            MIN(close_price) OVER (PARTITION BY ticker) as fifty_two_week_low,
            MAX(close_price) OVER (PARTITION BY ticker) as fifty_two_week_high,
            FIRST_VALUE(close_price) OVER (PARTITION BY ticker ORDER BY trade_date) as year_start_price,
            FIRST_VALUE(close_price) OVER (PARTITION BY ticker ORDER BY trade_date DESC) as one_year_ago_price,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date) as rn
        FROM silver_stock_daily_prices
        WHERE trade_date >= DATE_SUB(CURRENT_DATE(), 365)
    ) t
    WHERE rn = 1
),
volume_10d AS (
    SELECT 
        ticker,
        ROUND(AVG(volume), 0) as avg_volume_10d
    FROM (
        SELECT ticker, volume,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) as rn
        FROM silver_stock_daily_prices
    )
    WHERE rn <= 10
    GROUP BY ticker
)
SELECT
    lp.ticker,
    -- Company name from registry (derived from ai_extract) — no hardcoded CASE
    COALESCE(r.company_name, lp.ticker) AS company,
    
    -- Current price info
    lp.close_price as current_price,
    lp.trade_date as price_date,
    
    -- Placeholder for market data (would come from yfinance info)
    -- These would be updated by a separate process or estimated
    CAST(NULL AS DOUBLE) as market_cap_billions,
    CAST(NULL AS DOUBLE) as pe_ratio,
    CAST(NULL AS DOUBLE) as forward_pe,
    CAST(NULL AS DOUBLE) as peg_ratio,
    CAST(NULL AS DOUBLE) as price_to_sales,
    CAST(NULL AS DOUBLE) as price_to_book,
    CAST(NULL AS DOUBLE) as enterprise_value_billions,
    CAST(NULL AS DOUBLE) as ev_to_revenue,
    CAST(NULL AS DOUBLE) as ev_to_ebitda,
    CAST(NULL AS DOUBLE) as dividend_yield_pct,
    CAST(NULL AS DOUBLE) as beta,
    
    -- 52-week range
    yd.fifty_two_week_high,
    yd.fifty_two_week_low,
    
    -- Returns
    ROUND((lp.close_price - yd.year_start_price) / NULLIF(yd.year_start_price, 0) * 100, 2) as ytd_return_pct,
    ROUND((lp.close_price - yd.one_year_ago_price) / NULLIF(yd.one_year_ago_price, 0) * 100, 2) as one_year_return_pct,
    
    -- Volume
    v10.avg_volume_10d,
    
    -- Technical signals
    lp.close_price > lp.sma_50 as above_sma_50,
    lp.close_price > lp.sma_200 as above_sma_200,
    
    CASE 
        WHEN lp.close_price > lp.sma_50 AND lp.close_price > lp.sma_200 THEN 'Bullish'
        WHEN lp.close_price < lp.sma_50 AND lp.close_price < lp.sma_200 THEN 'Bearish'
        ELSE 'Neutral'
    END as trend_signal,
    
    CURRENT_TIMESTAMP() as last_updated

FROM latest_prices lp
LEFT JOIN yearly_data yd          ON lp.ticker = yd.ticker
LEFT JOIN volume_10d v10          ON lp.ticker = v10.ticker
LEFT JOIN company_tickers_registry r ON lp.ticker = r.ticker;
