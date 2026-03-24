# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Create Stock Analysis Views
# MAGIC 
# MAGIC This notebook creates analytical views that combine financial data with stock data.
# MAGIC These views join the pipeline-generated financial tables with Yahoo Finance stock data.
# MAGIC 
# MAGIC **Prerequisites:**
# MAGIC - Run `uv run deploy-sdp-pipeline` first (SDP pipeline creates financial tables + stock history)

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# Set context
spark.sql(f"USE CATALOG {UC_CATALOG}")
spark.sql(f"USE SCHEMA {UC_SCHEMA}")
print(f"✅ Set catalog: {UC_CATALOG}, schema: {UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify Source Tables

# COMMAND ----------

print("Source Table Status:")
print("-" * 50)

source_tables = ['company_financials', 'revenue_by_segment', 'revenue_by_geography', 
                 'stock_daily_prices', 'stock_summary']

for table_key in source_tables:
    table_name = TABLES[table_key]
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]['cnt']
        print(f"✅ {table_name}: {count} rows")
    except Exception as e:
        print(f"❌ {table_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Company Overview View
# MAGIC 
# MAGIC Combines financial metrics with current stock data.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}company_overview AS
SELECT 
    f.company,
    f.ticker,
    f.fiscal_year,
    f.fiscal_year_end_date,
    
    -- Revenue Metrics
    f.total_revenue_billions,
    f.revenue_growth_yoy_pct,
    f.cost_of_revenue_billions,
    
    -- Profitability
    f.gross_profit_billions,
    f.gross_margin_pct,
    f.operating_income_billions,
    f.operating_margin_pct,
    f.net_income_billions,
    f.net_margin_pct,
    f.eps_diluted as earnings_per_share,
    
    -- Balance Sheet
    f.total_assets_billions,
    f.total_liabilities_billions,
    f.total_equity_billions as stockholders_equity_billions,
    f.cash_and_equivalents_billions,
    
    -- Stock Metrics
    s.current_price,
    s.price_date,
    s.market_cap_billions,
    s.pe_ratio,
    s.forward_pe,
    s.peg_ratio,
    s.price_to_sales,
    s.price_to_book,
    s.enterprise_value_billions,
    s.ev_to_revenue,
    s.ev_to_ebitda,
    s.dividend_yield_pct,
    s.beta,
    s.fifty_two_week_high,
    s.fifty_two_week_low,
    s.ytd_return_pct,
    s.one_year_return_pct,
    s.trend_signal,
    
    -- Valuation Metrics (calculated)
    ROUND(s.market_cap_billions / NULLIF(f.net_income_billions, 0), 2) as price_to_earnings,
    ROUND(s.market_cap_billions / NULLIF(f.total_revenue_billions, 0), 2) as market_cap_to_revenue,
    
    f.last_updated as financial_data_updated,
    s.last_updated as stock_data_updated

FROM {TABLES['company_financials']} f
LEFT JOIN {TABLES['stock_summary']} s ON f.ticker = s.ticker
""")
print(f"✅ Created view: {TABLE_PREFIX}company_overview")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Peer Comparison View

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}peer_comparison AS
SELECT 
    company,
    ticker,
    fiscal_year,
    
    -- Size metrics
    total_revenue_billions,
    market_cap_billions,
    
    -- Growth
    revenue_growth_yoy_pct,
    ytd_return_pct,
    
    -- Profitability
    gross_margin_pct,
    operating_margin_pct,
    net_margin_pct,
    
    -- Valuation
    pe_ratio,
    price_to_sales,
    ev_to_revenue,
    
    -- Performance rankings within peer group
    RANK() OVER (ORDER BY total_revenue_billions DESC) as revenue_rank,
    RANK() OVER (ORDER BY net_margin_pct DESC) as margin_rank,
    RANK() OVER (ORDER BY revenue_growth_yoy_pct DESC NULLS LAST) as growth_rank,
    RANK() OVER (ORDER BY pe_ratio ASC NULLS LAST) as pe_rank
    
FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}company_overview
""")
print(f"✅ Created view: {TABLE_PREFIX}peer_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Stock Performance View

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}stock_performance AS
SELECT 
    sp.ticker,
    c.company,
    sp.trade_date,
    sp.close_price,
    sp.volume,
    sp.daily_return_pct,
    sp.sma_20,
    sp.sma_50,
    sp.sma_200,
    sp.volatility_20d,
    
    -- Technical signals
    CASE 
        WHEN sp.close_price > sp.sma_50 AND sp.close_price > sp.sma_200 THEN 'Bullish'
        WHEN sp.close_price < sp.sma_50 AND sp.close_price < sp.sma_200 THEN 'Bearish'
        ELSE 'Neutral'
    END as trend_signal,
    
    -- Price vs moving averages
    ROUND((sp.close_price - sp.sma_50) / NULLIF(sp.sma_50, 0) * 100, 2) as pct_above_sma_50,
    ROUND((sp.close_price - sp.sma_200) / NULLIF(sp.sma_200, 0) * 100, 2) as pct_above_sma_200

FROM {TABLES['stock_daily_prices']} sp
LEFT JOIN {TABLES['company_financials']} c ON sp.ticker = c.ticker
""")
print(f"✅ Created view: {TABLE_PREFIX}stock_performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Revenue Breakdown View

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}revenue_breakdown AS
SELECT 
    'segment' as breakdown_type,
    company,
    ticker,
    fiscal_year,
    segment_name as category,
    segment_revenue_billions as revenue_billions,
    segment_pct_of_total as pct_of_total,
    segment_rank as category_rank
FROM {TABLES['revenue_by_segment']}

UNION ALL

SELECT 
    'geography' as breakdown_type,
    company,
    ticker,
    fiscal_year,
    region as category,
    region_revenue_billions as revenue_billions,
    region_pct_of_total as pct_of_total,
    ROW_NUMBER() OVER (PARTITION BY company ORDER BY region_revenue_billions DESC) as category_rank
FROM {TABLES['revenue_by_geography']}
""")
print(f"✅ Created view: {TABLE_PREFIX}revenue_breakdown")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

print("=" * 60)
print("Verification: Created Views")
print("=" * 60)

views = [
    f"{TABLE_PREFIX}company_overview",
    f"{TABLE_PREFIX}peer_comparison",
    f"{TABLE_PREFIX}stock_performance",
    f"{TABLE_PREFIX}revenue_breakdown"
]

for view in views:
    try:
        count = spark.sql(f"SELECT COUNT(*) FROM {UC_CATALOG}.{UC_SCHEMA}.{view}").collect()[0][0]
        print(f"✅ {view}: {count} rows")
    except Exception as e:
        print(f"❌ {view}: {e}")

# COMMAND ----------

# Show sample from company_overview
print("\n📊 Company Overview Sample:")
display(spark.sql(f"""
SELECT company, ticker, 
       ROUND(total_revenue_billions, 1) as revenue_B,
       ROUND(gross_margin_pct, 1) as gross_margin,
       ROUND(market_cap_billions, 1) as market_cap_B,
       pe_ratio,
       trend_signal
FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}company_overview
ORDER BY total_revenue_billions DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"""
{'=' * 60}
Stock Views - Complete
{'=' * 60}

Views Created:
  • {TABLE_PREFIX}company_overview - Financial + stock data combined
  • {TABLE_PREFIX}peer_comparison - Company rankings and comparisons
  • {TABLE_PREFIX}stock_performance - Technical analysis of stock prices
  • {TABLE_PREFIX}revenue_breakdown - Segment and geography breakdown

Next Steps:
  1. Run 06_create_uc_functions.py - Create UC functions
  2. Run 07_create_genie_space.py - Create Genie Space
""")
