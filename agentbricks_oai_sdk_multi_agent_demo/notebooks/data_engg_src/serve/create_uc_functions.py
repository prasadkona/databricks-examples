# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Create UC Functions
# MAGIC 
# MAGIC Creates Unity Catalog SQL functions for financial analysis.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# Set the catalog and schema context
spark.sql(f"USE CATALOG {UC_CATALOG}")
spark.sql(f"USE SCHEMA {UC_SCHEMA}")
print(f"✅ Set catalog: {UC_CATALOG}, schema: {UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Valuation Score Function

# COMMAND ----------

# Create valuation score function
valuation_sql = f"""
CREATE OR REPLACE FUNCTION {FUNCTIONS['valuation_score']}(ticker_param STRING)
RETURNS TABLE (
    ticker STRING,
    company STRING,
    valuation_score INT,
    pe_component INT,
    growth_component INT,
    margin_component INT
)
RETURN
SELECT 
    f.ticker,
    f.company,
    CAST(
        CASE WHEN s.pe_ratio < 25 THEN 30 WHEN s.pe_ratio < 50 THEN 20 ELSE 10 END +
        CASE WHEN f.revenue_growth_yoy_pct > 50 THEN 40 WHEN f.revenue_growth_yoy_pct > 10 THEN 25 ELSE 10 END +
        CASE WHEN f.gross_margin_pct > 60 THEN 30 WHEN f.gross_margin_pct > 40 THEN 20 ELSE 10 END
    AS INT) as valuation_score,
    CASE WHEN s.pe_ratio < 25 THEN 30 WHEN s.pe_ratio < 50 THEN 20 ELSE 10 END as pe_component,
    CASE WHEN f.revenue_growth_yoy_pct > 50 THEN 40 WHEN f.revenue_growth_yoy_pct > 10 THEN 25 ELSE 10 END as growth_component,
    CASE WHEN f.gross_margin_pct > 60 THEN 30 WHEN f.gross_margin_pct > 40 THEN 20 ELSE 10 END as margin_component
FROM {TABLES['company_financials']} f
JOIN {TABLES['stock_summary']} s ON f.ticker = s.ticker
WHERE f.ticker = ticker_param
"""

try:
    spark.sql(valuation_sql)
    print(f"✅ Created: {FUNCTIONS['valuation_score']}")
except Exception as e:
    print(f"❌ Error creating valuation_score: {e}")

# COMMAND ----------

# Test valuation score
try:
    display(spark.sql(f"SELECT * FROM {FUNCTIONS['valuation_score']}('NVDA')"))
except Exception as e:
    print(f"Test error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compare Peers Function

# COMMAND ----------

peers_sql = f"""
CREATE OR REPLACE FUNCTION {FUNCTIONS['compare_peers']}(ticker_param STRING)
RETURNS TABLE (
    metric STRING,
    company_value DOUBLE,
    peer_avg DOUBLE
)
RETURN
WITH company AS (
    SELECT revenue_growth_yoy_pct, gross_margin_pct, net_margin_pct
    FROM {TABLES['company_financials']}
    WHERE ticker = ticker_param
),
peers AS (
    SELECT 
        AVG(revenue_growth_yoy_pct) as avg_growth,
        AVG(gross_margin_pct) as avg_margin,
        AVG(net_margin_pct) as avg_net
    FROM {TABLES['company_financials']}
    WHERE ticker != ticker_param
)
SELECT 'Revenue Growth %', c.revenue_growth_yoy_pct, ROUND(p.avg_growth, 1) FROM company c, peers p
UNION ALL
SELECT 'Gross Margin %', c.gross_margin_pct, ROUND(p.avg_margin, 1) FROM company c, peers p
UNION ALL
SELECT 'Net Margin %', c.net_margin_pct, ROUND(p.avg_net, 1) FROM company c, peers p
"""

try:
    spark.sql(peers_sql)
    print(f"✅ Created: {FUNCTIONS['compare_peers']}")
except Exception as e:
    print(f"❌ Error creating compare_peers: {e}")

# COMMAND ----------

# Test compare peers
try:
    display(spark.sql(f"SELECT * FROM {FUNCTIONS['compare_peers']}('NVDA')"))
except Exception as e:
    print(f"Test error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Investment Thesis Function

# COMMAND ----------

thesis_sql = f"""
CREATE OR REPLACE FUNCTION {FUNCTIONS['investment_thesis']}(ticker_param STRING)
RETURNS STRING
RETURN (
    SELECT CONCAT(
        company, ' (', ticker, ') Summary: ',
        'Revenue $', ROUND(total_revenue_billions, 1), 'B, ',
        'Growth ', ROUND(revenue_growth_yoy_pct, 1), '%, ',
        'Margin ', ROUND(gross_margin_pct, 1), '%'
    )
    FROM {TABLES['company_financials']}
    WHERE ticker = ticker_param
)
"""

try:
    spark.sql(thesis_sql)
    print(f"✅ Created: {FUNCTIONS['investment_thesis']}")
except Exception as e:
    print(f"❌ Error creating investment_thesis: {e}")

# COMMAND ----------

# Test investment thesis
try:
    result = spark.sql(f"SELECT {FUNCTIONS['investment_thesis']}('NVDA') as thesis").collect()
    print(result[0]['thesis'])
except Exception as e:
    print(f"Test error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Growth Trajectory Function

# COMMAND ----------

growth_sql = f"""
CREATE OR REPLACE FUNCTION {FUNCTIONS['growth_trajectory']}(ticker_param STRING)
RETURNS TABLE (
    year_offset INT,
    projected_revenue DOUBLE,
    growth_rate DOUBLE
)
RETURN
WITH base AS (
    SELECT total_revenue_billions, revenue_growth_yoy_pct
    FROM {TABLES['company_financials']}
    WHERE ticker = ticker_param
)
SELECT 1, ROUND(total_revenue_billions * 1.15, 1), 15.0 FROM base
UNION ALL
SELECT 2, ROUND(total_revenue_billions * 1.25, 1), 10.0 FROM base
UNION ALL
SELECT 3, ROUND(total_revenue_billions * 1.33, 1), 8.0 FROM base
"""

try:
    spark.sql(growth_sql)
    print(f"✅ Created: {FUNCTIONS['growth_trajectory']}")
except Exception as e:
    print(f"❌ Error creating growth_trajectory: {e}")

# COMMAND ----------

# Test growth trajectory
try:
    display(spark.sql(f"SELECT * FROM {FUNCTIONS['growth_trajectory']}('NVDA')"))
except Exception as e:
    print(f"Test error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Risk Summary Function

# COMMAND ----------

risk_sql = f"""
CREATE OR REPLACE FUNCTION {FUNCTIONS['risk_summary']}(ticker_param STRING)
RETURNS TABLE (
    risk_type STRING,
    severity STRING,
    description STRING
)
RETURN
SELECT 
    'Valuation' as risk_type,
    CASE WHEN s.pe_ratio > 50 THEN 'HIGH' WHEN s.pe_ratio > 30 THEN 'MEDIUM' ELSE 'LOW' END as severity,
    CONCAT('P/E ratio is ', ROUND(s.pe_ratio, 1)) as description
FROM {TABLES['stock_summary']} s
WHERE s.ticker = ticker_param
UNION ALL
SELECT 
    'Growth Sustainability',
    CASE WHEN f.revenue_growth_yoy_pct > 80 THEN 'HIGH' WHEN f.revenue_growth_yoy_pct > 30 THEN 'MEDIUM' ELSE 'LOW' END,
    CONCAT('Growth rate of ', ROUND(f.revenue_growth_yoy_pct, 1), '% may not sustain')
FROM {TABLES['company_financials']} f
WHERE f.ticker = ticker_param
UNION ALL
SELECT 
    'Debt',
    CASE WHEN f.total_liabilities_billions > f.cash_and_equivalents_billions THEN 'MEDIUM' ELSE 'LOW' END,
    CONCAT('Liabilities: $', ROUND(f.total_liabilities_billions, 1), 'B vs Cash: $', ROUND(f.cash_and_equivalents_billions, 1), 'B')
FROM {TABLES['company_financials']} f
WHERE f.ticker = ticker_param
"""

try:
    spark.sql(risk_sql)
    print(f"✅ Created: {FUNCTIONS['risk_summary']}")
except Exception as e:
    print(f"❌ Error creating risk_summary: {e}")

# COMMAND ----------

# Test risk summary
try:
    display(spark.sql(f"SELECT * FROM {FUNCTIONS['risk_summary']}('NVDA')"))
except Exception as e:
    print(f"Test error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("UC FUNCTIONS CREATION COMPLETE")
print("=" * 60)

print(f"\nFunctions in {UC_CATALOG}.{UC_SCHEMA}:")
for name, full_name in FUNCTIONS.items():
    print(f"  • {full_name}")

print("\n" + "=" * 60)
