-- ============================================================================
-- Gold Layer: Revenue by Geography
-- ============================================================================
--
-- This materialized view extracts and flattens geographic revenue data from
-- the silver layer extracted metrics.
--
-- Target: sec_fin_revenue_by_geography (for Genie Space queries)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_revenue_by_geography
COMMENT 'Geographic revenue breakdown — company identity from AI-extracted registry, no hardcoded mappings'
AS
SELECT
    COALESCE(
        extracted_metrics:response:company_name::STRING,
        r.company_name,
        s.company_key
    ) AS company,
    COALESCE(
        extracted_metrics:response:ticker_symbol::STRING,
        r.ticker,
        s.company_key
    ) AS ticker,
    COALESCE(extracted_metrics:response:fiscal_year::INT, 2024) AS fiscal_year,
    geo.region,
    geo.region_revenue_billions,
    ROUND(
        geo.region_revenue_billions /
        NULLIF(extracted_metrics:response:total_revenue_billions::DOUBLE, 0) * 100,
        2
    ) AS region_pct_of_total,
    CAST(NULL AS DOUBLE) AS region_growth_yoy_pct
FROM silver_sec_financial_metrics s
LEFT JOIN company_tickers_registry r
    ON r.ticker = extracted_metrics:response:ticker_symbol::STRING
LATERAL VIEW EXPLODE(
    from_json(
        extracted_metrics:response:geographic_revenue::STRING,
        'array<struct<region:string,region_revenue_billions:double>>'
    )
) t AS geo
WHERE extracted_metrics:response:geographic_revenue IS NOT NULL
  AND geo.region IS NOT NULL
  AND geo.region_revenue_billions > 0;
