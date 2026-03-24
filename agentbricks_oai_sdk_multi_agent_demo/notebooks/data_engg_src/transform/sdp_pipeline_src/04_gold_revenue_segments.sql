-- ============================================================================
-- Gold Layer: Revenue by Business Segment
-- ============================================================================
--
-- This materialized view extracts and flattens business segment data from
-- the silver layer extracted metrics.
--
-- Target: sec_fin_revenue_by_segment (for Genie Space queries)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_revenue_by_segment
COMMENT 'Business segment revenue breakdown — company identity from AI-extracted registry, no hardcoded mappings'
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
    seg.segment_name,
    seg.segment_revenue_billions,
    ROUND(
        seg.segment_revenue_billions /
        NULLIF(extracted_metrics:response:total_revenue_billions::DOUBLE, 0) * 100,
        2
    ) AS segment_pct_of_total,
    CAST(NULL AS DOUBLE) AS segment_growth_yoy_pct,
    ROW_NUMBER() OVER (
        PARTITION BY COALESCE(extracted_metrics:response:ticker_symbol::STRING, s.company_key)
        ORDER BY seg.segment_revenue_billions DESC
    ) AS segment_rank
FROM silver_sec_financial_metrics s
LEFT JOIN company_tickers_registry r
    ON r.ticker = extracted_metrics:response:ticker_symbol::STRING
LATERAL VIEW EXPLODE(
    from_json(
        extracted_metrics:response:business_segments::STRING,
        'array<struct<segment_name:string,segment_revenue_billions:double>>'
    )
) t AS seg
WHERE extracted_metrics:response:business_segments IS NOT NULL
  AND seg.segment_name IS NOT NULL;
