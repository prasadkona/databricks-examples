-- ============================================================================
-- Gold Layer: Company Financials
-- ============================================================================
--
-- Final company financials table from extracted silver metrics.
-- Company name and ticker are sourced from company_tickers_registry
-- (derived from ai_extract) — no hardcoded CASE statements.
--
-- Target: gold_company_financials (for Genie Space and UC Functions)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_company_financials
COMMENT 'Final company financial metrics — company identity from AI-extracted registry, no hardcoded mappings'
AS
SELECT
    -- Company identity: prefer ai_extract result, fall back to registry, then file stem
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
    COALESCE(extracted_metrics:response:fiscal_year::INT, 2024)       AS fiscal_year,
    extracted_metrics:response:fiscal_year_end_date::STRING            AS fiscal_year_end_date,
    extracted_metrics:response:total_revenue_billions::DOUBLE          AS total_revenue_billions,
    extracted_metrics:response:cost_of_revenue_billions::DOUBLE        AS cost_of_revenue_billions,
    extracted_metrics:response:gross_profit_billions::DOUBLE           AS gross_profit_billions,
    extracted_metrics:response:operating_income_billions::DOUBLE       AS operating_income_billions,
    extracted_metrics:response:net_income_billions::DOUBLE             AS net_income_billions,
    extracted_metrics:response:total_assets_billions::DOUBLE           AS total_assets_billions,
    extracted_metrics:response:total_liabilities_billions::DOUBLE      AS total_liabilities_billions,
    extracted_metrics:response:total_equity_billions::DOUBLE           AS total_equity_billions,
    extracted_metrics:response:cash_and_equivalents_billions::DOUBLE   AS cash_and_equivalents_billions,
    extracted_metrics:response:eps_diluted::DOUBLE                     AS eps_diluted,
    -- Calculated margins
    ROUND(
        extracted_metrics:response:gross_profit_billions::DOUBLE /
        NULLIF(extracted_metrics:response:total_revenue_billions::DOUBLE, 0) * 100,
        2
    ) AS gross_margin_pct,
    ROUND(
        extracted_metrics:response:operating_income_billions::DOUBLE /
        NULLIF(extracted_metrics:response:total_revenue_billions::DOUBLE, 0) * 100,
        2
    ) AS operating_margin_pct,
    ROUND(
        extracted_metrics:response:net_income_billions::DOUBLE /
        NULLIF(extracted_metrics:response:total_revenue_billions::DOUBLE, 0) * 100,
        2
    ) AS net_margin_pct,
    extracted_metrics:response:revenue_growth_yoy_pct::DOUBLE          AS revenue_growth_yoy_pct,
    s.document_type,
    s.industry_sector,
    current_timestamp()                                                 AS last_updated
FROM silver_sec_financial_metrics s
LEFT JOIN company_tickers_registry r
    ON r.ticker = extracted_metrics:response:ticker_symbol::STRING;
