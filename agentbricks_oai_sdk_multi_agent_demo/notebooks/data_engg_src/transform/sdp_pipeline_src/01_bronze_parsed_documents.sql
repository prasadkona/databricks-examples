-- ============================================================================
-- Bronze Layer: Parse SEC PDF Documents
-- ============================================================================
--
-- Reads PDF files from the configured UC Volume path and produces structured
-- text + classification using three AI functions (one parse, two classify):
--
--   ai_parse_document  — called ONCE per document in the CTE inner query.
--   ai_classify x2     — called on parsed_content (not raw bytes) in the
--                        outer SELECT, reusing the CTE result.
--
-- Volume path is driven by pipeline variables: catalog, schema, volume,
-- docs_subfolder — no hardcoded paths or company names.
--
-- Source: /Volumes/${catalog}/${schema}/${volume}/${docs_subfolder}/*.pdf
-- ============================================================================

CREATE OR REFRESH STREAMING TABLE bronze_sec_parsed_documents
TBLPROPERTIES (
  'delta.feature.variantType-preview' = 'supported'
)
COMMENT 'Parsed SEC PDF documents using ai_parse_document v2.0 with document type and industry classification'
AS
-- CTE: parse each PDF once. ai_classify in the outer SELECT reuses parsed_content.
WITH raw_parsed AS (
  SELECT
    path,
    _metadata.file_name                                  AS file_name,
    _metadata.file_size                                  AS file_size,
    -- Simple file-stem fallback for company_key (e.g. NVDA from NVDA_FY2024_10K.pdf).
    -- Accurate company name and ticker come from ai_extract in the silver layer.
    regexp_extract(_metadata.file_name, '^([^_\\.]+)', 1) AS company_key,
    -- ai_parse_document: converts raw binary PDF to structured text variant.
    -- Called ONCE per document here; downstream layers read parsed_content directly.
    ai_parse_document(
      content,
      map(
        'version',                  '2.0',
        'descriptionElementTypes',  '*'
      )
    )                                                    AS parsed_content,
    current_timestamp()                                  AS _ingested_at
  FROM STREAM(read_files(
    '/Volumes/${catalog}/${schema}/${volume}/${docs_subfolder}',
    format => 'binaryFile'
  ))
)
SELECT
  *,
  -- ai_classify: document type — operates on parsed_content from CTE (no re-parse)
  -- Cast VARIANT to STRING since ai_classify expects STRING input
  ai_classify(
    CAST(parsed_content AS STRING),
    ARRAY(
      'SEC_10K_Annual_Report',
      'SEC_10Q_Quarterly_Report',
      'Annual_Report',
      'Earnings_Release',
      'Other_Financial_Document'
    )
  ) AS document_type,
  -- ai_classify: industry sector — same parsed_content, no additional parse call
  ai_classify(
    CAST(parsed_content AS STRING),
    ARRAY(
      'Semiconductor',
      'Consumer_Electronics',
      'Software',
      'Financial_Services',
      'Healthcare',
      'Energy',
      'Retail',
      'Other'
    )
  ) AS industry_sector
FROM raw_parsed;
