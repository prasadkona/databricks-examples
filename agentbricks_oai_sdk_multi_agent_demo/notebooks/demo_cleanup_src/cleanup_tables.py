# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup: Drop Tables, Views, and SDP Pipeline
# MAGIC
# MAGIC Drops all demo tables, optionally drops views, and deletes the SDP pipeline.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run demo-cleanup tables
# MAGIC uv run demo-cleanup tables --include-views
# MAGIC uv run demo-cleanup tables --dry-run
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `SQL_WAREHOUSE_ID`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from notebooks.demo_shared import api_request, run_sql
from notebooks.demo_shared.bootstrap import get_project_root

_project_root = get_project_root(__file__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables and Views to Drop

# COMMAND ----------

TABLES_TO_DROP = [
    # Bronze — SEC documents (DLT-managed streaming table)
    "bronze_sec_parsed_documents",
    # Silver
    "silver_sec_financial_metrics",
    # Company / ticker registry (DLT MV, drives stock loading)
    "company_tickers_registry",
    # Gold — financials
    "gold_company_financials",
    "gold_revenue_by_segment",
    "gold_revenue_by_geography",
    # Stock pipeline — DLT-managed initial load
    "bronze_stock_initial",
    # Stock pipeline — external incremental refresh table (written by refresh-stocks job)
    "bronze_stock_daily_refresh",
    # Silver + Gold stock views
    "silver_stock_daily_prices",
    "gold_stock_summary",
    # Legacy: pre-Sprint-2 standalone bronze stock table (kept for backward compat)
    "bronze_stock_daily_prices",
]

VIEWS_TO_DROP = [
    "sec_fin_company_overview",
    "sec_fin_peer_comparison",
    "sec_fin_stock_performance",
    "sec_fin_revenue_breakdown",
]

PIPELINE_NAME = "sec_financial_analyst_pipeline"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Host Helper

# COMMAND ----------

def _get_bundle_ws_host() -> str | None:
    yml = _project_root / "notebooks" / "sdp_pipeline_src" / "databricks.yml"
    if not yml.exists():
        return None
    text = yml.read_text(encoding="utf-8")
    m = re.search(r"workspace:\s*\n\s*host:\s*(\S+)", text)
    return m.group(1).strip() if m else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Drop Tables, Views, and Pipeline

# COMMAND ----------

def run(include_views: bool = False, dry_run: bool = False) -> int:
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    wh_id = os.environ.get("SQL_WAREHOUSE_ID", "").strip()
    catalog = os.environ.get("UC_CATALOG", "your_catalog")
    schema = os.environ.get("UC_SCHEMA", "your_schema")

    if not host or not token:
        print("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN required", file=sys.stderr)
        return 1
    if not wh_id:
        print("ERROR: SQL_WAREHOUSE_ID required", file=sys.stderr)
        return 1

    stmts = [f"DROP TABLE IF EXISTS {catalog}.{schema}.{t};" for t in TABLES_TO_DROP]
    if include_views:
        stmts += [f"DROP VIEW IF EXISTS {catalog}.{schema}.{v};" for v in VIEWS_TO_DROP]

    if dry_run:
        for s in stmts:
            print(s)
        return 0

    print("Dropping tables" + (" and views" if include_views else "") + "...")
    for s in stmts:
        try:
            run_sql(host, token, wh_id, catalog, schema, s)
            print(f"  OK: {s.strip()}")
        except Exception as e:
            print(f"  Skip: {s.strip()} -> {e}")

    pipeline_host = _get_bundle_ws_host() or host
    try:
        params: dict = {"max_results": 100}
        all_statuses: list = []
        while True:
            result = api_request("GET", pipeline_host, token, "/api/2.0/pipelines", params)
            all_statuses.extend(result.get("statuses", []))
            npt = result.get("next_page_token")
            if not npt:
                break
            params = {"page_token": npt, "max_results": 100}

        deleted = False
        for p in all_statuses:
            name = p.get("name", "")
            if name == PIPELINE_NAME or name.endswith("sec_financial_analyst_pipeline"):
                pid = p.get("pipeline_id")
                if pid:
                    api_request("DELETE", pipeline_host, token, f"/api/2.0/pipelines/{pid}")
                    print(f"  OK: Deleted pipeline '{name}' ({pid})")
                    deleted = True
                break
        if not deleted:
            print(f"  OK: No pipeline '{PIPELINE_NAME}' to delete")
    except Exception as e:
        print(f"  Skip: Delete pipeline -> {e}")

    print("Table cleanup done.")
    return 0
