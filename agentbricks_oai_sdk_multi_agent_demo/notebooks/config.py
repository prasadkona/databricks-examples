# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration for SEC Financial Analyst Multi-Agent Demo
# MAGIC 
# MAGIC Shared configuration for all notebooks in this project.
# MAGIC Values are read from environment variables (loaded from central config)
# MAGIC with hardcoded defaults as fallback.

# COMMAND ----------

import os
from pathlib import Path

# Load central config when running locally (no-op in Databricks notebooks)
try:
    from dotenv import load_dotenv
    _central_config = Path(__file__).resolve().parent.parent.parent / "_local" / "config" / "databricks.env"
    if _central_config.exists():
        load_dotenv(_central_config, override=False)
except Exception:
    pass

# COMMAND ----------

# Configuration Constants
UC_CATALOG = os.getenv("UC_CATALOG", "your_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "your_schema")
UC_VOLUME = os.getenv("UC_VOLUME", "ka_demo")
TABLE_PREFIX = os.getenv("TABLE_PREFIX", "sec_fin_")

# SQL Warehouse ID (for AI functions and statement execution)
SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "your-warehouse-id")

# Full paths
VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
SEC_DOCS_PATH = f"{VOLUME_PATH}/sec_2024"

# Table names: pipeline gold/silver tables (used by views, functions, Genie)
# No copy step - views/functions/Genie read directly from SDP output.
TABLES = {
    "company_financials": f"{UC_CATALOG}.{UC_SCHEMA}.gold_company_financials",
    "revenue_by_segment": f"{UC_CATALOG}.{UC_SCHEMA}.gold_revenue_by_segment",
    "revenue_by_geography": f"{UC_CATALOG}.{UC_SCHEMA}.gold_revenue_by_geography",
    "stock_daily_prices": f"{UC_CATALOG}.{UC_SCHEMA}.silver_stock_daily_prices",
    "stock_summary": f"{UC_CATALOG}.{UC_SCHEMA}.gold_stock_summary",
}

# UC Function names (with prefix)
FUNCTIONS = {
    "valuation_score": f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}valuation_score",
    "compare_peers": f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}compare_peers",
    "investment_thesis": f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}investment_thesis",
    "growth_trajectory": f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}growth_trajectory",
    "risk_summary": f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}risk_summary",
}

# Demo company list — used by KA example questions and local test scripts.
# NOTE: The SDP pipeline does NOT use this list. It auto-discovers companies
# from SEC PDF content via ai_extract → company_tickers_registry.
# Add new companies by dropping their 10-K PDFs into the UC Volume.
COMPANIES = [
    {"name": "NVIDIA Corporation", "ticker": "NVDA", "fy_end": "January", "country": "USA"},
    {"name": "Apple Inc.", "ticker": "AAPL", "fy_end": "September", "country": "USA"},
    {"name": "Samsung Electronics", "ticker": "005930.KS", "fy_end": "December", "country": "South Korea"},
]

# Stock data configuration (used by legacy load_stock_data.py; pipeline uses 00_bronze_stock_initial.py)
STOCK_DATA_YEARS = 2  # Years of historical data to fetch

# Genie Space configuration
GENIE_SPACE_NAME = os.getenv("GENIE_SPACE_NAME", "SEC_Financial_Data_Explorer")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "")

# Knowledge Assistant configuration
KA_NAME = os.getenv("KA_NAME", "SEC_Financial_Analyst_KA")
KA_TILE_ID = os.getenv("KA_TILE_ID", "")
KA_ENDPOINT = os.getenv("KA_ENDPOINT", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Authentication (for SDK/API calls)

# COMMAND ----------

def get_databricks_auth():
    """
    Get Databricks hostname and token when running in a Databricks notebook.
    Returns (hostname, token) tuple.
    """
    try:
        # Get workspace hostname (DBR 18.0+)
        hostname = spark.conf.get("spark.databricks.workspaceUrl")
        if not hostname.startswith("https://"):
            hostname = f"https://{hostname}"
        
        # Get access token from notebook context
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        
        return hostname, token
    except Exception as e:
        print(f"Warning: Could not get Databricks auth: {e}")
        return None, None

def get_workspace_client():
    """
    Get an authenticated WorkspaceClient for SDK operations.
    """
    from databricks.sdk import WorkspaceClient
    
    hostname, token = get_databricks_auth()
    
    if hostname and token:
        return WorkspaceClient(host=hostname, token=token)
    else:
        # Fall back to default auth (works if env vars are set)
        return WorkspaceClient()

# COMMAND ----------

def get_table_name(table_key: str) -> str:
    """Get fully qualified table name."""
    return TABLES.get(table_key, f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}{table_key}")

def get_function_name(function_key: str) -> str:
    """Get fully qualified function name."""
    return FUNCTIONS.get(function_key, f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}{function_key}")

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("SEC Financial Analyst Multi-Agent Demo Configuration")
    print("=" * 60)
    print(f"\nCatalog: {UC_CATALOG}")
    print(f"Schema: {UC_SCHEMA}")
    print(f"Volume: {UC_VOLUME}")
    print(f"Table Prefix: {TABLE_PREFIX}")
    print(f"SQL Warehouse: {SQL_WAREHOUSE_ID}")
    print(f"\nVolume Path: {VOLUME_PATH}")
    print(f"SEC Docs Path: {SEC_DOCS_PATH}")
    print(f"\nTables:")
    for key, value in TABLES.items():
        print(f"  - {key}: {value}")
    print(f"\nFunctions:")
    for key, value in FUNCTIONS.items():
        print(f"  - {key}: {value}")
    print(f"\nCompanies:")
    for company in COMPANIES:
        print(f"  - {company['name']} ({company['ticker']})")
    
    # Show auth info
    hostname, token = get_databricks_auth()
    if hostname:
        print(f"\nWorkspace: {hostname}")
        print(f"Token: {'***' + token[-4:] if token else 'Not available'}")
    print("=" * 60)

# COMMAND ----------

# Print config when this module is run directly
if __name__ == "__main__" or "dbutils" in dir():
    print_config()
