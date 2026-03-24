# Databricks notebook source
# MAGIC %md
# MAGIC # API Client: REST API Helpers
# MAGIC
# MAGIC Consolidates REST API helpers from `cleanup_tables.py` and `step04_local_deploy_sdp_pipeline.py`.
# MAGIC Provides generic Databricks REST API calls with Bearer auth.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```python
# MAGIC from notebooks.demo_shared.api_client import api_request, run_sql
# MAGIC result = api_request("GET", host, token, "/api/2.0/pipelines")
# MAGIC run_sql(host, token, warehouse_id, catalog, schema, "SELECT 1")
# MAGIC ```

# COMMAND ----------

from __future__ import annotations

import time
from typing import Any

import requests

# COMMAND ----------

# MAGIC %md
# MAGIC ## REST API Request

# COMMAND ----------

def api_request(
    method: str,
    host: str,
    token: str,
    endpoint: str,
    json_data: dict | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Generic Databricks REST API call with Bearer auth.
    
    Args:
        method: HTTP method (GET, POST, DELETE, PATCH, PUT).
        host: Databricks workspace host (e.g., https://xxx.cloud.databricks.com).
        token: PAT token for authentication.
        endpoint: API endpoint (e.g., /api/2.0/pipelines).
        json_data: Optional JSON body for POST/PATCH/PUT, or query params for GET.
        timeout: Request timeout in seconds.
        
    Returns:
        Parsed JSON response as dict, or empty dict if no content.
        
    Raises:
        RuntimeError: If the API returns a 4xx/5xx status code.
    """
    url = f"{host.rstrip('/')}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    method = method.upper()
    if method == "POST":
        r = requests.post(url, headers=headers, json=json_data, timeout=timeout)
    elif method == "GET":
        r = requests.get(url, headers=headers, params=json_data, timeout=timeout)
    elif method == "DELETE":
        r = requests.delete(url, headers=headers, timeout=timeout)
    elif method == "PATCH":
        r = requests.patch(url, headers=headers, json=json_data, timeout=timeout)
    elif method == "PUT":
        r = requests.put(url, headers=headers, json=json_data, timeout=timeout)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    if r.status_code >= 400:
        raise RuntimeError(f"API {method} {endpoint}: {r.status_code} {r.text}")
    
    return r.json() if r.text else {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Statement Execution

# COMMAND ----------

def run_sql(
    host: str,
    token: str,
    warehouse_id: str,
    catalog: str,
    schema: str,
    statement: str,
    wait_timeout: str = "30s",
    poll_timeout: int = 60,
) -> dict[str, Any]:
    """Execute SQL via /api/2.0/sql/statements and wait for completion.
    
    Args:
        host: Databricks workspace host.
        token: PAT token.
        warehouse_id: SQL warehouse ID.
        catalog: Default catalog for the statement.
        schema: Default schema for the statement.
        statement: SQL statement to execute.
        wait_timeout: How long the API should wait before returning PENDING.
        poll_timeout: Max seconds to poll for completion.
        
    Returns:
        Final statement result dict.
        
    Raises:
        RuntimeError: If the statement fails or times out.
    """
    payload = {
        "warehouse_id": warehouse_id,
        "catalog": catalog,
        "schema": schema,
        "statement": statement,
        "wait_timeout": wait_timeout,
    }
    result = api_request("POST", host, token, "/api/2.0/sql/statements", payload)
    
    stmt_id = result.get("statement_id")
    status = (result.get("status") or {}).get("state", "")
    
    if stmt_id and status in ("PENDING", "RUNNING"):
        for _ in range(poll_timeout):
            time.sleep(1)
            r = api_request("GET", host, token, f"/api/2.0/sql/statements/{stmt_id}")
            status = (r.get("status") or {}).get("state", "")
            if status in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"):
                result = r
                break
        
        if status not in ("SUCCEEDED", "CLOSED"):
            raise RuntimeError(f"Statement ended with state: {status}")
    
    return result
