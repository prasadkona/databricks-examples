# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup: Delete Databricks App
# MAGIC
# MAGIC Deletes the deployed Databricks App and associated bundle state.
# MAGIC
# MAGIC **Cleanup actions:**
# MAGIC 1. Delete the Databricks App via REST API
# MAGIC 2. Delete local bundle state (app/.databricks/)
# MAGIC 3. Delete remote bundle state (/Workspace/.../bundle/)
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run demo-cleanup after-genie   # calls this module
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `APP_NAME`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import requests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bundle State Cleanup

# COMMAND ----------

def delete_local_bundle_state() -> bool:
    """Delete local .databricks/ folder in app/ directory.
    
    Returns True if deleted, False if not found.
    """
    try:
        from notebooks.demo_shared.paths import get_app_dir
        app_dir = get_app_dir()
        databricks_dir = app_dir / ".databricks"
        if databricks_dir.exists():
            shutil.rmtree(databricks_dir)
            print(f"  Deleted local bundle state: {databricks_dir}")
            return True
        else:
            print(f"  No local bundle state found at {databricks_dir}")
            return False
    except Exception as e:
        print(f"  Warning: Could not delete local bundle state: {e}")
        return False


def delete_remote_bundle_state(host: str, token: str, bundle_name: str = "sec_financial_analyst_agent") -> bool:
    """Delete remote bundle state from workspace.
    
    Returns True if deleted, False otherwise.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    try:
        me_resp = requests.get(f"{host}/api/2.0/preview/scim/v2/Me", headers=headers, timeout=30)
        if me_resp.status_code != 200:
            print(f"  Warning: Could not get current user: {me_resp.status_code}")
            return False
        
        user = me_resp.json().get("userName", "")
        if not user:
            print("  Warning: Could not determine current user")
            return False
        
        bundle_path = f"/Workspace/Users/{user}/.bundle/{bundle_name}"
        
        delete_url = f"{host}/api/2.0/workspace/delete"
        payload = {"path": bundle_path, "recursive": True}
        
        r = requests.post(delete_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            print(f"  Deleted remote bundle state: {bundle_path}")
            return True
        elif r.status_code == 404 or "does not exist" in r.text.lower():
            print(f"  No remote bundle state found at {bundle_path}")
            return False
        else:
            print(f"  Warning: Could not delete remote bundle state: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f"  Warning: Could not delete remote bundle state: {e}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete App

# COMMAND ----------

def run() -> int:
    host = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    app_name = os.environ.get("APP_NAME", "").strip()

    if not host or not token:
        print("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN required", file=sys.stderr)
        return 1
    if not app_name:
        print("APP_NAME not set -- nothing to delete")
        return 0

    print(f"Workspace: {host}")
    print(f"Deleting App: {app_name}")

    url = f"{host}/api/2.0/apps/{app_name}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        r = requests.delete(url, headers=headers, timeout=60)
        if r.status_code == 200:
            print(f"Deleted App: {app_name}")
        elif r.status_code == 404:
            print(f"App '{app_name}' not found -- already deleted")
        else:
            print(f"ERROR: {r.status_code} {r.text}")
            return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("\nCleaning up bundle state...")
    delete_local_bundle_state()
    delete_remote_bundle_state(host, token)

    return 0
