# Databricks notebook source
# MAGIC %md
# MAGIC # Demo Shared Utilities
# MAGIC
# MAGIC Common utilities for the SEC Financial Analyst demo project.
# MAGIC
# MAGIC - `bootstrap` - Environment loading and PAT-only auth setup
# MAGIC - `config_manager` - Central config file updates
# MAGIC - `api_client` - REST API helpers for Databricks APIs
# MAGIC - `subprocess_runner` - Subprocess execution with timing and banners
# MAGIC - `paths` - Centralized path resolution helpers

# COMMAND ----------

from .bootstrap import bootstrap, get_project_root, get_central_config, sync_cli_profile
from .config_manager import update_central_config
from .api_client import api_request, run_sql
from .subprocess_runner import run_step
from .paths import get_app_dir, get_notebooks_dir, get_bundle_workspace_path

__all__ = [
    "bootstrap",
    "get_project_root",
    "get_central_config",
    "sync_cli_profile",
    "update_central_config",
    "api_request",
    "run_sql",
    "run_step",
    "get_app_dir",
    "get_notebooks_dir",
    "get_bundle_workspace_path",
]
