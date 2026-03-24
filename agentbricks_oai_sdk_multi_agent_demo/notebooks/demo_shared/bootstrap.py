# Databricks notebook source
# MAGIC %md
# MAGIC # Bootstrap: Environment Loading + PAT-Only Auth
# MAGIC
# MAGIC Consolidates the common bootstrap pattern used across 11+ files:
# MAGIC - Resolve project root from any script's `__file__`
# MAGIC - Load central config from `_local/config/databricks.env`
# MAGIC - Fall back to project `.env` if central config is missing
# MAGIC - Strip OAuth/profile env vars to enforce PAT-only auth
# MAGIC
# MAGIC **Usage:**
# MAGIC ```python
# MAGIC from notebooks.demo_shared.bootstrap import bootstrap
# MAGIC _project_root, _central_config = bootstrap(__file__)
# MAGIC ```

# COMMAND ----------

from __future__ import annotations

import os
from pathlib import Path

import dotenv

# COMMAND ----------

# MAGIC %md
# MAGIC ## Path Resolution

# COMMAND ----------

def get_project_root(source_file: str) -> Path:
    """Resolve project root from any script's __file__.
    
    Walks up the directory tree until it finds pyproject.toml.
    """
    p = Path(source_file).resolve()
    while p.parent != p:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not find project root (pyproject.toml) from {source_file}")


def get_central_config(project_root: Path) -> Path:
    """Return path to the central config file."""
    return project_root.parent / "_local" / "config" / "databricks.env"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bootstrap Function

# COMMAND ----------

def sync_cli_profile(profile_name: str = "sec_financial_analyst") -> bool:
    """Sync Databricks CLI profile with current env vars.
    
    Creates or updates the CLI profile in ~/.databrickscfg to match
    the DATABRICKS_HOST and DATABRICKS_TOKEN from the environment.
    
    Args:
        profile_name: Name of the CLI profile to create/update.
        
    Returns:
        True if profile was updated, False otherwise.
    """
    import configparser
    
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    
    if not host or not token:
        return False
    
    cfg_path = Path.home() / ".databrickscfg"
    config = configparser.ConfigParser()
    
    if cfg_path.exists():
        config.read(cfg_path)
    
    if profile_name not in config:
        config[profile_name] = {}
    
    config[profile_name]["host"] = host
    config[profile_name]["token"] = token
    
    with open(cfg_path, "w") as f:
        config.write(f)
    
    # Set the profile env var so CLI uses it
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile_name
    
    return True


def bootstrap(source_file: str, override: bool = True, sync_profile: bool = False) -> tuple[Path, Path]:
    """Load central config, strip OAuth vars, return (project_root, central_config).
    
    Args:
        source_file: Pass `__file__` from the calling script.
        override: If True, env vars from config override existing vars.
        sync_profile: If True, sync CLI profile with env vars after loading.
        
    Returns:
        Tuple of (project_root, central_config) paths.
    """
    project_root = get_project_root(source_file)
    central_config = get_central_config(project_root)
    
    if central_config.exists():
        dotenv.load_dotenv(central_config, override=override)
    else:
        fallback = project_root / ".env"
        if fallback.exists():
            dotenv.load_dotenv(fallback, override=override)
    
    for k in ("DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET", "DATABRICKS_CONFIG_PROFILE"):
        os.environ.pop(k, None)
    
    if sync_profile:
        sync_cli_profile()
    
    return project_root, central_config
