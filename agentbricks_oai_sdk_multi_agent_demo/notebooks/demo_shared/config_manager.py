# Databricks notebook source
# MAGIC %md
# MAGIC # Config Manager: Central Config File Updates
# MAGIC
# MAGIC Consolidates the `_update_central_config` pattern used in multiple files.
# MAGIC Updates or appends key=value pairs in the central `.env` config file.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```python
# MAGIC from notebooks.demo_shared.config_manager import update_central_config
# MAGIC update_central_config(config_path, "GENIE_SPACE_ID", new_id)
# MAGIC ```

# COMMAND ----------

from __future__ import annotations

import re
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Central Config

# COMMAND ----------

def update_central_config(config_path: Path, key: str, value: str) -> bool:
    """Update or append a key=value in the central config file.
    
    Args:
        config_path: Path to the .env config file.
        key: Environment variable name.
        value: Value to set.
        
    Returns:
        True if the update was successful, False if the file doesn't exist.
    """
    if not config_path.exists():
        print(f"  WARNING: Config file not found: {config_path}")
        return False
    
    text = config_path.read_text(encoding="utf-8")
    pattern = rf'(?m)^({re.escape(key)}=).*$'
    new_text, n = re.subn(pattern, rf'\g<1>{value}', text)
    
    if n == 0:
        new_text = text.rstrip("\n") + f"\n{key}={value}\n"
    
    config_path.write_text(new_text, encoding="utf-8")
    return True
