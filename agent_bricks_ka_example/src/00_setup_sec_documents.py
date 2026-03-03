# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Download FY2024 Annual Reports and Upload to Volume
# MAGIC 
# MAGIC This script downloads FY2024 annual reports (PDFs) locally, then uploads them
# MAGIC to a Databricks Unity Catalog Volume for use with Knowledge Assistant.
# MAGIC 
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC cd agent_bricks_ka_example/src
# MAGIC python 00_setup_sec_documents.py
# MAGIC ```
# MAGIC 
# MAGIC ## Authentication
# MAGIC - **OAuth M2M** (default): Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
# MAGIC - **PAT** (fallback): Uses DATABRICKS_TOKEN if OAuth not configured
# MAGIC 
# MAGIC ## Companies
# MAGIC - **NVIDIA Corporation** (NVDA) - FY2024 Annual Report
# MAGIC - **Apple Inc.** (AAPL) - FY2024 10-K
# MAGIC - **Samsung Electronics** - FY2024 Consolidated Financial Statements
# MAGIC 
# MAGIC ## Output
# MAGIC - Local: `_local/datasets/sec_2024/*.pdf`
# MAGIC - Volume: `/Volumes/{catalog}/{schema}/{volume}/sec_2024/*.pdf`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import sys
import requests
import time
import io
from pathlib import Path

# Import common config
from config import load_env_file, setup_databricks_auth, grant_volume_permissions

# Load configuration and set up authentication
config = load_env_file()
config, auth_type = setup_databricks_auth(config)

DATABRICKS_HOST = config.get('DATABRICKS_HOST')

# Unity Catalog configuration from env
UC_CATALOG = config.get('UC_CATALOG', 'main')
UC_SCHEMA = config.get('UC_SCHEMA', 'default')
UC_VOLUME = config.get('UC_VOLUME', 'documents')
UC_VOLUME_PATH = config.get('UC_VOLUME_PATH', f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}")

# Dataset folder name
DATASET_FOLDER = "sec_2024"

# Local output directory
LOCAL_BASE_DIR = Path("../../_local/datasets")
LOCAL_OUTPUT_DIR = LOCAL_BASE_DIR / DATASET_FOLDER

# Volume output path
VOLUME_OUTPUT_PATH = f"{UC_VOLUME_PATH.rstrip('/')}/{DATASET_FOLDER}"

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Auth: {auth_type.upper()}")
print(f"Local output: {LOCAL_OUTPUT_DIR}")
print(f"Volume output: {VOLUME_OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Workspace Client

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant Volume Permissions (OAuth only)

# COMMAND ----------

if auth_type == "oauth":
    grant_volume_permissions(w, config, UC_VOLUME_PATH)
else:
    print("Using PAT authentication - volume grants not needed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annual Report URLs

# COMMAND ----------

ANNUAL_REPORTS = {
    "nvidia": {
        "name": "NVIDIA Corporation",
        "ticker": "NVDA",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2024/ar/NVIDIA-2024-Annual-Report.pdf",
        "filename": "NVDA_FY2024_Annual_Report.pdf",
        "description": "NVIDIA FY2024 Annual Report (fiscal year ended Jan 2024)"
    },
    "apple": {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf",
        "filename": "AAPL_FY2024_10K.pdf",
        "description": "Apple FY2024 10-K (fiscal year ended Sep 2024)"
    },
    "samsung": {
        "name": "Samsung Electronics",
        "ticker": "SAMSUNG",
        "url": "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_con_quarter04_all.pdf",
        "filename": "SAMSUNG_FY2024_Annual_Report.pdf",
        "description": "Samsung FY2024 Consolidated Financial Statements"
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Functions

# COMMAND ----------

def download_pdf(url: str, local_path: Path, description: str = "") -> bool:
    """Download a PDF from URL to local path."""
    print(f"  Downloading: {description or url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {local_path} ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def ensure_volume_folder_exists(volume_path: str) -> bool:
    """Ensure the volume folder exists by creating a placeholder if needed."""
    try:
        list(w.files.list_directory_contents(volume_path))
        print(f"  Volume folder exists: {volume_path}")
        return True
    except Exception as e:
        if "NOT_FOUND" in str(e) or "404" in str(e):
            print(f"  Creating volume folder: {volume_path}")
            try:
                placeholder_path = f"{volume_path.rstrip('/')}/.placeholder"
                w.files.upload(placeholder_path, io.BytesIO(b""), overwrite=True)
                print(f"  Created folder: {volume_path}")
                return True
            except Exception as create_error:
                print(f"  Error creating folder: {create_error}")
                return False
        else:
            print(f"  Error checking folder: {e}")
            return False


def upload_to_volume(local_path: Path, volume_path: str, filename: str) -> bool:
    """Upload a local file to Databricks volume."""
    target_path = f"{volume_path.rstrip('/')}/{filename}"
    print(f"  Uploading to: {target_path}")
    
    try:
        with open(local_path, 'rb') as f:
            content = f.read()
        
        w.files.upload(target_path, io.BytesIO(content), overwrite=True)
        size_mb = len(content) / (1024 * 1024)
        print(f"  Uploaded: {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  Error uploading: {e}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Download PDFs Locally

# COMMAND ----------

print("=" * 60)
print("STEP 1: Download PDFs Locally")
print("=" * 60)
print(f"\nLocal directory: {LOCAL_OUTPUT_DIR}")

LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

downloaded_files = []

for company_key, company_info in ANNUAL_REPORTS.items():
    print(f"\n{company_info['name']} ({company_info['ticker']})")
    print("-" * 40)
    
    local_path = LOCAL_OUTPUT_DIR / company_info['filename']
    
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {local_path} ({size_mb:.2f} MB)")
        downloaded_files.append(local_path)
    else:
        success = download_pdf(
            company_info['url'],
            local_path,
            company_info['description']
        )
        if success:
            downloaded_files.append(local_path)
    
    time.sleep(0.5)

print(f"\n{'=' * 60}")
print(f"Downloaded: {len(downloaded_files)} of {len(ANNUAL_REPORTS)} files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Upload to Databricks Volume

# COMMAND ----------

print("=" * 60)
print("STEP 2: Upload to Databricks Volume")
print("=" * 60)
print(f"\nVolume path: {VOLUME_OUTPUT_PATH}")

print(f"\nChecking volume folder...")
ensure_volume_folder_exists(VOLUME_OUTPUT_PATH)

uploaded_files = []

for company_key, company_info in ANNUAL_REPORTS.items():
    print(f"\n{company_info['name']} ({company_info['ticker']})")
    print("-" * 40)
    
    local_path = LOCAL_OUTPUT_DIR / company_info['filename']
    
    if not local_path.exists():
        print(f"  Local file not found: {local_path}")
        continue
    
    success = upload_to_volume(
        local_path,
        VOLUME_OUTPUT_PATH,
        company_info['filename']
    )
    
    if success:
        uploaded_files.append(f"{VOLUME_OUTPUT_PATH}/{company_info['filename']}")

print(f"\n{'=' * 60}")
print(f"Uploaded: {len(uploaded_files)} of {len(ANNUAL_REPORTS)} files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Upload

# COMMAND ----------

print("=" * 60)
print("STEP 3: Verify Upload")
print("=" * 60)
print(f"\nFiles in {VOLUME_OUTPUT_PATH}:")
print("-" * 60)

try:
    files = list(w.files.list_directory_contents(VOLUME_OUTPUT_PATH))
    total_size = 0
    file_count = 0
    for f in files:
        if f.name.startswith('.'):
            continue
        if f.file_size:
            total_size += f.file_size
            file_count += 1
            size_mb = f.file_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {f.name}/")
    
    print(f"\nTotal: {file_count} files, {total_size / (1024*1024):.2f} MB")
except Exception as e:
    print(f"Error listing files: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"\n{'=' * 60}")
print("FY2024 Annual Reports - Setup Complete")
print(f"{'=' * 60}")
print(f"\nAuth type: {auth_type.upper()}")
print(f"Local path: {LOCAL_OUTPUT_DIR.absolute()}")
print(f"Volume path: {VOLUME_OUTPUT_PATH}")
print(f"\nFiles available for Knowledge Assistant:")
for company_info in ANNUAL_REPORTS.values():
    print(f"  - {company_info['filename']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The annual reports are now in your Databricks volume. You can now:
# MAGIC 
# MAGIC 1. **Create a Knowledge Assistant** using `01_ka_using_rest_api.py` or `02_ka_using_agent_bricks_manager.py`
# MAGIC 2. Point the KA to the volume path shown above
# MAGIC 3. Ask questions about NVIDIA, Apple, and Samsung financials
# MAGIC 
# MAGIC Example questions:
# MAGIC - "What was NVIDIA's revenue growth in 2024?"
# MAGIC - "Compare Apple and NVIDIA's business segments"
# MAGIC - "What are Samsung's key growth areas?"
