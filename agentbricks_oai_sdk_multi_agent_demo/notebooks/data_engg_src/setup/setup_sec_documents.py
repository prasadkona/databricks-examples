# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup SEC Documents
# MAGIC 
# MAGIC **SEC Financial Analyst Multi-Agent Demo - Data Setup**
# MAGIC 
# MAGIC This notebook downloads FY2024 SEC filings (10-K annual reports) and uploads them
# MAGIC to a Unity Catalog Volume. These documents serve two purposes:
# MAGIC 
# MAGIC 1. **Knowledge Assistant (KA)**: The PDFs are indexed for document Q&A
# MAGIC 2. **AI Extraction (SDP Pipeline)**: `ai_parse_document()` and `ai_extract()` extract structured data
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Unity Catalog volume must exist: `/Volumes/your_catalog/your_schema/ka_demo/`
# MAGIC - Cluster with internet access to download PDFs
# MAGIC 
# MAGIC ## Companies Covered
# MAGIC | Company | Ticker | Fiscal Year End | Document Type |
# MAGIC |---------|--------|-----------------|---------------|
# MAGIC | NVIDIA Corporation | NVDA | January 2024 | Annual Report |
# MAGIC | Apple Inc. | AAPL | September 2024 | 10-K Filing |
# MAGIC | Samsung Electronics | 005930.KS | December 2024 | Consolidated Financials |
# MAGIC 
# MAGIC ## Output Location
# MAGIC ```
# MAGIC /Volumes/your_catalog/your_schema/ka_demo/sec_2024/
# MAGIC ├── NVDA_FY2024_Annual_Report.pdf
# MAGIC ├── AAPL_FY2024_10K.pdf
# MAGIC └── SAMSUNG_FY2024_Annual_Report.pdf
# MAGIC ```
# MAGIC 
# MAGIC ## Filename Convention
# MAGIC Filenames follow the pattern `{TICKER}_FY{YEAR}_{TYPE}.pdf`.
# MAGIC The SDP pipeline reads any PDF in the volume and uses `ai_parse_document` + `ai_extract`
# MAGIC to auto-discover company names and tickers — no hardcoded filename mapping needed.
# MAGIC Knowledge Assistant also indexes all PDFs in the volume folder automatically.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Configuration

# COMMAND ----------

import os
import io
import time
import requests
from datetime import datetime

# Verify SEC_DOCS_PATH from config matches our expected output
print("=" * 60)
print("SEC Financial Analyst - Document Setup")
print("=" * 60)
print(f"\nConfiguration from config.py:")
print(f"  UC_CATALOG: {UC_CATALOG}")
print(f"  UC_SCHEMA:  {UC_SCHEMA}")
print(f"  UC_VOLUME:  {UC_VOLUME}")
print(f"  VOLUME_PATH: {VOLUME_PATH}")
print(f"  SEC_DOCS_PATH: {SEC_DOCS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annual Report Sources
# MAGIC 
# MAGIC These URLs point to official company investor relations PDFs.
# MAGIC 
# MAGIC Any PDF dropped into this volume folder will be processed by the SDP pipeline.
# MAGIC The pipeline uses `ai_parse_document` + `ai_extract` to discover company names
# MAGIC and stock tickers directly from document content — no filename mapping required.

# COMMAND ----------

ANNUAL_REPORTS = {
    "nvidia": {
        "name": "NVIDIA Corporation",
        "ticker": "NVDA",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2024/ar/NVIDIA-2024-Annual-Report.pdf",
        "filename": "NVDA_FY2024_Annual_Report.pdf",
        "description": "NVIDIA FY2024 Annual Report (fiscal year ended Jan 28, 2024)",
        "fiscal_year_end": "2024-01-28"
    },
    "apple": {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf",
        "filename": "AAPL_FY2024_10K.pdf",
        "description": "Apple FY2024 10-K (fiscal year ended Sep 28, 2024)",
        "fiscal_year_end": "2024-09-28"
    },
    "samsung": {
        "name": "Samsung Electronics",
        "ticker": "005930.KS",
        "url": "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_con_quarter04_all.pdf",
        "filename": "SAMSUNG_FY2024_Annual_Report.pdf",
        "description": "Samsung FY2024 Consolidated Financial Statements (calendar year 2024)",
        "fiscal_year_end": "2024-12-31"
    }
}

print(f"\nDocuments to download ({len(ANNUAL_REPORTS)} files):")
print("-" * 60)
for key, info in ANNUAL_REPORTS.items():
    print(f"  {info['ticker']:12} → {info['filename']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def download_pdf_to_bytes(url: str, description: str = "") -> bytes:
    """
    Download a PDF from URL and return as bytes.
    
    Uses a browser-like User-Agent to avoid being blocked by CDNs.
    Includes retry logic for transient failures.
    """
    print(f"  Downloading: {description or url[:60]}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=120, stream=True)
            response.raise_for_status()
            
            content = response.content
            size_mb = len(content) / (1024 * 1024)
            print(f"  ✅ Downloaded: {size_mb:.2f} MB")
            return content
            
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.HTTPError as e:
            print(f"  ❌ HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    print(f"  ❌ Failed after {max_retries} attempts")
    return None


def ensure_volume_folder_exists(volume_path: str) -> bool:
    """
    Ensure the volume folder exists.
    
    Unity Catalog volumes don't have traditional folder creation - 
    folders are created implicitly when files are written.
    This function just verifies the path is accessible.
    """
    try:
        files = dbutils.fs.ls(volume_path)
        print(f"  ✅ Folder exists: {volume_path} ({len(files)} items)")
        return True
    except Exception as e:
        if "FileNotFoundException" in str(e) or "does not exist" in str(e).lower():
            # Folder doesn't exist - will be created when first file is written
            print(f"  📁 Folder will be created: {volume_path}")
            return True
        else:
            print(f"  ❌ Error accessing folder: {e}")
            return False


def upload_to_volume(content: bytes, volume_path: str, filename: str) -> bool:
    """
    Upload binary content to a Unity Catalog Volume.
    
    Uses Python's native file operations which work with /Volumes/ paths
    on Databricks Runtime 13.3+.
    """
    target_path = f"{volume_path}/{filename}"
    print(f"  Uploading to: {target_path}")
    
    try:
        # /Volumes/ paths work like regular filesystem paths in DBR 13.3+
        with open(target_path, 'wb') as f:
            f.write(content)
        
        size_mb = len(content) / (1024 * 1024)
        print(f"  ✅ Uploaded: {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return False


def file_exists_in_volume(volume_path: str, filename: str) -> tuple:
    """
    Check if a file already exists in the volume.
    
    Returns (exists: bool, size_mb: float or None)
    """
    target_path = f"{volume_path}/{filename}"
    try:
        file_info = dbutils.fs.ls(target_path)
        if file_info:
            size_mb = file_info[0].size / (1024 * 1024)
            return True, size_mb
    except:
        pass
    return False, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Volume Access

# COMMAND ----------

print("=" * 60)
print("STEP 1: Verify Volume Access")
print("=" * 60)

# Check that the main volume exists
print(f"\nChecking volume: {VOLUME_PATH}")
try:
    files = dbutils.fs.ls(VOLUME_PATH)
    print(f"✅ Volume accessible with {len(files)} existing items")
except Exception as e:
    print(f"❌ Volume not accessible: {e}")
    print(f"\n⚠️ Please create the volume first:")
    print(f"   CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME};")
    dbutils.notebook.exit("ERROR: Volume not found. See instructions above.")

# Check/create the SEC docs subfolder
print(f"\nChecking SEC docs folder: {SEC_DOCS_PATH}")
ensure_volume_folder_exists(SEC_DOCS_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download and Upload Documents
# MAGIC 
# MAGIC For each company:
# MAGIC 1. Check if file already exists (skip if so)
# MAGIC 2. Download PDF from investor relations URL
# MAGIC 3. Upload to Unity Catalog Volume

# COMMAND ----------

print("=" * 60)
print("STEP 2: Download and Upload SEC Documents")
print("=" * 60)

uploaded_files = []
skipped_files = []
failed_files = []

for company_key, company_info in ANNUAL_REPORTS.items():
    print(f"\n{company_info['name']} ({company_info['ticker']})")
    print("-" * 50)
    
    filename = company_info['filename']
    
    # Check if file already exists
    exists, size_mb = file_exists_in_volume(SEC_DOCS_PATH, filename)
    if exists:
        print(f"  ✅ Already exists: {filename} ({size_mb:.2f} MB)")
        skipped_files.append(filename)
        continue
    
    # Download the PDF
    content = download_pdf_to_bytes(
        company_info['url'],
        company_info['description']
    )
    
    if content:
        # Upload to volume
        success = upload_to_volume(content, SEC_DOCS_PATH, filename)
        
        if success:
            uploaded_files.append(filename)
        else:
            failed_files.append(filename)
    else:
        failed_files.append(filename)
    
    # Small delay between downloads to be respectful to servers
    time.sleep(1)

# Summary
print(f"\n{'=' * 60}")
print(f"Download Summary:")
print(f"  ✅ Uploaded:  {len(uploaded_files)} files")
print(f"  ⏭️ Skipped:   {len(skipped_files)} files (already existed)")
print(f"  ❌ Failed:    {len(failed_files)} files")

if failed_files:
    print(f"\nFailed downloads: {failed_files}")
    print("Try re-running this notebook or download manually.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify All Documents Present

# COMMAND ----------

print("=" * 60)
print("STEP 3: Verify Documents in Volume")
print("=" * 60)
print(f"\nLocation: {SEC_DOCS_PATH}")
print("-" * 60)

try:
    files = dbutils.fs.ls(SEC_DOCS_PATH)
    
    # Filter out hidden files
    pdf_files = [f for f in files if f.name.endswith('.pdf')]
    
    total_size = 0
    for f in pdf_files:
        size_mb = f.size / (1024 * 1024)
        total_size += f.size
        print(f"  📄 {f.name:40} ({size_mb:.2f} MB)")
    
    print(f"\nTotal: {len(pdf_files)} PDF files, {total_size / (1024*1024):.2f} MB")
    
    # Verify all expected files are present
    expected_files = [info['filename'] for info in ANNUAL_REPORTS.values()]
    present_files = [f.name for f in pdf_files]
    missing = set(expected_files) - set(present_files)
    
    if missing:
        print(f"\n⚠️ Missing files: {missing}")
        print("   Re-run this notebook or check URLs above.")
    else:
        print(f"\n✅ All {len(expected_files)} expected documents are present!")
        
except Exception as e:
    print(f"❌ Error listing files: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation: PDF Files Present
# MAGIC 
# MAGIC Verify that all expected PDF files are present in the volume.
# MAGIC The SDP pipeline uses `ai_parse_document` + `ai_extract` to discover companies
# MAGIC automatically from the file contents — no hardcoded filename patterns required.

# COMMAND ----------

print("=" * 60)
print("PDF Files Validation")
print("=" * 60)

try:
    files = dbutils.fs.ls(SEC_DOCS_PATH)
    pdf_names = [f.name for f in files if f.name.endswith('.pdf')]
    
    print(f"\nPDF files found in {SEC_DOCS_PATH}:")
    print("-" * 60)
    for name in sorted(pdf_names):
        print(f"  📄 {name}")

    expected_files = [info['filename'] for info in ANNUAL_REPORTS.values()]
    missing = set(expected_files) - set(pdf_names)
    
    if missing:
        print(f"\n⚠️ Missing: {missing}")
        print("   Re-run Step 2 above or check URLs in ANNUAL_REPORTS.")
    else:
        print(f"\n✅ All {len(expected_files)} expected documents present.")
        print("   The SDP pipeline will auto-discover companies via ai_extract.")
        
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

print(f"\n{'=' * 60}")
print("SEC Documents Setup Complete!")
print(f"{'=' * 60}")

print(f"""
📁 Documents Location:
   {SEC_DOCS_PATH}

📄 Files Ready for Processing:""")
for info in ANNUAL_REPORTS.values():
    print(f"   • {info['filename']}")

print(f"""
🔗 These documents will be used by:
   • Knowledge Assistant (KA) - indexed for document Q&A / RAG
   • SDP Pipeline - ai_parse_document + ai_extract auto-discover companies & tickers
   • Genie Space - queries the gold tables produced by the pipeline

📋 Next Steps (run locally):
   uv run run-sequence --ka          # Build Knowledge Assistant only
   uv run run-sequence --data-eng    # Run SDP pipeline + views + functions + Genie
   uv run run-sequence --all         # Full lifecycle: KA + data + agent

   Or step by step:
   uv run run-ka-sequence            # KA: create → sync sources → test
   uv run deploy-sdp-pipeline        # SDP: SEC PDFs → ai_extract → gold tables + stock history
   uv run run-workspace-notebooks 05 06 07  # Views + UC functions + Genie Space
""")
