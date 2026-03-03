# Databricks notebook source
# MAGIC %md
# MAGIC # Add Examples to Knowledge Assistant
# MAGIC 
# MAGIC This script adds sample questions with guidelines to a Knowledge Assistant.
# MAGIC Examples help improve KA response quality by providing guidance for similar questions.
# MAGIC 
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC cd agent_bricks_ka_example/src
# MAGIC 
# MAGIC # Add examples to KA 01
# MAGIC python 05_add_ka_examples.py 1
# MAGIC 
# MAGIC # Add examples to KA 02
# MAGIC python 05_add_ka_examples.py 2
# MAGIC 
# MAGIC # Add to KA by name
# MAGIC python 05_add_ka_examples.py SEC_Financial_Analyst
# MAGIC ```
# MAGIC 
# MAGIC ## Authentication
# MAGIC - **OAuth M2M** (default): Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
# MAGIC - **PAT** (fallback): Uses DATABRICKS_TOKEN if OAuth not configured

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import sys
import requests

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from config import load_env_file, setup_databricks_auth

# Load configuration and set up authentication
config = load_env_file()
config, auth_type = setup_databricks_auth(config)

DATABRICKS_HOST = config.get('DATABRICKS_HOST')

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Auth: {auth_type.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## KA Selection

# COMMAND ----------

# KA configurations from env
KA_CONFIG = {
    "1": {
        "name": config.get("KA_NAME_01", "SEC_Financial_Analyst"),
        "tile_id": config.get("KA_TILE_ID_01", ""),
    },
    "2": {
        "name": config.get("KA_NAME_02", "SEC_Financial_Analyst_02"),
        "tile_id": config.get("KA_TILE_ID_02", ""),
    },
}

def get_ka_config(arg: str) -> dict:
    """Get KA config from argument."""
    if arg in KA_CONFIG:
        return KA_CONFIG[arg]
    for key, ka in KA_CONFIG.items():
        if ka['name'] == arg:
            return ka
    return {"name": arg, "tile_id": ""}

# Get argument or default to "1"
ka_arg = sys.argv[1] if len(sys.argv) > 1 else "1"

ka_config = get_ka_config(ka_arg)
KA_NAME = ka_config['name']
KA_TILE_ID = ka_config['tile_id']

if not KA_TILE_ID:
    print(f"Error: No tile_id found for KA '{KA_NAME}'")
    print("\nConfigured KAs:")
    for key, ka in KA_CONFIG.items():
        print(f"  [{key}] {ka['name']} (tile_id: {ka['tile_id'] or 'NOT SET'})")
    sys.exit(1)

print(f"\nKA Name: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Sample Questions

# COMMAND ----------

# Sample questions with guidelines for SEC Financial Analysis
EXAMPLES = [
    {
        "question": "What was NVIDIA's total revenue in FY2024?",
        "guidelines": "Look for total revenue figures in the financial highlights or consolidated statements. Include the exact dollar amount and year-over-year growth percentage if available."
    },
    {
        "question": "What are Apple's main business segments?",
        "guidelines": "Identify the reportable segments from the business description or segment information section. List each segment with a brief description of what products/services it includes."
    },
    {
        "question": "Compare the revenue growth of NVIDIA, Apple, and Samsung",
        "guidelines": "For each company, find the most recent fiscal year revenue and the prior year revenue. Calculate the growth rate. Present as a comparison table if possible."
    },
    {
        "question": "What are the key risk factors mentioned in NVIDIA's annual report?",
        "guidelines": "Look for the Risk Factors section. Summarize the top 3-5 most significant risks. Include specific risks related to AI/GPU market, supply chain, and competition."
    },
    {
        "question": "What is Samsung's semiconductor business performance?",
        "guidelines": "Focus on the semiconductor or chip division. Include revenue, operating profit, and any commentary on memory chip market conditions or AI chip demand."
    },
    {
        "question": "How much did Apple spend on R&D in FY2024?",
        "guidelines": "Find the R&D expense in the income statement or notes. Provide the absolute amount and as a percentage of revenue. Compare to prior year if available."
    },
    {
        "question": "What are the main products driving NVIDIA's growth?",
        "guidelines": "Identify the key product categories (Data Center, Gaming, etc.). Focus on which segments showed the strongest growth and why, particularly AI-related products."
    },
    {
        "question": "What is Apple's geographic revenue breakdown?",
        "guidelines": "Look for geographic or regional segment information. List revenue by region (Americas, Europe, Greater China, Japan, Rest of Asia Pacific) with percentages."
    },
]

print(f"Defined {len(EXAMPLES)} sample questions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_headers():
    """Get authenticated headers for REST API calls."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers

def api_post(path: str, body: dict) -> dict:
    """Make a POST request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.post(url, headers=get_headers(), json=body, timeout=60)
    if response.status_code >= 400:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()
    return response.json() if response.text else {}

def add_example(tile_id: str, question: str, guidelines: str = None) -> dict:
    """Add an example question to a KA."""
    payload = {"question": question}
    if guidelines:
        payload["guidelines"] = guidelines
    return api_post(f"/api/2.0/knowledge-assistants/{tile_id}/examples", payload)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Examples to KA

# COMMAND ----------

print("=" * 60)
print("ADDING EXAMPLES TO KA")
print("=" * 60)
print(f"\nKA: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")
print(f"\nAdding {len(EXAMPLES)} examples...")

added = []
failed = []

for i, example in enumerate(EXAMPLES, 1):
    question = example['question']
    guidelines = example.get('guidelines', '')
    
    print(f"\n[{i}/{len(EXAMPLES)}] Adding: {question[:60]}...")
    
    try:
        result = add_example(KA_TILE_ID, question, guidelines)
        example_id = result.get('example', {}).get('example_id', 'unknown')
        print(f"  Added (ID: {example_id})")
        added.append(question)
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg.lower() or 'duplicate' in error_msg.lower():
            print(f"  Skipped (already exists)")
            added.append(question)
        else:
            print(f"  Failed: {e}")
            failed.append(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nKA Name: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")
print(f"\nTotal examples: {len(EXAMPLES)}")
print(f"Added/Existing: {len(added)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed questions:")
    for q in failed:
        print(f"  - {q[:70]}...")

print("\n" + "=" * 60)
print("Examples improve KA quality by providing guidance for similar questions.")
print("=" * 60)
