# Databricks notebook source
# MAGIC %md
# MAGIC # Knowledge Assistant using REST API
# MAGIC 
# MAGIC This script demonstrates how to create a Knowledge Assistant (KA) Agent Brick
# MAGIC using the Databricks REST API directly. **Run locally** using credentials from
# MAGIC the `_local/{workspace}.env` file.
# MAGIC 
# MAGIC ## Authentication
# MAGIC - **OAuth M2M** (default): Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
# MAGIC - **PAT** (fallback): Uses DATABRICKS_TOKEN if OAuth not configured
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Serverless compute enabled
# MAGIC - Access to Foundation Models in `system.ai` schema
# MAGIC - Files in a Unity Catalog Volume (PDF, TXT, MD, DOC/DOCX, PPT/PPTX)
# MAGIC - Serverless budget policy with nonzero budget
# MAGIC 
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC cd agent_bricks_ka_example/src
# MAGIC python 01_ka_using_rest_api.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC This script uses OAuth M2M (Service Principal) authentication by default.
# MAGIC Falls back to PAT if OAuth credentials are not configured.
# MAGIC 
# MAGIC **Setup:**
# MAGIC 1. Copy `.env.template` to `_local/{workspace}.env`
# MAGIC 2. Fill in your Databricks credentials

# COMMAND ----------

import os
import json
import time
import requests
from pathlib import Path

# Import common config
from config import load_env_file, setup_databricks_auth

# Load configuration and set up authentication (OAuth by default, PAT fallback)
config = load_env_file()
config, auth_type = setup_databricks_auth(config)

DATABRICKS_HOST = config.get('DATABRICKS_HOST')

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Auth: {auth_type.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_headers():
    """Get authenticated headers for REST API calls using OAuth M2M."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers

def api_get(path: str, params: dict = None) -> dict:
    """Make a GET request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.get(url, headers=get_headers(), params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()

def api_post(path: str, body: dict) -> dict:
    """Make a POST request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.post(url, headers=get_headers(), json=body, timeout=300)
    if response.status_code >= 400:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()
    return response.json()

def api_patch(path: str, body: dict) -> dict:
    """Make a PATCH request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.patch(url, headers=get_headers(), json=body, timeout=30)
    response.raise_for_status()
    return response.json()

def api_delete(path: str) -> dict:
    """Make a DELETE request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.delete(url, headers=get_headers(), timeout=30)
    if response.status_code >= 400 and response.status_code != 404:
        response.raise_for_status()
    return response.json() if response.text else {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Volume with Documents Exists
# MAGIC 
# MAGIC Before creating a KA, ensure you have documents in a Unity Catalog Volume.

# COMMAND ----------

# Configure your volume path containing documents
# Supported file types: PDF, TXT, MD, DOC/DOCX, PPT/PPTX
CATALOG = config.get("UC_CATALOG", "prasad_kona_isv")
SCHEMA = config.get("UC_SCHEMA", "isv_demo")
VOLUME = config.get("UC_VOLUME", "ka_demo")
DATASET_FOLDER = "sec_2024"
VOLUME_PATH = config.get("UC_VOLUME_PATH", f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")
VOLUME_PATH = f"{VOLUME_PATH.rstrip('/')}/{DATASET_FOLDER}"
# Volume contains: FY2024 Annual Reports - NVIDIA, Apple, Samsung, and others

print(f"Volume path: {VOLUME_PATH}")

# List files in the volume (if it exists)
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    files = list(w.files.list_directory_contents(VOLUME_PATH))
    print(f"Found {len(files)} files in {VOLUME_PATH}:")
    for f in files[:10]:
        print(f"  - {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
except Exception as e:
    print(f"Note: Could not list volume contents: {e}")
    print(f"Make sure the volume exists at: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Knowledge Assistant via REST API

# COMMAND ----------

def create_knowledge_assistant(
    name: str,
    volume_path: str,
    description: str = None,
    instructions: str = None
) -> dict:
    """Create a Knowledge Assistant with documents from a volume.
    
    Args:
        name: Name for the KA (alphanumeric and hyphens only)
        volume_path: Path to Unity Catalog volume with documents
        description: Optional description of the KA
        instructions: Optional instructions for how the KA should respond
    
    Returns:
        API response with tile_id and other metadata
    """
    # Sanitize name (only alphanumeric, hyphens, underscores)
    import re
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.replace(' ', '_'))
    sanitized_name = re.sub(r'[_-]{2,}', '_', sanitized_name).strip('_-')
    
    # Build knowledge source
    source_name = volume_path.rstrip('/').split('/')[-1]
    knowledge_sources = [
        {
            "files_source": {
                "name": source_name,
                "type": "files",
                "files": {"path": volume_path}
            }
        }
    ]
    
    payload = {
        "name": sanitized_name,
        "knowledge_sources": knowledge_sources
    }
    
    if description:
        payload["description"] = description
    if instructions:
        payload["instructions"] = instructions
    
    print(f"Creating Knowledge Assistant: {sanitized_name}")
    print(f"Volume path: {volume_path}")
    
    result = api_post("/api/2.0/knowledge-assistants", payload)
    return result

# COMMAND ----------

# Create the Knowledge Assistant
KA_NAME = "SEC_Financial_Analyst"
KA_DESCRIPTION = """Financial analyst assistant that answers questions about annual reports, 
10-K filings, and financial statements for companies like NVIDIA, Apple, Samsung, and others. 
Covers revenue, earnings, business segments, risk factors, and strategic initiatives."""

KA_INSTRUCTIONS = """You are a financial analyst assistant specializing in company annual reports and SEC filings.
You have access to financial filings that may include 10-K reports, annual reports, and financial statements.

Guidelines:
1. Always cite the specific company and document section when providing financial data
2. When comparing companies, present data in a structured format (tables or bullet points)
3. For revenue/earnings questions, include year-over-year changes when available
4. Be aware that different companies have different fiscal year end dates
5. If information is not in the documents, clearly state that and list which companies are available
6. Use precise financial terminology (revenue, net income, operating income, etc.)
7. When discussing business segments, explain what each segment includes
"""

# Uncomment to create the KA:
# result = create_knowledge_assistant(
#     name=KA_NAME,
#     volume_path=VOLUME_PATH,
#     description=KA_DESCRIPTION,
#     instructions=KA_INSTRUCTIONS
# )
# print(json.dumps(result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wait for KA Endpoint to be Online

# COMMAND ----------

def get_ka_status(tile_id: str) -> dict:
    """Get Knowledge Assistant status by tile_id."""
    return api_get(f"/api/2.0/knowledge-assistants/{tile_id}")

def wait_for_ka_online(tile_id: str, timeout_s: int = 600, poll_s: int = 15) -> dict:
    """Wait for KA endpoint to be ONLINE.
    
    Args:
        tile_id: The KA tile ID
        timeout_s: Maximum seconds to wait (default 10 minutes)
        poll_s: Seconds between status checks
    
    Returns:
        Final KA status dict
    """
    start = time.time()
    last_status = None
    
    while time.time() - start < timeout_s:
        ka = get_ka_status(tile_id)
        status = ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
        
        if status != last_status:
            elapsed = int(time.time() - start)
            print(f"[{elapsed}s] KA status: {status}")
            last_status = status
        
        if status == "ONLINE":
            print("KA endpoint is now ONLINE!")
            return ka
        
        time.sleep(poll_s)
    
    raise TimeoutError(f"KA {tile_id} did not come online within {timeout_s}s")

# Example usage (uncomment when you have a tile_id):
# tile_id = result["knowledge_assistant"]["tile"]["tile_id"]
# ka = wait_for_ka_online(tile_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Add Example Questions

# COMMAND ----------

def add_ka_example(tile_id: str, question: str, guidelines: list = None) -> dict:
    """Add an example question to the KA.
    
    Args:
        tile_id: The KA tile ID
        question: The example question
        guidelines: Optional list of guidelines for evaluating answers
    
    Returns:
        Created example dict
    """
    payload = {
        "tile_id": tile_id,
        "question": question
    }
    if guidelines:
        payload["guidelines"] = guidelines
    
    return api_post(f"/api/2.0/knowledge-assistants/{tile_id}/examples", payload)

def add_ka_examples_batch(tile_id: str, examples: list) -> list:
    """Add multiple examples to a KA.
    
    Args:
        tile_id: The KA tile ID
        examples: List of dicts with 'question' and optional 'guideline' keys
    
    Returns:
        List of created examples
    """
    created = []
    for ex in examples:
        question = ex.get("question")
        guideline = ex.get("guideline")
        guidelines = [guideline] if guideline else None
        
        try:
            result = add_ka_example(tile_id, question, guidelines)
            created.append(result)
            print(f"Added: {question[:60]}...")
        except Exception as e:
            print(f"Failed to add example: {e}")
    
    return created

# Example questions to seed the KA with relevant financial queries:
EXAMPLE_QUESTIONS = [
    {
        "question": "What was NVIDIA's total revenue for fiscal year 2024?",
        "guideline": "Should state $60.9 billion, up 126% year-over-year, with Data Center as the primary driver"
    },
    {
        "question": "How does NVIDIA's Data Center revenue compare to Gaming revenue?",
        "guideline": "Should compare Data Center ($47.5B, up 217%) vs Gaming ($10.4B, up 15%)"
    },
    {
        "question": "What are Apple's main product categories?",
        "guideline": "Should list iPhone, Mac, iPad, Wearables/Home/Accessories, and Services"
    },
    {
        "question": "What are the key risk factors mentioned in Apple's 10-K filing?",
        "guideline": "Should reference macroeconomic conditions, supply chain, competition, and regulatory risks"
    },
    {
        "question": "What are Samsung's two main business divisions?",
        "guideline": "Should explain DS (Device Solutions - semiconductors) and DX (Device eXperience - consumer electronics)"
    },
    {
        "question": "Compare the fiscal year 2024 performance of NVIDIA, Apple, and Samsung",
        "guideline": "Should compare revenue figures and highlight key growth drivers for each company"
    },
    {
        "question": "What is NVIDIA's Blackwell platform?",
        "guideline": "Should describe it as NVIDIA's most powerful AI platform for generative AI"
    },
    {
        "question": "What is Apple's Services segment and how did it perform?",
        "guideline": "Should describe Services (App Store, iCloud, Apple Music, etc.) and note growth"
    }
]

# Uncomment to add examples:
# add_ka_examples_batch(tile_id, EXAMPLE_QUESTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Sync Knowledge Sources (Re-index)

# COMMAND ----------

def sync_ka_sources(tile_id: str) -> None:
    """Trigger re-indexing of KA knowledge sources.
    
    Call this after adding/modifying files in the volume.
    """
    api_post(f"/api/2.0/knowledge-assistants/{tile_id}/sync-knowledge-sources", {})
    print(f"Triggered sync for KA {tile_id}")

# Uncomment to sync:
# sync_ka_sources(tile_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Query the KA Endpoint

# COMMAND ----------

def query_ka_endpoint(endpoint_name: str, question: str) -> dict:
    """Query a KA endpoint with a question.
    
    Args:
        endpoint_name: The serving endpoint name (usually 'ka-{tile_id}-endpoint')
        question: The question to ask
    
    Returns:
        Response from the endpoint
    """
    payload = {
        "input": [
            {"role": "user", "content": question}
        ]
    }
    
    return api_post(f"/serving-endpoints/{endpoint_name}/invocations", payload)

# Example:
# endpoint_name = f"ka-{tile_id}-endpoint"
# response = query_ka_endpoint(endpoint_name, "What are the key features?")
# print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: List All Knowledge Assistants

# COMMAND ----------

def list_knowledge_assistants() -> list:
    """List all Knowledge Assistants in the workspace."""
    all_kas = []
    page_token = None
    
    while True:
        params = {"filter": "tile_type=KA", "page_size": 100}
        if page_token:
            params["page_token"] = page_token
        
        resp = api_get("/api/2.0/tiles", params=params)
        
        for tile in resp.get("tiles", []):
            all_kas.append(tile)
        
        page_token = resp.get("next_page_token")
        if not page_token:
            break
    
    return all_kas

# List all KAs
kas = list_knowledge_assistants()
print(f"Found {len(kas)} Knowledge Assistants:")
for ka in kas[:10]:
    print(f"  - {ka.get('name')} (tile_id: {ka.get('tile_id')})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Delete a Knowledge Assistant

# COMMAND ----------

def delete_knowledge_assistant(tile_id: str) -> None:
    """Delete a Knowledge Assistant by tile_id."""
    api_delete(f"/api/2.0/tiles/{tile_id}")
    print(f"Deleted KA: {tile_id}")

# Uncomment to delete:
# delete_knowledge_assistant(tile_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complete Example: End-to-End KA Creation

# COMMAND ----------

def create_ka_end_to_end(
    name: str,
    volume_path: str,
    description: str = None,
    instructions: str = None,
    examples: list = None,
    wait_for_online: bool = True
) -> dict:
    """Create a KA end-to-end with optional examples.
    
    Args:
        name: KA name
        volume_path: Path to volume with documents
        description: Optional description
        instructions: Optional instructions
        examples: Optional list of example questions
        wait_for_online: Whether to wait for endpoint to be online
    
    Returns:
        Final KA status dict
    """
    # Step 1: Create the KA
    result = create_knowledge_assistant(
        name=name,
        volume_path=volume_path,
        description=description,
        instructions=instructions
    )
    
    tile_id = result["knowledge_assistant"]["tile"]["tile_id"]
    print(f"Created KA with tile_id: {tile_id}")
    
    # Step 2: Wait for online (optional)
    if wait_for_online:
        ka = wait_for_ka_online(tile_id, timeout_s=600)
        
        # Step 3: Add examples after online
        if examples:
            print(f"\nAdding {len(examples)} examples...")
            add_ka_examples_batch(tile_id, examples)
        
        return ka
    
    return result

# Uncomment to run end-to-end:
# ka = create_ka_end_to_end(
#     name="SEC_Financial_Analyst",
#     volume_path=VOLUME_PATH,
#     description=KA_DESCRIPTION,
#     instructions=KA_INSTRUCTIONS,
#     examples=EXAMPLE_QUESTIONS,
#     wait_for_online=True
# )
# 
# # Example query after KA is online:
# tile_id = ka["knowledge_assistant"]["tile"]["tile_id"]
# endpoint_name = f"ka-{tile_id}-endpoint"
# response = query_ka_endpoint(endpoint_name, "What was NVIDIA's total revenue in FY2024?")
# print(response["choices"][0]["message"]["content"])
