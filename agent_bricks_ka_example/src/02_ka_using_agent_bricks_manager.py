# Databricks notebook source
# MAGIC %md
# MAGIC # Knowledge Assistant using AgentBricksManager
# MAGIC 
# MAGIC This notebook demonstrates how to create a Knowledge Assistant (KA) Agent Brick
# MAGIC using the `AgentBricksManager` wrapper class.
# MAGIC 
# MAGIC This approach provides a higher-level, more convenient API compared to raw REST calls.
# MAGIC 
# MAGIC ## Authentication
# MAGIC - **OAuth M2M** (default): Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
# MAGIC - **PAT** (fallback): Uses DATABRICKS_TOKEN if OAuth not configured
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Serverless compute enabled  
# MAGIC - Access to Foundation Models in `system.ai` schema
# MAGIC - Files in a Unity Catalog Volume
# MAGIC - Serverless budget policy with nonzero budget

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install databricks-sdk requests

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
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from databricks.sdk import WorkspaceClient

# Import common config
from config import load_env_file, setup_databricks_auth

# Load configuration and set up authentication (OAuth by default, PAT fallback)
config = load_env_file()
config, auth_type = setup_databricks_auth(config)

print(f"Configured for workspace: {config.get('DATABRICKS_HOST')}")
print(f"Auth: {auth_type.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AgentBricksManager Class
# MAGIC 
# MAGIC This is a self-contained version of the AgentBricksManager for use in notebooks.

# COMMAND ----------

class AgentBricksManager:
    """Unified wrapper for Agent Bricks tiles (KA, MAS, Genie).
    
    Provides convenient methods for:
    - Creating and managing Knowledge Assistants
    - Creating and managing Multi-Agent Supervisors
    - Creating and managing Genie Spaces
    """
    
    def __init__(self, client: Optional[WorkspaceClient] = None):
        """Initialize the Agent Bricks Manager."""
        self.w = client or WorkspaceClient()
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitize a name for Databricks API compliance."""
        sanitized = name.replace(" ", "_")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
        sanitized = re.sub(r"[_-]{2,}", "_", sanitized)
        return sanitized.strip("_-") or "knowledge_assistant"
    
    # ========================================================================
    # HTTP Helpers
    # ========================================================================
    
    def _get_headers(self) -> dict:
        headers = self.w.config.authenticate()
        headers["Content-Type"] = "application/json"
        return headers
    
    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.w.config.host}{path}"
        response = requests.get(url, headers=self._get_headers(), params=params or {}, timeout=30)
        if response.status_code >= 400:
            raise Exception(f"GET {path} failed: {response.text}")
        return response.json()
    
    def _post(self, path: str, body: dict, timeout: int = 300) -> dict:
        url = f"{self.w.config.host}{path}"
        response = requests.post(url, headers=self._get_headers(), json=body, timeout=timeout)
        if response.status_code >= 400:
            raise Exception(f"POST {path} failed: {response.text}")
        return response.json()
    
    def _patch(self, path: str, body: dict) -> dict:
        url = f"{self.w.config.host}{path}"
        response = requests.patch(url, headers=self._get_headers(), json=body, timeout=30)
        if response.status_code >= 400:
            raise Exception(f"PATCH {path} failed: {response.text}")
        return response.json()
    
    def _delete(self, path: str) -> dict:
        url = f"{self.w.config.host}{path}"
        response = requests.delete(url, headers=self._get_headers(), timeout=30)
        if response.status_code >= 400 and response.status_code != 404:
            raise Exception(f"DELETE {path} failed: {response.text}")
        return response.json() if response.text else {}
    
    # ========================================================================
    # Knowledge Assistant Operations
    # ========================================================================
    
    def ka_create(
        self,
        name: str,
        knowledge_sources: List[Dict[str, Any]],
        description: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Knowledge Assistant.
        
        Args:
            name: Name for the KA
            knowledge_sources: List of knowledge source dictionaries
            description: Optional description
            instructions: Optional instructions
        
        Returns:
            KA creation response
        """
        payload = {
            "name": self.sanitize_name(name),
            "knowledge_sources": knowledge_sources,
        }
        if instructions:
            payload["instructions"] = instructions
        if description:
            payload["description"] = description
        
        print(f"Creating KA: {payload['name']}")
        return self._post("/api/2.0/knowledge-assistants", payload)
    
    def ka_get(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Get KA by tile_id."""
        try:
            return self._get(f"/api/2.0/knowledge-assistants/{tile_id}")
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    def ka_get_endpoint_status(self, tile_id: str) -> Optional[str]:
        """Get endpoint status of a KA."""
        ka = self.ka_get(tile_id)
        if not ka:
            return None
        return ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
    
    def ka_wait_until_endpoint_online(
        self,
        tile_id: str,
        timeout_s: int = 600,
        poll_s: float = 15.0
    ) -> Dict[str, Any]:
        """Wait for KA endpoint to be ONLINE."""
        deadline = time.time() + timeout_s
        start_time = time.time()
        last_status = None
        
        while True:
            ka = self.ka_get(tile_id)
            status = ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
            
            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] KA status: {status}")
                last_status = status
            
            if status == "ONLINE":
                return ka
            
            if time.time() >= deadline:
                return ka
            
            time.sleep(poll_s)
    
    def ka_create_or_update(
        self,
        name: str,
        knowledge_sources: List[Dict[str, Any]],
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a Knowledge Assistant."""
        existing_ka = None
        operation = "created"
        
        if tile_id:
            existing_ka = self.ka_get(tile_id)
            if existing_ka:
                operation = "updated"
        
        if existing_ka:
            result = self.ka_update(
                tile_id,
                name=self.sanitize_name(name),
                description=description,
                instructions=instructions,
                knowledge_sources=knowledge_sources,
            )
        else:
            result = self.ka_create(
                name=name,
                knowledge_sources=knowledge_sources,
                description=description,
                instructions=instructions,
            )
        
        if result:
            result["operation"] = operation
        return result
    
    def ka_update(
        self,
        tile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        knowledge_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Update KA metadata and/or knowledge sources."""
        if name is not None or description is not None or instructions is not None:
            body = {}
            if name is not None:
                body["name"] = name
            if description is not None:
                body["description"] = description
            if instructions is not None:
                body["instructions"] = instructions
            self._patch(f"/api/2.0/knowledge-assistants/{tile_id}", body)
        
        return self.ka_get(tile_id)
    
    def ka_sync_sources(self, tile_id: str) -> None:
        """Trigger re-indexing of knowledge sources."""
        self._post(f"/api/2.0/knowledge-assistants/{tile_id}/sync-knowledge-sources", {})
        print(f"Triggered sync for KA {tile_id}")
    
    def ka_create_example(
        self,
        tile_id: str,
        question: str,
        guidelines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create an example question for the KA."""
        payload = {"tile_id": tile_id, "question": question}
        if guidelines:
            payload["guidelines"] = guidelines
        return self._post(f"/api/2.0/knowledge-assistants/{tile_id}/examples", payload)
    
    def ka_add_examples_batch(
        self,
        tile_id: str,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add multiple examples in parallel."""
        created = []
        
        def create_example(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            question_text = q.get("question", "")
            guideline = q.get("guideline")
            guidelines = [guideline] if guideline else None
            
            if not question_text:
                return None
            try:
                example = self.ka_create_example(tile_id, question_text, guidelines)
                print(f"Added: {question_text[:50]}...")
                return example
            except Exception as e:
                print(f"Failed: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(create_example, q): q for q in questions}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    created.append(result)
        
        return created
    
    def ka_list_examples(self, tile_id: str, page_size: int = 100) -> Dict[str, Any]:
        """List all examples for a KA."""
        return self._get(f"/api/2.0/knowledge-assistants/{tile_id}/examples", {"page_size": page_size})
    
    def delete(self, tile_id: str) -> None:
        """Delete any tile (KA or MAS)."""
        self._delete(f"/api/2.0/tiles/{tile_id}")
        print(f"Deleted tile: {tile_id}")
    
    def find_by_name(self, name: str) -> Optional[Dict[str, str]]:
        """Find a KA by exact name."""
        page_token = None
        while True:
            params = {"filter": f"name_contains={name}&&tile_type=KA"}
            if page_token:
                params["page_token"] = page_token
            resp = self._get("/api/2.0/tiles", params=params)
            for t in resp.get("tiles", []):
                if t.get("name") == name:
                    return {"tile_id": t["tile_id"], "name": name}
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return None
    
    def list_all_knowledge_assistants(self) -> List[Dict[str, Any]]:
        """List all Knowledge Assistants."""
        all_kas = []
        page_token = None
        
        while True:
            params = {"filter": "tile_type=KA", "page_size": 100}
            if page_token:
                params["page_token"] = page_token
            
            resp = self._get("/api/2.0/tiles", params=params)
            all_kas.extend(resp.get("tiles", []))
            
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        
        return all_kas
    
    @staticmethod
    def get_knowledge_sources_from_volumes(
        volume_paths: List[Tuple[str, Optional[str]]]
    ) -> List[Dict[str, Any]]:
        """Convert volume paths to knowledge source dictionaries.
        
        Args:
            volume_paths: List of (volume_path, description) tuples
        
        Returns:
            List of knowledge source dictionaries
        """
        knowledge_sources = []
        
        for idx, (volume_path, description) in enumerate(volume_paths):
            path_parts = volume_path.rstrip("/").split("/")
            source_name = path_parts[-1] if path_parts else f"source_{idx + 1}"
            source_name = source_name.replace(" ", "_").replace(".", "_")
            
            knowledge_source = {
                "files_source": {
                    "name": source_name,
                    "type": "files",
                    "files": {"path": volume_path},
                }
            }
            knowledge_sources.append(knowledge_source)
        
        return knowledge_sources

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Manager

# COMMAND ----------

# Initialize the manager
manager = AgentBricksManager()
print(f"Connected to: {manager.w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define Knowledge Sources

# COMMAND ----------

# Define your volume paths containing documents (from .env config)
CATALOG = config.get("UC_CATALOG", "prasad_kona_isv")
SCHEMA = config.get("UC_SCHEMA", "isv_demo")
VOLUME = config.get("UC_VOLUME", "ka_demo")
DATASET_FOLDER = "sec_2024_md"
VOLUME_PATH = config.get("UC_VOLUME_PATH", f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")
VOLUME_PATH = f"{VOLUME_PATH.rstrip('/')}/{DATASET_FOLDER}"

print(f"Volume path: {VOLUME_PATH}")

# Create knowledge sources from volume paths
volume_paths = [
    (VOLUME_PATH, "FY2024 Annual Reports - NVIDIA, Apple, Samsung, and others"),
]
knowledge_sources = AgentBricksManager.get_knowledge_sources_from_volumes(volume_paths)

print("Knowledge Sources:")
print(json.dumps(knowledge_sources, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Knowledge Assistant

# COMMAND ----------

# Define KA configuration
KA_NAME = "SEC_Financial_Analyst_02"
KA_DESCRIPTION = """Financial analyst assistant that answers questions about annual reports, 
10-K filings, and financial statements for companies like NVIDIA, Apple, Samsung, and others. 
Covers revenue, earnings, business segments, risk factors, and strategic initiatives."""

KA_INSTRUCTIONS = """You are a financial analyst assistant specializing in company annual reports and SEC filings.
You have access to financial filings that may include 10-K reports, annual reports, and financial statements.

Guidelines:
1. **Cite Sources**: Always cite the specific company and document section when providing financial data
2. **Structured Comparisons**: When comparing companies, present data in tables or bullet points
3. **Year-over-Year**: For revenue/earnings questions, include YoY changes when available
4. **Fiscal Year Clarity**: Be aware that different companies have different fiscal year end dates
5. **Be Honest**: If information is not in the documents, clearly state that and list available companies
6. **Financial Terminology**: Use precise terms (revenue, net income, operating income, etc.)
7. **Segment Details**: When discussing business segments, explain what each segment includes
"""

# Create the KA (uncomment to run)
# result = manager.ka_create_or_update(
#     name=KA_NAME,
#     knowledge_sources=knowledge_sources,
#     description=KA_DESCRIPTION,
#     instructions=KA_INSTRUCTIONS
# )
# 
# tile_id = result.get("knowledge_assistant", {}).get("tile", {}).get("tile_id")
# print(f"Created KA with tile_id: {tile_id}")
# print(json.dumps(result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wait for Endpoint to be Online

# COMMAND ----------

# Wait for endpoint to be ready (uncomment when you have tile_id)
# ka = manager.ka_wait_until_endpoint_online(tile_id, timeout_s=600)
# print(f"Final status: {ka.get('knowledge_assistant', {}).get('status', {}).get('endpoint_status')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Add Example Questions

# COMMAND ----------

# Define example questions for seeding the KA with relevant financial queries
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
    },
    {
        "question": "What is NVIDIA's net income for FY2024?",
        "guideline": "Should state $33.0 billion, up significantly year-over-year"
    },
    {
        "question": "What accounting standards does Samsung follow?",
        "guideline": "Should mention Korean IFRS (Korean International Financial Reporting Standards)"
    }
]

# Add examples (uncomment when KA is online)
# created_examples = manager.ka_add_examples_batch(tile_id, EXAMPLE_QUESTIONS)
# print(f"Added {len(created_examples)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: List All Knowledge Assistants

# COMMAND ----------

# List all KAs in the workspace
kas = manager.list_all_knowledge_assistants()
print(f"Found {len(kas)} Knowledge Assistants:\n")

for ka in kas[:10]:
    print(f"Name: {ka.get('name')}")
    print(f"  Tile ID: {ka.get('tile_id')}")
    print(f"  Created: {ka.get('create_time')}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Find KA by Name

# COMMAND ----------

# Find a specific KA by name
# found = manager.find_by_name("My_Document_Assistant")
# if found:
#     print(f"Found KA: {found}")
#     ka = manager.ka_get(found["tile_id"])
#     print(json.dumps(ka, indent=2))
# else:
#     print("KA not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Sync Knowledge Sources (Re-index)

# COMMAND ----------

# Trigger re-indexing after updating documents (uncomment when needed)
# manager.ka_sync_sources(tile_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Delete a Knowledge Assistant

# COMMAND ----------

# Delete a KA (uncomment with caution)
# manager.delete(tile_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complete End-to-End Example

# COMMAND ----------

def create_ka_complete(
    manager: AgentBricksManager,
    name: str,
    volume_path: str,
    description: str = None,
    instructions: str = None,
    examples: List[Dict[str, Any]] = None,
    wait_for_online: bool = True,
    timeout_s: int = 600
) -> Dict[str, Any]:
    """Create a complete KA with all configurations.
    
    Args:
        manager: AgentBricksManager instance
        name: KA name
        volume_path: Path to volume with documents
        description: Optional description
        instructions: Optional instructions
        examples: Optional list of example questions
        wait_for_online: Whether to wait for endpoint
        timeout_s: Timeout for waiting
    
    Returns:
        Final KA configuration
    """
    print("=" * 60)
    print(f"Creating Knowledge Assistant: {name}")
    print("=" * 60)
    
    # Create knowledge sources
    volume_paths = [(volume_path, None)]
    knowledge_sources = AgentBricksManager.get_knowledge_sources_from_volumes(volume_paths)
    
    # Create the KA
    result = manager.ka_create_or_update(
        name=name,
        knowledge_sources=knowledge_sources,
        description=description,
        instructions=instructions
    )
    
    tile_id = result.get("knowledge_assistant", {}).get("tile", {}).get("tile_id")
    print(f"\nCreated KA with tile_id: {tile_id}")
    
    # Wait for online
    if wait_for_online:
        print("\nWaiting for endpoint to be online...")
        ka = manager.ka_wait_until_endpoint_online(tile_id, timeout_s=timeout_s)
        status = ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
        print(f"\nEndpoint status: {status}")
        
        # Add examples if provided and online
        if examples and status == "ONLINE":
            print(f"\nAdding {len(examples)} examples...")
            created = manager.ka_add_examples_batch(tile_id, examples)
            print(f"Added {len(created)} examples successfully")
        
        return ka
    
    return result

# Run complete example (uncomment to execute)
# ka = create_ka_complete(
#     manager=manager,
#     name="SEC_Financial_Analyst",
#     volume_path=VOLUME_PATH,
#     description=KA_DESCRIPTION,
#     instructions=KA_INSTRUCTIONS,
#     examples=EXAMPLE_QUESTIONS,
#     wait_for_online=True
# )
# 
# # Example queries after KA is online:
# tile_id = ka["tile_id"]
# endpoint_name = f"ka-{tile_id}-endpoint"
# 
# # Query 1: NVIDIA revenue
# response = manager.ka_query(endpoint_name, "What was NVIDIA's total revenue in FY2024?")
# print(response["choices"][0]["message"]["content"])
# 
# # Query 2: Compare companies
# response = manager.ka_query(endpoint_name, "Compare NVIDIA and Apple's FY2024 performance")
# print(response["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the KA Endpoint

# COMMAND ----------

def query_ka(manager: AgentBricksManager, tile_id: str, question: str) -> Dict[str, Any]:
    """Query a KA endpoint with a question.
    
    Args:
        manager: AgentBricksManager instance
        tile_id: KA tile ID
        question: Question to ask
    
    Returns:
        Response from the endpoint
    """
    endpoint_name = f"ka-{tile_id}-endpoint"
    
    payload = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    
    return manager._post(f"/serving-endpoints/{endpoint_name}/invocations", payload)

# Query example (uncomment when KA is ready)
# response = query_ka(manager, tile_id, "What are the main features described in the documentation?")
# print(json.dumps(response, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Report to _local

# COMMAND ----------

import os
from datetime import datetime

def save_ka_report(kas: List[Dict[str, Any]], report_dir: str = None):
    """Save KA list report to _local folder."""
    if report_dir is None:
        # Try to find _local/reports relative to current working directory
        possible_paths = [
            Path("../../_local/reports"),        # When running from src/
            Path("../_local/reports"),           # When running from agent_bricks_ka_example/
            Path("_local/reports"),              # When running from project root
        ]
        for p in possible_paths:
            if p.parent.exists():
                report_dir = str(p)
                break
        else:
            report_dir = "_local/reports"
    
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_path = os.path.join(report_dir, f"ka_inventory_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(kas, f, indent=2, default=str)
    print(f"Saved JSON report: {json_path}")
    
    # Save summary markdown
    md_path = os.path.join(report_dir, f"ka_inventory_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write(f"# Knowledge Assistant Inventory\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total KAs: {len(kas)}\n\n")
        f.write("| Name | Tile ID | Created |\n")
        f.write("|------|---------|----------|\n")
        for ka in kas:
            f.write(f"| {ka.get('name', 'N/A')} | {ka.get('tile_id', 'N/A')} | {ka.get('create_time', 'N/A')} |\n")
    print(f"Saved markdown report: {md_path}")

# Save report (uncomment to run)
# save_ka_report(kas)
