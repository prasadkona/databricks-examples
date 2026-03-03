# Databricks notebook source
# MAGIC %md
# MAGIC # Test Knowledge Assistant - Multi-turn Conversation
# MAGIC 
# MAGIC This script tests a Knowledge Assistant endpoint with a multi-turn conversation.
# MAGIC It demonstrates how to maintain conversation context across multiple questions.
# MAGIC 
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC # Test using KA name (generic - works with any KA)
# MAGIC python 03_test_ka_conversation.py SEC_Financial_Analyst
# MAGIC 
# MAGIC # Test using shortcut from env config
# MAGIC python 03_test_ka_conversation.py 1   # KA_NAME_01
# MAGIC python 03_test_ka_conversation.py 2   # KA_NAME_02
# MAGIC 
# MAGIC # Default: KA_NAME_01
# MAGIC python 03_test_ka_conversation.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import sys
import json
import requests
from pathlib import Path

def load_env_file(env_path: str = None, env_name: str = "e2-demo-field-eng") -> dict:
    """Load environment variables from a .env file."""
    if env_path is None:
        search_dirs = [
            Path("../../_local"),
            Path("../_local"),
            Path("_local"),
        ]
        for d in search_dirs:
            env_specific = d / f"{env_name}.env"
            if env_specific.exists():
                env_path = env_specific
                break
            generic = d / ".env"
            if generic.exists():
                env_path = generic
                break
    
    config = {}
    if env_path and Path(env_path).exists():
        print(f"Loading config from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    return config

# Load config from local env file
config = load_env_file()

# Set environment variables for OAuth M2M authentication
os.environ['DATABRICKS_HOST'] = config.get('DATABRICKS_HOST', '')
os.environ['DATABRICKS_CLIENT_ID'] = config.get('DATABRICKS_CLIENT_ID', '')
os.environ['DATABRICKS_CLIENT_SECRET'] = config.get('DATABRICKS_CLIENT_SECRET', '')

DATABRICKS_HOST = config.get('DATABRICKS_HOST')

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"Workspace: {DATABRICKS_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## KA Selection from Config

# COMMAND ----------

# KA configurations from env (name + tile_id pairs)
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
    """
    Get KA config from argument.
    
    Args:
        arg: Either a shortcut ("1", "2") or a KA name
        
    Returns:
        dict with 'name' and 'tile_id'
    """
    # Check if it's a shortcut
    if arg in KA_CONFIG:
        return KA_CONFIG[arg]
    
    # Check if it matches a KA name in config
    for key, ka in KA_CONFIG.items():
        if ka['name'] == arg:
            return ka
    
    # Not found in config
    return {"name": arg, "tile_id": ""}

def get_endpoint_from_tile_id(tile_id: str) -> str:
    """Construct endpoint name from tile_id."""
    # Endpoint format: ka-{first_segment_of_tile_id}-endpoint
    first_segment = tile_id.split('-')[0]
    return f"ka-{first_segment}-endpoint"

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

KA_ENDPOINT_NAME = get_endpoint_from_tile_id(KA_TILE_ID)

print(f"KA Name: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")
print(f"Endpoint: {KA_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_headers():
    """Get authenticated headers for REST API calls."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers

def query_ka(endpoint_name: str, messages: list) -> dict:
    """
    Query a KA endpoint with conversation history.
    
    Args:
        endpoint_name: The serving endpoint name
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        Response from the endpoint
    """
    url = f"{DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations"
    
    payload = {
        "input": messages
    }
    
    response = requests.post(url, headers=get_headers(), json=payload, timeout=120)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def extract_response_text(response: dict) -> str:
    """Extract the text content from KA response."""
    if not response or 'output' not in response:
        return ""
    
    for item in response.get('output', []):
        if item.get('type') == 'message':
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    return content.get('text', '')
    return ""

def print_conversation_turn(role: str, content: str):
    """Pretty print a conversation turn."""
    prefix = "USER:" if role == "user" else "ASSISTANT:"
    print(f"\n{prefix}")
    print("-" * 60)
    print(content)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Multi-turn Conversation Test

# COMMAND ----------

# Define test questions
QUESTION_1 = "What documents do you have access to? List the companies and fiscal years."
QUESTION_2 = "For the companies you mentioned, compare their main business segments. What are the key differences?"

print("\n" + "=" * 70)
print(f"MULTI-TURN CONVERSATION TEST")
print(f"Endpoint: {KA_ENDPOINT_NAME}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 1: Initial Question

# COMMAND ----------

# Turn 1
conversation = [
    {"role": "user", "content": QUESTION_1}
]

print_conversation_turn("user", QUESTION_1)

response1 = query_ka(KA_ENDPOINT_NAME, conversation)
assistant_response_1 = extract_response_text(response1)

print_conversation_turn("assistant", assistant_response_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 2: Follow-up Question

# COMMAND ----------

# Turn 2 - Add previous exchange to conversation history
conversation.append({"role": "assistant", "content": assistant_response_1})
conversation.append({"role": "user", "content": QUESTION_2})

print_conversation_turn("user", QUESTION_2)

response2 = query_ka(KA_ENDPOINT_NAME, conversation)
assistant_response_2 = extract_response_text(response2)

print_conversation_turn("assistant", assistant_response_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversation Summary

# COMMAND ----------

print("=" * 70)
print("CONVERSATION SUMMARY")
print("=" * 70)
print(f"\nKA Name: {KA_NAME}")
print(f"Endpoint: {KA_ENDPOINT_NAME}")
print(f"Total turns: 2")
print(f"\nQ1: {QUESTION_1[:80]}...")
print(f"Q2: {QUESTION_2[:80]}...")
print(f"\nConversation history maintained: {len(conversation)} messages")
