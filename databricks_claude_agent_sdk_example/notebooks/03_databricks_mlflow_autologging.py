# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install claude-agent-sdk>=0.1.19 python-dotenv>=1.0.0 databricks-sdk>=0.30.0 mlflow>=2.16.0 anthropic>=0.39.0

# COMMAND ----------

# Restart Python to ensure new packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Get workspace hostname using spark.conf (the correct method for DBR 18.0)
DATABRICKS_HOSTNAME = spark.conf.get("spark.databricks.workspaceUrl")

# Get Databricks access token from notebook context using dbutils (recommended for notebooks)
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Set environment variables for Claude Agent SDK
import os
os.environ["DATABRICKS_HOST"] = f"https://{DATABRICKS_HOSTNAME}"
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["ANTHROPIC_AUTH_TOKEN"] = DATABRICKS_TOKEN
os.environ["ANTHROPIC_BASE_URL"] = f"https://{DATABRICKS_HOSTNAME}/serving-endpoints/anthropic"
os.environ["ANTHROPIC_CUSTOM_HEADERS"] = "x-databricks-disable-beta-headers: true"
os.environ["ANTHROPIC_MODEL"] = "databricks-claude-haiku-4-5"

# COMMAND ----------

"""
Claude Agent SDK with MLflow Autologging

This example demonstrates:
- Using MLflow autologging for Claude Agent SDK
- Automatic tracing of all Claude Agent SDK interactions
- ClaudeSDKClient for direct client usage
- Experiment tracking without manual logging

NOTE: This notebook is designed to run in a Databricks workspace for best
      MLflow integration. For local execution, ensure your PAT token has
      proper permissions to access MLflow tracking APIs.
"""

import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables from .env (for local execution only)
# In Databricks, credentials are already set above
load_dotenv()


async def run_agent_with_autologging():
    """Run agent with MLflow autologging enabled."""
    print("="*60)
    print("Claude Agent SDK with MLflow Autologging")
    print("="*60)
    
    # Setup MLflow with autologging
    try:
        import mlflow
        import mlflow.anthropic
        from claude_agent_sdk import ClaudeSDKClient
        
        print("\n🔧 Configuring MLflow with autologging...")
        
        # Clear cluster ID if set (can cause auth issues for local execution)
        if "DATABRICKS_CLUSTER_ID" in os.environ:
            print("   Clearing DATABRICKS_CLUSTER_ID...")
            del os.environ["DATABRICKS_CLUSTER_ID"]
        
        mlflow.set_tracking_uri("databricks")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/claude-agent-demo")
        
        # Test the connection by trying to set experiment
        print(f"   Connecting to: {os.getenv('DATABRICKS_HOST', 'default')}")
        print(f"   Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ MLflow configured successfully")
        
        # Enable autologging for Claude Agent SDK
        print("\n🔍 Enabling MLflow autologging for Claude Agent SDK...")
        mlflow.anthropic.autolog()
        print("✅ Autologging enabled - all interactions will be traced automatically")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package: {e}")
        print("   Install with: pip install mlflow")
        return
    except Exception as e:
        print(f"\n❌ ERROR: MLflow configuration failed")
        print(f"   Error: {e}")
        print("\n   💡 Recommended Solution:")
        print("   This notebook works best when run inside a Databricks workspace.")
        print("   Upload this file to Databricks and run it there for full MLflow integration.")
        print("\n   For local execution, ensure:")
        print("   - DATABRICKS_HOST in .env is correct")
        print("   - DATABRICKS_TOKEN has workspace and MLflow permissions")
        print("   - You have access to the MLflow experiment path")
        return
    
    # Run agent queries with autologging
    print("\n" + "="*60)
    print("🤖 Running Claude Agent with Autologging")
    print("="*60)
    
    try:
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            
            print(f"\n📊 MLflow Run Started: {run_id}")
            print(f"   Experiment ID: {experiment_id}")
            
            # Log additional parameters
            mlflow.log_param("agent_type", "autolog_demo")
            mlflow.log_param("model", os.getenv("ANTHROPIC_MODEL", "databricks-claude-haiku-4-5"))
            mlflow.log_param("databricks_host", os.getenv("DATABRICKS_HOST", "unknown"))
            
            # Example 1: Simple query
            print("\n" + "-"*60)
            print("📝 Example 1: Simple Question")
            print("-"*60)
            
            async with ClaudeSDKClient() as client:
                await client.query("What is the capital of France? Please provide a brief answer.")
                
                response_text = ""
                async for message in client.receive_response():
                    response_text += str(message)
                
                print(f"\n🤖 Claude Response:\n{response_text}\n")
                mlflow.log_metric("query_1_complete", 1)
            
            # Example 2: File analysis
            print("-"*60)
            print("📝 Example 2: Directory Analysis")
            print("-"*60)
            
            async with ClaudeSDKClient() as client:
                await client.query(
                    "List the Python files in the current directory. "
                    "Use the appropriate tools to accomplish this task."
                )
                
                response_text = ""
                async for message in client.receive_response():
                    response_text += str(message)
                
                print(f"\n🤖 Claude Response:\n{response_text}\n")
                mlflow.log_metric("query_2_complete", 1)
            
            # Log summary
            mlflow.log_metric("total_queries", 2)
            
            # After run completes, print MLflow details
            databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            mlflow_url = f"{databricks_host}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
            
            print("="*60)
            print("✅ SUCCESS: Autologging captured all interactions!")
            print("="*60)
            print(f"\n📈 MLflow Run Details:")
            print(f"   Run ID: {run_id}")
            print(f"   Experiment: {experiment_name}")
            print(f"   View in Databricks: {mlflow_url}")
            print("\n🔍 Autologged Information:")
            print("   - All Claude API calls")
            print("   - Request/response payloads")
            print("   - Token usage and costs")
            print("   - Latency and performance metrics")
            print("   - Tool usage statistics")
            print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR during agent execution: {e}")
        import traceback
        traceback.print_exc()


# COMMAND ----------

# Run the async function directly in the notebook
await run_agent_with_autologging()
