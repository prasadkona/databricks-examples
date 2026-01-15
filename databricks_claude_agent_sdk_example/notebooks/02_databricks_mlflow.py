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
Claude Agent with Databricks MLflow Tracking

This example demonstrates:
- Using Claude Agent SDK with Databricks
- MLflow experiment tracking
- Logging agent interactions and results

NOTE: This notebook is designed to run in a Databricks workspace for best
      MLflow integration. For local execution, ensure your PAT token has
      proper permissions to access MLflow tracking APIs.
"""

import asyncio
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions

# Load environment variables from .env (for local execution only)
# In Databricks, credentials are already set above
load_dotenv()

async def run_agent_with_mlflow():
    """Run agent with MLflow tracking."""
    print("="*60)
    print("Claude Agent with Databricks MLflow Tracking")
    print("="*60)
    
    # Setup MLflow - this is REQUIRED for this notebook
    try:
        import mlflow
        
        print("\n🔧 Configuring MLflow...")
        
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
        print(f"   Ready to log agent runs")
        
    except ImportError:
        print("\n❌ ERROR: MLflow not installed")
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
    
    # Example task
    prompt = """
    Analyze the current directory and provide:
    1. Count of Python files
    2. List of main directories
    3. Summary of what this project does
    4. Key dependencies from requirements.txt
    """
    
    print(f"\n📝 Task: {prompt.strip()}")
    print("\n" + "="*60)
    print("🤖 Agent working...\n")
    
    try:
        result = None
        run_id = None
        
        # Run agent WITH MLflow tracking
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            
            print(f"📊 MLflow Run Started: {run_id}")
            print(f"   Experiment ID: {experiment_id}\n")
            
            # Log parameters
            mlflow.log_param("agent_type", "file_analyzer")
            mlflow.log_param("tools", "Bash,Glob,Read")
            mlflow.log_param("model", os.getenv("ANTHROPIC_MODEL", "databricks-claude-haiku-4-5"))
            mlflow.log_param("databricks_host", os.getenv("DATABRICKS_HOST", "unknown"))
            
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(allowed_tools=["Bash", "Glob", "Read"])
            ):
                if hasattr(message, "result"):
                    result = message.result
                    
                    # Log result to MLflow
                    mlflow.log_text(result, "analysis_result.txt")
                    mlflow.log_metric("analysis_complete", 1)
                    mlflow.log_metric("result_length", len(result))
                    
                    print("📊 Agent Result:")
                    print("-"*60)
                    print(result)
                    print("-"*60)
        
        # After run completes, print MLflow details
        if run_id and result:
            databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            mlflow_url = f"{databricks_host}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
            
            print("\n" + "="*60)
            print("✅ SUCCESS: Example completed with MLflow tracking")
            print("="*60)
            print(f"\n📈 MLflow Run Details:")
            print(f"   Run ID: {run_id}")
            print(f"   Experiment: {experiment_name}")
            print(f"   View in Databricks: {mlflow_url}")
            print("\n" + "="*60)
        else:
            print("\n❌ ERROR: Agent did not produce a result")
        
    except Exception as e:
        print(f"\n❌ ERROR during agent execution: {e}")
        import traceback
        traceback.print_exc()


# COMMAND ----------

# Run the async function directly in the notebook
await run_agent_with_mlflow()
