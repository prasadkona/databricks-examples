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
Claude Agent SDK with MLflow GenAI Evaluation

This example demonstrates:
- Using Claude Agent SDK with MLflow GenAI evaluation framework
- Creating custom judges/scorers for response quality
- Running evaluations with automatic tracing
- Batch evaluation of agent responses
- Performance and quality metrics

NOTE: This notebook is designed to run in a Databricks workspace for best
      MLflow integration. Requires access to an LLM judge (e.g., GPT-4, Claude)
      for evaluation scoring.
"""

import asyncio
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env (for local execution only)
# In Databricks, credentials are already set above
load_dotenv()


async def run_agent(query_text: str) -> str:
    """
    Run Claude Agent SDK and return response.
    
    Args:
        query_text: User query to send to the agent
        
    Returns:
        str: Agent response text
    """
    from claude_agent_sdk import ClaudeSDKClient
    
    async with ClaudeSDKClient() as client:
        await client.query(query_text)
        
        response_text = ""
        async for message in client.receive_response():
            response_text += str(message) + "\n\n"
        
        return response_text.strip()


def predict_fn(query: str) -> str:
    """
    Synchronous wrapper for evaluation.
    MLflow evaluate() requires a synchronous function.
    The parameter name 'query' must match the key in the inputs dictionary.
    
    Args:
        query: The query string to send to the agent
        
    Returns:
        str: Agent response
    """
    if not query:
        return "Error: No query provided"
    
    try:
        response = asyncio.run(run_agent(query))
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def run_evaluation():
    """Run Claude Agent evaluation with MLflow GenAI framework."""
    print("="*60)
    print("Claude Agent SDK with MLflow GenAI Evaluation")
    print("="*60)
    
    # Setup MLflow
    try:
        import mlflow
        import mlflow.anthropic
        from mlflow.genai import evaluate
        from mlflow.genai.scorers import scorer
        
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
        experiment = mlflow.set_experiment(experiment_name)
        
        print(f"✅ MLflow configured successfully")
        
        # Enable autologging for Claude Agent SDK
        print("\n🔍 Enabling MLflow autologging...")
        mlflow.anthropic.autolog()
        print("✅ Autologging enabled")
        
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
        return
    
    # Create evaluation dataset
    print("\n" + "="*60)
    print("📊 Creating Evaluation Dataset")
    print("="*60)
    
    eval_data = pd.DataFrame({
        "inputs": [
            {"query": "What is machine learning?"},
            {"query": "Explain neural networks in simple terms"},
            {"query": "What are the main types of AI?"},
            {"query": "How does deep learning differ from traditional ML?"},
        ]
    })
    
    print(f"\n✅ Created evaluation dataset with {len(eval_data)} queries:")
    for i, row in eval_data.iterrows():
        print(f"   {i+1}. {row['inputs']['query']}")
    
    # Create custom scorers
    print("\n" + "="*60)
    print("⚖️  Creating Custom Scorers")
    print("="*60)
    
    @scorer
    def response_length(outputs: str) -> int:
        """Measure response length in characters."""
        return len(outputs)
    
    @scorer
    def has_content(outputs: str) -> int:
        """Check if response has meaningful content (not an error)."""
        return 0 if outputs.startswith("Error:") else 1
    
    print("✅ Created custom scorers:")
    print("   - response_length: Measures response length")
    print("   - has_content: Checks for successful responses")
    
    # Run evaluation
    print("\n" + "="*60)
    print("🚀 Running Evaluation")
    print("="*60)
    print("\nThis will:")
    print("   1. Run each query through Claude Agent SDK")
    print("   2. Collect responses")
    print("   3. Score responses with custom scorers")
    print("   4. Log everything to MLflow with automatic tracing")
    print("\n" + "-"*60)
    
    try:
        # Run evaluation with automatic tracing
        print("\n🔄 Running evaluation (this may take a few minutes)...\n")
        
        results = evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[response_length, has_content],
        )
        
        print("\n" + "="*60)
        print("✅ SUCCESS: Evaluation Complete!")
        print("="*60)
        
        # Display results
        if hasattr(results, 'tables') and 'eval_results_table' in results.tables:
            print("\n📊 Evaluation Results:")
            print("-"*60)
            results_df = results.tables['eval_results_table']
            # Display summary for each query
            for idx, row in results_df.iterrows():
                query = row['inputs'].get('query', 'N/A') if isinstance(row['inputs'], dict) else 'N/A'
                output = row['outputs'] if 'outputs' in row else 'N/A'
                length = row.get('response_length', 'N/A')
                has_content_val = row.get('has_content', 'N/A')
                
                print(f"\nQuery {idx+1}: {query}")
                print(f"  Response length: {length} characters")
                print(f"  Has content: {'Yes' if has_content_val == 1 else 'No'}")
                print(f"  Response preview: {str(output)[:150]}...")
        
        if hasattr(results, 'metrics'):
            print("\n📈 Aggregate Metrics:")
            print("-"*60)
            for metric_name, metric_value in results.metrics.items():
                print(f"   {metric_name}: {metric_value}")
        
        # Get MLflow run info
        databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        print(f"\n🔗 View detailed results in Databricks MLflow:")
        print(f"   {databricks_host}/#mlflow/experiments/{experiment.experiment_id}")
        
        print("\n" + "="*60)
        print("✨ What was logged to MLflow:")
        print("   - All agent queries and responses")
        print("   - Custom scorer metrics (response_length, has_content)")
        print("   - Token usage and costs (via autologging)")
        print("   - Performance metrics")
        print("   - Evaluation summary tables")
        print("   - Full trace of agent interactions")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()


# COMMAND ----------

# Run the evaluation directly in the notebook
run_evaluation()
