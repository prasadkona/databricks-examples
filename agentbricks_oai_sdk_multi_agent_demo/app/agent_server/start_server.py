"""
FastAPI server entry point for the SEC Financial Analyst Multi-Agent.
"""

import os
import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer

load_dotenv(dotenv_path=".env", override=True, verbose=False)


def setup_mlflow_experiment():
    """Set up MLflow experiment for tracing."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    try:
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")

        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
            print(f"Using MLflow experiment ID: {experiment_id}")
            return experiment_id

        # Derive workspace user from SDK if MLFLOW_EXPERIMENT_NAME is not set
        _default_user = "your-user@databricks.com"
        try:
            from databricks.sdk import WorkspaceClient
            _default_user = WorkspaceClient().current_user.me().user_name or _default_user
        except Exception:
            pass
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            f"/Users/{_default_user}/sec-financial-analyst-agent"
        )

        experiment = mlflow.set_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using MLflow experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    except Exception as e:
        print(f"Warning: Could not set up MLflow experiment: {e}")
        print("Tracing may not work correctly.")
        return None


EXPERIMENT_ID = setup_mlflow_experiment()

import agent_server.agent  # noqa: E402, F401

print(f"MLflow experiment set to: {EXPERIMENT_ID}")

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app


def main():
    if EXPERIMENT_ID:
        print(f"\n*** MLflow Experiment ID: {EXPERIMENT_ID} ***\n")

    agent_server.run(app_import_string="agent_server.start_server:app")


if __name__ == "__main__":
    main()
