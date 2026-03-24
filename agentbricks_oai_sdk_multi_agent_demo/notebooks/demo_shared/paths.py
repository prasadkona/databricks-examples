"""
Centralized path resolution helpers.

Provides reliable path resolution regardless of where scripts are located,
avoiding issues when files are moved between folders.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Get project root (agentbricks_custom_agent/) regardless of caller location.
    
    Walks up the directory tree until it finds a directory containing both
    pyproject.toml and an app/ subdirectory.
    
    Returns:
        Path to the project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "app").is_dir():
            return parent
    raise RuntimeError(
        "Could not find project root. Expected directory with pyproject.toml and app/ folder."
    )


def get_app_dir() -> Path:
    """Get the app/ directory containing the Databricks App.
    
    Returns:
        Path to the app/ directory
        
    Raises:
        FileNotFoundError: If app/ directory doesn't exist or is missing required files
    """
    app_dir = get_project_root() / "app"
    if not app_dir.is_dir():
        raise FileNotFoundError(f"app/ directory not found at {app_dir}")
    if not (app_dir / "pyproject.toml").exists():
        raise FileNotFoundError(f"app/pyproject.toml not found at {app_dir}")
    if not (app_dir / "databricks.yml").exists():
        raise FileNotFoundError(f"app/databricks.yml not found at {app_dir}")
    return app_dir


def get_notebooks_dir() -> Path:
    """Get the notebooks/ directory.
    
    Returns:
        Path to the notebooks/ directory
    """
    return get_project_root() / "notebooks"


def get_bundle_workspace_path(app_name: str = "sec_financial_analyst_agent", target: str = "dev") -> str:
    """Get the workspace path where bundle files are deployed.
    
    Args:
        app_name: Bundle name (default: sec_financial_analyst_agent)
        target: Deployment target (default: dev)
        
    Returns:
        Workspace path string like /Workspace/Users/user@domain/.bundle/app_name/target/files
    """
    import os
    user = os.environ.get("DATABRICKS_USER", "")
    if not user:
        from databricks.sdk import WorkspaceClient
        try:
            w = WorkspaceClient()
            user = w.current_user.me().user_name
        except Exception:
            user = "unknown"
    return f"/Workspace/Users/{user}/.bundle/{app_name}/{target}/files"
