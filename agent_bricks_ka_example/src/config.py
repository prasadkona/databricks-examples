"""
Configuration loader for Agent Bricks examples.

Loads environment variables from _local/.env for authentication.
Supports OAuth M2M (default) with PAT fallback.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_env_file(env_path: str = None, env_name: str = "e2-demo-field-eng") -> Dict[str, str]:
    """Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file. If None, searches for {env_name}.env or .env
        env_name: Environment name prefix (default: e2-demo-field-eng)
    
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Try multiple search paths to support running from different directories
        search_dirs = [
            Path("../../_local"),
            Path("../_local"),
            Path("_local"),
        ]
        
        # Also try relative to this file
        this_file = Path(__file__).resolve()
        project_root = this_file.parent.parent.parent
        search_dirs.append(project_root / "_local")
        
        for d in search_dirs:
            env_specific = d / f"{env_name}.env"
            if env_specific.exists():
                env_path = env_specific
                break
            generic = d / ".env"
            if generic.exists():
                env_path = generic
                break
    else:
        env_path = Path(env_path)
    
    config = {}
    
    if env_path and Path(env_path).exists():
        print(f"Loading config from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    else:
        print(f"Warning: Config file not found")
        print("Please copy .env.template to _local/{workspace}.env and fill in your credentials")
    
    return config


def setup_databricks_auth(config: Dict[str, str] = None) -> Tuple[Dict[str, str], str]:
    """Set up Databricks authentication environment variables.
    
    Uses OAuth M2M by default. Falls back to PAT if OAuth credentials not configured.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
    
    Returns:
        Tuple of (config dict, auth_type string)
        auth_type is either "oauth" or "pat"
    """
    if config is None:
        config = load_env_file()
    
    # Set host
    if 'DATABRICKS_HOST' in config:
        os.environ['DATABRICKS_HOST'] = config['DATABRICKS_HOST']
    
    # Check if OAuth credentials are configured
    has_oauth = (
        config.get('DATABRICKS_CLIENT_ID') and 
        config.get('DATABRICKS_CLIENT_SECRET')
    )
    
    # Check if PAT is configured
    has_pat = bool(config.get('DATABRICKS_TOKEN'))
    
    if has_oauth:
        # Use OAuth M2M (preferred)
        os.environ['DATABRICKS_CLIENT_ID'] = config['DATABRICKS_CLIENT_ID']
        os.environ['DATABRICKS_CLIENT_SECRET'] = config['DATABRICKS_CLIENT_SECRET']
        # Clear PAT to avoid conflicts
        os.environ.pop('DATABRICKS_TOKEN', None)
        return config, "oauth"
    elif has_pat:
        # Fall back to PAT
        os.environ['DATABRICKS_TOKEN'] = config['DATABRICKS_TOKEN']
        # Clear OAuth to avoid conflicts
        os.environ.pop('DATABRICKS_CLIENT_ID', None)
        os.environ.pop('DATABRICKS_CLIENT_SECRET', None)
        return config, "pat"
    else:
        print("Warning: No authentication configured.")
        print("Set DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (OAuth M2M)")
        print("Or set DATABRICKS_TOKEN (PAT)")
        return config, "none"


def get_workspace_client():
    """Create and return a WorkspaceClient with proper authentication.
    
    Returns:
        Tuple of (WorkspaceClient, config dict, auth_type string)
    """
    from databricks.sdk import WorkspaceClient
    
    config = load_env_file()
    config, auth_type = setup_databricks_auth(config)
    
    w = WorkspaceClient()
    
    print(f"Workspace: {w.config.host}")
    print(f"Auth: {auth_type.upper()}")
    
    return w, config, auth_type


def grant_volume_permissions(w, config: Dict[str, str], volume_path: str = None) -> bool:
    """Grant READ and WRITE permissions on a volume to the service principal.
    
    Only applicable when using OAuth M2M authentication.
    
    Args:
        w: WorkspaceClient instance
        config: Configuration dictionary
        volume_path: Volume path (e.g., /Volumes/catalog/schema/volume). 
                     If None, uses UC_VOLUME_PATH from config.
    
    Returns:
        True if successful or not needed, False on error
    """
    import requests
    
    # Get service principal ID
    sp_id = config.get('DATABRICKS_CLIENT_ID')
    if not sp_id:
        print("No service principal ID found - skipping volume grants")
        return True
    
    # Get volume path
    if volume_path is None:
        volume_path = config.get('UC_VOLUME_PATH')
    
    if not volume_path:
        print("No volume path configured - skipping volume grants")
        return True
    
    # Parse volume path: /Volumes/catalog/schema/volume
    parts = volume_path.strip('/').split('/')
    if len(parts) < 4 or parts[0].lower() != 'volumes':
        print(f"Invalid volume path format: {volume_path}")
        return False
    
    catalog = parts[1]
    schema = parts[2]
    volume = parts[3]
    volume_fqn = f"{catalog}.{schema}.{volume}"
    
    # Get warehouse ID for SQL execution
    warehouse_id = config.get('SQL_WAREHOUSE_ID')
    if not warehouse_id:
        print("No SQL_WAREHOUSE_ID configured - skipping volume grants")
        print("To grant permissions, add SQL_WAREHOUSE_ID to your env file")
        return True
    
    host = config.get('DATABRICKS_HOST', '').rstrip('/')
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    
    print(f"\nGranting volume permissions to service principal...")
    print(f"Volume: {volume_fqn}")
    print(f"Service Principal: {sp_id}")
    
    grants = ["READ_VOLUME", "WRITE_VOLUME"]
    success = True
    
    for grant in grants:
        sql = f"GRANT {grant} ON VOLUME {volume_fqn} TO `{sp_id}`"
        
        payload = {
            "warehouse_id": warehouse_id,
            "statement": sql,
            "wait_timeout": "30s"
        }
        
        try:
            response = requests.post(
                f"{host}/api/2.0/sql/statements",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', {}).get('state', '')
                if status == 'SUCCEEDED':
                    print(f"  Granted {grant}")
                elif status == 'FAILED':
                    error = result.get('status', {}).get('error', {}).get('message', 'Unknown error')
                    if 'already has' in error.lower() or 'already granted' in error.lower():
                        print(f"  {grant} already granted")
                    else:
                        print(f"  Failed to grant {grant}: {error}")
                        success = False
                else:
                    print(f"  {grant}: {status}")
            else:
                print(f"  Error granting {grant}: {response.status_code} - {response.text[:200]}")
                success = False
                
        except Exception as e:
            print(f"  Error granting {grant}: {e}")
            success = False
    
    return success


def get_volume_path(config: Dict[str, str] = None) -> str:
    """Get the volume path from config or construct from components.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
    
    Returns:
        Volume path string like /Volumes/catalog/schema/volume
    """
    if config is None:
        config = load_env_file()
    
    # Use explicit path if provided
    if 'UC_VOLUME_PATH' in config:
        return config['UC_VOLUME_PATH']
    
    # Construct from components
    catalog = config.get('UC_CATALOG', 'main')
    schema = config.get('UC_SCHEMA', 'default')
    volume = config.get('UC_VOLUME', 'ka_documents')
    
    return f"/Volumes/{catalog}/{schema}/{volume}"


# Module-level config (loaded on import)
_config: Optional[Dict[str, str]] = None
_auth_type: Optional[str] = None


def get_config() -> Tuple[Dict[str, str], str]:
    """Get the loaded configuration, loading it if necessary.
    
    Returns:
        Tuple of (config dict, auth_type string)
    """
    global _config, _auth_type
    if _config is None:
        _config = load_env_file()
        _config, _auth_type = setup_databricks_auth(_config)
    return _config, _auth_type
