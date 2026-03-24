# Databricks notebook source
# MAGIC %md
# MAGIC # AgentBricksManager
# MAGIC
# MAGIC Reusable wrapper for Databricks Knowledge Assistant REST APIs.
# MAGIC
# MAGIC Provides CRUD operations for Knowledge Assistants:
# MAGIC - Create, get, update, delete KA tiles
# MAGIC - Wait for endpoint to come ONLINE
# MAGIC - Sync (re-index) knowledge sources
# MAGIC - Create/list example questions
# MAGIC - Find KA by name
# MAGIC - Query a KA endpoint
# MAGIC
# MAGIC This module is imported by the step scripts; it is not executed directly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import re
import time
import requests
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## AgentBricksManager Class

# COMMAND ----------

class AgentBricksManager:
    """Unified wrapper for Agent Bricks KA operations via REST API."""

    def __init__(self, client: Optional[WorkspaceClient] = None):
        self.w = client or WorkspaceClient()

    @staticmethod
    def sanitize_name(name: str) -> str:
        sanitized = name.replace(" ", "_")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
        sanitized = re.sub(r"[_-]{2,}", "_", sanitized)
        return sanitized.strip("_-") or "knowledge_assistant"

# COMMAND ----------

# MAGIC %md
# MAGIC ### HTTP Helpers
# MAGIC
# MAGIC Low-level methods for authenticated GET / POST / PATCH / DELETE
# MAGIC against the Databricks REST API.

# COMMAND ----------

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _get_headers(self) -> dict:
        headers = self.w.config.authenticate()
        headers["Content-Type"] = "application/json"
        return headers

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.w.config.host}{path}"
        resp = requests.get(url, headers=self._get_headers(), params=params or {}, timeout=30)
        if resp.status_code >= 400:
            raise Exception(f"GET {path} failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def _post(self, path: str, body: dict, timeout: int = 300) -> dict:
        url = f"{self.w.config.host}{path}"
        resp = requests.post(url, headers=self._get_headers(), json=body, timeout=timeout)
        if resp.status_code >= 400:
            raise Exception(f"POST {path} failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def _patch(self, path: str, body: dict) -> dict:
        url = f"{self.w.config.host}{path}"
        resp = requests.patch(url, headers=self._get_headers(), json=body, timeout=30)
        if resp.status_code >= 400:
            raise Exception(f"PATCH {path} failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def _delete(self, path: str) -> dict:
        url = f"{self.w.config.host}{path}"
        resp = requests.delete(url, headers=self._get_headers(), timeout=30)
        if resp.status_code >= 400 and resp.status_code != 404:
            raise Exception(f"DELETE {path} failed ({resp.status_code}): {resp.text}")
        return resp.json() if resp.text else {}

# COMMAND ----------

# MAGIC %md
# MAGIC ### KA Lifecycle
# MAGIC
# MAGIC Create, get, update, and wait for Knowledge Assistants.

# COMMAND ----------

    def ka_create(
        self,
        name: str,
        knowledge_sources: List[Dict[str, Any]],
        description: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
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
        try:
            return self._get(f"/api/2.0/knowledge-assistants/{tile_id}")
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            raise

    def ka_get_endpoint_status(self, tile_id: str) -> Optional[str]:
        ka = self.ka_get(tile_id)
        if not ka:
            return None
        return ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")

    def ka_wait_until_endpoint_online(
        self, tile_id: str, timeout_s: int = 600, poll_s: float = 15.0
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        start = time.time()
        last_status = None
        while True:
            ka = self.ka_get(tile_id)
            status = ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
            elapsed = int(time.time() - start)
            if status != last_status:
                print(f"[{elapsed}s] KA endpoint status: {status}")
                last_status = status
            else:
                print(f"[{elapsed}s] Still waiting... (status: {status})")
            if status == "ONLINE":
                return ka
            if time.time() >= deadline:
                host = self.w.config.host.rstrip("/")
                print(f"\n{'!' * 60}")
                print(f"Timeout after {timeout_s}s -- endpoint provisioning is still in progress.")
                print(f"Last status: {status}")
                print(f"\nThe KA was created successfully but the endpoint is not yet ONLINE.")
                print(f"Knowledge source sync and endpoint provisioning can take 10-20+ minutes.")
                print(f"\nCheck endpoint status in the Databricks UI:")
                print(f"  {host}/ml/endpoints/{tile_id}")
                print(f"\nOr run:  uv run sync-ka     (to check sync status)")
                print(f"         uv run test-ka     (to test once ONLINE)")
                print(f"{'!' * 60}")
                return ka
            time.sleep(poll_s)

    def ka_update(
        self,
        tile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if instructions is not None:
            body["instructions"] = instructions
        if body:
            self._patch(f"/api/2.0/knowledge-assistants/{tile_id}", body)
        return self.ka_get(tile_id)

    def ka_create_or_update(
        self,
        name: str,
        knowledge_sources: List[Dict[str, Any]],
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = self.ka_get(tile_id) if tile_id else None
        if existing:
            result = self.ka_update(tile_id, name=self.sanitize_name(name),
                                    description=description, instructions=instructions)
            result["operation"] = "updated"
        else:
            result = self.ka_create(name, knowledge_sources, description, instructions)
            result["operation"] = "created"
        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Knowledge Source Sync
# MAGIC
# MAGIC Trigger re-indexing after document changes in the UC Volume.

# COMMAND ----------

    def ka_sync_sources(self, tile_id: str) -> None:
        self._post(f"/api/2.0/knowledge-assistants/{tile_id}/sync-knowledge-sources", {})
        print(f"Triggered sync for KA {tile_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Questions
# MAGIC
# MAGIC Add example questions with optional guidelines to improve KA response quality.

# COMMAND ----------

    def ka_create_example(
        self, tile_id: str, question: str, guidelines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"tile_id": tile_id, "question": question}
        if guidelines:
            payload["guidelines"] = guidelines
        return self._post(f"/api/2.0/knowledge-assistants/{tile_id}/examples", payload)

    def ka_add_examples_batch(
        self, tile_id: str, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []

        def _create(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            text = q.get("question", "")
            guideline = q.get("guideline")
            guidelines = [guideline] if guideline else None
            if not text:
                return None
            try:
                ex = self.ka_create_example(tile_id, text, guidelines)
                print(f"  Added: {text[:60]}...")
                return ex
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"  Skipped (exists): {text[:60]}...")
                    return None
                print(f"  Failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_create, q): q for q in questions}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    created.append(r)
        return created

    def ka_list_examples(self, tile_id: str, page_size: int = 100) -> Dict[str, Any]:
        return self._get(f"/api/2.0/knowledge-assistants/{tile_id}/examples",
                         {"page_size": page_size})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query a KA Endpoint

# COMMAND ----------

    def ka_query(self, endpoint_name: str, messages: list) -> Dict[str, Any]:
        """Query a KA serving endpoint. ``messages`` is a list of role/content dicts."""
        return self._post(f"/serving-endpoints/{endpoint_name}/invocations",
                          {"input": messages})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discovery and Deletion

# COMMAND ----------

    def find_by_name(self, name: str) -> Optional[Dict[str, str]]:
        page_token = None
        while True:
            params: Dict[str, Any] = {"filter": f"name_contains={name}&&tile_type=KA"}
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
        all_kas: List[Dict[str, Any]] = []
        page_token = None
        while True:
            params: Dict[str, Any] = {"filter": "tile_type=KA", "page_size": 100}
            if page_token:
                params["page_token"] = page_token
            resp = self._get("/api/2.0/tiles", params=params)
            all_kas.extend(resp.get("tiles", []))
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return all_kas

    def delete(self, tile_id: str) -> None:
        self._delete(f"/api/2.0/tiles/{tile_id}")
        print(f"Deleted tile: {tile_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Static Helpers

# COMMAND ----------

    @staticmethod
    def get_knowledge_sources_from_volumes(
        volume_paths: List[Tuple[str, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """Convert (volume_path, description) tuples to knowledge source dicts."""
        sources: List[Dict[str, Any]] = []
        for idx, (vol_path, _desc) in enumerate(volume_paths):
            parts = vol_path.rstrip("/").split("/")
            src_name = (parts[-1] if parts else f"source_{idx + 1}").replace(" ", "_").replace(".", "_")
            sources.append({
                "files_source": {
                    "name": src_name,
                    "type": "files",
                    "files": {"path": vol_path},
                }
            })
        return sources

    @staticmethod
    def extract_response_text(response: dict) -> str:
        """Extract text content from a KA endpoint response."""
        if not response or "output" not in response:
            return ""
        for item in response.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
        return ""
