"""
Ollama Client Module for Experiment 4
Course: 24CS2506 – Building Applications using GPT-4
Experiment 4: Personalized Recommendation System
"""

import json
import requests
from typing import Dict, Any, Optional, List


class OllamaClient:
    """
    Client for interacting with local Ollama API (http://localhost:11434).
    Supports generation and model querying for Llama 3.2, Llama 3, or Mistral.
    """
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._last_conn_check = 0.0
        self._conn_cached = False

    def check_connection(self, force: bool = False) -> bool:
        """Check if local Ollama daemon is reachable."""
        import time
        now = time.time()
        if not force and (now - self._last_conn_check < 4.0):
            return self._conn_cached

        self._last_conn_check = now
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            self._conn_cached = (resp.status_code == 200)
            return self._conn_cached
        except Exception:
            self._conn_cached = False
            return False

    @property
    def is_connected(self) -> bool:
        return self.check_connection()

    def list_models(self) -> List[str]:
        """List available local models in Ollama."""
        if not self.check_connection():
            return []
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return [m for m in models if m]
            return []
        except Exception:
            return []

    def get_best_available_model(self) -> str:
        """Return llama3.2 if available, or first available model, or default."""
        models = self.list_models()
        if not models:
            return self.default_model
        for m in models:
            if "llama3.2" in m.lower():
                return m
        for m in models:
            if any(k in m.lower() for k in ["llama", "qwen", "mistral", "phi", "gemma"]):
                return m
        return models[0]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
        format_json: bool = False
    ) -> Optional[str]:
        """Send generation request to Ollama."""
        if not self.check_connection():
            return None

        chosen_model = model or self.default_model
        payload = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024
            }
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=45.0
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("response", "").strip()
            return None
        except Exception:
            return None
