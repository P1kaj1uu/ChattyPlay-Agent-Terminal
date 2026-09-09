from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any


CHAT_MODEL = "qwen3.5:9b-q4_K_M"
EMBED_MODEL = "embeddinggemma"
BASE_URL = "http://127.0.0.1:11434"


def status(base_url: str = BASE_URL) -> dict[str, Any]:
    result: dict[str, Any] = {"installed": bool(shutil.which("ollama")), "running": False, "models": []}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=2) as response:
            data = json.load(response)
        result["running"] = True
        result["models"] = [item.get("name", "") for item in data.get("models", [])]
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return result


def pull(model: str) -> None:
    models = status().get("models", [])
    if model in models or (":" not in model and f"{model}:latest" in models):
        return
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("Ollama is not installed; see https://ollama.com/download")
    subprocess.run([executable, "pull", model], check=True)


def provider(model: str = CHAT_MODEL) -> dict[str, Any]:
    return {"api_style": "openai", "base_url": BASE_URL + "/v1", "api_key_env": "", "model": model}
