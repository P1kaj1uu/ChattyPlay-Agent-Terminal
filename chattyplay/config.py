from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "provider": {
        "api_style": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    "models": [
        {"name": "gpt-4.1-mini", "label": "GPT-4.1 mini"},
        {"name": "qwen3.5:9b-q4_K_M", "label": "Qwen3.5 9B Q4 (local, recommended)"},
        {"name": "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M", "label": "MiniCPM5 2B Q4 (local)"},
    ],
    "providerProfiles": {
        "ollama-coder": {
            "api_style": "openai",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key_env": "",
            "model": "qwen3.5:9b-q4_K_M",
        },
    },
    "permissions": {
        "read": "allow",
        "write": "ask",
        "shell": "ask",
        "browser": "ask",
        "clipboard": "ask",
        "mcp": "ask",
        "interaction": "allow",
        "delegate": "ask",
    },
    "skills": {"enabled": [], "dirs": [".chattyplay/skills", ".agents/skills"]},
    "mcpServers": {},
    "rag": {
        "enabled": True,
        "base_url": "http://127.0.0.1:11434",
        "embedding_model": "embeddinggemma",
        "top_k": 6,
        "chunk_lines": 80,
        "overlap_lines": 10,
    },
    "agent": {"max_steps": 30, "max_tool_output": 30000, "max_context_chars": 500000, "max_file_mention_chars": 30000, "max_image_bytes": 5000000, "max_images": 4},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config {path}: root must be an object")
    return data


class ConfigStore:
    def __init__(self, workspace: Path | str = ".", user_path: Path | None = None):
        self.workspace = Path(workspace).resolve()
        self.user_path = user_path or Path.home() / ".chattyplay" / "config.json"
        self.project_path = self.workspace / ".chattyplay" / "config.json"
        self._write_lock = Lock()

    def load(self) -> dict[str, Any]:
        config = _merge(_merge(DEFAULT_CONFIG, _read_json(self.user_path)), _read_json(self.project_path))
        self._validate(config)
        return config

    def project_config(self) -> dict[str, Any]:
        return _read_json(self.project_path)

    def save_project(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("Config root must be an object")
        self._validate(data)
        with self._write_lock:
            self.project_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.project_path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temp, self.project_path)

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        merged = _merge(DEFAULT_CONFIG, data)
        provider = merged["provider"]
        if not isinstance(provider, dict):
            raise ValueError("provider must be an object")
        for key in ("base_url", "model"):
            if not isinstance(provider.get(key), str) or not provider[key].strip():
                raise ValueError(f"provider.{key} must be a non-empty string")
        if not isinstance(provider.get("api_key_env"), str):
            raise ValueError("provider.api_key_env must be a string")
        if provider.get("api_style") not in {"openai", "anthropic"}:
            raise ValueError("provider.api_style must be openai or anthropic")
        if not isinstance(provider.get("thinking_enabled", False), bool):
            raise ValueError("provider.thinking_enabled must be boolean")
        if provider.get("reasoning_effort", "medium") not in {"low", "medium", "high", "max"}:
            raise ValueError("provider.reasoning_effort must be low, medium, high, or max")
        if provider.get("api_style") == "anthropic" and not provider["api_key_env"].strip():
            raise ValueError("Anthropic requires provider.api_key_env")
        if not str(provider["base_url"]).startswith(("http://", "https://")):
            raise ValueError("provider.base_url must be an HTTP(S) URL")
        permissions = merged.get("permissions")
        if not isinstance(permissions, dict):
            raise ValueError("permissions must be an object")
        for name, policy in permissions.items():
            if policy not in {"allow", "ask", "deny"}:
                raise ValueError(f"permissions.{name} must be allow, ask, or deny")
        if not isinstance(merged.get("mcpServers"), dict):
            raise ValueError("mcpServers must be an object")
        profiles = merged.get("providerProfiles")
        if not isinstance(profiles, dict) or any(not isinstance(name, str) or not isinstance(value, dict) for name, value in profiles.items()):
            raise ValueError("providerProfiles must map names to provider objects")
        for name, profile in profiles.items():
            resolved = {**provider, **profile}
            if resolved.get("api_style") not in {"openai", "anthropic"}:
                raise ValueError(f"providerProfiles.{name}.api_style is invalid")
            if any(not isinstance(resolved.get(key), str) or not resolved[key].strip() for key in ("base_url", "model")):
                raise ValueError(f"providerProfiles.{name} requires base_url and model")
            if not isinstance(resolved.get("api_key_env"), str):
                raise ValueError(f"providerProfiles.{name}.api_key_env must be a string")
        if not isinstance(merged.get("skills"), dict):
            raise ValueError("skills must be an object")
        if not isinstance(merged.get("agent"), dict):
            raise ValueError("agent must be an object")
        rag = merged.get("rag")
        if not isinstance(rag, dict) or not isinstance(rag.get("enabled"), bool):
            raise ValueError("rag must be an object and rag.enabled must be boolean")
        if not isinstance(rag.get("base_url"), str) or not rag["base_url"].startswith(("http://", "https://")):
            raise ValueError("rag.base_url must be an HTTP(S) URL")
        if not isinstance(rag.get("embedding_model"), str) or not rag["embedding_model"].strip():
            raise ValueError("rag.embedding_model must be a non-empty string")
        try:
            if int(provider.get("max_tokens", 0)) <= 0:
                raise ValueError
            if int(merged["agent"].get("max_steps", 0)) <= 0:
                raise ValueError
            if int(merged["agent"].get("max_tool_output", 0)) <= 0:
                raise ValueError
            if int(merged["agent"].get("max_context_chars", 0)) < 1000:
                raise ValueError
            if int(merged["agent"].get("max_file_mention_chars", 0)) <= 0:
                raise ValueError
            if int(merged["agent"].get("max_image_bytes", 0)) <= 0 or int(merged["agent"].get("max_images", 0)) <= 0:
                raise ValueError
            if int(rag.get("top_k", 0)) <= 0 or int(rag.get("chunk_lines", 0)) <= 0:
                raise ValueError
            if not 0 <= int(rag.get("overlap_lines", -1)) < int(rag["chunk_lines"]):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric limits must be valid positive integers") from exc
        skill_config = merged["skills"]
        if not isinstance(skill_config.get("enabled", []), list) or not isinstance(skill_config.get("dirs", []), list):
            raise ValueError("skills.enabled and skills.dirs must be arrays")
