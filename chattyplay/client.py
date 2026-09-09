from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable


class ModelError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, provider: dict[str, Any]):
        self.provider = provider

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        env_name = self.provider["api_key_env"]
        api_key = os.environ.get(env_name, "") if env_name else ""
        if env_name and not api_key:
            raise ModelError(f"Missing API key: set environment variable {env_name}")
        payload: dict[str, Any] = {
            "model": self.provider["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": int(self.provider.get("max_tokens", 8192)),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.provider.get("temperature") is not None:
            payload["temperature"] = float(self.provider["temperature"])
        if self.provider.get("thinking_enabled"):
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = str(self.provider.get("reasoning_effort", "medium"))
        url = self.provider["base_url"].rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    data = json.loads(response.read().decode("utf-8"))
                    message = data.get("choices", [{}])[0].get("message", {})
                    if not message:
                        raise ModelError("Model returned an empty response")
                    text = message.get("content") or ""
                    if text and on_text:
                        on_text(text)
                    message["_usage"] = data.get("usage") or {}
                    return message
                return self._read_stream(response, on_text)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode("utf-8", errors="replace")
            raise ModelError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"Cannot reach model API: {exc.reason}") from exc

    @staticmethod
    def _read_stream(response: Any, on_text: Callable[[str], None] | None) -> dict[str, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage.update(event["usage"])
                delta = event["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            text = delta.get("content") or ""
            if text:
                content.append(text)
                if on_text:
                    on_text(text)
            thought = delta.get("reasoning_content") or ""
            if thought:
                reasoning.append(thought)
            for part in delta.get("tool_calls") or []:
                index = int(part.get("index", 0))
                call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] += part.get("id") or ""
                function = part.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        if usage:
            message["_usage"] = usage
        if not message.get("content") and not message.get("tool_calls"):
            raise ModelError("Model returned an empty response")
        return message


class AnthropicClient:
    def __init__(self, provider: dict[str, Any]):
        self.provider = provider

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        env_name = self.provider["api_key_env"]
        api_key = os.environ.get(env_name, "") if env_name else ""
        if not env_name or not api_key:
            raise ModelError(f"Missing API key: set environment variable {env_name}")
        system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        payload = {
            "model": self.provider["model"],
            "system": system,
            "messages": self._messages([m for m in messages if m.get("role") != "system"]),
            "max_tokens": int(self.provider.get("max_tokens", 8192)),
            "stream": True,
        }
        if tools:
            payload["tools"] = [self._tool(tool) for tool in tools]
        if self.provider.get("temperature") is not None:
            payload["temperature"] = float(self.provider["temperature"])
        request = urllib.request.Request(
            self.provider["base_url"].rstrip("/") + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": str(self.provider.get("anthropic_version", "2023-06-01")),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    return self._response(json.loads(response.read().decode("utf-8")), on_text)
                return self._read_stream(response, on_text)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode("utf-8", errors="replace")
            raise ModelError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"Cannot reach model API: {exc.reason}") from exc

    @staticmethod
    def _tool(tool: dict[str, Any]) -> dict[str, Any]:
        function = tool["function"]
        return {"name": function["name"], "description": function.get("description", ""), "input_schema": function["parameters"]}

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                item = {"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""), "content": message.get("content", "")}
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(item)
                else:
                    converted.append({"role": "user", "content": [item]})
                continue
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    function = call.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    blocks.append({"type": "tool_use", "id": call.get("id", ""), "name": function.get("name", ""), "input": arguments})
                converted.append({"role": "assistant", "content": blocks})
            elif role == "user" and isinstance(message.get("content"), list):
                blocks = []
                for block in message["content"]:
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": str(block.get("text", ""))})
                        continue
                    url = (block.get("image_url") or {}).get("url", "")
                    match = re.fullmatch(r"data:(image/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/=]+)", str(url))
                    if match:
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}})
                converted.append({"role": "user", "content": blocks})
            elif role in {"user", "assistant"}:
                converted.append({"role": role, "content": message.get("content") or ""})
        return converted

    @staticmethod
    def _response(data: dict[str, Any], on_text: Callable[[str], None] | None) -> dict[str, Any]:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                texts.append(text)
                if text and on_text:
                    on_text(text)
            elif block.get("type") == "tool_use":
                calls.append({"id": block.get("id", ""), "type": "function", "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}})
        message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
        if calls:
            message["tool_calls"] = calls
        if isinstance(data.get("usage"), dict):
            message["_usage"] = data["usage"]
        if not message.get("content") and not calls:
            raise ModelError("Model returned an empty response")
        return message

    @staticmethod
    def _read_stream(response: Any, on_text: Callable[[str], None] | None) -> dict[str, Any]:
        texts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "error":
                raise ModelError(str(event.get("error", "Anthropic stream error")))
            if event_type == "message_start" and isinstance((event.get("message") or {}).get("usage"), dict):
                usage.update(event["message"]["usage"])
            if event_type == "message_delta" and isinstance(event.get("usage"), dict):
                usage.update(event["usage"])
            if event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    calls[int(event.get("index", 0))] = {"id": block.get("id", ""), "name": block.get("name", ""), "arguments": ""}
            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    texts.append(text)
                    if text and on_text:
                        on_text(text)
                elif delta.get("type") == "input_json_delta":
                    calls.setdefault(int(event.get("index", 0)), {"id": "", "name": "", "arguments": ""})["arguments"] += delta.get("partial_json", "")
        message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
        if calls:
            message["tool_calls"] = [
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"] or "{}"}}
                for _, call in sorted(calls.items())
            ]
        if usage:
            message["_usage"] = usage
        if not message.get("content") and not calls:
            raise ModelError("Model returned an empty response")
        return message


def create_client(provider: dict[str, Any]) -> OpenAIClient | AnthropicClient:
    return AnthropicClient(provider) if provider.get("api_style") == "anthropic" else OpenAIClient(provider)
