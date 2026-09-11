from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from . import __version__
from .tools import Tool, ToolRegistry


class MCPClient:
    def __init__(self, name: str, config: dict[str, Any], workspace: Path):
        command = config.get("command")
        args = config.get("args", [])
        if not isinstance(command, str) or not command or not isinstance(args, list):
            raise ValueError(f"MCP server {name}: command and args are required")
        env = os.environ.copy()
        configured_env = config.get("env", {})
        if not isinstance(configured_env, dict):
            raise ValueError(f"MCP server {name}: env must be an object")
        env.update({str(k): str(v) for k, v in configured_env.items()})
        self.name = name
        self.next_id = 1
        executable = shutil.which(command) or command
        self.process = subprocess.Popen(
            [executable, *map(str, args)], cwd=workspace, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read_lines, daemon=True).start()
        try:
            self.request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "chattyplay", "version": __version__},
            })
            self.notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def _read_lines(self) -> None:
        if self.process.stdout:
            for line in self.process.stdout:
                self.lines.put(line)

    def _send(self, message: dict[str, Any]) -> None:
        if not self.process.stdin:
            raise RuntimeError("MCP stdin is closed")
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            try:
                line = self.lines.get(timeout=15)
            except queue.Empty as exc:
                raise TimeoutError(f"MCP server {self.name} did not respond in 15s") from exc
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            result = message.get("result", {})
            return result if isinstance(result, dict) else {"value": result}

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list", {}).get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


class MCPManager:
    def __init__(self):
        self.clients: list[MCPClient] = []
        self.status: dict[str, str] = {}

    def connect(self, configs: dict[str, Any], registry: ToolRegistry, workspace: Path) -> None:
        for server_name, config in configs.items():
            if not isinstance(config, dict) or config.get("disabled") is True:
                self.status[server_name] = "disabled"
                continue
            try:
                client = MCPClient(server_name, config, workspace)
                self.clients.append(client)
                remote_tools = client.list_tools()
                for remote in remote_tools:
                    original = str(remote.get("name", ""))
                    safe_server = re.sub(r"\W+", "_", server_name)
                    safe_tool = re.sub(r"\W+", "_", original)
                    local_name = f"mcp__{safe_server}__{safe_tool}"
                    schema = remote.get("inputSchema") or {"type": "object", "properties": {}}
                    def call(args: dict[str, Any], client: MCPClient = client, tool_name: str = original) -> dict[str, Any]:
                        return client.call_tool(tool_name, args)
                    registry.add(Tool(
                        local_name, str(remote.get("description", f"MCP tool {original}")), schema, "mcp",
                        call,
                    ))
                self.status[server_name] = f"connected ({len(remote_tools)} tools)"
            except Exception as exc:
                self.status[server_name] = f"error: {exc}"

    def close(self) -> None:
        for client in self.clients:
            client.close()
