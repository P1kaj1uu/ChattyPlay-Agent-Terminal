from __future__ import annotations

import base64
import json
import platform
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .client import create_client
from .mcp import MCPManager
from .rag import RAGIndex
from .sessions import SessionStore
from .skills import discover, enabled_prompt
from .tools import Tool, ToolRegistry
from .wiki import ProjectWiki


BASE_PROMPT = """You are ChattyPlay, a local AI coding agent.
Work directly in the current workspace and finish the user's task end to end.
Inspect existing code before editing. Prefer small root-cause fixes and existing project patterns.
Use tools whenever facts depend on local files or commands. Never invent tool results.
Keep edits inside the workspace. Run the smallest relevant verification after non-trivial edits.
Do not perform destructive, irreversible, privileged, or externally visible actions without explicit user approval.
When done, summarize the result and verification concisely.
When activate_tools is the only available tool, call it before answering any request that needs current workspace facts or actions.
"""

ACTIVATE_TOOLS = {
    "type": "function",
    "function": {
        "name": "activate_tools",
        "description": "Enable repository inspection, editing, shell, browser, RAG, Wiki, Skills, delegation, and MCP tools. Call before using workspace facts or taking actions; skip for greetings and self-contained questions.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def _project_instructions(workspace: Path) -> str:
    path = (workspace / "AGENTS.md").resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:30000]


class Agent:
    def __init__(
        self,
        workspace: Path,
        config: dict[str, Any],
        confirm: Callable[[str, str], bool | str] | None = None,
        ask_user: Callable[[str], str] | None = None,
        on_plan: Callable[[list[dict[str, Any]]], None] | None = None,
        session_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        depth: int = 0,
    ):
        self.workspace = workspace.resolve()
        self.depth = depth
        self.config = config
        self.sessions = SessionStore(self.workspace)
        self.session_id = session_id or self.sessions.new_id()
        self.messages = messages or []
        self.turns: list[tuple[int, int]] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "requests": 0}
        self.last_output_tokens = 0
        agent_config = config.get("agent", {})
        self.registry = ToolRegistry(
            self.workspace,
            permissions=config.get("permissions", {}),
            confirm=confirm,
            ask_user=ask_user,
            on_plan=on_plan,
            max_output=int(agent_config.get("max_tool_output", 30000)),
        )
        self.mcp = MCPManager()
        mcp_configs = config.get("mcpServers", {})
        enabled_mcp = [name for name, value in mcp_configs.items() if not isinstance(value, dict) or value.get("disabled") is not True]
        mcp_policy = self.registry.permissions.get("mcp", "ask")
        mcp_decision: bool | str = mcp_policy == "allow"
        if enabled_mcp and mcp_policy == "ask":
            mcp_decision = self.registry.confirm("mcp", "start servers: " + ", ".join(enabled_mcp)) if self.registry.confirm else False
            if mcp_decision == "always":
                self.registry.always_allowed.add("mcp")
        if mcp_decision:
            self.mcp.connect(mcp_configs, self.registry, self.workspace)
        else:
            self.mcp.status.update({
                name: "disabled" if isinstance(value, dict) and value.get("disabled") is True else "permission denied"
                for name, value in mcp_configs.items()
            })
        self.rag = RAGIndex(self.workspace, config.get("rag", {}))
        self.wiki = ProjectWiki(self.workspace)
        self.registry.add(Tool(
            "read_project_wiki",
            "Read the LLM-compiled project wiki for architecture, components, workflows, configuration, and known risks. Prefer it for broad project questions; verify exact code with file or RAG tools.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            "read",
            lambda _: self.wiki.read(),
        ))
        if config.get("rag", {}).get("enabled", True):
            self.registry.add(Tool(
                "search_codebase",
                "Semantically search the local project RAG index. Use this when keywords are unknown or concepts span files.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "read",
                lambda args: self.rag.search(str(args["query"]), args.get("top_k")),
            ))
        skill_config = config.get("skills", {})
        self.skills = discover(self.workspace, skill_config.get("dirs", []))
        if depth < 1:
            self.registry.add(Tool(
                "delegate_task",
                "Run an independent read-only sub-agent for focused codebase exploration or review. It cannot write, use shell, network, or MCP.",
                {
                    "type": "object",
                    "properties": {"task": {"type": "string"}, "skill": {"type": "string"}},
                    "required": ["task"],
                    "additionalProperties": False,
                },
                "delegate",
                self._delegate,
            ))
        skill_text = enabled_prompt(self.skills, skill_config.get("enabled", []))
        project_text = _project_instructions(self.workspace)
        self.system = BASE_PROMPT + (
            f"\nWorkspace: {self.workspace}\nPlatform: {platform.system()}\n"
            + (f"\n# Project instructions (AGENTS.md)\n{project_text}\n" if project_text else "")
            + (f"\n{skill_text}\n" if skill_text else "")
        )
        self.client = create_client(config["provider"])
        self.plan_mode = False

    def run(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict[str, Any]], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        content = self._attach_mentions(prompt)
        self.turns.append((self.registry.checkpoint(), len(self.messages)))
        self.messages.append({"role": "user", "content": content})
        max_steps = min(100, max(1, int(self.config.get("agent", {}).get("max_steps", 30))))
        final = ""
        tools_active = False
        try:
            for _ in range(max_steps):
                system = self.system
                if self.plan_mode:
                    system += "\nPLAN MODE: Inspect and reason only. Do not modify files, run shell commands, access networks, or call MCP. Return an implementation plan.\n"
                request_messages = [{"role": "system", "content": system}, *self.context_messages()]
                if on_status:
                    on_status("thinking")
                message = self.client.complete(request_messages, self.registry.schemas() if tools_active else [ACTIVATE_TOOLS], on_text)
                self._track_usage(message)
                self.messages.append(message)
                final = message.get("content") or final
                calls = message.get("tool_calls") or []
                self.usage["tool_calls"] += len(calls)
                if not calls:
                    self.sessions.save(self.session_id, self.messages)
                    return final
                gate_phase = not tools_active
                if gate_phase:
                    tools_active = True
                for call in calls:
                    function = call.get("function") or {}
                    name = str(function.get("name", ""))
                    if gate_phase:
                        args = {}
                        result = "Tools enabled. Retry the task using the newly available tools."
                    else:
                        try:
                            args = json.loads(function.get("arguments") or "{}")
                        except json.JSONDecodeError as exc:
                            result = f"error: invalid tool arguments: {exc}"
                            args = {}
                        else:
                            if on_tool:
                                on_tool(name, args)
                            result = self.registry.execute(name, args)
                    self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
                self.sessions.save(self.session_id, self.messages)
            final = f"Stopped after {max_steps} tool steps. Ask me to continue if needed."
            self.messages.append({"role": "assistant", "content": final})
            self.sessions.save(self.session_id, self.messages)
            return final
        except Exception:
            self.sessions.save(self.session_id, self.messages)
            raise

    def undo(self) -> str:
        if not self.turns:
            return "error: nothing to undo in this process"
        checkpoint, message_index = self.turns[-1]
        result = self.registry.undo(checkpoint)
        if result.startswith("error:"):
            return result
        self.turns.pop()
        del self.messages[message_index:]
        self.sessions.save(self.session_id, self.messages)
        return result

    def context_messages(self) -> list[dict[str, Any]]:
        limit = int(self.config.get("agent", {}).get("max_context_chars", 500000))
        turns: list[list[dict[str, Any]]] = []
        for message in self.messages:
            if message.get("role") == "user":
                turns.append([message])
            elif not turns:
                turns.append([message])
            else:
                turns[-1].append(message)
        selected: list[list[dict[str, Any]]] = []
        used = 0
        for turn in reversed(turns):
            size = len(json.dumps(turn, ensure_ascii=False))
            if selected and used + size > limit:
                break
            selected.append(turn)
            used += size
        return [message for turn in reversed(selected) for message in turn]

    def context_status(self) -> str:
        active = self.context_messages()
        chars = len(json.dumps(active, ensure_ascii=False))
        return f"{len(active)}/{len(self.messages)} messages · about {chars} chars in model context"

    def stats(self) -> str:
        return (
            f"requests: {self.usage['requests']} · tools: {self.usage['tool_calls']} · "
            f"input tokens: {self.usage['input_tokens']} · output tokens: {self.usage['output_tokens']}"
        )

    def set_plan_mode(self, enabled: bool) -> None:
        self.plan_mode = enabled
        self.registry.plan_mode = enabled

    def set_provider(self, provider: dict[str, Any]) -> None:
        self.config["provider"] = provider
        self.client = create_client(provider)

    def fork(self) -> str:
        self.session_id = self.sessions.new_id()
        self.turns.clear()
        self.sessions.save(self.session_id, self.messages)
        return self.session_id

    def compact(self) -> str:
        if len(self.messages) < 4:
            return "error: conversation is already compact"
        source = json.dumps(self.context_messages(), ensure_ascii=False)
        prompt = (
            "Summarize this coding session for another agent. Preserve user requirements, decisions, "
            "files changed, commands/results, unresolved problems, and next steps. Be concise and factual.\n\n" + source
        )
        response = self.client.complete(
            [{"role": "system", "content": "You create loss-minimizing coding-session summaries."}, {"role": "user", "content": prompt}],
            [],
        )
        self._track_usage(response)
        summary = response.get("content")
        if not summary:
            return "error: model returned no summary"
        self.messages = [
            {"role": "user", "content": "Previous session summary:\n" + summary},
            {"role": "assistant", "content": "Understood. I will continue from this summary."},
        ]
        self.turns.clear()
        self.sessions.save(self.session_id, self.messages)
        return f"ok: compacted conversation to {len(summary)} characters"

    def build_wiki(self) -> dict[str, object]:
        provider = {**self.config["provider"], "thinking_enabled": True, "reasoning_effort": "low"}
        wiki_client = create_client(provider)

        def generate(prompt: str) -> str:
            response = wiki_client.complete(
                [{"role": "system", "content": "You compile source-grounded project documentation."}, {"role": "user", "content": prompt}],
                [],
            )
            self._track_usage(response)
            return str(response.get("content") or "")

        return self.wiki.build(generate)

    def export(self, raw_path: str | None = None) -> Path:
        path = Path(raw_path) if raw_path else Path(".chattyplay") / "exports" / f"{self.session_id}.md"
        path = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError("export path must be inside the workspace") from exc
        lines = [f"# ChattyPlay session {self.session_id}\n"]
        for message in self.messages:
            role = message.get("role", "unknown")
            lines.append(f"\n## {role}\n")
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        lines.append(str(block.get("text", "")))
                    elif block.get("type") == "image_url":
                        lines.append("[local image attachment]")
            elif content:
                lines.append(str(content))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                lines.append(f"\n`{function.get('name', '')}`\n```json\n{function.get('arguments', '{}')}\n```")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _attach_mentions(self, prompt: str) -> str | list[dict[str, Any]]:
        limit = int(self.config.get("agent", {}).get("max_file_mention_chars", 30000))
        image_limit = int(self.config.get("agent", {}).get("max_image_bytes", 5_000_000))
        max_images = int(self.config.get("agent", {}).get("max_images", 4))
        attachments: list[str] = []
        images: list[dict[str, Any]] = []
        image_bytes = 0
        seen: set[Path] = set()
        image_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
        for raw in re.findall(r"(?<!\w)@([\w./\\-]+)", prompt):
            candidate = (self.workspace / raw).resolve()
            try:
                candidate.relative_to(self.workspace)
            except ValueError:
                continue
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            media_type = image_types.get(candidate.suffix.lower())
            if media_type:
                if len(images) >= max_images:
                    raise ValueError(f"too many image attachments; maximum is {max_images}")
                size = candidate.stat().st_size
                if image_bytes + size > image_limit:
                    raise ValueError(f"image attachments exceed {image_limit} bytes")
                data = candidate.read_bytes()
                image_bytes += len(data)
                encoded = base64.b64encode(data).decode("ascii")
                images.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}})
                continue
            with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(limit)
            attachments.append(f'<file path="{candidate.relative_to(self.workspace)}">\n{text}\n</file>')
        text = prompt + ("\n\n<attached_files>\n" + "\n".join(attachments) + "\n</attached_files>" if attachments else "")
        return [{"type": "text", "text": text}, *images] if images else text

    def _track_usage(self, message: dict[str, Any]) -> None:
        usage = message.pop("_usage", {})
        self.last_output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        self.usage["input_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        self.usage["output_tokens"] += self.last_output_tokens
        self.usage["requests"] += 1

    def _delegate(self, args: dict[str, Any]) -> str:
        task = str(args["task"]).strip()
        if not task:
            raise ValueError("task must not be empty")
        skill = str(args.get("skill", "")).strip()
        if skill and skill not in self.skills:
            raise ValueError(f"unknown skill: {skill}")
        config = deepcopy(self.config)
        config["mcpServers"] = {}
        config["permissions"] = {
            **config.get("permissions", {}),
            "read": "allow",
            "write": "deny",
            "shell": "deny",
            "browser": "deny",
            "clipboard": "deny",
            "mcp": "deny",
            "delegate": "deny",
            "interaction": "allow",
        }
        if skill:
            config.setdefault("skills", {})["enabled"] = [skill]
        child = Agent(self.workspace, config, depth=self.depth + 1)
        try:
            result = child.run(task)
            for key in self.usage:
                self.usage[key] += child.usage[key]
            return result
        finally:
            child.close()

    def close(self) -> None:
        self.mcp.close()
