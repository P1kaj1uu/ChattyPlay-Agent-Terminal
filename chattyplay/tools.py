from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import webbrowser
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from . import __version__


HARD_DENY = re.compile(
    r"(?:^|[;&|\r\n])\s*(?:sudo\s+)?(?:rm\s+-[^\n]*(?:r[^\n]*f|f[^\n]*r)\s+(?:/|~|\$HOME|%USERPROFILE%)|mkfs|format\s+[a-z]:|"
    r"shutdown|reboot|diskpart|dd\s+if=)|:\(\)\s*\{",
    re.IGNORECASE,
)


def _clipboard_program() -> tuple[list[str], list[str]]:
    system = platform.system()
    if system == "Darwin":
        return ["pbpaste"], ["pbcopy"]
    if system == "Windows":
        executable = shutil.which("pwsh") or shutil.which("powershell.exe") or shutil.which("powershell")
        if not executable:
            raise RuntimeError("PowerShell is required for clipboard access")
        prefix = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
        return prefix + ["[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); Get-Clipboard -Raw"], prefix + ["[Console]::InputEncoding=[Text.UTF8Encoding]::new(); Set-Clipboard -Value ([Console]::In.ReadToEnd())"]
    for read_name, write_name, read_args, write_args in (
        ("wl-paste", "wl-copy", ["--no-newline"], []),
        ("xclip", "xclip", ["-selection", "clipboard", "-o"], ["-selection", "clipboard"]),
        ("xsel", "xsel", ["--clipboard", "--output"], ["--clipboard", "--input"]),
    ):
        reader, writer = shutil.which(read_name), shutil.which(write_name)
        if reader and writer:
            return [reader, *read_args], [writer, *write_args]
    raise RuntimeError("install wl-clipboard, xclip, or xsel for clipboard access")


def read_clipboard() -> str:
    read_command, _ = _clipboard_program()
    return subprocess.run(read_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=True).stdout


def write_clipboard(text: str) -> None:
    _, write_command = _clipboard_program()
    subprocess.run(write_command, input=text, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=True)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    handler: Callable[[dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Change:
    path: Path
    before: bytes | None
    after: bytes | None
    before_mode: int | None = None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
        permissions: dict[str, str] | None = None,
        confirm: Callable[[str, str], bool | str] | None = None,
        ask_user: Callable[[str], str] | None = None,
        on_plan: Callable[[list[dict[str, Any]]], None] | None = None,
        max_output: int = 30000,
    ):
        self.workspace = workspace.resolve()
        self.permissions = permissions or {}
        self.confirm = confirm
        self.ask_user = ask_user
        self.on_plan = on_plan
        self.max_output = max_output
        self.tools: dict[str, Tool] = {}
        self.always_allowed: set[str] = set()
        self.changes: list[Change] = []
        self.plan_mode = False
        self.plan: list[dict[str, Any]] = []
        self._register_builtin()

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def checkpoint(self) -> int:
        return len(self.changes)

    def undo(self, checkpoint: int) -> str:
        pending = self.changes[checkpoint:]
        expected: dict[Path, bytes | None] = {}
        for change in pending:
            expected[change.path] = change.after
        for path, after in expected.items():
            current = path.read_bytes() if path.exists() and path.is_file() else None
            if current != after:
                return f"error: {path.relative_to(self.workspace)} changed after the agent edit; undo cancelled"
        for change in reversed(pending):
            if change.before is None:
                change.path.unlink(missing_ok=True)
            else:
                self._replace_bytes(change.path, change.before, change.before_mode)
        del self.changes[checkpoint:]
        return f"ok: reverted {len(pending)} file operation(s)"

    def execute(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"error: unknown tool: {name}"
        if not isinstance(args, dict):
            return "error: tool arguments must be an object"
        if name == "shell" and HARD_DENY.search(str(args.get("command", ""))):
            return "error: command blocked by hard safety rule"
        if self.plan_mode and tool.category in {"write", "shell", "browser", "clipboard", "mcp"}:
            return f"error: {name} is unavailable in plan mode"
        policy = self.permissions.get(tool.category, "ask")
        if tool.category not in self.always_allowed:
            if policy == "deny":
                return f"error: {tool.category} permission denied"
            if policy == "ask":
                decision = self.confirm(tool.category, self._preview(name, args)) if self.confirm else False
                if decision == "always":
                    self.always_allowed.add(tool.category)
                elif not decision:
                    return "error: user denied permission"
        try:
            result = tool.handler(args)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            if len(text) > self.max_output:
                text = text[: self.max_output] + f"\n... truncated ({len(text)} chars total)"
            return text
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    @staticmethod
    def _preview(name: str, args: dict[str, Any]) -> str:
        first = next(iter(args.values()), "")
        return f"{name}({str(first)[:160]})"

    def _path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("path must be a non-empty string")
        path = Path(raw).expanduser()
        path = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(f"path is outside workspace: {path}") from exc
        return path

    def _register_builtin(self) -> None:
        obj = {"type": "object", "additionalProperties": False}
        self.add(Tool("read_file", "Read a UTF-8 text file with line numbers and a SHA-256 version. Pass that version to edit_file.", {
            **obj,
            "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": 2000}},
            "required": ["path"],
        }, "read", self._read))
        self.add(Tool("write_file", "Atomically create or replace a UTF-8 text file. Pass the SHA-256 from read_file when overwriting to reject stale writes.", {
            **obj, "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "expected_sha256": {"type": "string"}}, "required": ["path", "content"]
        }, "write", self._write))
        self.add(Tool("edit_file", "Replace an exact, unique string in a file. Read first and pass expected_sha256 to reject stale edits.", {
            **obj, "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "expected_sha256": {"type": "string"}}, "required": ["path", "old_text", "new_text"]
        }, "write", self._edit))
        self.add(Tool("move_file", "Move one file within the workspace. The destination must not exist and the move can be reverted with /undo.", {
            **obj, "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]
        }, "write", self._move))
        self.add(Tool("delete_file", "Delete one file inside the workspace. The deletion can be reverted with /undo.", {
            **obj, "properties": {"path": {"type": "string"}}, "required": ["path"]
        }, "write", self._delete))
        self.add(Tool("glob_files", "List workspace files matching a glob pattern.", {
            **obj, "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]
        }, "read", self._glob))
        self.add(Tool("grep_files", "Search text files with a regular expression.", {
            **obj, "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]
        }, "read", self._grep))
        self.add(Tool("shell", "Run a shell command in the workspace. Use PowerShell/cmd syntax on Windows and shell syntax on macOS/Linux.", {
            **obj, "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "minimum": 1, "maximum": 600}}, "required": ["command"]
        }, "shell", self._shell))
        self.add(Tool("open_browser", "Open an HTTP(S) URL in the user's default browser.", {
            **obj, "properties": {"url": {"type": "string"}}, "required": ["url"]
        }, "browser", self._browser))
        self.add(Tool("fetch_url", "Fetch an HTTP(S) page and return readable text. Use for web research and documentation.", {
            **obj, "properties": {"url": {"type": "string"}}, "required": ["url"]
        }, "browser", self._fetch))
        self.add(Tool("read_clipboard", "Read text from the local system clipboard on macOS, Windows, or Linux.", {
            **obj, "properties": {}
        }, "clipboard", lambda _: read_clipboard()))
        self.add(Tool("copy_to_clipboard", "Copy text to the local system clipboard on macOS, Windows, or Linux.", {
            **obj, "properties": {"text": {"type": "string"}}, "required": ["text"]
        }, "clipboard", lambda args: self._copy(args)))
        self.add(Tool("ask_user", "Ask the user one concise question when their input is required to continue.", {
            **obj, "properties": {"question": {"type": "string"}}, "required": ["question"]
        }, "interaction", self._ask_user))
        self.add(Tool("update_plan", "Publish the current task plan and step statuses.", {
            **obj,
            "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {"step": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["step", "status"], "additionalProperties": False}}},
            "required": ["steps"],
        }, "interaction", self._update_plan))

    def _read(self, args: dict[str, Any]) -> str:
        path = self._path(args["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        offset = max(1, int(args.get("offset", 1)))
        limit = min(2000, max(1, int(args.get("limit", 400))))
        selected = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{offset + i:>5} | {line}" for i, line in enumerate(selected)) or "(empty)"
        return f"path: {path.relative_to(self.workspace)}\nsha256: {hashlib.sha256(raw).hexdigest()}\nlines: {offset}-{offset + len(selected) - 1}/{len(lines)}\n{body}"

    def _write(self, args: dict[str, Any]) -> str:
        path = self._path(args["path"])
        if path.exists() and not path.is_file():
            raise IsADirectoryError(path)
        before = path.read_bytes() if path.exists() and path.is_file() else None
        before_mode = path.stat().st_mode & 0o7777 if before is not None else None
        after = str(args["content"]).encode("utf-8")
        expected = args.get("expected_sha256")
        actual = hashlib.sha256(before).hexdigest() if before is not None else "missing"
        if expected and str(expected) != actual:
            raise ValueError(f"stale file version: expected {expected}, current {actual}")
        if before == after:
            return f"ok: unchanged {path.relative_to(self.workspace)}"
        self._replace_bytes(path, after)
        self.changes.append(Change(path, before, after, before_mode))
        return f"ok: wrote {path.relative_to(self.workspace)}"

    def _edit(self, args: dict[str, Any]) -> str:
        path = self._path(args["path"])
        before = path.read_bytes()
        before_mode = path.stat().st_mode & 0o7777
        expected = args.get("expected_sha256")
        actual = hashlib.sha256(before).hexdigest()
        if expected and str(expected) != actual:
            raise ValueError(f"stale file version: expected {expected}, current {actual}")
        content = before.decode("utf-8")
        old = str(args["old_text"])
        count = content.count(old)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once; found {count}")
        after = content.replace(old, str(args["new_text"]), 1).encode("utf-8")
        if before == after:
            return f"ok: unchanged {path.relative_to(self.workspace)}"
        self._replace_bytes(path, after)
        self.changes.append(Change(path, before, after, before_mode))
        return f"ok: edited {path.relative_to(self.workspace)}"

    def _move(self, args: dict[str, Any]) -> str:
        source = self._path(args["source"])
        destination = self._path(args["destination"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        content = source.read_bytes()
        mode = source.stat().st_mode & 0o7777
        self._replace_bytes(destination, content, mode)
        try:
            source.unlink()
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self.changes.extend((Change(destination, None, content), Change(source, content, None, mode)))
        return f"ok: moved {source.relative_to(self.workspace)} to {destination.relative_to(self.workspace)}"

    def _delete(self, args: dict[str, Any]) -> str:
        path = self._path(args["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        before = path.read_bytes()
        before_mode = path.stat().st_mode & 0o7777
        path.unlink()
        self.changes.append(Change(path, before, None, before_mode))
        return f"ok: deleted {path.relative_to(self.workspace)}"

    @staticmethod
    def _replace_bytes(path: Path, content: bytes, mode: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode is None and path.exists():
            mode = path.stat().st_mode & 0o7777
        fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp = Path(raw_temp)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if mode is not None:
                temp.chmod(mode)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def _glob(self, args: dict[str, Any]) -> str:
        pattern = str(args["pattern"])
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise PermissionError("glob must stay inside workspace")
        matches = [p for p in self.workspace.glob(pattern) if p.is_file() and not p.is_symlink()]
        return "\n".join(str(p.relative_to(self.workspace)) for p in sorted(matches)[:1000]) or "none"

    def _grep(self, args: dict[str, Any]) -> str:
        regex = re.compile(str(args["pattern"]))
        glob_pattern = str(args.get("glob", "**/*"))
        if Path(glob_pattern).is_absolute() or ".." in Path(glob_pattern).parts:
            raise PermissionError("glob must stay inside workspace")
        hits: list[str] = []
        for path in self.workspace.glob(glob_pattern):
            if path.is_symlink() or not path.is_file() or any(part in {".git", "node_modules", ".venv"} for part in path.parts):
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{path.relative_to(self.workspace)}:{number}:{line}")
                        if len(hits) >= 500:
                            return "\n".join(hits) + "\n... truncated"
            except (OSError, UnicodeError):
                continue
        return "\n".join(hits) or "none"

    def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        timeout = min(600, max(1, int(args.get("timeout", 120))))
        completed = subprocess.run(
            str(args["command"]), shell=True, cwd=self.workspace, capture_output=True,
            text=True, errors="replace", timeout=timeout, env=os.environ.copy(),
        )
        return {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}

    @staticmethod
    def _browser(args: dict[str, Any]) -> str:
        url = str(args["url"])
        if not url.startswith(("http://", "https://")):
            raise ValueError("only HTTP(S) URLs are allowed")
        return "ok: opened" if webbrowser.open(url) else "error: browser could not be opened"

    @staticmethod
    def _fetch(args: dict[str, Any]) -> str:
        url = str(args["url"])
        if not url.startswith(("http://", "https://")):
            raise ValueError("only HTTP(S) URLs are allowed")
        request = urllib.request.Request(url, headers={"User-Agent": f"ChattyPlay/{__version__} (+local coding agent)"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(1_000_001)
            if len(body) > 1_000_000:
                body = body[:1_000_000]
                suffix = "\n... response truncated at 1 MB"
            else:
                suffix = ""
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            if content_type == "text/html":
                parser = _TextExtractor()
                parser.feed(text)
                text = "\n".join(parser.parts)
            return f"URL: {response.url}\nStatus: {response.status}\nContent-Type: {content_type}\n\n{text}{suffix}"

    def _ask_user(self, args: dict[str, Any]) -> str:
        question = str(args["question"]).strip()
        if not question:
            raise ValueError("question must not be empty")
        return self.ask_user(question) if self.ask_user else "error: interactive input is unavailable"

    @staticmethod
    def _copy(args: dict[str, Any]) -> str:
        text = str(args["text"])
        write_clipboard(text)
        return f"ok: copied {len(text)} characters"

    def _update_plan(self, args: dict[str, Any]) -> str:
        steps = args["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a non-empty array")
        self.plan = [{"step": str(item["step"]), "status": str(item["status"])} for item in steps]
        if self.on_plan:
            self.on_plan(self.plan)
        return "ok: plan updated"
