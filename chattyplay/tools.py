from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
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

FETCH_TIMEOUT_SECONDS = 10
FETCH_ATTEMPTS = 2
FETCH_RETRY_DELAY_SECONDS = 0.25
FETCH_MAX_BYTES = 1_000_000
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
DISCOVERY_TIMEOUT_SECONDS = 5
DISCOVERY_MAX_CANDIDATES = 5
DISCOVERY_VERIFY_BYTES = 4096


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


@dataclass
class StreamingWrite:
    raw_path: str
    path: Path
    before: bytes | None
    before_mode: int | None
    handle: Any = None
    content: str = ""


@dataclass
class _HttpResponse:
    url: str
    status: int
    content_type: str
    charset: str
    body: bytes
    truncated: bool


class _TextExtractor(HTMLParser):
    _HIDDEN_TAGS = {"head", "script", "style", "noscript", "svg", "template", "nav", "footer", "aside", "form"}
    _BREAK_TAGS = {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
        "figure", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
        "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._HIDDEN_TAGS:
            self.hidden += 1
        elif not self.hidden and tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._HIDDEN_TAGS and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(re.sub(r"\s+", " ", data).strip())

    def text(self) -> str:
        lines: list[str] = []
        for raw in " ".join(self.parts).splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line or (lines and line == lines[-1]):
                continue
            lines.append(line)
        return "\n".join(lines)


class _DiscoveryExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.in_title = False
        self.base_href: str | None = None
        self.links: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {name.lower(): value for name, value in attrs if value is not None}
        if tag == "title":
            self.in_title = True
        elif tag == "base" and not self.base_href and values.get("href"):
            self.base_href = values["href"]
        elif tag == "a" and values.get("href"):
            self.links.append(values["href"])
        elif tag == "link" and values.get("href"):
            rel = {item.lower() for item in values.get("rel", "").split()}
            if rel & {"canonical", "alternate"}:
                self.links.append(values["href"])
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and data.strip():
            self.title_parts.append(data.strip())

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()


def _normalize_http_url(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("URL must be a non-empty string")
    value = raw.strip()
    if value.startswith("<") or value.endswith(">"):
        if not (value.startswith("<") and value.endswith(">") and value.count("<") == 1 and value.count(">") == 1):
            raise ValueError("malformed HTTP(S) autolink")
        value = value[1:-1].strip()
    elif value.startswith("["):
        match = re.fullmatch(r"\[(?:[^\\\]]|\\.)*\]\(([^\s]+)\)", value)
        if not match:
            raise ValueError("malformed Markdown HTTP(S) link")
        value = match.group(1)
    if any(character.isspace() for character in value):
        raise ValueError("URL must be a plain HTTP(S) target, Markdown link, or autolink")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("malformed HTTP(S) URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("only HTTP(S) URLs with a hostname are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, parsed.fragment))


def _response_content_type(headers: Any) -> tuple[str, str]:
    try:
        content_type = headers.get_content_type().lower()
        charset = headers.get_content_charset() or "utf-8"
    except AttributeError:
        raw = str(headers.get("Content-Type", "text/plain"))
        content_type = raw.split(";", 1)[0].strip().lower() or "text/plain"
        match = re.search(r"charset\s*=\s*['\"]?([^;'\"\s]+)", raw, re.IGNORECASE)
        charset = match.group(1) if match else "utf-8"
    return content_type, charset


def _decode_response(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _readable_html(text: str) -> str:
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text()


def _content_quality_error(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 40:
        return "page contained too little readable text"
    lowered = compact.lower()
    blocked_markers = (
        "enable javascript", "javascript is required", "verify you are human", "access denied",
        "checking your browser", "just a moment...", "sign in to continue", "log in to continue",
    )
    if len(compact) < 500 and any(marker in lowered for marker in blocked_markers):
        return "page appears to be a JavaScript, verification, or access-block page"
    return None


def _request_url(
    url: str,
    *,
    attempts: int = FETCH_ATTEMPTS,
    timeout: int = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = FETCH_MAX_BYTES,
) -> tuple[_HttpResponse | None, str | None]:
    request_url, _ = urllib.parse.urldefrag(url)
    request = urllib.request.Request(request_url, headers={"User-Agent": f"ChattyPlay/{__version__} (+local coding agent)"})
    last_error = "request failed"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                truncated = len(body) > max_bytes
                if truncated:
                    body = body[:max_bytes]
                content_type, charset = _response_content_type(response.headers)
                final_url = response.geturl() if hasattr(response, "geturl") else response.url
                return _HttpResponse(
                    final_url,
                    getattr(response, "status", 200),
                    content_type,
                    charset,
                    body,
                    truncated,
                ), None
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            exc.close()
            if exc.code not in RETRYABLE_HTTP_STATUS:
                return None, f"{last_error} (not retryable)"
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionResetError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            last_error = f"{type(reason).__name__}: {reason}"
        if attempt + 1 < attempts:
            time.sleep(FETCH_RETRY_DELAY_SECONDS)
    return None, f"unable to fetch URL after {attempts} attempts: {last_error}"


def _same_origin(left: str, right: str) -> bool:
    a, b = urllib.parse.urlsplit(left), urllib.parse.urlsplit(right)
    return (a.scheme.lower(), a.hostname, a.port) == (b.scheme.lower(), b.hostname, b.port)


def _discovery_metadata(html: str, page_url: str, requested_url: str, limit: int) -> tuple[str, str | None, list[str]]:
    parser = _DiscoveryExtractor()
    parser.feed(html)
    parser.close()
    base_url = urllib.parse.urljoin(page_url, parser.base_href) if parser.base_href else page_url
    if not _same_origin(page_url, base_url):
        base_url = page_url
    candidates: list[str] = []

    def add(raw: str) -> None:
        if len(candidates) >= limit:
            return
        try:
            candidate = _normalize_http_url(urllib.parse.urljoin(base_url, raw))
        except ValueError:
            return
        candidate, _ = urllib.parse.urldefrag(candidate)
        page_without_fragment, _ = urllib.parse.urldefrag(page_url)
        if not _same_origin(page_url, candidate) or candidate == page_without_fragment or candidate in candidates:
            return
        candidates.append(candidate)

    for link in parser.links:
        add(link)

    docsify = "window.$docsify" in html or any("docsify" in script.lower() for script in parser.scripts)
    framework = "docsify" if docsify else None
    if docsify:
        homepage = re.search(r"\bhomepage\s*:\s*(['\"])([^'\"]+)\1", html)
        fragment = urllib.parse.urlsplit(requested_url).fragment
        route = urllib.parse.unquote(fragment[1:] if fragment.startswith("/") else fragment).strip("/")
        if homepage:
            add(homepage.group(2))
        elif route and ".." not in route.split("/"):
            add(route if Path(route).suffix else f"{route}.md")
        else:
            add("README.md")
        sidebar = re.search(r"\bloadSidebar\s*:\s*(?:(['\"])([^'\"]+)\1|(true))", html, re.IGNORECASE)
        if sidebar:
            add(sidebar.group(2) or "_sidebar.md")
    return parser.title, framework, candidates


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
        self._stream_write: StreamingWrite | None = None
        self._stream_denied: str | None = None
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
        streamed = name == "write_file" and self._stream_write is not None and args.get("path") == self._stream_write.raw_path
        if name == "write_file" and args.get("path") == self._stream_denied:
            self._stream_denied = None
            return "error: user denied permission"
        policy = self.permissions.get(tool.category, "ask")
        if not streamed and tool.category not in self.always_allowed:
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
        self.add(Tool("write_file", "Atomically create or fully replace a UTF-8 text file. Prefer edit_file for targeted changes. Pass the SHA-256 from read_file when overwriting to reject stale writes.", {
            **obj, "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "expected_sha256": {"type": "string"}}, "required": ["path", "content"]
        }, "write", self._write))
        self.add(Tool("edit_file", "Replace an exact, unique string in a file. Read first and pass expected_sha256 to reject stale edits.", {
            **obj, "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "expected_sha256": {"type": "string"}}, "required": ["path", "old_text", "new_text"]
        }, "write", self._edit))
        self.add(Tool("move_file", "Rename or move an existing file within the workspace without reading or rebuilding its contents. It supports case-only renames, never overwrites a different destination, and must not be simulated with create/write/delete tools if it fails. Contents and permissions are preserved, and a successful move can be reverted with /undo.", {
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
        self.add(Tool("open_browser", "Open an HTTP(S) URL in the user's default browser. Accepts a plain URL, a complete Markdown link, or an autolink.", {
            **obj, "properties": {"url": {"type": "string"}}, "required": ["url"]
        }, "browser", self._browser))
        self.add(Tool("fetch_url", "Fetch an HTTP(S) page and return readable text. Accepts a plain URL, a complete Markdown link, or an autolink. Use discover_url if a page reports too little readable content or a JavaScript app shell.", {
            **obj, "properties": {"url": {"type": "string"}}, "required": ["url"]
        }, "browser", self._fetch))
        self.add(Tool("discover_url", "Find a small, verified set of content URLs evidenced by an HTML page, including documented SPA resources. Use after fetch_url reports too little readable content; candidates are leads and must be fetched before answering.", {
            **obj,
            "properties": {"url": {"type": "string"}, "max_candidates": {"type": "integer", "minimum": 1, "maximum": DISCOVERY_MAX_CANDIDATES}},
            "required": ["url"],
        }, "browser", self._discover))
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
        streaming = self._stream_write if self._stream_write and self._stream_write.path == path else None
        if streaming and streaming.handle:
            streaming.handle.close()
        before = streaming.before if streaming else path.read_bytes() if path.exists() and path.is_file() else None
        before_mode = streaming.before_mode if streaming else path.stat().st_mode & 0o7777 if before is not None else None
        after = str(args["content"]).encode("utf-8")
        expected = args.get("expected_sha256")
        actual = hashlib.sha256(before).hexdigest() if before is not None else "missing"
        if expected and str(expected) != actual:
            self.cancel_stream_write()
            raise ValueError(f"stale file version: expected {expected}, current {actual}")
        if before == after:
            if streaming and streaming.content.encode("utf-8") != after:
                self._replace_bytes(path, after, before_mode)
            self._stream_write = None
            return f"ok: unchanged {path.relative_to(self.workspace)}"
        if not streaming or streaming.content.encode("utf-8") != after:
            self._replace_bytes(path, after, before_mode)
        self._stream_write = None
        self.changes.append(Change(path, before, after, before_mode))
        return f"ok: wrote {path.relative_to(self.workspace)}"

    @staticmethod
    def _partial_string(arguments: str, key: str, partial: bool = False) -> str | None:
        match = re.search(rf'"{key}"\s*:\s*', arguments)
        if not match or match.end() >= len(arguments) or arguments[match.end()] != '"':
            return None
        raw = arguments[match.end():]
        try:
            value, _ = json.JSONDecoder().raw_decode(raw)
            return value if isinstance(value, str) else None
        except json.JSONDecodeError:
            if not partial:
                return None
        raw = raw[1:]
        for cut in range(min(6, len(raw)) + 1):
            try:
                return json.loads('"' + (raw[:-cut] if cut else raw) + '"')
            except json.JSONDecodeError:
                continue
        return None

    def stream_write(self, arguments: str, before_confirm: Callable[[], None] | None = None) -> bool:
        raw_path = self._partial_string(arguments, "path")
        if not raw_path or raw_path == self._stream_denied:
            return False
        if self._stream_write is None:
            path = self._path(raw_path)
            if path.exists() and not path.is_file():
                return False
            policy = self.permissions.get("write", "ask")
            if "write" not in self.always_allowed and policy != "allow":
                if policy == "deny":
                    return False
                if before_confirm:
                    before_confirm()
                decision = self.confirm("write", f"write_file({raw_path})") if self.confirm else False
                if not decision:
                    self._stream_denied = raw_path
                    return False
                if decision == "always":
                    self.always_allowed.add("write")
            before = path.read_bytes() if path.is_file() else None
            mode = path.stat().st_mode & 0o7777 if before is not None else None
            self._stream_write = StreamingWrite(raw_path, path, before, mode)
        stream = self._stream_write
        if stream.raw_path != raw_path:
            return False  # ponytail: one streamed file at a time; batch writes still commit normally.
        content = self._partial_string(arguments, "content", partial=True)
        if content is None or not content.startswith(stream.content):
            return True
        if stream.handle is None:
            stream.path.parent.mkdir(parents=True, exist_ok=True)
            stream.handle = stream.path.open("wb")
        stream.handle.write(content[len(stream.content):].encode("utf-8"))
        stream.handle.flush()
        stream.content = content
        return True

    def cancel_stream_write(self) -> None:
        stream, self._stream_write = self._stream_write, None
        if not stream:
            return
        if stream.handle:
            stream.handle.close()
        if stream.before is None:
            stream.path.unlink(missing_ok=True)
        else:
            self._replace_bytes(stream.path, stream.before, stream.before_mode)

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
        if source == destination:
            return f"ok: unchanged {source.relative_to(self.workspace)}"
        case_only = self._is_case_only_rename(source, destination)
        if destination.exists() and not case_only:
            raise FileExistsError(f"destination already exists: {destination}")
        content = source.read_bytes()
        mode = source.stat().st_mode & 0o7777
        destination.parent.mkdir(parents=True, exist_ok=True)
        if case_only:
            temporary = source.with_name(f".{source.name}.{uuid.uuid4().hex}.rename")
            while temporary.exists():
                temporary = source.with_name(f".{source.name}.{uuid.uuid4().hex}.rename")
            os.rename(source, temporary)
            try:
                os.rename(temporary, destination)
            except Exception as rename_error:
                try:
                    os.rename(temporary, source)
                except Exception as recovery_error:
                    raise RuntimeError(
                        f"case-only rename failed: {rename_error}; recovery failed: {recovery_error}; "
                        f"file may remain at {temporary}"
                    ) from rename_error
                raise OSError(f"case-only rename failed; source restored: {rename_error}") from rename_error
        else:
            os.rename(source, destination)
        self.changes.extend((Change(destination, None, content), Change(source, content, None, mode)))
        return f"ok: moved {source.relative_to(self.workspace)} to {destination.relative_to(self.workspace)}"

    @staticmethod
    def _is_case_only_rename(source: Path, destination: Path) -> bool:
        if source.parent != destination.parent or source.name == destination.name or source.name.casefold() != destination.name.casefold():
            return False
        try:
            names = {entry.name for entry in source.parent.iterdir()}
            if source.name in names and destination.name in names:
                return False
            return destination.exists() and os.path.samefile(source, destination)
        except OSError:
            return False

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
            if mode is not None:
                temp.chmod(mode)
            for delay in (0.05, 0.1, 0.2, None):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError:
                    if delay is None:
                        raise
                    time.sleep(delay)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

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
        url = _normalize_http_url(args["url"])
        return "ok: opened" if webbrowser.open(url) else "error: browser could not be opened"

    @staticmethod
    def _fetch(args: dict[str, Any]) -> str:
        url = _normalize_http_url(args["url"])
        response, error = _request_url(url)
        if response is None:
            return f"error: {error}\nURL: {url}"
        if response.content_type in {"text/html", "application/xhtml+xml"}:
            html = _decode_response(response.body, response.charset)
            text = _readable_html(html)
            quality_error = _content_quality_error(text)
            if quality_error:
                title, framework, candidates = _discovery_metadata(
                    html, response.url, url, DISCOVERY_MAX_CANDIDATES,
                )
                metadata = []
                if title:
                    metadata.append(f"Title (metadata only): {title}")
                if framework:
                    metadata.append(f"Detected framework: {framework}")
                if candidates:
                    metadata.append("Discovery candidates: " + ", ".join(candidates))
                suffix = "\n" + "\n".join(metadata) if metadata else ""
                return (
                    f"error: unable to read this page reliably: {quality_error}\n"
                    f"URL: {response.url}\nStatus: {response.status}\nContent-Type: {response.content_type}{suffix}"
                )
        elif response.content_type.startswith("text/") or response.content_type in {
            "application/json", "application/xml",
        }:
            text = _decode_response(response.body, response.charset)
        else:
            return (
                f"error: unsupported Content-Type: {response.content_type}\n"
                f"URL: {response.url}\nStatus: {response.status}\nContent-Type: {response.content_type}"
            )
        suffix = "\n... response truncated at 1 MB" if response.truncated else ""
        return f"URL: {response.url}\nStatus: {response.status}\nContent-Type: {response.content_type}\n\n{text}{suffix}"

    @staticmethod
    def _discover(args: dict[str, Any]) -> str:
        url = _normalize_http_url(args["url"])
        limit = min(DISCOVERY_MAX_CANDIDATES, max(1, int(args.get("max_candidates", DISCOVERY_MAX_CANDIDATES))))
        response, error = _request_url(url, attempts=1, timeout=DISCOVERY_TIMEOUT_SECONDS)
        if response is None:
            return f"error: unable to inspect page for content discovery: {error}\nURL: {url}"
        if response.content_type not in {"text/html", "application/xhtml+xml"}:
            return f"error: content discovery requires HTML, got {response.content_type}\nURL: {response.url}"
        html = _decode_response(response.body, response.charset)
        title, framework, candidates = _discovery_metadata(html, response.url, url, limit)
        lines = [f"URL: {response.url}"]
        if title:
            lines.append(f"Title (metadata only): {title}")
        if framework:
            lines.append(f"Detected framework: {framework}")
        if not candidates:
            lines.append("no reliable content candidates found")
            return "\n".join(lines)
        lines.append("Candidates (fetch a verified URL before answering):")
        for candidate in candidates:
            check, check_error = _request_url(
                candidate,
                attempts=1,
                timeout=DISCOVERY_TIMEOUT_SECONDS,
                max_bytes=DISCOVERY_VERIFY_BYTES,
            )
            if check is None:
                lines.append(f"- {candidate} [unverified: {check_error}]")
            else:
                lines.append(f"- {candidate} [verified: HTTP {check.status}, {check.content_type}]")
        return "\n".join(lines)

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
