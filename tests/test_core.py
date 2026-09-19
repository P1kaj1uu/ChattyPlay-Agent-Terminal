from __future__ import annotations

import json
import io
import hashlib
import os
import re
import tempfile
import threading
import time
import sys
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.utils import get_cwidth

from chattyplay.config import ConfigStore
from chattyplay.client import AnthropicClient, OpenAIClient, create_client
from chattyplay.cli import _format_shell_result, _is_command, _reload_agent
from chattyplay.agent import Agent, BASE_PROMPT, _tool_status
from chattyplay.sessions import SessionStore
from chattyplay.mcp import MCPManager
from chattyplay.rag import RAGIndex
from chattyplay.skills import discover
from chattyplay.terminal import Spinner, welcome_screen
from chattyplay.tools import ToolRegistry
from chattyplay.webui import create_server
from chattyplay.wiki import ProjectWiki


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_welcome_screen_adapts_to_terminal_width(self) -> None:
        wide = welcome_screen("qwen3.5:9b-q4_K_M", self.root, "current", ["previous"], 100)
        narrow = welcome_screen("本地模型", self.root, "current", [], 36)
        tiny = welcome_screen("本地模型", self.root, "current", [], 24)
        self.assertIn("ChattyPlay", wide)
        self.assertNotIn("v0.8.0", wide)
        self.assertIn("previous", wide)
        self.assertIn("No recent activity", narrow)
        self.assertTrue(all(get_cwidth(line) <= 100 for line in wide.splitlines()))
        self.assertTrue(all(get_cwidth(line) <= 36 for line in narrow.splitlines()))
        self.assertTrue(all(get_cwidth(line) <= 24 for line in tiny.splitlines()))

    def test_spinner_renders_and_clears_on_a_tty(self) -> None:
        class TTYBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        output = TTYBuffer()
        spinner = Spinner(stream=output)
        spinner.start()
        time.sleep(0.02)
        spinner.stop()
        self.assertIn("thinking...", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\r"))

        calls = 0
        def clock() -> float:
            nonlocal calls
            calls += 1
            return 0.0 if calls == 1 else 3.0
        delayed = TTYBuffer()
        with patch("chattyplay.terminal.time.monotonic", side_effect=clock):
            spinner = Spinner(stream=delayed)
            spinner.start()
            time.sleep(0.02)
            spinner.stop()
        self.assertIn("Ctrl+C to cancel", delayed.getvalue())

        streamed = TTYBuffer()
        spinner = Spinner(stream=streamed)
        spinner._phase_started = spinner._stream_started = time.monotonic() - 2
        spinner.write("你好", "agent ❯ ")
        spinner.finish(20)
        self.assertNotIn("\033[s", streamed.getvalue())
        self.assertIn("10.0 tok/s", streamed.getvalue())
        self.assertNotIn("n/a", streamed.getvalue())

    def test_file_tools_and_workspace_boundary(self) -> None:
        tools = ToolRegistry(self.root, {"read": "allow", "write": "allow"})
        checkpoint = tools.checkpoint()
        self.assertIn("ok: wrote", tools.execute("write_file", {"path": "src/a.py", "content": "x = 1\n"}))
        read = tools.execute("read_file", {"path": "src/a.py"})
        self.assertIn("x = 1", read)
        version = hashlib.sha256(b"x = 1\n").hexdigest()
        self.assertIn("ok: unchanged", tools.execute("write_file", {"path": "src/a.py", "content": "x = 1\n", "expected_sha256": version}))
        self.assertIn("stale file version", tools.execute("write_file", {"path": "src/a.py", "content": "changed", "expected_sha256": "old"}))
        self.assertIn("ok: edited", tools.execute("edit_file", {"path": "src/a.py", "old_text": "1", "new_text": "2", "expected_sha256": version}))
        (self.root / "src" / "a.py").chmod(0o755)
        self.assertIn("stale file version", tools.execute("edit_file", {"path": "src/a.py", "old_text": "2", "new_text": "3", "expected_sha256": version}))
        self.assertIn("ok: moved", tools.execute("move_file", {"source": "src/a.py", "destination": "src/b.py"}))
        self.assertEqual((self.root / "src" / "b.py").stat().st_mode & 0o777, 0o755)
        changes_before_noop = len(tools.changes)
        self.assertIn("ok: unchanged", tools.execute("move_file", {"source": "src/b.py", "destination": "src/b.py"}))
        self.assertEqual(len(tools.changes), changes_before_noop)
        self.assertIn("ok: deleted", tools.execute("delete_file", {"path": "src/b.py"}))
        self.assertIn("outside workspace", tools.execute("read_file", {"path": "../secret"}))
        self.assertIn("reverted 5", tools.undo(checkpoint))
        self.assertFalse((self.root / "src" / "a.py").exists())
        self.assertFalse((self.root / "src" / "b.py").exists())

    def test_move_file_renames_without_rewriting_and_creates_parent(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "source.bin"
        destination = self.root / "nested" / "renamed.bin"
        content = b"\x00rename me\xff"
        source.write_bytes(content)
        source.chmod(0o755)

        with patch.object(tools, "_replace_bytes") as replace_bytes:
            result = tools.execute("move_file", {"source": "source.bin", "destination": "nested/renamed.bin"})

        self.assertIn("ok: moved", result)
        replace_bytes.assert_not_called()
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o755)

    def test_move_file_rejects_existing_missing_and_outside_paths(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "source.txt"
        destination = self.root / "destination.txt"
        source.write_text("source")
        destination.write_text("destination")

        self.assertIn("FileExistsError", tools.execute("move_file", {"source": "source.txt", "destination": "destination.txt"}))
        self.assertEqual(source.read_text(), "source")
        self.assertEqual(destination.read_text(), "destination")
        self.assertIn("FileNotFoundError", tools.execute("move_file", {"source": "missing.txt", "destination": "new.txt"}))
        self.assertFalse((self.root / "new.txt").exists())
        self.assertIn("outside workspace", tools.execute("move_file", {"source": "../outside.txt", "destination": "inside.txt"}))
        self.assertIn("outside workspace", tools.execute("move_file", {"source": "source.txt", "destination": "../outside.txt"}))
        self.assertTrue(source.exists())

    def test_move_file_undo_restores_path_content_and_mode(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "before.sh"
        destination = self.root / "after.sh"
        content = b"#!/bin/sh\necho moved\n"
        source.write_bytes(content)
        source.chmod(0o755)

        self.assertIn("ok: moved", tools.execute("move_file", {"source": "before.sh", "destination": "after.sh"}))
        self.assertIn("reverted 2", tools.undo(0))
        self.assertTrue(source.exists())
        self.assertEqual(source.read_bytes(), content)
        self.assertEqual(source.stat().st_mode & 0o777, 0o755)
        self.assertFalse(destination.exists())

    def test_move_file_rename_failure_leaves_source_untouched(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "source.txt"
        destination = self.root / "nested" / "destination.txt"
        source.write_text("unchanged")

        with patch("chattyplay.tools.os.rename", side_effect=OSError("rename failed")):
            result = tools.execute("move_file", {"source": "source.txt", "destination": "nested/destination.txt"})

        self.assertIn("rename failed", result)
        self.assertEqual(source.read_text(), "unchanged")
        self.assertFalse(destination.exists())

    def test_move_file_case_only_rename_uses_temporary_path(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "name.txt"
        destination = self.root / "Name.txt"
        source.write_text("same file")

        with patch.object(tools, "_is_case_only_rename", return_value=True), patch("chattyplay.tools.os.rename") as rename:
            result = tools.execute("move_file", {"source": "name.txt", "destination": "Name.txt"})

        self.assertIn("ok: moved", result)
        self.assertEqual(rename.call_count, 2)
        first_source, temporary = rename.call_args_list[0].args
        second_temporary, final_destination = rename.call_args_list[1].args
        self.assertEqual(first_source, source.resolve())
        self.assertEqual(second_temporary, temporary)
        self.assertEqual(temporary.parent, source.resolve().parent)
        self.assertNotIn(temporary, {source.resolve(), destination.resolve()})
        self.assertEqual(final_destination, destination.resolve())
        self.assertEqual(len(tools.changes), 2)

    def test_move_file_case_only_failure_rolls_back_or_reports_recovery_path(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        source = self.root / "name.txt"
        destination = self.root / "Name.txt"
        source.write_text("same file")

        with patch.object(tools, "_is_case_only_rename", return_value=True), patch(
            "chattyplay.tools.os.rename", side_effect=[None, PermissionError("locked"), None],
        ) as rename:
            result = tools.execute("move_file", {"source": "name.txt", "destination": "Name.txt"})
        self.assertIn("source restored", result)
        self.assertEqual(rename.call_count, 3)
        self.assertEqual(rename.call_args_list[2].args[1], source.resolve())
        self.assertEqual(tools.changes, [])

        with patch.object(tools, "_is_case_only_rename", return_value=True), patch(
            "chattyplay.tools.os.rename",
            side_effect=[None, PermissionError("locked"), PermissionError("rollback locked")],
        ) as rename:
            result = tools.execute("move_file", {"source": "name.txt", "destination": "Name.txt"})
        self.assertIn("recovery failed", result)
        self.assertIn("file may remain at", result)
        self.assertEqual(rename.call_count, 3)
        self.assertEqual(tools.changes, [])

    def test_file_write_retries_transient_windows_lock_without_fsync(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        real_replace = os.replace
        attempts = 0

        def replace(source: Path, destination: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("file is temporarily in use")
            real_replace(source, destination)

        with patch("chattyplay.tools.os.replace", side_effect=replace), patch("chattyplay.tools.os.fsync") as fsync, patch("chattyplay.tools.time.sleep"):
            self.assertIn("ok: wrote", tools.execute("write_file", {"path": "locked.txt", "content": "done"}))

        self.assertEqual(attempts, 3)
        fsync.assert_not_called()
        self.assertEqual((self.root / "locked.txt").read_text(), "done")

    def test_file_content_is_written_while_tool_arguments_stream(self) -> None:
        tools = ToolRegistry(self.root, {"write": "allow"})
        self.assertTrue(tools.stream_write('{"path":"live.txt","content":"first\\n'))
        self.assertEqual((self.root / "live.txt").read_text(), "first\n")
        arguments = '{"path":"live.txt","content":"first\\nsecond"}'
        self.assertTrue(tools.stream_write(arguments))
        self.assertEqual((self.root / "live.txt").read_text(), "first\nsecond")
        self.assertIn("ok: wrote", tools.execute("write_file", json.loads(arguments)))

        (self.root / "live.txt").write_text("original")
        tools.stream_write('{"path":"live.txt","content":"partial')
        self.assertEqual((self.root / "live.txt").read_text(), "partial")
        tools.cancel_stream_write()
        self.assertEqual((self.root / "live.txt").read_text(), "original")

    def test_create_file_exists_before_generation_and_grows_between_chunks(self) -> None:
        config = ConfigStore(self.root, self.root / "user.json").load()
        config["permissions"]["write"] = "allow"
        agent = Agent(self.root, config)
        target = self.root / "generated.txt"
        class Generator:
            def complete(inner, messages, tools, on_text):
                self.assertEqual(target.read_bytes(), b"")
                self.assertEqual(tools, [])
                on_text("first\n")
                self.assertEqual(target.read_bytes(), b"first\n")
                on_text("second\n")
                self.assertEqual(target.read_bytes(), b"first\nsecond\n")
                return {"role": "assistant", "content": "first\nsecond\n"}
        agent.client = Generator()
        try:
            self.assertIn("ok: created", agent.registry.execute("create_file", {"path": "generated.txt", "instructions": "Write two lines"}))
            self.assertIn("FileExistsError", agent.registry.execute("create_file", {"path": "generated.txt", "instructions": "Replace"}))
            self.assertIn("reverted 1", agent.registry.undo(0))
            self.assertFalse(target.exists())
        finally:
            agent.close()

    def test_permissions_and_hard_deny(self) -> None:
        tools = ToolRegistry(self.root, {"shell": "allow"})
        self.assertIn("blocked", tools.execute("shell", {"command": "rm -rf /"}))
        self.assertIn("blocked", tools.execute("shell", {"command": "  rm -rf /"}))
        self.assertIn("blocked", tools.execute("shell", {"command": "echo ok\nrm -rf /"}))
        denied = ToolRegistry(self.root, {"write": "deny"})
        self.assertIn("denied", denied.execute("write_file", {"path": "x", "content": "x"}))
        asked: list[str] = []
        allowed = ToolRegistry(self.root, {"write": "ask"}, lambda category, _: asked.append(category) or "always")
        allowed.execute("write_file", {"path": "one", "content": "1"})
        allowed.execute("write_file", {"path": "two", "content": "2"})
        self.assertEqual(asked, ["write"])

    def test_clipboard_tools_use_windows_powershell_without_a_shell(self) -> None:
        tools = ToolRegistry(self.root, {"clipboard": "allow"})
        with patch("chattyplay.tools.platform.system", return_value="Windows"), patch("chattyplay.tools.shutil.which", side_effect=lambda name: "C:/pwsh.exe" if name == "pwsh" else None), patch("chattyplay.tools.subprocess.run", return_value=SimpleNamespace(stdout="copied text")) as run:
            self.assertEqual(tools.execute("read_clipboard", {}), "copied text")
            self.assertIn("copied 5", tools.execute("copy_to_clipboard", {"text": "hello"}))
        self.assertFalse(run.call_args_list[0].kwargs.get("shell", False))
        self.assertEqual(run.call_args_list[1].kwargs["input"], "hello")

    def test_config_sessions_and_skills(self) -> None:
        store = ConfigStore(self.root, self.root / "user.json")
        store.save_project({"provider": {"model": "local-model"}, "skills": {"enabled": ["demo"]}})
        self.assertEqual(store.load()["provider"]["model"], "local-model")
        skill_path = self.root / ".chattyplay" / "skills" / "demo" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("---\nname: demo\ndescription: Demo\n---\nDo it.\n", encoding="utf-8")
        self.assertIn("demo", discover(self.root, [".chattyplay/skills"]))
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "outside"
            outside.mkdir()
            (outside / "SKILL.md").write_text("---\nname: outside\n---\nDo it.\n", encoding="utf-8")
            self.assertNotIn("outside", discover(self.root, [external]))
        sessions = SessionStore(self.root)
        sessions.save("test", [{"role": "user", "content": "hi"}])
        self.assertEqual(sessions.load("test")[0]["content"], "hi")
        with self.assertRaisesRegex(ValueError, "permissions"):
            store.save_project({"permissions": []})
        with self.assertRaisesRegex(ValueError, "overlap_lines"):
            store.save_project({"rag": {"chunk_lines": 10, "overlap_lines": 10}})
        with self.assertRaisesRegex(ValueError, "max_steps"):
            store.save_project({"agent": {"max_steps": 0}})
        store.save_project({"providerProfiles": {"local": {"base_url": "http://127.0.0.1:11434/v1", "api_key_env": "", "model": "local"}}})
        self.assertEqual(store.load()["providerProfiles"]["local"]["model"], "local")

    def test_web_config_requires_token_for_writes(self) -> None:
        store = ConfigStore(self.root, self.root / "user.json")
        server, url = create_server(store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(url) as response:
                page = response.read().decode()
            token = re.search(r"const token='([^']+)'", page).group(1)
            with urllib.request.urlopen(url + "api/state") as response:
                self.assertEqual(response.status, 200)
            fetcher = ToolRegistry(self.root, {"browser": "allow"})
            self.assertIn("ChattyPlay Control Room", fetcher.execute("fetch_url", {"url": url}))
            request = urllib.request.Request(url + "api/config", data=b"{}", method="PUT", headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            skill_request = urllib.request.Request(
                url + "api/skill", method="PUT",
                data=json.dumps({"name": "web-skill", "content": "---\nname: web-skill\n---\nDo it."}).encode(),
                headers={"Content-Type": "application/json", "X-ChattyPlay-Token": token},
            )
            with urllib.request.urlopen(skill_request) as response:
                self.assertEqual(response.status, 200)
            self.assertTrue((self.root / ".chattyplay" / "skills" / "web-skill" / "SKILL.md").is_file())
            ollama_request = urllib.request.Request(
                url + "api/ollama/use", method="PUT", data=b'{"model":"local-test:latest"}',
                headers={"Content-Type": "application/json", "X-ChattyPlay-Token": token},
            )
            with urllib.request.urlopen(ollama_request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(store.load()["provider"]["model"], "local-test:latest")
            self.assertEqual(store.load()["provider"]["api_key_env"], "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_fetch_url_extracts_readable_html_and_reports_final_url(self) -> None:
        class Headers:
            def get_content_type(self) -> str:
                return "text/html"

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            headers = Headers()
            status = 200
            url = "https://example.test/final"

            def geturl(self) -> str:
                return self.url

        html = b"""<html><head><style>hidden css</style></head><body>
            <nav>navigation noise</nav><header><h1>Useful title</h1></header>
            <main><p>This is the useful article body with enough detail to answer the user's question reliably.</p>
            <ul><li>First fact</li><li>Second fact</li></ul></main>
            <footer>footer noise</footer><script>hidden script</script></body></html>"""
        tools = ToolRegistry(self.root, {"browser": "allow"})
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response(html)):
            result = tools.execute("fetch_url", {"url": "https://example.test/redirect"})
        self.assertIn("URL: https://example.test/final", result)
        self.assertIn("Useful title", result)
        self.assertIn("First fact", result)
        self.assertNotIn("navigation noise", result)
        self.assertNotIn("footer noise", result)
        self.assertNotIn("hidden script", result)

    def test_http_tools_normalize_complete_markdown_links_and_autolinks(self) -> None:
        class Headers:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            headers = Headers()
            status = 200

            def __init__(self, url: str) -> None:
                super().__init__(b"verified page content")
                self.url = url

            def geturl(self) -> str:
                return self.url

        requested: list[str] = []

        def open_url(request, timeout=0):
            requested.append(request.full_url)
            return Response(request.full_url)

        tools = ToolRegistry(self.root, {"browser": "allow"})
        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=open_url):
            tools.execute("fetch_url", {"url": "[Hello Agents](https://hello-agents.datawhale.cc/#/)"})
            tools.execute("fetch_url", {"url": "<https://example.test/docs>"})
        self.assertEqual(requested, ["https://hello-agents.datawhale.cc/", "https://example.test/docs"])

        with patch("chattyplay.tools.webbrowser.open", return_value=True) as opened:
            self.assertIn("ok: opened", tools.execute("open_browser", {"url": "[Docs (v2)](https://example.test/guide)"}))
        opened.assert_called_once_with("https://example.test/guide")

    def test_http_tools_reject_ambiguous_or_unsafe_url_values(self) -> None:
        tools = ToolRegistry(self.root, {"browser": "allow"})
        invalid = (
            "please visit https://example.test/a",
            "[bad](javascript:alert(1))",
            "https:///missing-host",
            "[one](https://one.test) [two](https://two.test)",
        )
        with patch("chattyplay.tools.urllib.request.urlopen") as request:
            for value in invalid:
                self.assertIn("error: ValueError", tools.execute("fetch_url", {"url": value}))
        request.assert_not_called()

    def test_discover_url_finds_docsify_content_from_page_evidence(self) -> None:
        class Headers:
            def __init__(self, content_type: str) -> None:
                self.content_type = content_type

            def get_content_type(self) -> str:
                return self.content_type

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            status = 200

            def __init__(self, url: str, body: bytes, content_type: str) -> None:
                super().__init__(body)
                self.url = url
                self.headers = Headers(content_type)

            def geturl(self) -> str:
                return self.url

        shell = b"""<html><head><title>Hello Agents</title></head><body><div id=app></div>
            <script>window.$docsify = {loadSidebar: true};</script>
            <script src=\"https://cdn.example.test/docsify.min.js\"></script></body></html>"""
        calls: list[str] = []

        def open_url(request, timeout=0):
            calls.append(request.full_url)
            if request.full_url == "https://docs.example.test/":
                return Response(request.full_url, shell, "text/html")
            if request.full_url == "https://docs.example.test/README.md":
                return Response(request.full_url, b"# Real content\nThis text came from the documented SPA resource.", "text/markdown")
            raise AssertionError(f"unexpected request: {request.full_url}")

        tools = ToolRegistry(self.root, {"browser": "allow"})
        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=open_url):
            failed_fetch = tools.execute("fetch_url", {"url": "[Docs](https://docs.example.test/#/)"})
            discovered = tools.execute("discover_url", {"url": "https://docs.example.test/#/", "max_candidates": 1})
        self.assertIn("unable to read this page reliably", failed_fetch)
        self.assertIn("Discovery candidates: https://docs.example.test/README.md", failed_fetch)
        self.assertIn("Detected framework: docsify", discovered)
        self.assertIn("https://docs.example.test/README.md [verified: HTTP 200, text/markdown]", discovered)
        self.assertEqual(calls, [
            "https://docs.example.test/",
            "https://docs.example.test/",
            "https://docs.example.test/README.md",
        ])

        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response(
            "https://docs.example.test/README.md",
            b"# Real content\nThis text came from the documented SPA resource.",
            "text/markdown",
        )):
            fetched = tools.execute("fetch_url", {"url": "https://docs.example.test/README.md"})
        self.assertIn("This text came from the documented SPA resource", fetched)

    def test_discover_url_does_not_guess_from_title_and_limits_safe_candidates(self) -> None:
        class Headers:
            def get_content_type(self) -> str:
                return "text/html"

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            headers = Headers()
            status = 200

            def __init__(self, url: str, body: bytes) -> None:
                super().__init__(body)
                self.url = url

            def geturl(self) -> str:
                return self.url

        tools = ToolRegistry(self.root, {"browser": "allow"})
        empty_shell = b"<html><title>Amazing AI Course</title><div id=app></div></html>"
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response("https://example.test/", empty_shell)) as request:
            result = tools.execute("discover_url", {"url": "https://example.test/#/"})
        self.assertIn("Title (metadata only): Amazing AI Course", result)
        self.assertIn("no reliable content candidates found", result)
        self.assertNotIn("README.md", result)
        self.assertEqual(request.call_count, 1)

        linked_shell = b"""<html><body>
            <a href=\"/one.md\">one</a><a href=\"/one.md\">duplicate</a>
            <a href=\"/two.md\">two</a><a href=\"javascript:alert(1)\">bad</a>
            <a href=\"mailto:test@example.test\">mail</a><a href=\"https://other.test/out.md\">external</a>
            </body></html>"""
        calls: list[str] = []

        def open_url(request, timeout=0):
            calls.append(request.full_url)
            if request.full_url == "https://example.test/":
                return Response(request.full_url, linked_shell)
            return Response(request.full_url, b"verified")

        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=open_url):
            result = tools.execute("discover_url", {"url": "https://example.test/#/", "max_candidates": 1})
        self.assertIn("https://example.test/one.md", result)
        self.assertNotIn("two.md", result)
        self.assertNotIn("other.test", result)
        self.assertEqual(calls, ["https://example.test/", "https://example.test/one.md"])

    def test_fetch_url_rejects_empty_shell_and_bad_content_type(self) -> None:
        class Headers:
            def __init__(self, content_type: str) -> None:
                self.content_type = content_type

            def get_content_type(self) -> str:
                return self.content_type

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            status = 200
            url = "https://example.test/page"

            def __init__(self, body: bytes, content_type: str) -> None:
                super().__init__(body)
                self.headers = Headers(content_type)

            def geturl(self) -> str:
                return self.url

        tools = ToolRegistry(self.root, {"browser": "allow"})
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response(b"<html><body>Enable JavaScript</body></html>", "text/html")):
            result = tools.execute("fetch_url", {"url": "https://example.test/page"})
        self.assertIn("unable to read this page reliably", result)
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response(b"binary", "application/octet-stream")):
            result = tools.execute("fetch_url", {"url": "https://example.test/file"})
        self.assertIn("unsupported Content-Type", result)

    def test_fetch_url_retries_only_transient_failures(self) -> None:
        class Headers:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "utf-8"

        class Response(io.BytesIO):
            headers = Headers()
            status = 200
            url = "https://example.test/ok"

            def geturl(self) -> str:
                return self.url

        tools = ToolRegistry(self.root, {"browser": "allow"})
        transient = urllib.error.HTTPError("https://example.test", 503, "temporary", {}, io.BytesIO())
        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=[transient, Response(b"eventually successful response")]) as request, patch("chattyplay.tools.time.sleep") as sleep:
            result = tools.execute("fetch_url", {"url": "https://example.test"})
        self.assertIn("eventually successful", result)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()

        missing = urllib.error.HTTPError("https://example.test/missing", 404, "missing", {}, io.BytesIO())
        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=missing) as request, patch("chattyplay.tools.time.sleep") as sleep:
            result = tools.execute("fetch_url", {"url": "https://example.test/missing"})
        self.assertIn("not retryable", result)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

        with patch("chattyplay.tools.urllib.request.urlopen", side_effect=TimeoutError("slow")) as request, patch("chattyplay.tools.time.sleep"):
            result = tools.execute("fetch_url", {"url": "https://example.test/slow"})
        self.assertIn("after 2 attempts", result)
        self.assertEqual(request.call_count, 2)

    def test_fetch_url_falls_back_from_unknown_charset_and_keeps_size_limit(self) -> None:
        class Headers:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "not-a-real-charset"

        class Response(io.BytesIO):
            headers = Headers()
            status = 200
            url = "https://example.test/text"

            def geturl(self) -> str:
                return self.url

        tools = ToolRegistry(self.root, {"browser": "allow"}, max_output=2_000_000)
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response("fallback text".encode())):
            self.assertIn("fallback text", tools.execute("fetch_url", {"url": "https://example.test/text"}))
        with patch("chattyplay.tools.urllib.request.urlopen", return_value=Response(b"x" * 1_000_001)):
            result = tools.execute("fetch_url", {"url": "https://example.test/large"})
        self.assertIn("response truncated at 1 MB", result)

    def test_base_prompt_requires_grounded_url_fetching(self) -> None:
        self.assertIn("call fetch_url before answering", BASE_PROMPT)
        self.assertIn("real URL target", BASE_PROMPT)
        self.assertIn("call discover_url", BASE_PROMPT)
        self.assertIn("never use it to guess page contents", BASE_PROMPT)
        self.assertIn("could not be read reliably", BASE_PROMPT)

    def test_move_file_schema_and_prompt_direct_rename_requests(self) -> None:
        tools = ToolRegistry(self.root)
        description = tools.tools["move_file"].description
        self.assertIn("Rename or move an existing file", description)
        self.assertIn("without reading or rebuilding its contents", description)
        self.assertIn("call move_file directly", BASE_PROMPT)
        self.assertIn("Do not read the file contents first", BASE_PROMPT)
        self.assertIn("never simulate a failed move with create_file, write_file, edit_file, delete_file", BASE_PROMPT)
        self.assertIn("Only report rename/move success when move_file returns `ok:`", BASE_PROMPT)
        self.assertIn("do not overwrite: report the conflict", BASE_PROMPT)

    def test_streamed_tool_call_is_reassembled(self) -> None:
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "write_", "arguments": "{\"path\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": "\"a.txt\",\"content\":\"ok\"}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        ]
        stream = io.BytesIO("".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n")
        progress: list[tuple[str, str]] = []
        message = OpenAIClient._read_stream(stream, None, lambda name, args: progress.append((name, args)))
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "write_file")
        self.assertEqual(json.loads(call["function"]["arguments"])["path"], "a.txt")
        self.assertEqual(message["_usage"]["prompt_tokens"], 10)
        self.assertEqual(progress[-1], ("write_file", '{"path":"a.txt","content":"ok"}'))
        self.assertEqual(_tool_status(*progress[-1]), "writing a.txt · 31 chars...")

    def test_openai_reasoning_options_are_only_sent_when_enabled(self) -> None:
        response = io.BytesIO(b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')
        response.headers = {"Content-Type": "application/json"}
        provider = {"api_key_env": "", "model": "test", "base_url": "http://localhost/v1", "max_tokens": 10, "thinking_enabled": True, "reasoning_effort": "high", "request_timeout": 12}
        with patch("chattyplay.client.urllib.request.urlopen", return_value=response) as request:
            OpenAIClient(provider).complete([], [])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(request.call_args.kwargs["timeout"], 12)
        self.assertNotIn("thinking", payload)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["reasoning_effort"], "high")
        no_thought = io.BytesIO(b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')
        no_thought.headers = {"Content-Type": "application/json"}
        provider.update({"thinking_enabled": False, "reasoning_effort": "none"})
        with patch("chattyplay.client.urllib.request.urlopen", return_value=no_thought) as request:
            OpenAIClient(provider).complete([], [])
        payload = json.loads(request.call_args.args[0].data)
        self.assertNotIn("thinking", payload)
        self.assertEqual(payload["reasoning_effort"], "none")
        local = io.BytesIO(b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')
        local.headers = {"Content-Type": "application/json"}
        provider.update({"base_url": "http://127.0.0.1:11434/v1", "reasoning_effort": "medium"})
        with patch("chattyplay.client.urllib.request.urlopen", return_value=local) as request:
            OpenAIClient(provider).complete([], [])
        self.assertEqual(json.loads(request.call_args.args[0].data)["reasoning_effort"], "none")

    def test_anthropic_messages_and_stream_are_normalized(self) -> None:
        messages = AnthropicClient._messages([
            {"role": "user", "content": [{"type": "text", "text": "inspect"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        ])
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["source"]["media_type"], "image/png")
        self.assertEqual(messages[1]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":\"a\"}"}},
        ]
        stream = io.BytesIO("".join(f"data: {json.dumps(e)}\n\n" for e in events).encode())
        progress: list[tuple[str, str]] = []
        call = AnthropicClient._read_stream(stream, None, lambda name, args: progress.append((name, args)))["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a"})
        self.assertEqual(progress, [("read_file", '{"path":"a"}')])
        self.assertIsInstance(create_client({"api_style": "anthropic"}), AnthropicClient)

    def test_stdio_mcp_tool_registration_and_call(self) -> None:
        registry = ToolRegistry(self.root, {"mcp": "allow"})
        manager = MCPManager()
        server = Path(__file__).with_name("fake_mcp_server.py")
        try:
            manager.connect({"demo": {"command": sys.executable, "args": [str(server)]}}, registry, self.root)
            self.assertEqual(manager.status["demo"], "connected (1 tools)")
            result = json.loads(registry.execute("mcp__demo__echo", {"text": "hello"}))
            self.assertEqual(result["content"][0]["text"], "hello")
        finally:
            manager.close()

    def test_context_trimming_keeps_complete_recent_turn(self) -> None:
        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 1000
        agent = Agent(self.root, config)
        try:
            agent.messages = [
                {"role": "user", "content": "original goal"},
                {"role": "assistant", "content": "starting"},
                {"role": "user", "content": "middle" * 500},
                {"role": "assistant", "content": "middle answer"},
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "x", "content": "result"},
            ]
            active = agent.context_messages()
            self.assertEqual(active[0]["content"], "new")
            self.assertEqual(active[-1]["role"], "tool")
            self.assertLessEqual(len(json.dumps(active, ensure_ascii=False)), 1000)
            self.assertNotIn("original goal", [message.get("content") for message in active])
            self.assertNotIn("middle" * 500, [message.get("content") for message in active])
        finally:
            agent.close()

    def test_context_does_not_truncate_oversized_current_turn(self) -> None:
        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 1000
        agent = Agent(self.root, config)
        try:
            agent.messages = [{"role": "user", "content": "x" * 2000}]
            self.assertEqual(agent.context_messages(), agent.messages)
        finally:
            agent.close()

    def test_auto_compact_summarizes_old_turns_and_keeps_recent_turns(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.source = ""

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.source = messages[-1]["content"]
                return {"role": "assistant", "content": "old work summary"}

        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 1000
        agent = Agent(self.root, config)
        fake = FakeClient()
        try:
            agent.messages = [
                {"role": "user", "content": "oldest " * 100},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
            ]
            with patch("chattyplay.agent.create_client", return_value=fake) as create:
                self.assertTrue(agent._auto_compact())
            self.assertIn("oldest", fake.source)
            compact_provider = create.call_args.args[0]
            self.assertFalse(compact_provider["thinking_enabled"])
            self.assertEqual(compact_provider["reasoning_effort"], "none")
            self.assertLessEqual(compact_provider["max_tokens"], 1536)
            self.assertEqual(compact_provider["request_timeout"], 30)
            self.assertEqual(agent.messages[-2]["content"], "recent question")
            self.assertEqual(agent.messages[-1]["content"], "recent answer")
            self.assertIn("old work summary", agent.messages[0]["content"])
            self.assertEqual(SessionStore(self.root).load(agent.session_id), agent.messages)
            self.assertFalse(agent._auto_compact())
            agent.messages.extend([{"role": "user", "content": "small follow-up"}, {"role": "assistant", "content": "small answer"}])
            self.assertFalse(agent._auto_compact())
        finally:
            agent.close()

    def test_auto_compact_bounds_large_tool_history_and_preserves_key_context(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.source = ""

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.source = messages[-1]["content"]
                return {"role": "assistant", "content": "bounded summary"}

        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 10000
        config["provider"]["max_tokens"] = 8192
        agent = Agent(self.root, config)
        fake = FakeClient()
        try:
            agent.messages = [
                {"role": "user", "content": "original user goal must survive"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "function": {"name": "read_file", "arguments": "{\"path\":\"large.txt\"}"}}]},
                {"role": "tool", "tool_call_id": "x", "content": "tool-start\n" + "x" * 50000 + "\ntool-end"},
                {"role": "user", "content": "important old decision " + "d" * 6000},
                {"role": "assistant", "content": "decision recorded"},
                {"role": "user", "content": "recent active question"},
                {"role": "assistant", "content": "recent active answer"},
            ]
            with patch("chattyplay.agent.create_client", return_value=fake):
                self.assertTrue(agent._auto_compact())
            self.assertIn("original user goal must survive", fake.source)
            self.assertIn("important old decision", fake.source)
            self.assertIn("tool-start", fake.source)
            self.assertIn("tool-end", fake.source)
            self.assertLess(len(fake.source), 5500)
            self.assertEqual(agent.messages[-2]["content"], "recent active question")
            self.assertEqual(agent.messages[-1]["content"], "recent active answer")
            self.assertLess(len(json.dumps(agent.messages, ensure_ascii=False)), 5000)
        finally:
            agent.close()

    def test_auto_compact_failure_waits_for_context_growth_before_retry(self) -> None:
        class FailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.calls += 1
                raise RuntimeError("summary unavailable")

        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 10000
        agent = Agent(self.root, config)
        failing = FailingClient()
        try:
            agent.messages = [
                {"role": "user", "content": "old goal " + "x" * 10000},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "recent"},
                {"role": "assistant", "content": "answer"},
            ]
            with patch("chattyplay.agent.create_client", return_value=failing):
                self.assertFalse(agent._auto_compact())
                self.assertFalse(agent._auto_compact())
                self.assertEqual(failing.calls, 1)
                agent.messages.append({"role": "user", "content": "growth " + "y" * 2500})
                self.assertFalse(agent._auto_compact())
                self.assertEqual(failing.calls, 2)
        finally:
            agent.close()

    def test_manual_compact_includes_history_outside_active_context(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.source = ""

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.source = messages[-1]["content"]
                return {"role": "assistant", "content": "summary"}

        config = ConfigStore(self.root, self.root / "user.json").load()
        config["agent"]["max_context_chars"] = 1000
        agent = Agent(self.root, config)
        fake = FakeClient()
        agent.client = fake
        try:
            agent.messages = [
                {"role": "user", "content": "forgotten " * 300},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "latest"},
                {"role": "assistant", "content": "latest answer"},
            ]
            self.assertIn("ok: compacted", agent.compact())
            self.assertIn("forgotten", fake.source)
        finally:
            agent.close()

    def test_agent_sends_previous_turn_with_follow_up(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.requests.append(messages)
                return {"role": "assistant", "content": "first answer" if len(self.requests) == 1 else "second answer"}

        agent = Agent(self.root, ConfigStore(self.root, self.root / "user.json").load())
        fake = FakeClient()
        agent.client = fake
        try:
            agent.run("first question")
            agent.run("follow up")
            history = fake.requests[1][1:]
            self.assertEqual([message["content"] for message in history], ["first question", "first answer", "follow up"])
            self.assertEqual(SessionStore(self.root).load(agent.session_id), agent.messages)
        finally:
            agent.close()

    def test_agent_uses_lightweight_tool_gate_before_full_schema(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.schemas: list[list[dict]] = []

            def complete(self, messages: list[dict], tools: list[dict], on_text=None, on_tool_delta=None) -> dict:
                self.schemas.append(tools)
                if len(self.schemas) == 1:
                    return {"role": "assistant", "content": None, "tool_calls": [{"id": "gate", "function": {"name": "activate_tools", "arguments": "{}"}}]}
                return {"role": "assistant", "content": "done"}

        config = ConfigStore(self.root, self.root / "user.json").load()
        agent = Agent(self.root, config)
        fake = FakeClient()
        agent.client = fake
        try:
            self.assertEqual(agent.run("inspect the repository"), "done")
            self.assertEqual([tool["function"]["name"] for tool in fake.schemas[0]], ["activate_tools"])
            self.assertIn("read_file", [tool["function"]["name"] for tool in fake.schemas[1]])
        finally:
            agent.close()

    def test_agent_plan_mentions_fork_and_export(self) -> None:
        (self.root / "note.txt").write_text("important context", encoding="utf-8")
        (self.root / "screen.png").write_bytes(b"\x89PNG\r\n")
        config = ConfigStore(self.root, self.root / "user.json").load()
        agent = Agent(self.root, config)
        try:
            self.assertIn("delegate_task", agent.registry.tools)
            attached = agent._attach_mentions("review @note.txt")
            self.assertIn("important context", attached)
            multimodal = agent._attach_mentions("inspect @screen.png and @note.txt")
            self.assertIsInstance(multimodal, list)
            self.assertIn("important context", multimodal[0]["text"])
            self.assertTrue(multimodal[1]["image_url"]["url"].startswith("data:image/png;base64,"))
            agent.config["agent"]["max_image_bytes"] = 4
            with self.assertRaisesRegex(ValueError, "exceed"):
                agent._attach_mentions("inspect @screen.png")
            agent.messages = [{"role": "user", "content": multimodal}]
            exported_image = agent.export("image-session.md").read_text(encoding="utf-8")
            self.assertIn("[local image attachment]", exported_image)
            self.assertNotIn("iVBOR", exported_image)
            agent.set_plan_mode(True)
            self.assertIn("unavailable in plan mode", agent.registry.execute("write_file", {"path": "blocked", "content": "x"}))
            agent.messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
            original = agent.session_id
            self.assertNotEqual(agent.fork(), original)
            exported = agent.export("session.md")
            self.assertIn("## assistant", exported.read_text(encoding="utf-8"))
        finally:
            agent.close()

    def test_interaction_tools(self) -> None:
        plans: list[list[dict]] = []
        registry = ToolRegistry(
            self.root,
            {"interaction": "allow"},
            ask_user=lambda question: f"answer to {question}",
            on_plan=plans.append,
        )
        self.assertEqual(registry.execute("ask_user", {"question": "continue?"}), "answer to continue?")
        result = registry.execute("update_plan", {"steps": [{"step": "test", "status": "in_progress"}]})
        self.assertIn("plan updated", result)
        self.assertEqual(plans[0][0]["step"], "test")

    def test_shell_result_is_readable_in_terminal(self) -> None:
        result = _format_shell_result('{"exit_code": 2, "stdout": "built\\n", "stderr": "failed\\n"}')
        self.assertEqual(result, "built\nfailed\n[exit 2]")
        self.assertEqual(_format_shell_result("error: blocked"), "error: blocked")
        self.assertTrue(_is_command("/model local", "/model"))
        self.assertFalse(_is_command("/modelx", "/model"))

    def test_rag_indexes_and_semantically_searches_project(self) -> None:
        (self.root / "alpha.py").write_text("def login():\n    return 'session token'\n", encoding="utf-8")
        (self.root / "beta.py").write_text("def invoice():\n    return 'payment total'\n", encoding="utf-8")
        (self.root / ".mypy_cache").mkdir()
        (self.root / ".mypy_cache" / "noise.json").write_text('{"noise": true}', encoding="utf-8")
        (self.root / "demo.egg-info").mkdir()
        (self.root / "demo.egg-info" / "sources.txt").write_text("noise", encoding="utf-8")

        def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "login" in text.lower() else [0.0, 1.0] for text in texts]

        rag = RAGIndex(self.root, {"chunk_lines": 80, "overlap_lines": 10, "top_k": 1}, embed)
        self.assertEqual(rag.index(), {"files": 2, "chunks": 2})
        self.assertIn("alpha.py:1", rag.search("login flow"))
        self.assertEqual(rag.status()["chunks"], 2)
        (self.root / "alpha.py").write_text("def login():\n    return 'changed session'\n", encoding="utf-8")
        self.assertTrue(rag.status()["stale"])
        self.assertIn("stale", rag.search("login flow"))
        rag.path.write_bytes(b"not a database")
        self.assertIn("error", rag.status())
        self.assertEqual(rag.index()["files"], 2)

    def test_project_wiki_build_read_and_staleness(self) -> None:
        (self.root / "README.md").write_text("# Demo\n", encoding="utf-8")
        wiki = ProjectWiki(self.root)
        result = wiki.build(lambda prompt: "```markdown\n# Generated\n\n## Overview\nSee [README.md](../../README.md).\n```")
        self.assertEqual(result["files"], 1)
        self.assertTrue(wiki.status()["built"])
        self.assertIn("# Project Wiki", wiki.read())
        (self.root / "README.md").write_text("# Changed\n", encoding="utf-8")
        self.assertTrue(wiki.status()["stale"])
        self.assertIn("WARNING", wiki.read())
        with self.assertRaisesRegex(RuntimeError, "empty wiki section"):
            wiki.build(lambda _: "")
        with self.assertRaisesRegex(RuntimeError, "without sufficient valid source citations"):
            wiki.build(lambda _: "See [missing](../../missing.py).")
        linked = wiki.build(lambda _: "See `README.md` for details.")
        self.assertEqual(linked["files"], 1)
        self.assertIn("[README.md](../../README.md)", wiki.read())
        self.assertIn("# Project Wiki", wiki.read())
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(RuntimeError, "no source files"):
                ProjectWiki(Path(empty)).build(lambda _: "unused")

    def test_agent_does_not_start_mcp_without_permission(self) -> None:
        config = ConfigStore(self.root, self.root / "user.json").load()
        config["mcpServers"] = {"untrusted": {"command": "anything"}, "off": {"disabled": True}}
        config["permissions"]["mcp"] = "ask"
        with patch("chattyplay.agent.MCPManager.connect") as connect:
            agent = Agent(self.root, config, confirm=lambda *_: False)
            try:
                connect.assert_not_called()
                self.assertEqual(agent.mcp.status["untrusted"], "permission denied")
                self.assertEqual(agent.mcp.status["off"], "disabled")
            finally:
                agent.close()

    def test_rag_skips_symlinks_and_reload_preserves_undo(self) -> None:
        outside = Path(self.temp.name).parent / (self.root.name + "-outside.py")
        outside.write_text("secret = 'outside'\n", encoding="utf-8")
        try:
            try:
                (self.root / "linked.py").symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")
            rag = RAGIndex(self.root, {"chunk_lines": 80, "overlap_lines": 10}, lambda texts: [[1.0] for _ in texts])
            self.assertEqual(rag.index()["files"], 0)
            tools = ToolRegistry(self.root, {"read": "allow"})
            self.assertEqual(tools.execute("glob_files", {"pattern": "*.py"}), "none")
            self.assertEqual(tools.execute("grep_files", {"pattern": "secret", "glob": "*.py"}), "none")
            store = ConfigStore(self.root, self.root / "user.json")
            store.save_project({"permissions": {"write": "allow"}})
            agent = Agent(self.root, store.load())
            agent.registry.execute("write_file", {"path": "created.txt", "content": "x"})
            agent.turns.append((0, 0))
            replacement = _reload_agent(agent, self.root, store, True)
            try:
                self.assertIn("reverted 1", replacement.undo())
                self.assertFalse((self.root / "created.txt").exists())
            finally:
                replacement.close()
        finally:
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
