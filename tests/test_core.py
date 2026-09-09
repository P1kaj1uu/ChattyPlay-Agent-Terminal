from __future__ import annotations

import json
import io
import hashlib
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
from chattyplay.cli import _format_shell_result, _reload_agent
from chattyplay.agent import Agent
from chattyplay.sessions import SessionStore
from chattyplay.mcp import MCPManager
from chattyplay.rag import RAGIndex
from chattyplay.skills import discover
from chattyplay.terminal import Spinner, welcome_screen
from chattyplay.tools import ToolRegistry
from chattyplay.webui import create_server


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_welcome_screen_adapts_to_terminal_width(self) -> None:
        wide = welcome_screen("qwen3.5:9b-q4_K_M", self.root, "current", ["previous"], 100)
        narrow = welcome_screen("本地模型", self.root, "current", [], 36)
        self.assertIn("ChattyPlay", wide)
        self.assertNotIn("v0.8.0", wide)
        self.assertIn("previous", wide)
        self.assertIn("No recent activity", narrow)
        self.assertTrue(all(get_cwidth(line) <= 100 for line in wide.splitlines()))
        self.assertTrue(all(get_cwidth(line) <= 36 for line in narrow.splitlines()))

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

    def test_file_tools_and_workspace_boundary(self) -> None:
        tools = ToolRegistry(self.root, {"read": "allow", "write": "allow"})
        checkpoint = tools.checkpoint()
        self.assertIn("ok: wrote", tools.execute("write_file", {"path": "src/a.py", "content": "x = 1\n"}))
        read = tools.execute("read_file", {"path": "src/a.py"})
        self.assertIn("x = 1", read)
        version = hashlib.sha256(b"x = 1\n").hexdigest()
        self.assertIn("ok: edited", tools.execute("edit_file", {"path": "src/a.py", "old_text": "1", "new_text": "2", "expected_sha256": version}))
        (self.root / "src" / "a.py").chmod(0o755)
        self.assertIn("stale file version", tools.execute("edit_file", {"path": "src/a.py", "old_text": "2", "new_text": "3", "expected_sha256": version}))
        self.assertIn("ok: moved", tools.execute("move_file", {"source": "src/a.py", "destination": "src/b.py"}))
        self.assertEqual((self.root / "src" / "b.py").stat().st_mode & 0o777, 0o755)
        self.assertIn("FileExistsError", tools.execute("move_file", {"source": "src/b.py", "destination": "src/b.py"}))
        self.assertIn("ok: deleted", tools.execute("delete_file", {"path": "src/b.py"}))
        self.assertIn("outside workspace", tools.execute("read_file", {"path": "../secret"}))
        self.assertIn("reverted 5", tools.undo(checkpoint))
        self.assertFalse((self.root / "src" / "a.py").exists())
        self.assertFalse((self.root / "src" / "b.py").exists())

    def test_permissions_and_hard_deny(self) -> None:
        tools = ToolRegistry(self.root, {"shell": "allow"})
        self.assertIn("blocked", tools.execute("shell", {"command": "rm -rf /"}))
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
        sessions = SessionStore(self.root)
        sessions.save("test", [{"role": "user", "content": "hi"}])
        self.assertEqual(sessions.load("test")[0]["content"], "hi")
        with self.assertRaisesRegex(ValueError, "permissions"):
            store.save_project({"permissions": []})
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

    def test_streamed_tool_call_is_reassembled(self) -> None:
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "write_", "arguments": "{\"path\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": "\"a.txt\",\"content\":\"ok\"}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        ]
        stream = io.BytesIO("".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n")
        message = OpenAIClient._read_stream(stream, None)
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "write_file")
        self.assertEqual(json.loads(call["function"]["arguments"])["path"], "a.txt")
        self.assertEqual(message["_usage"]["prompt_tokens"], 10)

    def test_openai_reasoning_options_are_only_sent_when_enabled(self) -> None:
        response = io.BytesIO(b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')
        response.headers = {"Content-Type": "application/json"}
        provider = {"api_key_env": "", "model": "test", "base_url": "http://localhost/v1", "max_tokens": 10, "thinking_enabled": True, "reasoning_effort": "high"}
        with patch("chattyplay.client.urllib.request.urlopen", return_value=response) as request:
            OpenAIClient(provider).complete([], [])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")

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
        call = AnthropicClient._read_stream(stream, None)["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a"})
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
                {"role": "user", "content": "old" * 500},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "x", "content": "result"},
            ]
            active = agent.context_messages()
            self.assertEqual(active[0]["content"], "new")
            self.assertEqual(active[-1]["role"], "tool")
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

    def test_rag_indexes_and_semantically_searches_project(self) -> None:
        (self.root / "alpha.py").write_text("def login():\n    return 'session token'\n", encoding="utf-8")
        (self.root / "beta.py").write_text("def invoice():\n    return 'payment total'\n", encoding="utf-8")

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
