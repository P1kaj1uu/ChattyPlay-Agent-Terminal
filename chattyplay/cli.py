from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
import platform
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .agent import Agent
from .config import ConfigStore
from .ollama import CHAT_MODEL, EMBED_MODEL, provider as ollama_provider, pull as ollama_pull, status as ollama_status
from .sessions import SessionStore
from .terminal import Spinner, TerminalInput, welcome_screen
from .tools import write_clipboard
from .webui import create_server, run as run_web


GREEN, CYAN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[36m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = CYAN = YELLOW = RED = DIM = RESET = ""


def _enable_windows_ansi() -> None:
    if os.name != "nt" or not sys.stdout.isatty():
        return
    try:
        import ctypes
        kernel32 = getattr(ctypes, "windll").kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError):
        pass


_enable_windows_ansi()


def confirm(category: str, preview: str, store: ConfigStore | None = None) -> bool | str:
    try:
        answer = input(f"\n{YELLOW}? allow {category}: {preview} [y/N/a=session/p=project] {RESET}").strip().lower()
        if answer in {"p", "project"} and store:
            project = store.project_config()
            project.setdefault("permissions", {})[category] = "allow"
            store.save_project(project)
            return "always"
        return "always" if answer in {"a", "always"} else answer in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _print_help() -> None:
    print("""Commands:
  /help              show this help
  /new               start a new conversation
  /resume [id]       list or resume saved sessions
  /fork              copy the conversation into a new session
  /compact           summarize old context with the current model
  /export [path]     export the conversation as Markdown
  /model [name]      show or change the project model
  /provider [name]   list or hot-switch configured provider profiles
  /thinking [off|low|medium|high|max]  configure model reasoning
  /reload            reload model, permissions, Skills and MCP config
  /plan [on|off]     toggle read-only planning mode
  /skills [enable|disable name] configure project Skills
  /mcp               show MCP connection status
  /config            show the project config path
  /init              create a minimal AGENTS.md
  /undo              revert the last turn's file-tool edits and messages
  /copy              copy the latest assistant reply to the clipboard
  /context           show active conversation context size
  /stats             show model requests, tokens, and tool calls
  /doctor            check runtime, model, workspace, Skills and MCP
  /run <command>     run a shell command through the permission and safety layer
  /ollama [status|use [model]|pull [model]]  manage keyless local models
  /rag [status|index|search] manage project semantic search
  /wiki [status|build|show] manage the LLM-compiled project wiki
  /web                open the visual config editor
  /exit               quit
""")


def _ask_user(question: str) -> str:
    try:
        return input(f"\n{CYAN}? {question}\nanswer ❯ {RESET}")
    except (EOFError, KeyboardInterrupt):
        return "User cancelled the question."


def _show_plan(steps: list[dict[str, Any]]) -> None:
    symbols = {"completed": "✓", "in_progress": "→", "pending": "·"}
    print("\n" + "\n".join(f"  {symbols.get(item['status'], '·')} {item['step']}" for item in steps))


def _format_shell_result(result: str) -> str:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict) or "exit_code" not in payload:
        return result
    output = str(payload.get("stdout", "")).rstrip()
    error = str(payload.get("stderr", "")).rstrip()
    body = "\n".join(part for part in (output, error) if part)
    return f"{body + chr(10) if body else ''}[exit {payload['exit_code']}]"


def _is_command(prompt: str, name: str) -> bool:
    return prompt == name or prompt.startswith(name + " ")


def _confirmation(store: ConfigStore, yes: bool, interactive_io: bool = True) -> Callable[[str, str], bool | str]:
    def decide(category: str, preview: str) -> bool | str:
        return True if yes else confirm(category, preview, store) if interactive_io else False
    return decide


def _new_agent(workspace: Path, store: ConfigStore, yes: bool, session_id: str | None = None, interactive_io: bool = True) -> Agent:
    messages = SessionStore(workspace).load(session_id) if session_id else None
    return Agent(
        workspace,
        store.load(),
        confirm=_confirmation(store, yes, interactive_io),
        ask_user=_ask_user if interactive_io else None,
        on_plan=_show_plan if interactive_io else None,
        session_id=session_id,
        messages=messages,
    )


def _reload_agent(agent: Agent, workspace: Path, store: ConfigStore, yes: bool) -> Agent:
    replacement = Agent(
        workspace,
        store.load(),
        confirm=_confirmation(store, yes),
        ask_user=_ask_user,
        on_plan=_show_plan,
        session_id=agent.session_id,
        messages=agent.messages,
    )
    replacement.usage = agent.usage
    replacement.turns = agent.turns
    replacement.registry.changes = agent.registry.changes
    replacement.registry.always_allowed = agent.registry.always_allowed
    replacement.set_plan_mode(agent.plan_mode)
    agent.close()
    return replacement


def _doctor(agent: Agent) -> str:
    provider = agent.config["provider"]
    key_env = provider.get("api_key_env", "")
    local = ollama_status()
    rag = agent.rag.status()
    rag_health = "invalid: " + rag["error"] if rag.get("error") else "stale; rebuild with /rag index" if rag.get("stale") else f"{rag['chunks']} indexed chunk(s)" if rag.get("indexed") else "not indexed"
    wiki = agent.wiki.status()
    wiki_health = "invalid: " + str(wiki["error"]) if wiki.get("error") else "stale; rebuild with /wiki build" if wiki.get("stale") else f"compiled from {wiki.get('files', 0)} file(s)" if wiki.get("built") else "not built"
    checks = [
        f"python: {platform.python_version()} ({platform.system()})",
        f"workspace: {'writable' if os.access(agent.workspace, os.W_OK) else 'read-only'} · {agent.workspace}",
        f"provider: {provider.get('api_style', 'openai')} · {provider['model']} · {provider['base_url']}",
        f"api key: {'not required' if not key_env else 'set' if os.environ.get(key_env) else 'missing: ' + key_env}",
        f"skills: {len(agent.skills)} discovered · {len(agent.config.get('skills', {}).get('enabled', []))} enabled",
        "mcp: " + (", ".join(f"{name}={status}" for name, status in agent.mcp.status.items()) or "none"),
        f"ollama: {'running' if local['running'] else 'installed, stopped' if local['installed'] else 'not installed'} · {len(local['models'])} model(s)",
        f"rag: {rag_health}",
        f"wiki: {wiki_health}",
    ]
    return "\n".join(checks)


def interactive(workspace: Path, store: ConfigStore, yes: bool = False, resume_id: str | None = None) -> int:
    try:
        agent = _new_agent(workspace, store, yes, resume_id)
    except Exception as exc:
        print(f"{RED}startup error: {exc}{RESET}", file=sys.stderr)
        return 1
    print(welcome_screen(
        agent.config["provider"]["model"], workspace, agent.session_id,
        SessionStore(workspace).list(), accent=GREEN, muted=DIM, reset=RESET,
    ) + "\n")
    terminal = TerminalInput(
        workspace,
        lambda: f" {agent.config['provider']['model']} · {agent.session_id} · {'PLAN' if agent.plan_mode else 'BUILD'} · Ctrl+J newline · Shift+Tab mode ",
        lambda: agent.set_plan_mode(not agent.plan_mode),
    )
    server = None
    while True:
        try:
            prompt = terminal.prompt(f"{CYAN}you ❯ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit", "exit"}:
            break
        if prompt == "/help":
            _print_help(); continue
        if prompt == "/new":
            agent.close(); agent = _new_agent(workspace, store, yes)
            print(f"{GREEN}new session {agent.session_id}{RESET}"); continue
        if _is_command(prompt, "/resume"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                print("\n".join(SessionStore(workspace).list()[:30]) or "no sessions")
            else:
                try:
                    next_agent = _new_agent(workspace, store, yes, parts[1])
                    agent.close(); agent = next_agent
                    print(f"{GREEN}resumed {agent.session_id}{RESET}")
                except Exception as exc:
                    print(f"{RED}{exc}{RESET}")
            continue
        if prompt == "/fork":
            print(f"{GREEN}forked session {agent.fork()}{RESET}")
            continue
        if prompt == "/compact":
            try:
                print(f"{GREEN}{agent.compact()}{RESET}")
            except Exception as exc:
                print(f"{RED}compact failed: {exc}{RESET}")
            continue
        if _is_command(prompt, "/export"):
            parts = prompt.split(maxsplit=1)
            try:
                print(f"{GREEN}exported {agent.export(parts[1] if len(parts) == 2 else None)}{RESET}")
            except (OSError, PermissionError) as exc:
                print(f"{RED}{exc}{RESET}")
            continue
        if _is_command(prompt, "/model"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                print(agent.config["provider"]["model"])
            else:
                project = store.project_config()
                project.setdefault("provider", {})["model"] = parts[1]
                store.save_project(project)
                agent.set_provider(store.load()["provider"])
                print(f"{GREEN}model: {parts[1]}{RESET}")
            continue
        if _is_command(prompt, "/provider"):
            parts = prompt.split(maxsplit=1)
            profiles = agent.config.get("providerProfiles", {})
            if len(parts) == 1:
                print("\n".join(f"- {name}: {value.get('model', '(inherits current model)')}" for name, value in profiles.items()) or "no provider profiles; configure them in /web")
            elif parts[1] not in profiles:
                print(f"{RED}unknown provider profile: {parts[1]}{RESET}")
            else:
                provider = {**agent.config["provider"], **profiles[parts[1]]}
                project = store.project_config()
                project["provider"] = provider
                store.save_project(project)
                agent.set_provider(provider)
                print(f"{GREEN}provider: {parts[1]} · {provider['model']}{RESET}")
            continue
        if _is_command(prompt, "/thinking"):
            parts = prompt.split(maxsplit=1)
            provider = agent.config["provider"]
            if len(parts) == 1:
                print(provider.get("reasoning_effort", "medium") if provider.get("thinking_enabled") else "off")
            elif parts[1].lower() not in {"off", "on", "low", "medium", "high", "max"}:
                print(f"{RED}usage: /thinking [off|low|medium|high|max]{RESET}")
            else:
                value = parts[1].lower()
                project = store.project_config()
                configured = project.setdefault("provider", {})
                configured["thinking_enabled"] = value != "off"
                configured["reasoning_effort"] = "medium" if value == "on" else "none" if value == "off" else value
                store.save_project(project)
                agent.set_provider(store.load()["provider"])
                print(f"{GREEN}thinking: {'off' if value == 'off' else configured['reasoning_effort']}{RESET}")
            continue
        if _is_command(prompt, "/plan"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 2 and parts[1].lower() not in {"on", "off", "true", "false", "1", "0"}:
                print(f"{RED}usage: /plan [on|off]{RESET}")
                continue
            plan_enabled = (not agent.plan_mode) if len(parts) == 1 else parts[1].lower() in {"on", "true", "1"}
            agent.set_plan_mode(plan_enabled)
            print(f"{GREEN}plan mode {'on' if plan_enabled else 'off'}{RESET}")
            continue
        if prompt == "/reload":
            try:
                agent = _reload_agent(agent, workspace, store, yes)
                print(f"{GREEN}configuration reloaded · {agent.config['provider']['model']}{RESET}")
            except Exception as exc:
                print(f"{RED}reload failed: {exc}{RESET}")
            continue
        if _is_command(prompt, "/skills"):
            parts = prompt.split(maxsplit=2)
            enabled_skills = set(agent.config.get("skills", {}).get("enabled", []))
            if len(parts) == 1:
                print("\n".join(f"{'*' if name in enabled_skills else '-'} {name}: {skill.description}" for name, skill in agent.skills.items()) or "no skills")
            elif len(parts) == 3 and parts[1] in {"enable", "disable"}:
                name = parts[2]
                if parts[1] == "enable" and name not in agent.skills:
                    print(f"{RED}unknown skill: {name}{RESET}")
                    continue
                enabled_skills.add(name) if parts[1] == "enable" else enabled_skills.discard(name)
                project = store.project_config()
                project.setdefault("skills", {})["enabled"] = sorted(enabled_skills)
                try:
                    store.save_project(project)
                    agent = _reload_agent(agent, workspace, store, yes)
                    print(f"{GREEN}{parts[1]}d skill: {name}{RESET}")
                except Exception as exc:
                    print(f"{RED}skill update failed: {exc}{RESET}")
            else:
                print(f"{RED}usage: /skills [enable|disable name]{RESET}")
            continue
        if prompt == "/mcp":
            print("\n".join(f"{name}: {status}" for name, status in agent.mcp.status.items()) or "no MCP servers")
            continue
        if prompt == "/config":
            print(store.project_path); continue
        if prompt == "/init":
            instructions = workspace / "AGENTS.md"
            if instructions.exists():
                print(f"{YELLOW}{instructions} already exists{RESET}")
            else:
                instructions.write_text("# Project instructions\n\nDescribe build, test, style, and safety rules here.\n", encoding="utf-8")
                print(f"{GREEN}created {instructions}{RESET}")
            continue
        if prompt == "/undo":
            result = agent.undo()
            color = RED if result.startswith("error:") else GREEN
            print(f"{color}{result}{RESET}")
            continue
        if prompt == "/copy":
            answer = next((message.get("content") for message in reversed(agent.messages) if message.get("role") == "assistant" and isinstance(message.get("content"), str) and message.get("content")), None)
            if not answer:
                print(f"{YELLOW}no assistant reply to copy{RESET}")
            else:
                try:
                    write_clipboard(answer)
                    print(f"{GREEN}copied {len(answer)} characters{RESET}")
                except Exception as exc:
                    print(f"{RED}clipboard: {exc}{RESET}")
            continue
        if prompt == "/context":
            print(agent.context_status())
            continue
        if prompt == "/stats":
            print(agent.stats())
            continue
        if prompt == "/doctor":
            print(_doctor(agent))
            continue
        if _is_command(prompt, "/run"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                print(f"{RED}usage: /run <command>{RESET}")
            else:
                result = agent.registry.execute("shell", {"command": parts[1]})
                color = RED if result.startswith("error:") else ""
                print(f"{color}{_format_shell_result(result)}{RESET}")
            continue
        if _is_command(prompt, "/ollama"):
            parts = prompt.split(maxsplit=2)
            action = parts[1].lower() if len(parts) > 1 else "status"
            model = parts[2].strip() if len(parts) > 2 else CHAT_MODEL
            try:
                if action == "pull":
                    ollama_pull(model)
                    if len(parts) < 3:
                        ollama_pull(EMBED_MODEL)
                    print(f"{GREEN}local model ready: {model}{RESET}")
                elif action == "use":
                    provider = ollama_provider(model)
                    project = store.project_config(); project["provider"] = provider; store.save_project(project)
                    agent.set_provider(provider)
                    print(f"{GREEN}provider: Ollama · {model} · no API key{RESET}")
                elif action == "status":
                    local = ollama_status()
                    print(f"installed: {local['installed']} · running: {local['running']}\n" + ("\n".join(local["models"]) or "no models"))
                else:
                    print(f"{RED}usage: /ollama [status|use [model]|pull [model]]{RESET}")
            except Exception as exc:
                print(f"{RED}Ollama: {exc}{RESET}")
            continue
        if _is_command(prompt, "/rag"):
            parts = prompt.split(maxsplit=2)
            action = parts[1].lower() if len(parts) > 1 else "status"
            try:
                if action == "index":
                    rag_result = agent.rag.index()
                    print(f"{GREEN}indexed {rag_result['files']} files / {rag_result['chunks']} chunks{RESET}")
                elif action == "search" and len(parts) == 3:
                    print(agent.rag.search(parts[2]))
                elif action == "status":
                    print(json.dumps(agent.rag.status(), indent=2, ensure_ascii=False))
                else:
                    print(f"{RED}usage: /rag [status|index|search query]{RESET}")
            except Exception as exc:
                print(f"{RED}RAG: {exc}{RESET}")
            continue
        if _is_command(prompt, "/wiki"):
            parts = prompt.split(maxsplit=1)
            action = parts[1].lower() if len(parts) == 2 else "status"
            if action == "build":
                if agent.plan_mode:
                    print(f"{RED}wiki build is unavailable in plan mode{RESET}")
                    continue
                spinner = Spinner("compiling project wiki...", style=DIM, reset=RESET)
                spinner.start()
                try:
                    wiki_result = agent.build_wiki()
                    spinner.stop()
                    print(f"{GREEN}compiled {wiki_result['files']} files → {wiki_result['path']}{RESET}")
                except Exception as exc:
                    spinner.stop()
                    print(f"{RED}Wiki: {exc}{RESET}")
            elif action == "show":
                print(agent.wiki.read())
            elif action == "status":
                print(json.dumps(agent.wiki.status(), indent=2, ensure_ascii=False))
            else:
                print(f"{RED}usage: /wiki [status|build|show]{RESET}")
            continue
        if prompt == "/web":
            if server is None:
                server, url = create_server(store)
                threading.Thread(target=server.serve_forever, daemon=True).start()
            webbrowser.open(url)
            print(f"{GREEN}{url}{RESET}"); continue
        if prompt.startswith("/"):
            print(f"{YELLOW}unknown command; use /help{RESET}"); continue
        spinner = Spinner(style=DIM, reset=RESET)
        printed = False
        def on_text(text: str) -> None:
            nonlocal printed
            spinner.stop()
            if not printed:
                print(f"{GREEN}agent ❯ {RESET}", end="", flush=True)
                printed = True
            print(text, end="", flush=True)
        def on_tool(name: str, args: dict[str, Any]) -> None:
            spinner.stop()
            print(f"\n{DIM}  → {name} {json.dumps(args, ensure_ascii=False)[:180]}{RESET}")
        try:
            answer = agent.run(prompt, on_text, on_tool, lambda _: spinner.start())
            spinner.stop()
            if not printed and answer:
                print(f"{GREEN}agent ❯ {RESET}{answer}", end="")
            print("\n")
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{YELLOW}interrupted{RESET}\n")
        except Exception as exc:
            spinner.stop()
            print(f"\n{RED}{exc}{RESET}\n")
    agent.close()
    if server:
        server.shutdown(); server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chattyplay", description="Local terminal AI coding agent")
    parser.add_argument("prompt", nargs="?", help="run one prompt and exit")
    parser.add_argument("--workspace", "-C", default=".", help="workspace directory")
    parser.add_argument("--resume", metavar="ID", help="resume a saved session")
    parser.add_argument("--web", action="store_true", help="run the visual config editor")
    parser.add_argument("--port", type=int, default=0, help="config editor port (default: random)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("--yes", "-y", action="store_true", help="approve permission prompts for this run")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--doctor", action="store_true", help="check configuration and runtime")
    parser.add_argument("--setup-local", action="store_true", help="pull and select recommended Ollama models")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        parser.error(f"workspace does not exist: {workspace}")
    store = ConfigStore(workspace)
    if args.setup_local:
        try:
            ollama_pull(CHAT_MODEL); ollama_pull(EMBED_MODEL)
            project = store.project_config(); project["provider"] = ollama_provider(); store.save_project(project)
            print(f"ready: {CHAT_MODEL} + {EMBED_MODEL}; no API key required")
            return 0
        except Exception as exc:
            print(f"setup failed: {exc}", file=sys.stderr)
            return 1
    if args.web:
        run_web(store, args.port, not args.no_open)
        return 0
    if args.doctor:
        agent = _new_agent(workspace, store, args.yes, args.resume, interactive_io=False)
        try:
            print(_doctor(agent))
            return 0
        finally:
            agent.close()
    if args.prompt:
        agent = _new_agent(workspace, store, args.yes, args.resume, interactive_io=False)
        try:
            answer = agent.run(args.prompt, lambda text: print(text, end="", flush=True), lambda name, _: print(f"\n→ {name}", file=sys.stderr))
            if answer:
                print()
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            agent.close()
    return interactive(workspace, store, args.yes, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
