# Project Wiki

> Source-grounded project map. Exact behavior remains authoritative in the linked code.

## Overview

ChattyPlay is a Python terminal AI coding agent with an optional localhost configuration UI. It supports OpenAI-compatible and Anthropic APIs, local Ollama models, streaming, file and shell tools, Skills, MCP, sessions, RAG, and this project Wiki. The package entry point is `chattyplay = chattyplay.cli:main`. [README.md](../../README.md) [pyproject.toml](../../pyproject.toml)

## Runtime architecture

The CLI loads merged user/project configuration, creates an `Agent`, and then runs one prompt, the interactive terminal, the doctor, local-model setup, or the Web UI. Slash commands use exact boundaries, so `/modelx` is not mistaken for `/model`. [chattyplay/cli.py](../../chattyplay/cli.py)

`Agent` owns conversation context, usage counters, the model client, tool registry, MCP manager, RAG index, Wiki, Skills, and sessions. Each user turn initially exposes one lightweight activation tool; full tool schemas are loaded only when workspace facts or actions are needed, reducing ordinary-answer latency without removing capabilities. Tool results return to the conversation until a final answer or the step limit. [chattyplay/agent.py](../../chattyplay/agent.py)

`OpenAIClient` and `AnthropicClient` normalize provider requests, streamed text, reasoning, images, usage, and fragmented tool calls. API secrets are read from the configured environment-variable name; Ollama uses an empty name and needs no API key. [chattyplay/client.py](../../chattyplay/client.py) [chattyplay/ollama.py](../../chattyplay/ollama.py)

`ToolRegistry` provides file operations, glob/grep, shell, browser fetch/open, clipboard, questions, and plans. Paths resolve inside the workspace, writes are atomic, edits support SHA-256 stale-write checks, and file-tool changes can be undone. High-impact categories follow `allow`, `ask`, or `deny`; common destructive shell commands receive an extra hard-deny check. [chattyplay/tools.py](../../chattyplay/tools.py)

MCP servers use stdio JSON-RPC and expose remote tools as `mcp__<server>__<tool>`. Enabled servers start only after MCP permission evaluation; denied or non-interactive unapproved startup launches no process. [chattyplay/mcp.py](../../chattyplay/mcp.py) [chattyplay/agent.py](../../chattyplay/agent.py)

## Knowledge and persistence

Sessions are JSON files under `.chattyplay/sessions`; IDs are validated before load/save. Skills come from project and trusted user directories, with project-configured directories constrained to the workspace and oversized files skipped. [chattyplay/sessions.py](../../chattyplay/sessions.py) [chattyplay/skills.py](../../chattyplay/skills.py)

RAG scans supported text files while excluding symlinks, generated metadata, dependency trees, caches, and files over 1 MB. It chunks by lines, obtains embeddings through LangChain Ollama, stores vectors in SQLite, and detects stale indexes with a source fingerprint. [chattyplay/rag.py](../../chattyplay/rag.py)

The Wiki sends a bounded source snapshot to the model, requires links resolving to real workspace files, writes atomically, and records a source fingerprint. Missing, invalid, or stale states are reported without replacing a previous document on generation failure. [chattyplay/wiki.py](../../chattyplay/wiki.py)

## Configuration and interfaces

`ConfigStore` merges defaults, user configuration, and project configuration, then validates provider, profile, permission, Skill, Agent, and RAG settings. Project configuration is written atomically under `.chattyplay/config.json`; configuration stores API-key environment-variable names, not secret values. [chattyplay/config.py](../../chattyplay/config.py)

The terminal provides a responsive ASCII card, completion, history, multiline input, plan switching, streamed responses, and a TTY-only loading spinner that shows elapsed time plus the interrupt shortcut after two seconds. The Web UI listens on `127.0.0.1`, requires a random token for writes, and edits model, permission, Skill, MCP, provider-profile, RAG, and Agent configuration. [chattyplay/terminal.py](../../chattyplay/terminal.py) [chattyplay/webui.py](../../chattyplay/webui.py)

## Operations

Install with `python -m pip install -e .`; start with `chattyplay`; inspect with `chattyplay --doctor`; and open settings with `chattyplay --web`. `chattyplay --setup-local` selects `qwen3.5:9b-q4_K_M` plus `embeddinggemma` through Ollama without an API key. [README.md](../../README.md) [chattyplay/ollama.py](../../chattyplay/ollama.py)

Commands include `/run`, `/model`, `/provider`, `/thinking`, `/plan`, `/skills`, `/ollama`, `/rag`, `/wiki`, `/resume`, `/fork`, `/compact`, `/export`, `/undo`, `/doctor`, and `/web`; `/help` is authoritative. [chattyplay/cli.py](../../chattyplay/cli.py) [chattyplay/terminal.py](../../chattyplay/terminal.py)

Run `python -m unittest discover -s tests -v`, `python -m compileall -q chattyplay`, and `mypy chattyplay --ignore-missing-imports` for regression, syntax, and type checks. Tests cover providers, streams, permissions, files, MCP, RAG, Wiki, Web writes, sessions, terminal behavior, and boundaries. [tests/test_core.py](../../tests/test_core.py)

## Safety notes

Approval is the main trust boundary for shell, browser, clipboard, MCP, and writes. An approved shell command runs with the current OS user’s privileges, browser fetches can send network requests, and approved MCP servers are external processes; review previews and project configuration first. [chattyplay/tools.py](../../chattyplay/tools.py) [chattyplay/mcp.py](../../chattyplay/mcp.py)

## Source freshness

Compiled from 19 repository source files. Run `/wiki status` after source changes and `/wiki build` to regenerate. [chattyplay/wiki.py](../../chattyplay/wiki.py)
