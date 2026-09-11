from __future__ import annotations

import shutil
import sys
import threading
import time
from itertools import cycle
from pathlib import Path
from typing import Callable, TextIO

from prompt_toolkit.utils import get_cwidth


COMMANDS = [
    "/help", "/new", "/resume", "/fork", "/compact", "/export", "/model", "/provider", "/thinking",
    "/plan", "/skills", "/mcp", "/context", "/stats", "/undo", "/copy", "/doctor", "/config",
    "/init", "/reload", "/run", "/ollama", "/rag", "/wiki", "/web", "/exit",
]

CHATTYPLAY_ART = (
    "  ____ _           _   _        ",
    " / ___| |__   __ _| |_| |_ _   _ ",
    "| |   | '_ \\ / _` | __| __| | | |",
    "| |___| | | | (_| | |_| |_| |_| |",
    " \\____|_| |_|\\__,_|\\__|\\__|\\__, |",
    "                            |___/ ",
    "        ____  _             ",
    "       |  _ \\| | __ _ _   _ ",
    "       | |_) | |/ _` | | | |",
    "       |  __/| | (_| | |_| |",
    "       |_|   |_|\\__,_|\\__, |",
    "                     |___/ ",
)


def welcome_screen(
    model: str,
    workspace: Path,
    session_id: str,
    recent: list[str],
    width: int | None = None,
    accent: str = "",
    muted: str = "",
    reset: str = "",
) -> str:
    """Build a responsive startup card without taking over the terminal."""
    columns = max(20, min(width or shutil.get_terminal_size((100, 24)).columns, 120))
    inner = columns - 2

    def fit(value: str, size: int) -> str:
        if get_cwidth(value) <= size:
            return value + " " * (size - get_cwidth(value))
        clipped = ""
        for char in value:
            if get_cwidth(clipped + char + "...") > size:
                break
            clipped += char
        result = clipped + "..."
        return result + " " * max(0, size - get_cwidth(result))

    def paint(value: str, color: str) -> str:
        return f"{color}{value}{reset}" if color else value

    label = " ChattyPlay "
    lines = [paint("+" + label + "-" * max(0, inner - get_cwidth(label)) + "+", accent)]
    info = [
        "Recent activity",
        *(recent[:3] or ["No recent activity"]),
        "",
        "Quick start",
        "/help  commands",
        "/web   visual settings",
        "/run   shell command",
        "/thinking off  faster replies",
        "/doctor environment check",
        "",
        "Model",
        model,
        f"Session  {session_id}",
    ]
    if columns >= 88:
        left_size = min(42, (inner - 1) // 2)
        right_size = inner - left_size - 1
        left = ["Welcome back!", *CHATTYPLAY_ART]
        for index in range(max(len(left), len(info))):
            a = left[index] if index < len(left) else ""
            b = info[index] if index < len(info) else ""
            style_a = accent if index == 0 else ""
            style_b = accent if b in {"Recent activity", "Quick start", "Model"} else muted
            lines.append(
                paint("|", accent) + paint(fit(" " + a, left_size), style_a) + paint("|", accent)
                + paint(fit(" " + b, right_size), style_b) + paint("|", accent)
            )
    else:
        content = ["Welcome back!", *(CHATTYPLAY_ART if columns >= 40 else ("ChattyPlay",)), "", *info]
        for value in content:
            style = accent if value in {"Welcome back!", "Recent activity", "Quick start", "Model"} else muted
            lines.append(paint("|", accent) + paint(fit(" " + value, inner), style) + paint("|", accent))
    lines.append(paint("+" + "-" * inner + "+", accent))
    lines.append(paint(fit(f"workspace  {workspace}", columns), muted))
    return "\n".join(lines)


class Spinner:
    """A small TTY-only indicator for time-to-first-token waits."""

    def __init__(self, text: str = "thinking...", stream: TextIO | None = None, style: str = "", reset: str = ""):
        self.text = text
        self.stream = stream or sys.stdout
        self.style = style
        self.reset = reset
        self.enabled = self.stream.isatty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rendered_width = 0

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.stream.write("\r" + " " * self._rendered_width + "\r")
        self.stream.flush()

    def _spin(self) -> None:
        started = time.monotonic()
        for symbol in cycle("|/-\\"):
            if self._stop.is_set():
                return
            elapsed = int(time.monotonic() - started)
            label = self.text if elapsed < 2 else f"{self.text} {elapsed}s · Ctrl+C to cancel"
            self._rendered_width = max(self._rendered_width, get_cwidth(label) + 2)
            self.stream.write(f"\r{self.style}{symbol} {label}{self.reset}")
            self.stream.flush()
            if self._stop.wait(0.1):
                return


class TerminalInput:
    def __init__(self, workspace: Path, toolbar: Callable[[], str], toggle_plan: Callable[[], None] | None = None):
        self.toolbar = toolbar
        self.session = None
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings

            history_path = workspace / ".chattyplay" / "prompt_history"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            keys = KeyBindings()

            @keys.add("c-j")
            def insert_newline(event):  # type: ignore[no-untyped-def]
                event.current_buffer.insert_text("\n")

            if toggle_plan:
                @keys.add("s-tab")
                def switch_plan(event):  # type: ignore[no-untyped-def]
                    toggle_plan()
                    event.app.invalidate()

            self.session = PromptSession(
                history=FileHistory(str(history_path)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=WordCompleter(COMMANDS, sentence=True),
                key_bindings=keys,
                complete_while_typing=False,
            )
        except ImportError:
            pass

    def prompt(self, label: str) -> str:
        if self.session is None:
            return input(label)
        from prompt_toolkit.formatted_text import ANSI

        return self.session.prompt(ANSI(label), bottom_toolbar=self.toolbar)
