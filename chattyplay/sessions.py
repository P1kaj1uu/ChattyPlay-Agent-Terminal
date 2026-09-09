from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionStore:
    def __init__(self, workspace: Path):
        self.root = workspace / ".chattyplay" / "sessions"

    def new_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]

    def save(self, session_id: str, messages: list[dict]) -> None:
        if not re.fullmatch(r"[\w.-]+", session_id):
            raise ValueError("Invalid session id")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{session_id}.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    def load(self, session_id: str) -> list[dict]:
        if not re.fullmatch(r"[\w.-]+", session_id):
            raise ValueError("Invalid session id")
        data = json.loads((self.root / f"{session_id}.json").read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Invalid session")
        return data

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return [p.stem for p in sorted(self.root.glob("*.json"), reverse=True)]
