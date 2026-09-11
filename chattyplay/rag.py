from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java", ".js", ".json",
    ".jsx", ".kt", ".md", ".php", ".py", ".rb", ".rs", ".sh", ".sql", ".swift", ".toml", ".ts",
    ".tsx", ".txt", ".vue", ".xml", ".yaml", ".yml",
}
EXCLUDED = {
    ".git", ".chattyplay", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".tox", ".venv",
    "__pycache__", "build", "coverage", "dist", "node_modules", "vendor",
}


def source_files(workspace: Path) -> Iterable[Path]:
    workspace = workspace.resolve()
    for path in workspace.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            parts = path.relative_to(workspace).parts
            if any(part in EXCLUDED or part.endswith(".egg-info") for part in parts) or path.stat().st_size > 1_000_000:
                continue
            yield path
        except OSError:
            continue


class RAGIndex:
    def __init__(self, workspace: Path, config: dict[str, Any], embed: Callable[[list[str]], list[list[float]]] | None = None):
        self.workspace = workspace.resolve()
        self.config = config
        self.path = self.workspace / ".chattyplay" / "rag.sqlite3"
        self._custom_embed = embed
        self._embeddings: Any = None

    def _langchain(self) -> Any:
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(
                model=str(self.config.get("embedding_model", "embeddinggemma")),
                base_url=str(self.config.get("base_url", "http://127.0.0.1:11434")),
            )
        return self._embeddings

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._custom_embed(texts) if self._custom_embed else self._langchain().embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        return self._custom_embed([text])[0] if self._custom_embed else self._langchain().embed_query(text)

    def _files(self) -> Iterable[Path]:
        return source_files(self.workspace)

    def _chunks(self, paths: Iterable[Path]) -> list[tuple[str, int, str]]:
        size = int(self.config.get("chunk_lines", 80))
        step = size - int(self.config.get("overlap_lines", 10))
        chunks: list[tuple[str, int, str]] = []
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            relative = path.relative_to(self.workspace).as_posix()
            for start in range(0, len(lines), step):
                text = "\n".join(lines[start : start + size]).strip()[:6000]
                if text:
                    chunks.append((relative, start + 1, text))
                if start + size >= len(lines):
                    break
        return chunks

    def index(self) -> dict[str, int]:
        paths = list(self._files())
        chunks = self._chunks(paths)
        inputs = [f"{path}:{line}\n{text}" for path, line, text in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), 32):
            vectors.extend(self._embed_documents(inputs[start : start + 32]))
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.unlink(missing_ok=True)
        with closing(sqlite3.connect(temp)) as db:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS chunks(path TEXT, line INTEGER, text TEXT, vector TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
                # ponytail: full rebuild is simpler; add incremental hashing above ~50k chunks.
                db.execute("DELETE FROM chunks")
                db.executemany("INSERT INTO chunks VALUES(?,?,?,?)", [
                    (path, line, text, json.dumps(vector, separators=(",", ":")))
                    for (path, line, text), vector in zip(chunks, vectors)
                ])
                db.execute("INSERT OR REPLACE INTO meta VALUES('model', ?)", (str(self.config.get("embedding_model", "embeddinggemma")),))
                db.execute("INSERT OR REPLACE INTO meta VALUES('sources', ?)", (self._fingerprint(paths),))
        os.replace(temp, self.path)
        return {"files": len({item[0] for item in chunks}), "chunks": len(chunks)}

    def search(self, query: str, top_k: int | None = None) -> str:
        if not self.path.is_file():
            return "RAG index is missing. Run /rag index first."
        state = self.status()
        if state.get("error") or state.get("stale"):
            return "RAG index is invalid or stale. Run /rag index to rebuild it."
        try:
            with closing(sqlite3.connect(self.path)) as db:
                rows = db.execute("SELECT path,line,text,vector FROM chunks").fetchall()
        except sqlite3.DatabaseError:
            return "RAG index is invalid. Run /rag index to rebuild it."
        vector = self._embed_query(query[:6000])
        scored = sorted(
            ((self._cosine(vector, json.loads(raw)), path, line, text) for path, line, text, raw in rows),
            reverse=True,
        )[: int(top_k or self.config.get("top_k", 6))]
        return "\n\n".join(f"{path}:{line} (score {score:.3f})\n{text}" for score, path, line, text in scored) or "No indexed text."

    def status(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"indexed": False, "stale": False, "chunks": 0, "path": str(self.path)}
        try:
            with closing(sqlite3.connect(self.path)) as db:
                count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
                meta = dict(db.execute("SELECT key,value FROM meta").fetchall())
            stale = meta.get("model") != str(self.config.get("embedding_model", "embeddinggemma")) or meta.get("sources") != self._fingerprint(self._files())
            return {"indexed": not stale, "stale": stale, "chunks": count, "path": str(self.path)}
        except sqlite3.DatabaseError as exc:
            return {"indexed": False, "stale": True, "chunks": 0, "path": str(self.path), "error": str(exc)}

    def _fingerprint(self, paths: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for path in sorted(paths):
            try:
                stat = path.stat()
                relative = path.relative_to(self.workspace).as_posix()
                digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
            except OSError:
                continue
        return digest.hexdigest()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return -1.0
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0
