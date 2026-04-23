"""SQLite FTS5 storage wrapper for chunked markdown content."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from knowledge_mcp.types import Chunk, SearchHit

# Sanitisation strategy: drop every non-word character, split into tokens, then
# wrap each token in double quotes so FTS5 treats it as a literal term. This
# neutralises operators/keywords (AND, OR, NOT, NEAR) and punctuation that would
# otherwise raise a syntax error. Multiple tokens are implicitly AND-ed.
_MATCH_SAFE_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    heading_path,
    content='chunks',
    content_rowid='id',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, heading_path)
    VALUES (new.id, new.content, new.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path)
    VALUES ('delete', old.id, old.content, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path)
    VALUES ('delete', old.id, old.content, old.heading_path);
    INSERT INTO chunks_fts(rowid, content, heading_path)
    VALUES (new.id, new.content, new.heading_path);
END;
"""


class FTSStore:
    """SQLite-backed FTS5 store for chunks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def upsert_chunks(self, file_path: str, chunks: list[Chunk]) -> None:
        """Atomically replace all chunks for `file_path`."""
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            if not chunks:
                return
            self._conn.executemany(
                "INSERT INTO chunks (file_path, chunk_index, heading_path, content) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        file_path,
                        c.chunk_index,
                        json.dumps(list(c.heading_path)),
                        c.text,
                    )
                    for c in chunks
                ],
            )

    def delete_file(self, file_path: str) -> None:
        """Remove all chunks belonging to `file_path`."""
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Full-text search via bm25. Returns hits with score = -bm25 (higher is better)."""
        if not query or not query.strip():
            return []
        cleaned = _MATCH_SAFE_RE.sub(" ", query)
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            return []
        # Quote each token so keywords like AND/OR/NEAR are treated as terms.
        sanitised = " ".join(f'"{t}"' for t in tokens)
        cur = self._conn.execute(
            """
            SELECT c.file_path, c.heading_path, c.content, c.chunk_index,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts) ASC
            LIMIT ?
            """,
            (sanitised, limit),
        )
        hits: list[SearchHit] = []
        for file_path, heading_path_json, content, chunk_index, bm25_score in cur.fetchall():
            try:
                heading_path = json.loads(heading_path_json)
            except (json.JSONDecodeError, TypeError):
                heading_path = []
            hits.append(
                SearchHit(
                    path=file_path,
                    heading_path=heading_path,
                    content=content,
                    score=-float(bm25_score),
                    chunk_index=chunk_index,
                )
            )
        return hits

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
