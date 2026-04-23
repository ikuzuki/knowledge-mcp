"""LanceDB-backed vector store for chunk embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
import pyarrow as pa

from knowledge_mcp.types import Chunk, SearchHit


def _build_schema(dim: int) -> pa.Schema:
    """PyArrow schema used to create and validate the LanceDB table."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("heading_path", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("model_version", pa.string()),
        ]
    )


def _escape_sql_literal(value: str) -> str:
    """Escape a string for use inside a single-quoted SQL literal."""
    return value.replace("'", "''")


class VectorStore:
    """LanceDB-backed vector store for chunks + embeddings."""

    def __init__(
        self,
        db_path: Path,
        *,
        table_name: str = "chunks",
        dim: int = 768,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.dim = dim
        self._schema = _build_schema(dim)
        self._db = lancedb.connect(str(self.db_path))
        self._table = self._open_or_create_table()

    def _open_or_create_table(self):
        if self.table_name in self._db.list_tables():
            return self._db.open_table(self.table_name)
        empty = pa.Table.from_pylist([], schema=self._schema)
        return self._db.create_table(self.table_name, data=empty, schema=self._schema)

    # ------------------------------------------------------------------ writes

    def upsert_chunks(
        self,
        file_path: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        model_version: str,
    ) -> None:
        """Atomically replace all rows for ``file_path`` with the provided chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"len(chunks)={len(chunks)} does not match len(embeddings)={len(embeddings)}"
            )
        for i, emb in enumerate(embeddings):
            if len(emb) != self.dim:
                raise ValueError(
                    f"embedding[{i}] has length {len(emb)}, expected {self.dim}"
                )

        # Delete first so an empty `chunks` still purges stale rows.
        self.delete_file(file_path)

        if not chunks:
            return

        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append(
                {
                    "id": f"{file_path}::{chunk.chunk_index}",
                    "file_path": file_path,
                    "chunk_index": int(chunk.chunk_index),
                    "heading_path": json.dumps(list(chunk.heading_path)),
                    "content": chunk.text,
                    "embedding": [float(x) for x in emb],
                    "model_version": model_version,
                }
            )
        table = pa.Table.from_pylist(rows, schema=self._schema)
        self._table.add(table)

    def delete_file(self, file_path: str) -> None:
        """Delete all rows matching ``file_path``."""
        escaped = _escape_sql_literal(file_path)
        self._table.delete(f"file_path = '{escaped}'")

    def clear(self) -> None:
        """Drop and recreate the table, leaving it empty."""
        empty = pa.Table.from_pylist([], schema=self._schema)
        self._table = self._db.create_table(
            self.table_name, data=empty, schema=self._schema, mode="overwrite"
        )

    # ------------------------------------------------------------------ reads

    def search(self, query_embedding: list[float], limit: int = 20) -> list[SearchHit]:
        """Vector search. Returns hits scored ``1 / (1 + distance)`` (higher is better)."""
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"query embedding has length {len(query_embedding)}, expected {self.dim}"
            )
        # Empty table: LanceDB may error; guard via count.
        if self._table.count_rows() == 0:
            return []
        rows = (
            self._table.search(list(query_embedding))
            .limit(limit)
            .to_list()
        )
        hits: list[SearchHit] = []
        for row in rows:
            try:
                heading_path = json.loads(row.get("heading_path", "[]"))
            except (json.JSONDecodeError, TypeError):
                heading_path = []
            distance = float(row.get("_distance", 0.0))
            hits.append(
                SearchHit(
                    path=row["file_path"],
                    heading_path=heading_path,
                    content=row["content"],
                    score=1.0 / (1.0 + distance),
                    chunk_index=int(row["chunk_index"]),
                )
            )
        # LanceDB returns ascending distance; score descending. Sort defensively.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def model_versions_present(self) -> set[str]:
        """Return the distinct set of ``model_version`` values currently in the table."""
        if self._table.count_rows() == 0:
            return set()
        arrow_tbl = self._table.to_arrow()
        col = arrow_tbl.column("model_version").to_pylist()
        return {v for v in col if v is not None}

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """No-op; LanceDB manages its own file handles."""
        return None
