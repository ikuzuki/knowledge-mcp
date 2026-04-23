"""CLI: garbage-collect index rows whose source file no longer exists on disk.

Run occasionally (weekly cron, or after large reorganisations) to keep the
indices trim. The file watcher handles individual deletes live, so this is
mostly a safety net for edits made with the server offline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from knowledge_mcp.config import load_settings
from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.storage.vectors import VectorStore


def _fts_file_paths(fts: FTSStore) -> set[str]:
    cur = fts._conn.execute("SELECT DISTINCT file_path FROM chunks")  # pyright: ignore[reportPrivateUsage]
    return {row[0] for row in cur.fetchall()}


def _vector_file_paths(vectors: VectorStore) -> set[str]:
    # Use the table handle the store already opened at init.
    table = vectors._table  # pyright: ignore[reportPrivateUsage]
    return {row["file_path"] for row in table.to_arrow().to_pylist()}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    index_dir = settings.vault_path / ".index"

    fts = FTSStore(index_dir / "fts.db")

    vectors: VectorStore | None = None
    if (index_dir / "vectors.lance").exists():
        # dim is only needed for creation; use a best-effort value here.
        vectors = VectorStore(index_dir / "vectors.lance", dim=768)

    try:
        orphans_fts = {
            path
            for path in _fts_file_paths(fts)
            if not (settings.vault_path / Path(path)).exists()
        }
        for path in orphans_fts:
            fts.delete_file(path)
        print(f"fts: removed {len(orphans_fts)} orphan files")

        if vectors is not None:
            try:
                orphans_v = {
                    path
                    for path in _vector_file_paths(vectors)
                    if not (settings.vault_path / Path(path)).exists()
                }
            except Exception:
                logging.exception("could not enumerate vector store; skipping")
                orphans_v = set()
            for path in orphans_v:
                vectors.delete_file(path)
            print(f"vectors: removed {len(orphans_v)} orphan files")
    finally:
        fts.close()
        if vectors is not None:
            vectors.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
