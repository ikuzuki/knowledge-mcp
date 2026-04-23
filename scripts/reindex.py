"""CLI: force a full rebuild of the FTS and vector indices.

Usage:
    python -m scripts.reindex               # full reindex (FTS + vectors)
    python -m scripts.reindex --embeddings-only   # wipe and rebuild vectors only
"""

from __future__ import annotations

import argparse
import logging
import sys

from knowledge_mcp.config import load_settings
from knowledge_mcp.embed import OllamaProvider
from knowledge_mcp.indexer import Indexer
from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.storage.vectors import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild knowledge-mcp indices.")
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Wipe and rebuild vectors only; leave FTS intact.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    index_dir = settings.vault_path / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)

    fts = FTSStore(index_dir / "fts.db")
    embeddings = OllamaProvider(
        endpoint=settings.embedding_endpoint,
        model_version=settings.embedding_model,
    )
    vectors = VectorStore(index_dir / "vectors.lance", dim=embeddings.dim)
    indexer = Indexer(settings.vault_path, fts, vectors=vectors, embeddings=embeddings)

    try:
        if args.embeddings_only:
            count = indexer.reindex_all_embeddings()
            print(f"re-embedded {count} files")
        else:
            # Clear both indices so we get a true rebuild, not an upsert-over-stale.
            vectors.clear()
            # FTS has no public truncate; drop+recreate via schema by reopening.
            fts.close()
            (index_dir / "fts.db").unlink(missing_ok=True)
            fts = FTSStore(index_dir / "fts.db")
            indexer.fts = fts
            count = indexer.reindex_all()
            print(f"reindexed {count} files")
    finally:
        fts.close()
        vectors.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
