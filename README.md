# knowledge-mcp

A local [MCP](https://modelcontextprotocol.io/) server that exposes a markdown
knowledge vault to Claude Desktop, Claude Code, and any other MCP client.

Provides hybrid retrieval — BM25 keyword search via SQLite FTS5 fused with
dense vector search via LanceDB — with a file watcher that keeps the indices
in sync as you edit notes in your favourite editor. Embeddings run locally
through [Ollama](https://ollama.com/).

> **Status:** v0 — plumbing only. `list_documents` and `get_document` work.
> Search, indexing, and vectors arrive in v1 and v2.

## Install

```bash
git clone https://github.com/ikuzuki/knowledge-mcp.git
cd knowledge-mcp
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev]
```

Python 3.11+ required.

## Configure

The server needs to know where your vault lives. Either:

**Env var**
```bash
export KNOWLEDGE_MCP_VAULT_PATH="/path/to/your/vault"
```

**Or a TOML file** — copy `knowledge_mcp.toml.example` to `knowledge_mcp.toml`
and fill in `vault_path`. Env vars override TOML.

## Register with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "knowledge-mcp",
      "env": {
        "KNOWLEDGE_MCP_VAULT_PATH": "/absolute/path/to/your/vault"
      }
    }
  }
}
```

Restart Claude Desktop. Ask it to *"list documents in my knowledge vault"* to
verify.

Config file locations:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

## Run standalone

```bash
KNOWLEDGE_MCP_VAULT_PATH=/path/to/vault knowledge-mcp
```

The server speaks MCP over stdio — useful only when driven by a client.

## Tools (v0)

| Tool | Purpose |
|---|---|
| `list_documents(prefix="")` | List every `.md` file in the vault. Optional path-prefix filter. |
| `get_document(path)` | Read a file; returns `{path, title, frontmatter, content}`. |

## Development

```bash
pytest
ruff check .
```

## Architecture

See [`ARCHITECTURE.md`](../ClaudeMaxing/knowledge-base/ARCHITECTURE.md) in the
sibling planning repo for full design rationale. In brief:

- **Keyword index:** SQLite FTS5 at `<vault>/.index/fts.db` (v1)
- **Vector store:** LanceDB at `<vault>/.index/vectors.lance/` (v2)
- **Embeddings:** Ollama + `nomic-embed-text:v1.5`, 768-dim (v2)
- **Fusion:** Reciprocal Rank Fusion with `k=60` (v2)
- **Chunker:** `MarkdownHeaderTextSplitter` at h2/h3, 512-token fallback
- **Watcher:** `watchdog`, 500 ms per-file debounce, blake3 hash dedup
- **Transport:** stdio only

Every vector row is stamped with `embedding_model_version`. On startup, a
mismatch between configured and stored versions surfaces a warning plus a
`reindex_all_embeddings` tool.

## Licence

MIT — see [LICENSE](LICENSE).
