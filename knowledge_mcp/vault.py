"""Vault filesystem helpers: safe path resolution and markdown reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import frontmatter


@dataclass(frozen=True)
class DocumentSummary:
    path: str
    title: str


@dataclass(frozen=True)
class Document:
    path: str
    title: str
    frontmatter: dict
    content: str


def _resolve_inside_vault(vault_root: Path, relative: str) -> Path:
    """Resolve a vault-relative path, rejecting traversal outside the vault."""
    # Reject absolute paths and drive letters; force vault-relative semantics.
    candidate = (vault_root / relative).resolve()
    try:
        candidate.relative_to(vault_root)
    except ValueError as e:
        raise ValueError(f"path escapes vault root: {relative}") from e
    return candidate


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def list_markdown(vault_root: Path, prefix: str = "") -> list[DocumentSummary]:
    """Walk the vault, returning summaries for every ``*.md`` file.

    ``prefix`` is matched against the vault-relative POSIX path.
    """
    results: list[DocumentSummary] = []
    for md_path in sorted(vault_root.rglob("*.md")):
        rel = md_path.relative_to(vault_root).as_posix()
        if prefix and not rel.startswith(prefix):
            continue
        # Skip hidden index directories like .index/.
        if any(part.startswith(".") for part in md_path.relative_to(vault_root).parts):
            continue
        try:
            post = frontmatter.load(md_path)
        except Exception:
            # Corrupt frontmatter — fall back to raw read for title extraction.
            post = frontmatter.Post(md_path.read_text(encoding="utf-8", errors="replace"))
        title = _extract_title(post.content, md_path.stem)
        results.append(DocumentSummary(path=rel, title=title))
    return results


def read_markdown(vault_root: Path, relative_path: str) -> Document:
    target = _resolve_inside_vault(vault_root, relative_path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"document not found: {relative_path}")
    post = frontmatter.load(target)
    rel = target.relative_to(vault_root).as_posix()
    title = _extract_title(post.content, target.stem)
    return Document(
        path=rel,
        title=title,
        frontmatter=dict(post.metadata),
        content=post.content,
    )
