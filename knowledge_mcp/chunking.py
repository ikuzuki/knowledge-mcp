"""Markdown chunking for knowledge-mcp.

Splits markdown documents into `Chunk` objects using heading-aware splitting,
with a recursive character fallback for oversized sections.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from knowledge_mcp.types import Chunk

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_markdown(
    content: str,
    frontmatter: dict | None = None,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Chunk a markdown document into heading-aware `Chunk` objects.

    Args:
        content: Raw markdown body (no frontmatter).
        frontmatter: Parsed frontmatter dict, accepted for forward-compat but
            not currently used by the chunker.
        max_tokens: Maximum tokens per chunk before fallback splitting kicks in.
        overlap_tokens: Overlap tokens used by the fallback splitter.

    Returns:
        List of `Chunk` objects in document order.
    """
    del frontmatter  # reserved for future use

    if not content or not content.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    sections = header_splitter.split_text(content)

    fallback_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
    )

    chunks: list[Chunk] = []
    index = 0

    for section in sections:
        text = (section.page_content or "").strip()
        if not text:
            continue

        metadata = section.metadata or {}
        heading_path: list[str] = []
        for _, key in _HEADERS_TO_SPLIT_ON:
            value = metadata.get(key)
            if value:
                heading_path.append(str(value))

        token_count = _token_len(text)
        if token_count <= max_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    heading_path=heading_path,
                    chunk_index=index,
                    token_count=token_count,
                )
            )
            index += 1
            continue

        for sub_text in fallback_splitter.split_text(text):
            sub_stripped = sub_text.strip()
            if not sub_stripped:
                continue
            chunks.append(
                Chunk(
                    text=sub_stripped,
                    heading_path=list(heading_path),
                    chunk_index=index,
                    token_count=_token_len(sub_stripped),
                )
            )
            index += 1

    return chunks
