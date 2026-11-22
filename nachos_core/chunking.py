"""Text chunking helpers for semantic embeddings.

Pure functions over strings; no store or index access. Each entry gets a head
chunk (title + summary + start of body) plus overlapping title-prefixed body
chunks when the body exceeds ``CHUNK_SIZE`` characters.
"""

from __future__ import annotations

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_ENTRY = 16


def embed_text(title: str, summary: str, body: str) -> str:
    """Head text embedded for an entry (chunk 0): title, summary, then body head."""
    return f"{title}\n{summary}\n{body}"


def _find_split(text: str, target_end: int, min_end: int) -> int:
    """Choose a clean paragraph/line/sentence/word split in ``text[min_end:target_end]``."""
    if target_end >= len(text):
        return len(text)
    window = text[min_end:target_end]
    for sep in ("\n\n", "\n"):
        pos = window.rfind(sep)
        if pos != -1:
            return min_end + pos + len(sep)
    best_sent = max((window.rfind(sep) for sep in (". ", "! ", "? ")), default=-1)
    if best_sent != -1:
        return min_end + best_sent + 2
    space = window.rfind(" ")
    if space != -1:
        return min_end + space + 1
    return target_end


def _snap_start(text: str, raw_start: int, limit: int) -> int:
    """Avoid starting an overlapped chunk mid-word."""
    if raw_start <= 0:
        return 0
    if not text[raw_start - 1].isspace():
        window = text[raw_start:limit]
        for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
            pos = window.find(sep)
            if pos != -1:
                raw_start += pos + len(sep)
                break
    while raw_start < limit and text[raw_start].isspace():
        raw_start += 1
    return raw_start


def chunk_body(
    body: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_ENTRY,
) -> list[str]:
    """Split ``body`` into up to ``max_chunks`` overlapping ~``chunk_size``-char chunks."""
    text = (body or "").strip()
    if not text or max_chunks <= 0:
        return []
    if len(text) <= chunk_size or max_chunks == 1:
        return [text]
    step_floor = max(1, chunk_size // 2)
    effective_overlap = min(overlap, step_floor - 1) if step_floor > 1 else 0
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        target_end = start + chunk_size
        min_end = min(len(text), start + step_floor)
        end = _find_split(text, target_end, min_end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        raw_next = max(start + 1, end - effective_overlap)
        next_start = _snap_start(
            text, raw_next, min(end, raw_next + max(1, effective_overlap // 2))
        )
        start = next_start if next_start > start else end
    return chunks


def embed_chunks(
    title: str,
    summary: str,
    body: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_ENTRY,
) -> list[str]:
    """All texts to embed for an entry: chunk 0 is head, chunks 1..N-1 are titled body chunks."""
    head = embed_text(title, summary, body)
    stripped = (body or "").strip()
    if len(stripped) <= chunk_size or max_chunks <= 1:
        return [head]
    body_chunks = chunk_body(
        stripped, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks
    )
    prefix = f"{title}\n" if title else ""
    return [head, *(f"{prefix}{piece}" for piece in body_chunks[1:max_chunks])]
