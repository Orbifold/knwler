"""
Text chunking utilities.
"""

from uuid import uuid4

import tiktoken

from knwler.config import Config
from knwler.models import Chunk

# ---------------------------------------------------------------------------
# Cached encoder
# ---------------------------------------------------------------------------
_ENCODER: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Return a cached tiktoken encoder (cl100k_base)."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, config: Config) -> list[Chunk]:
    """Split text into overlapping chunks by token count."""
    enc = get_encoder()
    tokens = enc.encode(text)

    if len(tokens) <= config.max_tokens:
        return [Chunk(chunk_idx=0, text=text, id=str(uuid4()))]

    chunks: list[Chunk] = []
    start = 0

    while start < len(tokens):
        end = min(start + config.max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # Try to break at sentence boundary
        if end < len(tokens):
            for i in range(len(chunk_tokens) - 1, max(len(chunk_tokens) - 100, 0), -1):
                if enc.decode([chunk_tokens[i]]).rstrip().endswith((".", "!", "?")):
                    chunk_tokens = chunk_tokens[: i + 1]
                    end = start + i + 1
                    break

        chunks.append(
            Chunk(chunk_idx=len(chunks), text=enc.decode(chunk_tokens), id=str(uuid4()))
        )
        if end >= len(tokens):
            break
        start = end - config.overlap_tokens

    return chunks
