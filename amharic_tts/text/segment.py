"""Sentence splitting and chunking for long-form synthesis."""
from __future__ import annotations

import re

# Sentence terminators after normalisation (Ethiopic ። is already mapped to '.')
_SENT_END = re.compile(r"(?<=[.!?;።፧])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,:፣፤፥])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return parts


def chunk_text(text: str, max_chars: int = 220, min_chars: int = 20) -> list[str]:
    """Group sentences into chunks of at most ``max_chars`` characters.

    Ethiopic characters are syllables, so 220 characters is roughly 35-45
    seconds of speech; the engine may split further internally.  Sentences
    longer than ``max_chars`` are split at clause punctuation, and as a last
    resort at word boundaries.
    """
    chunks: list[str] = []
    cur = ""
    for sent in split_sentences(text):
        pieces = [sent] if len(sent) <= max_chars else _split_long(sent, max_chars)
        for p in pieces:
            if cur and len(cur) + 1 + len(p) > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    # merge very short trailing chunks into the previous one
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + len(c) + 1 <= max_chars * 1.2:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    return merged


def _split_long(sent: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for clause in _CLAUSE_END.split(sent):
        if len(clause) > max_chars:
            words = clause.split()
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            continue
        if cur and len(cur) + 1 + len(clause) > max_chars:
            out.append(cur)
            cur = clause
        else:
            cur = f"{cur} {clause}".strip()
    if cur:
        out.append(cur)
    return out


__all__ = ["split_sentences", "chunk_text"]
