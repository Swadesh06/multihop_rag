"""Phase 9: supporting chunk mapping + adversarial distractor selection."""
from __future__ import annotations
import re
import numpy as np
from typing import Any


def map_span_to_chunk(span: str, chunks: list[dict]) -> dict | None:
    """Exact-substring mapping; returns chunk dict + char range within it, or None."""
    if not span or not chunks:
        return None
    s_clean = span.strip().strip('"\'')
    for ch in chunks:
        ct = ch.get("chunk_text") or ""
        i = ct.find(s_clean)
        if i >= 0:
            return {
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "start_char_in_chunk": i,
                "end_char_in_chunk": i + len(s_clean),
                "start_char": ch.get("start_char", 0) + i,
                "end_char": ch.get("start_char", 0) + i + len(s_clean),
                "text": s_clean,
            }
    return None


def embed_similarity_best_chunk(span_emb: np.ndarray, chunk_embs: np.ndarray,
                                chunk_meta: list[dict]) -> tuple[dict, float]:
    s = span_emb.reshape(1, -1).astype(np.float32)
    c = chunk_embs.astype(np.float32)
    sims = (s @ c.T).ravel()
    i = int(np.argmax(sims))
    return chunk_meta[i], float(sims[i])
