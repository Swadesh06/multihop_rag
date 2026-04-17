"""Sentence split + word-packed chunking (Phase 2). blingfire primary, pysbd fallback."""
from __future__ import annotations
import re
from typing import Iterable
from .corpus import section_at_offset

try:
    import blingfire
    _BF = True
except Exception:
    _BF = False

try:
    import pysbd
    _PS = True
    _PS_SEG = None
except Exception:
    _PS = False
    _PS_SEG = None


def _pysbd_seg():
    global _PS_SEG
    if _PS_SEG is None and _PS:
        _PS_SEG = pysbd.Segmenter(language="en", clean=False)
    return _PS_SEG


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    if _BF:
        try:
            s = blingfire.text_to_sentences(text)
            sents = [x.strip() for x in s.split("\n") if x.strip()]
            if sents:
                return sents
        except Exception:
            pass
    if _PS:
        try:
            seg = _pysbd_seg()
            return [s.strip() for s in seg.segment(text) if s.strip()]
        except Exception:
            pass
    # last resort: split on . ! ?
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _word_count(s: str) -> int:
    return len(s.split())


def pack_chunks(sents: list[str], size_min: int, size_max: int,
                overlap_min: int, overlap_max: int) -> list[tuple[int, int]]:
    """Pack a sentence list into chunk index ranges (start_sent, end_sent)
    targeting size_min..size_max words, with `overlap_min..overlap_max`
    word-overlap between chunks.
    """
    if not sents:
        return []
    wc = [_word_count(s) for s in sents]
    total = sum(wc)
    if total <= size_max:
        return [(0, len(sents))]

    chunks = []
    n = len(sents)
    i = 0
    target = (size_min + size_max) // 2
    overlap_target = (overlap_min + overlap_max) // 2
    while i < n:
        j = i
        words = 0
        while j < n and words + wc[j] <= size_max:
            words += wc[j]
            j += 1
            if words >= target:
                break
        if j == i:
            j = i + 1  # force progress on mega-sentence
        chunks.append((i, j))
        if j >= n:
            break
        # Back up for overlap
        o_words = 0
        k = j
        while k > i + 1 and o_words < overlap_target:
            k -= 1
            o_words += wc[k]
        i = max(i + 1, k)
    return chunks


def _coerce_str(x) -> str:
    """Coerce potentially-ndarray / None / bytes to a plain string."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(x, str):
        return x
    # numpy / pandas object: take first element if 1-d
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            if x.ndim == 0:
                return str(x.item())
            return str(x.tolist())
    except Exception:
        pass
    return str(x)


def chunk_doc(doc: dict, size_min: int = 150, size_max: int = 300,
              overlap_min: int = 30, overlap_max: int = 50) -> list[dict]:
    """Doc -> list[chunk]. Records section_title and char offsets in clean_text."""
    clean = _coerce_str(doc.get("clean_text"))
    raw = _coerce_str(doc.get("raw_text"))
    did = doc["doc_id"]
    sents = split_sentences(clean)
    if not sents:
        return []

    # Locate each sentence's char span in clean_text (approx).
    starts = []
    cur = 0
    for s in sents:
        pos = clean.find(s, cur)
        if pos < 0:
            # tolerate minor whitespace mismatch
            pos = cur
        starts.append(pos)
        cur = pos + len(s)
    ends = [s + len(t) for s, t in zip(starts, sents)]

    ranges = pack_chunks(sents, size_min, size_max, overlap_min, overlap_max)
    out = []
    for idx, (a, b) in enumerate(ranges):
        txt = " ".join(sents[a:b])
        s_char = starts[a] if a < len(starts) else 0
        e_char = ends[b - 1] if 0 <= b - 1 < len(ends) else s_char + len(txt)
        sec = section_at_offset(raw, s_char) if raw else None
        out.append({
            "chunk_id": f"c_{doc['source_id']:06d}_{idx:03d}",
            "doc_id": did,
            "chunk_index": idx,
            "chunk_text": txt,
            "section_title": sec,
            "page_no": None,
            "word_count": _word_count(txt),
            "char_count": len(txt),
            "start_char": s_char,
            "end_char": e_char,
        })
    return out
