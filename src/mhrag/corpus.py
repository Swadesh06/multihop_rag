"""Corpus streaming + filter + schema conversion (Phase 1)."""
from __future__ import annotations
import os, re, time, json
from urllib.parse import unquote
from pathlib import Path
from typing import Iterator, Any


def norm_target(href: str) -> str | None:
    """URL-decode + lowercase + strip. Return None for non-article hrefs."""
    if not href:
        return None
    try:
        t = unquote(href).strip()
    except Exception:
        return None
    if not t:
        return None
    tl = t.lower()
    if tl.startswith("wikt:"):
        return None
    if "#" in tl:
        return None
    # commons/wiki prefixes
    if tl.startswith(("file:", "image:", "category:", "help:", "portal:", "template:",
                      "user:", "talk:", "wikipedia:", "book:", "special:")):
        return None
    return tl


def doc_passes_basic(row: dict, min_char: int, min_links: int) -> bool:
    art = row.get("article") or ""
    if len(art) < min_char:
        return False
    links = row.get("links") or []
    if len(links) < min_links:
        return False
    return True


def extract_link_targets(row: dict) -> list[str]:
    """Return normalized link targets for a row (dedup keeping order)."""
    seen = set()
    out = []
    for l in row.get("links") or []:
        t = norm_target(l.get("href", ""))
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def row_to_doc(row: dict, doc_idx: int) -> dict:
    """HotpotQA row -> internal doc schema."""
    art = row.get("article") or ""
    raw = row.get("text") or ""
    title = row.get("title") or ""
    links = []
    for l in row.get("links") or []:
        t = norm_target(l.get("href", ""))
        if not t:
            continue
        links.append({"href": t, "text": (l.get("text") or "").strip()})
    src_id = row.get("id", doc_idx)
    return {
        "doc_id": f"doc_{doc_idx:06d}",
        "source_id": int(src_id) if str(src_id).isdigit() else doc_idx,
        "source": "hotpotqa-wiki",
        "title": title,
        "url": row.get("url") or f"https://en.wikipedia.org/wiki?curid={src_id}",
        "date": "2017-10-01",
        "language": "en",
        "raw_text": raw,
        "clean_text": art,
        "links": links,
        "char_count": len(art),
        "word_count": len(art.split()),
    }


# --- Section-title detection in raw_text ---
# HotpotQA raw text markup occasionally has "== Section ==" headers. Section
# detection is best-effort and isn't load-bearing.
SEC_RE = re.compile(r"^\s*==+\s*([^=]+?)\s*==+\s*$", re.M)


def section_at_offset(raw_text: str, char_off: int) -> str | None:
    """Return most-recent section header before char_off in raw_text (if any)."""
    if not raw_text:
        return None
    m = None
    for match in SEC_RE.finditer(raw_text):
        if match.start() > char_off:
            break
        m = match
    if m:
        return m.group(1).strip()
    return None
