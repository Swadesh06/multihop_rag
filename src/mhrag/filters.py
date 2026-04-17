"""Filter battery + dedup + quality score (Phase 8)."""
from __future__ import annotations
import re, math
from typing import Iterable

STOP = set(("a an the and or but if then when while of at by for from in on to with as "
            "is are was were be been being do does did have has had can could should would "
            "may might must will shall this that these those who whom whose which what where "
            "when why how not no yes it its his her their them they he she him them our your "
            "we us you i me my mine yours ours theirs both each every such any some more most "
            "other another few many much less").split())

Q_WORDS_OK = ("who", "what", "where", "when", "which", "how", "why", "in what",
              "by what", "during what", "into what", "of what", "from what", "on what")


def normalize_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def jaccard(a: str, b: str) -> float:
    wa = set(normalize_words(a)) - STOP
    wb = set(normalize_words(b)) - STOP
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def lexical_overlap_fraction(q: str, chunk_text: str) -> float:
    """Fraction of question content-words that appear in chunk (excl. stop words)."""
    qw = [w for w in normalize_words(q) if w not in STOP]
    if not qw:
        return 0.0
    cset = set(normalize_words(chunk_text)) - STOP
    hit = sum(1 for w in qw if w in cset)
    return hit / len(qw)


PRONOUN_ANS = {"it", "they", "he", "she", "him", "her", "them", "us", "you", "we",
               "yes", "no", "maybe", "true", "false"}


def apply_filter_battery(q: dict, chunks_by_doc: dict[str, list[dict]],
                         cfg: dict) -> tuple[bool, list[str], list[str]]:
    """Return (accept, reject_reasons, flag_reasons)."""
    rej, flag = [], []
    query = (q.get("query") or "").strip()
    short = (q.get("short_answer") or "").strip()
    long_a = (q.get("long_answer") or "").strip()
    qw = len(normalize_words(query))
    if qw < cfg.get("min_question_words", 8):
        rej.append("short_question")
    if qw > cfg.get("max_question_words", 60):
        flag.append("long_question")
    if not query.endswith("?"):
        flag.append("no_qmark")
    if not any(query.lower().startswith(x) for x in Q_WORDS_OK):
        flag.append("nonstd_qword")
    if short.lower() in PRONOUN_ANS and cfg.get("reject_pronoun_short_answers", True):
        rej.append("pronoun_answer")
    sw = len(short.split())
    if sw > cfg.get("max_short_answer_words", 8):
        flag.append("long_short_answer")
    if sw == 0:
        rej.append("empty_short_answer")
    if cfg.get("reject_if_short_answer_in_question", True) and short and short.lower() in query.lower():
        # verbatim substring: reject
        if re.search(r"\b" + re.escape(short.lower()) + r"\b", query.lower()):
            rej.append("answer_in_question")
    la_w = len(long_a.split())
    if la_w < cfg.get("min_long_answer_words", 40):
        flag.append("short_long_answer")
    if la_w > cfg.get("max_long_answer_words", 150):
        flag.append("too_long_long_answer")
    # at least 2 docs cited in long answer
    cited_docs = set(re.findall(r"\(Document\s+[A-Z0-9]+\)", long_a))
    if len(cited_docs) < cfg.get("min_docs_cited_in_long_answer", 2):
        rej.append("insufficient_citations")
    # lexical leakage against any single chunk
    max_ov = 0.0
    for did in (d["doc_id"] for d in q.get("supporting_docs", [])):
        chs = chunks_by_doc.get(did) or []
        for ch in chs:
            ov = lexical_overlap_fraction(query, ch["chunk_text"])
            if ov > max_ov:
                max_ov = ov
    if max_ov > cfg.get("max_lexical_overlap_with_any_chunk", 0.6):
        rej.append("lexical_leakage")
    return (len(rej) == 0), rej, flag


def composite_quality_score(q: dict, judge_conf: float, sol_conf: float,
                            docs_by_id: dict[str, dict]) -> float:
    """Implements pipeline_description Section 9.3 composite formula.

    Components (each 0..1):
      solvability_confidence  -- from Phase 7
      answer_specificity      -- short non-pronoun, title-case or numeric
      reasoning_chain_quality -- steps count >= 2, cites 2+ docs
      question_naturalness    -- starts with standard q-word, has ?
      document_diversity      -- different titles + grades spread
    """
    short = (q.get("short_answer") or "").strip()
    w = short.split()
    asp = 0.3
    if short and short.lower() not in PRONOUN_ANS:
        asp = 0.7
        if 1 <= len(w) <= 4:
            asp = 0.95
    rc = q.get("reasoning_chain") or ""
    steps = len(re.findall(r"step\s*\d|\bstep\b", rc, re.I))
    docs_in_rc = len(set(re.findall(r"document\s+[a-z0-9]+", rc, re.I)))
    rcq = min(1.0, 0.4 + 0.2 * min(steps, 3) + 0.2 * min(docs_in_rc, 2))
    query = (q.get("query") or "").strip()
    natural = 0.4
    if query.endswith("?"):
        natural += 0.3
    if any(query.lower().startswith(x) for x in Q_WORDS_OK):
        natural += 0.3
    natural = min(1.0, natural)
    titles = [(docs_by_id.get(d["doc_id"]) or {}).get("title", "") for d in q.get("supporting_docs", [])]
    distinct_titles = len(set(t for t in titles if t))
    grades = {d.get("grade", 0) for d in q.get("supporting_docs", [])}
    div = min(1.0, 0.3 + 0.2 * distinct_titles + 0.1 * len(grades))
    s = 0.30 * sol_conf + 0.20 * asp + 0.20 * rcq + 0.15 * natural + 0.15 * div
    # blend judge confidence (replaces solvability if not available)
    s = 0.85 * s + 0.15 * judge_conf
    return max(0.0, min(1.0, s))
