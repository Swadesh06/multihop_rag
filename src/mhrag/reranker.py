"""LLM-as-judge best-of-N reranker (Phase 6)."""
from __future__ import annotations
import asyncio, json, re
from typing import Any
from .gemini import GeminiPool, parse_json_lenient, RerankerOutput, validate_obj


def build_reranker_prompt(template: str, docs_block: str, candidates: list[dict]) -> str:
    """candidates: list of {idx, query, short_answer, long_answer, reasoning_chain, ...}.

    Uses .replace() not .format() because the prompt template contains JSON
    braces that would trip str.format.
    """
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"--- CANDIDATE {i} ---")
        lines.append(f"query: {c.get('query','')}")
        lines.append(f"short_answer: {c.get('short_answer','')}")
        lines.append(f"long_answer: {c.get('long_answer','')}")
        lines.append(f"bridge_entity: {c.get('bridge_entity','')}")
        lines.append(f"reasoning_type: {c.get('reasoning_type','')}")
        lines.append(f"reasoning_chain: {c.get('reasoning_chain','')}")
        lines.append("")
    candidates_block = "\n".join(lines)
    return (template
            .replace("{n_candidates}", str(len(candidates)))
            .replace("{documents_block}", docs_block)
            .replace("{candidates_block}", candidates_block))


def heuristic_candidate_score(cand: dict) -> float:
    """Local 0..1 score from lengths + schema health; replaces full quality score here."""
    q = (cand.get("query") or "").strip()
    sa = (cand.get("short_answer") or "").strip()
    la = (cand.get("long_answer") or "").strip()
    rc = (cand.get("reasoning_chain") or "").strip()
    score = 0.0
    # valid lengths
    if 8 <= len(q.split()) <= 60:
        score += 0.2
    if q.endswith("?"):
        score += 0.1
    if 1 <= len(sa.split()) <= 6:
        score += 0.2
    if 40 <= len(la.split()) <= 150:
        score += 0.2
    if rc:
        score += 0.1
    # citations in long_answer
    cites = len(set(re.findall(r"\(Document\s+[A-Z0-9]+\)", la)))
    if cites >= 2:
        score += 0.2
    return min(1.0, score)
