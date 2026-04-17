"""Solvability driver (Phase 7)."""
from __future__ import annotations
from typing import Any
from .gemini import SolvOutput, validate_obj, parse_json_lenient


def build_solv_prompt(template: str, q: str, short_answer: str,
                      doc_id: str, title: str, chunks_text: str) -> str:
    """Uses .replace() not .format() because prompt contains JSON braces."""
    return (template
            .replace("{question}", q)
            .replace("{short_answer}", short_answer)
            .replace("{doc_id}", doc_id)
            .replace("{title}", title)
            .replace("{document_chunks_text}", chunks_text))


def decide(solv_per_doc: list[dict]) -> tuple[str, float]:
    """Per-doc judgments -> overall decision + confidence score (0..1)."""
    n_yes = sum(1 for s in solv_per_doc if s.get("solvable_from_this_document_alone"))
    if n_yes == 0:
        # multi-hop -- pass
        conf_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
        confs = [conf_map.get((s.get("confidence") or "medium").lower(), 0.5) for s in solv_per_doc]
        return "pass", (sum(confs) / len(confs)) if confs else 0.7
    if n_yes == 1:
        return "reject_single_hop", 0.0
    return "reject_trivial", 0.0
