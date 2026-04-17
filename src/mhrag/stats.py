"""Dataset stats + reference comparison (Phase 11)."""
from __future__ import annotations
import re, json
from collections import Counter


def answer_type(short: str) -> str:
    s = (short or "").strip()
    if not s:
        return "empty"
    if re.match(r"^-?\d+(\.\d+)?$", s):
        return "numeric"
    if re.match(r"^\d{4}$", s):
        return "date"
    if re.match(r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+", s, re.I):
        return "date"
    if re.search(r"^\d{1,4}[/-]\d{1,4}([/-]\d{2,4})?$", s):
        return "date"
    if s.lower() in ("yes", "no", "true", "false"):
        return "boolean"
    w = s.split()
    if len(w) <= 3 and all(x[:1].isupper() for x in w if x):
        return "entity"
    return "phrase"


def distributions(rows: list[dict]) -> dict:
    q_len = [len((r.get("query") or "").split()) for r in rows]
    sa_len = [len((r.get("short_answer") or "").split()) for r in rows]
    la_len = [len((r.get("long_answer") or "").split()) for r in rows]
    rt = Counter(r.get("reasoning_type", "unknown") for r in rows)
    diff = Counter(r.get("difficulty", "unknown") for r in rows)
    hops = Counter(r.get("hop_count", 2) for r in rows)
    atype = Counter(answer_type(r.get("short_answer", "")) for r in rows)
    return {
        "n": len(rows),
        "q_len_mean": _m(q_len), "q_len_hist": _hist(q_len, [0, 8, 12, 16, 20, 30, 60]),
        "sa_len_mean": _m(sa_len), "sa_len_hist": _hist(sa_len, [0, 1, 2, 3, 4, 6, 10]),
        "la_len_mean": _m(la_len), "la_len_hist": _hist(la_len, [0, 40, 60, 80, 100, 120, 200]),
        "reasoning_type_share": _share(rt),
        "difficulty_share": _share(diff),
        "hops_share": _share(hops),
        "answer_type_share": _share(atype),
    }


def _m(xs: list[int]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def _hist(xs: list[int], bins: list[int]) -> list[int]:
    out = [0] * (len(bins) - 1)
    for x in xs:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                out[i] += 1
                break
        else:
            if x >= bins[-1]:
                out[-1] += 1
    return out


def _share(c: Counter) -> dict[str, float]:
    tot = sum(c.values()) or 1
    return {k: round(v / tot, 4) for k, v in c.most_common()}
