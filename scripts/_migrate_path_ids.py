#!/usr/bin/env python
"""One-shot migration: rewrite pilot_qg_raw.jsonl / pilot_qg_best.jsonl /
pilot_solv.jsonl so path_id is a content hash of doc_ids + bridge_entity.
This matches the stable hash-based path_id introduced in 03_graph.py.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path

OUT = Path("/workspace/swadesh/multihop_rag/outputs")


def _pid(docs: list[str], bridge: str) -> str:
    h = hashlib.sha1(("|".join(sorted(docs)) + "||" + bridge).encode()).hexdigest()
    return f"p_{h[:12]}"


def migrate(p: Path):
    if not p.exists():
        print(f"skip {p}: missing")
        return
    rows = []
    with open(p) as f:
        for ln in f:
            r = json.loads(ln)
            dids = r.get("doc_ids")
            bridge = r.get("bridge_entity")
            if dids is not None and bridge is not None:
                r["path_id"] = _pid(dids, bridge)
            rows.append(r)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)
    print(f"migrated {p.name}: {len(rows)} rows")


if __name__ == "__main__":
    migrate(OUT / "pilot_qg_raw.jsonl")
    migrate(OUT / "pilot_qg_best.jsonl")
    migrate(OUT / "pilot_solv.jsonl")
