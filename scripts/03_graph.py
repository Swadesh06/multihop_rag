#!/usr/bin/env python
"""Phase 3 -- graph + path enumeration.

Load docs + chunks, build inverted index from links, IDF-weighted edges
(including direct title-links), enumerate 2-hop + 3-hop paths, sample
`paths_sampled` stratified by reasoning_type_caps.

Outputs:
  data/graphs/pilot_edges.parquet
  data/graphs/pilot_paths.jsonl
"""
from __future__ import annotations
import os, sys, time, argparse, json, random
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd

from mhrag.env import load_cfg, get_logger, DATA
from mhrag.progress import phase_start, phase_done, ProgressEmitter
from mhrag.graph import (build_inverted_index, doc_title_index, idf_weights,
                          build_edges, adjacency, infer_reasoning_type,
                          score_path, enumerate_3hop)


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--out_edges", default=str(DATA / "graphs" / "pilot_edges.parquet"))
    p.add_argument("--out_paths", default=str(DATA / "graphs" / "pilot_paths.jsonl"))
    return p.parse_args()


def chunks_for_doc_entity(chunks_by_doc: dict[str, list[dict]], did: str,
                          entity: str, k: int = 3) -> list[dict]:
    """Return up to k chunks from doc `did` that contain `entity` (case-insensitive)."""
    chs = chunks_by_doc.get(did) or []
    lo = entity.lower()
    hits = []
    for ch in chs:
        if lo in (ch.get("chunk_text") or "").lower():
            hits.append(ch)
            if len(hits) >= k:
                break
    if hits:
        return hits
    # fallback: first k chunks
    return chs[:k]


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("03_graph")
    phase_start("03_graph")
    t0 = time.time()

    g = cfg["graph"]
    corp = cfg["corpus"]

    log.info(f"loading docs from {args.docs}")
    df = pd.read_parquet(args.docs)
    docs = df.to_dict("records")
    n = len(docs)
    log.info(f"docs={n}")

    cdf = pd.read_parquet(args.chunks)
    log.info(f"chunks={len(cdf)}")
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    doc_by_id = {d["doc_id"]: d for d in docs}
    t_idx = doc_title_index(docs)

    log.info("building inverted index")
    inv = build_inverted_index(docs)
    log.info(f"inverted entries={len(inv)}")

    pct = float(corp.get("generic_percentile_cutoff", 0.98))
    idf = idf_weights(inv, n, min_df=2, generic_percentile=pct)
    log.info(f"idf entries (after percentile cutoff={pct})={len(idf)}")

    out_e = Path(args.out_edges)
    if out_e.exists():
        log.info(f"edges file exists at {out_e}; rebuilding from scratch anyway")
    log.info("building edges")
    edges = build_edges(docs, inv, idf, t_idx)
    log.info(f"edges={len(edges)}")

    # Persist edges in one shot
    rows = [{"doc_a": a, "doc_b": b, "weight": float(ed["weight"]),
             "direct": bool(ed["direct"]),
             "shared_entities": ed["shared_entities"][:20],
             "n_shared": len(ed["shared_entities"])}
            for (a, b), ed in edges.items()]
    edf = pd.DataFrame(rows)
    out_e.parent.mkdir(parents=True, exist_ok=True)
    edf.to_parquet(out_e, engine="pyarrow", index=False)
    log.info(f"wrote {out_e} rows={len(edf)}")
    del rows
    del edf

    # Sample paths: 2-hop (direct pairs with >=1 shared entity) + some 3-hop
    n_paths_target = int(g["paths_sampled"])
    rng = random.Random(23)

    # Pre-filter edges by weight to reduce scoring cost -- 798K edges are too
    # many to score individually. Take top N by raw weight, then score.
    PRE_SCORE_N = max(n_paths_target * 30, 8000)
    filtered_edges = []
    for (a, b), ed in edges.items():
        if not ed["shared_entities"] and not ed["direct"]:
            continue
        filtered_edges.append(((a, b), ed))
    filtered_edges.sort(key=lambda kv: -kv[1]["weight"])
    filtered_edges = filtered_edges[:PRE_SCORE_N]
    log.info(f"pre-filtered edges for scoring: {len(filtered_edges)} (from {len(edges)})")

    edge_scored = []
    pem_s = ProgressEmitter("03_graph_score", total=len(filtered_edges),
                             every_n=1000, every_s=10.0)
    for (a, b), ed in filtered_edges:
        if ed["shared_entities"]:
            bridge = max(ed["shared_entities"], key=lambda e: idf.get(e, 0.0))
        else:
            bridge = (doc_by_id[b].get("title") or "").lower()
        shared = list(ed["shared_entities"])
        rt = infer_reasoning_type(doc_by_id[a], doc_by_id[b], shared, idf)
        s = score_path([doc_by_id[a], doc_by_id[b]], shared, bridge, idf,
                       float(g.get("bridge_entity_specificity_min", 0.2)),
                       float(g.get("bridge_entity_specificity_max", 0.9)),
                       bool(g.get("prefer_indirect_bridges", True)))
        edge_scored.append({"doc_ids": [a, b], "bridge_entity": bridge,
                            "shared_entities": shared,
                            "reasoning_type_inferred": rt,
                            "path_quality_score": s,
                            "weight": float(ed["weight"]),
                            "direct": ed["direct"]})
        pem_s.tick()
    pem_s.done()
    edge_scored.sort(key=lambda r: -r["path_quality_score"])
    log.info(f"scored 2-hop candidates={len(edge_scored)}")

    # Stratified sample per reasoning_type_caps
    caps = g.get("reasoning_type_caps", {})
    target_per_type = {k: max(1, int(v * n_paths_target)) for k, v in caps.items()}

    # First pass: fill each bucket with its cap from highest-scored.
    picked: list[dict] = []
    per_type_count = Counter()
    picked_set: set[tuple] = set()
    def _key(r: dict) -> tuple:
        return (tuple(r["doc_ids"]), r["bridge_entity"])
    for ty, cap in target_per_type.items():
        take_n = 0
        for r in edge_scored:
            if take_n >= cap:
                break
            if r["reasoning_type_inferred"] != ty:
                continue
            k = _key(r)
            if k in picked_set:
                continue
            picked.append(r)
            picked_set.add(k)
            per_type_count[ty] += 1
            take_n += 1
    # Top up to target from remaining
    for r in edge_scored:
        if len(picked) >= n_paths_target:
            break
        k = _key(r)
        if k in picked_set:
            continue
        picked.append(r)
        picked_set.add(k)
        per_type_count[r["reasoning_type_inferred"]] += 1
    log.info(f"picked 2-hop={len(picked)} by_type={dict(per_type_count)}")

    # Optionally add some 3-hop paths (if caps leave room and topology supports)
    # Keep 3-hop small (at most 10% of target) to avoid awkward Qs.
    n_3hop_target = min(int(0.1 * n_paths_target), max(0, n_paths_target - len(picked)))
    if n_3hop_target > 0:
        adj = adjacency(edges)
        tri = enumerate_3hop(adj, max_paths=n_3hop_target * 5)
        tri.sort(key=lambda t: -((t[3]["weight"] + t[4]["weight"]) / 2))
        added = 0
        for a, m, b, ed_am, ed_mb in tri:
            if added >= n_3hop_target:
                break
            shared = list(set(ed_am["shared_entities"]) | set(ed_mb["shared_entities"]))
            if not shared:
                continue
            bridge = max(shared, key=lambda e: idf.get(e, 0.0))
            s = score_path([doc_by_id[a], doc_by_id[m], doc_by_id[b]], shared, bridge, idf,
                           float(g.get("bridge_entity_specificity_min", 0.2)),
                           float(g.get("bridge_entity_specificity_max", 0.9)),
                           bool(g.get("prefer_indirect_bridges", True)))
            picked.append({"doc_ids": [a, m, b], "bridge_entity": bridge,
                           "shared_entities": shared,
                           "reasoning_type_inferred": "bridge_entity",
                           "path_quality_score": s,
                           "weight": float(ed_am["weight"] + ed_mb["weight"]),
                           "direct": False})
            added += 1
        log.info(f"added 3-hop={added}")

    picked = picked[:n_paths_target]
    # Content-hash path_id -- stable across re-samples so Phase 5/6/7 resume
    # logic can skip paths already processed in a prior run.
    import hashlib
    def _pid(docs: list[str], bridge: str) -> str:
        h = hashlib.sha1(("|".join(sorted(docs)) + "||" + bridge).encode()).hexdigest()
        return f"p_{h[:12]}"

    out_p = Path(args.out_paths)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for i, p in enumerate(picked):
            pid = _pid(p["doc_ids"], p["bridge_entity"])
            hint = {}
            for did in p["doc_ids"]:
                hint_chunks = chunks_for_doc_entity(chunks_by_doc, did, p["bridge_entity"], k=3)
                hint[did] = [c["chunk_id"] for c in hint_chunks]
            rec = {"path_id": pid, **p, "chunk_hints": hint}
            f.write(json.dumps(rec) + "\n")

    log.info(f"wrote {out_p} paths={len(picked)}")
    dur = int(time.time() - t0)
    phase_done("03_graph", {"n_edges": len(edges), "n_paths": len(picked),
                             "duration_s": dur, "by_type": dict(per_type_count)})


if __name__ == "__main__":
    main()
