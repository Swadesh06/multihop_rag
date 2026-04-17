#!/usr/bin/env python
"""Phase 9 -- supporting chunks + adversarial distractors.

For each filtered Q:
  - Map each quoted_span to a chunk (exact match first; embedding fallback).
  - Mine distractors: embedding top-k (2) + BM25 top-k (1); remove gold.
  - Write pilot_annotated.jsonl with the full final schema.
"""
from __future__ import annotations
import os, sys, time, argparse, json, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
import numpy as np

from mhrag.env import load_cfg, get_logger, DATA, OUT
from mhrag.progress import phase_start, phase_done, append_jsonl, read_jsonl
from mhrag.annotate import map_span_to_chunk
from mhrag.embed import encode_texts
from mhrag.baselines import BM25, tok as _bm25_tok


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--filtered", default=str(OUT / "pilot_filtered.jsonl"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--embed_dir", default=str(DATA / "embeddings"))
    p.add_argument("--prefix", default="pilot")
    p.add_argument("--out", default=str(OUT / "pilot_annotated.jsonl"))
    return p.parse_args()


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("09_annotate")
    phase_start("09_annotate")
    t0 = time.time()

    dcfg = cfg["distractors"]
    per_q = int(dcfg["per_question"])
    mix = dcfg["mix"]
    emb_k = int(mix["embedding_topk"])
    bm25_k = int(mix["bm25_topk"])

    filtered = read_jsonl(Path(args.filtered))
    log.info(f"filtered: {len(filtered)}")

    # Load paths to recover graph-inferred reasoning_type per path_id.
    # The QG model's reasoning_type is heavily biased toward bridge_entity --
    # we trust the graph-level heuristic (matched to reasoning_type_caps in
    # Phase 3 sampling) for distribution diversity.
    paths_rt: dict[str, str] = {}
    paths_fp = Path(str(Path(args.filtered).parent.parent / "data" / "graphs" / "pilot_paths.jsonl"))
    if not paths_fp.exists():
        from mhrag.env import DATA as _D
        paths_fp = _D / "graphs" / "pilot_paths.jsonl"
    if paths_fp.exists():
        with open(paths_fp) as f:
            for ln in f:
                p = json.loads(ln)
                paths_rt[p["path_id"]] = p.get("reasoning_type_inferred", "bridge_entity")
        log.info(f"loaded {len(paths_rt)} path reasoning_types")

    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    all_doc_ids = [r["doc_id"] for r in docs_df.to_dict("records")]
    cdf = pd.read_parquet(args.chunks)
    chunks_by_doc = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    # Load doc embeddings and idx
    d_emb = np.load(Path(args.embed_dir) / f"{args.prefix}_doc_embeds.npy")
    d_idx = read_jsonl(Path(args.embed_dir) / f"{args.prefix}_doc_idx.jsonl")
    row_to_did = {i["row"]: i["id"] for i in d_idx}
    did_to_row = {v: k for k, v in row_to_did.items()}
    log.info(f"doc embeds shape={d_emb.shape}")

    # BM25 over doc-level (title + first 2000 chars of clean_text).
    # Use docs as retrieval units -- consistent with doc-level retrieval tasks.
    bm25_texts = [((r.get("title") or "") + " " + (r.get("clean_text") or "")[:2000]) for r in docs_df.to_dict("records")]
    bm25_docids = [r["doc_id"] for r in docs_df.to_dict("records")]
    log.info(f"building BM25 over docs n={len(bm25_texts)}")
    bm25_tokens = [_bm25_tok(t) for t in bm25_texts]
    bm25 = BM25(bm25_tokens, k1=1.5, b=0.75)

    # Query embeddings (one per question)
    queries = [r["response_json"]["query"] for r in filtered]
    log.info(f"embedding {len(queries)} queries")
    q_emb = encode_texts(queries, batch_size=64).astype(np.float32)
    d_emb_f = d_emb.astype(np.float32)
    sims = q_emb @ d_emb_f.T  # (Q, Ndocs)

    out = Path(args.out)
    if out.exists():
        out.unlink()

    annotated = []
    for qi, r in enumerate(filtered):
        rj = r["response_json"]
        query_id = f"q_{qi:05d}"
        short_answer = rj.get("short_answer", "")
        long_answer = rj.get("long_answer", "")
        answer_points = rj.get("answer_points", [])
        supporting_docs = rj.get("supporting_docs", [])
        # Prefer graph-level reasoning_type (stable distribution) over model's
        # (model strongly biased toward bridge_entity).
        pid_r = r.get("path_id")
        reasoning_type = paths_rt.get(pid_r) or rj.get("reasoning_type", "bridge_entity")
        reasoning_chain = rj.get("reasoning_chain", "")
        bridge_entity = rj.get("bridge_entity", "")
        quoted_spans = rj.get("quoted_spans", {})
        difficulty = rj.get("difficulty", "medium")

        # supporting chunks via exact span matching
        sup_chunks = []
        for did, spans in (quoted_spans or {}).items():
            for sp in spans or []:
                m = map_span_to_chunk(sp, chunks_by_doc.get(did, []))
                if m:
                    sup_chunks.append(m)

        # Distractors
        gold_ids = {d["doc_id"] for d in supporting_docs}
        # embedding
        topk_sims = sims[qi]
        order = np.argsort(-topk_sims)
        emb_picks = []
        for r_idx in order:
            did = row_to_did[int(r_idx)]
            if did in gold_ids:
                continue
            emb_picks.append(did)
            if len(emb_picks) >= emb_k * 2:  # pull extra for mixing
                break
        # bm25
        q_tok = _bm25_tok(queries[qi])
        bm25_idx, _ = bm25.topk(q_tok, k=emb_k + bm25_k + 4)
        bm25_picks = [bm25_docids[i] for i in bm25_idx if bm25_docids[i] not in gold_ids]
        # final mix
        distractors = []
        for d in emb_picks[:emb_k]:
            distractors.append({"doc_id": d, "source": "embedding", "grade": 0})
        for d in bm25_picks:
            if any(x["doc_id"] == d for x in distractors):
                continue
            distractors.append({"doc_id": d, "source": "bm25", "grade": 0})
            if len([x for x in distractors if x["source"] == "bm25"]) >= bm25_k:
                break
        distractors = distractors[:per_q]

        # Compose supporting_docs with titles
        sup_out = []
        for sd in supporting_docs:
            did = sd.get("doc_id", "")
            title = doc_by_id.get(did, {}).get("title", "")
            sup_out.append({"doc_id": did, "title": title, "grade": int(sd.get("grade", 0))})
        # + distractors with titles
        for d in distractors:
            d["title"] = doc_by_id.get(d["doc_id"], {}).get("title", "")
        final_docs = sup_out + [{"doc_id": d["doc_id"], "title": d["title"], "grade": 0,
                                  "source": d["source"]} for d in distractors]

        # hop_count: number of grade>=2 supporting docs
        hop_count = sum(1 for s in sup_out if s["grade"] >= 2)

        # answer_type
        from mhrag.stats import answer_type
        atype = answer_type(short_answer)

        rec = {
            "query_id": query_id,
            "query": rj.get("query", ""),
            "short_answer": short_answer,
            "long_answer": long_answer,
            "answer_points": answer_points,
            "supporting_docs": final_docs,
            "supporting_chunks": sup_chunks,
            "reasoning_type": reasoning_type,
            "reasoning_chain": reasoning_chain,
            "bridge_entity": bridge_entity,
            "difficulty": difficulty,
            "hop_count": hop_count,
            "answer_type": atype,
            "quoted_spans": quoted_spans,
            "generation_metadata": {
                "path_id": r.get("path_id"),
                "candidate_idx": r.get("best_candidate_idx"),
                "quality_score": r.get("quality_score"),
                "final_score": r.get("final_score"),
                "judge_confidence": r.get("judge_confidence"),
                "solvability_confidence": r.get("solvability_confidence"),
                "distractor_sources": [d["source"] for d in distractors],
            },
        }
        annotated.append(rec)

    append_jsonl(out, annotated)
    log.info(f"wrote {out} rows={len(annotated)}")
    phase_done("09_annotate", {"duration_s": int(time.time() - t0), "n_annotated": len(annotated)})


if __name__ == "__main__":
    main()
