#!/usr/bin/env python
"""Phase 10 -- baselines.

- BM25 Recall@10, MRR over the filtered Q set.
- Dense retrieval (bge-m3) Recall@10, MRR.
- Qwen2.5-3B reader F1/EM on a subsample.
"""
from __future__ import annotations
import os, sys, time, argparse, json, gc
from pathlib import Path

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
import numpy as np

from mhrag.env import load_cfg, load_models_cfg, get_logger, DATA, OUT
from mhrag.progress import phase_start, phase_done, read_jsonl, ProgressEmitter
from mhrag.baselines import BM25, tok, mrr, recall_at_k, f1, em, reader_answer


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--annotated", default=str(OUT / "pilot_annotated.jsonl"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--embed_dir", default=str(DATA / "embeddings"))
    p.add_argument("--prefix", default="pilot")
    p.add_argument("--out", default=str(OUT / "pilot_baselines.json"))
    return p.parse_args()


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    mcfg = load_models_cfg()
    log = get_logger("10_baselines")
    phase_start("10_baselines")
    t0 = time.time()

    bcfg = cfg["baselines"]
    reader_subsample = int(bcfg.get("reader_eval_subsample", 200))

    annotated = read_jsonl(Path(args.annotated))
    log.info(f"annotated: {len(annotated)}")
    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    all_doc_ids = [r["doc_id"] for r in docs_df.to_dict("records")]

    # --- BM25 over docs ---
    bm25_texts = [((r.get("title") or "") + " " + (r.get("clean_text") or "")[:4000])
                   for r in docs_df.to_dict("records")]
    bm25_docids = [r["doc_id"] for r in docs_df.to_dict("records")]
    log.info("building BM25")
    tokens = [tok(t) for t in bm25_texts]
    bm25 = BM25(tokens, k1=float(mcfg["retrieval_baselines"]["bm25"]["k1"]),
                b=float(mcfg["retrieval_baselines"]["bm25"]["b"]))

    # gold per question = grade>=2 supporting docs
    gold = []
    gold_any = []
    for r in annotated:
        g3 = {s["doc_id"] for s in r.get("supporting_docs", []) if s.get("grade", 0) >= 3}
        g_any = {s["doc_id"] for s in r.get("supporting_docs", []) if s.get("grade", 0) >= 2}
        gold.append(g3)
        gold_any.append(g_any)

    # BM25 eval
    k_top = int(mcfg["retrieval_baselines"]["bm25"]["top_k"])
    bm25_r10 = []
    bm25_mrr = []
    bm25_g3 = []
    log.info(f"BM25 eval (top_k={k_top})")
    pem = ProgressEmitter("10_bm25", total=len(annotated), every_n=100, every_s=10.0)
    for i, r in enumerate(annotated):
        q = r["query"]
        q_tok = tok(q)
        idx, _ = bm25.topk(q_tok, k=k_top)
        ranked = [bm25_docids[x] for x in idx]
        bm25_r10.append(recall_at_k(gold_any[i], ranked, k_top))
        bm25_mrr.append(mrr(gold_any[i], ranked))
        if gold[i]:
            bm25_g3.append(recall_at_k(gold[i], ranked, k_top))
        pem.tick()
    pem.done()

    bm25_r10_mean = float(np.mean(bm25_r10)) if bm25_r10 else 0.0
    bm25_mrr_mean = float(np.mean(bm25_mrr)) if bm25_mrr else 0.0
    bm25_g3_mean = float(np.mean(bm25_g3)) if bm25_g3 else 0.0
    log.info(f"BM25 R@{k_top}={bm25_r10_mean:.3f} MRR={bm25_mrr_mean:.3f} G3={bm25_g3_mean:.3f}")

    # --- Dense retrieval (bge-m3) ---
    d_emb = np.load(Path(args.embed_dir) / f"{args.prefix}_doc_embeds.npy").astype(np.float32)
    d_idx = read_jsonl(Path(args.embed_dir) / f"{args.prefix}_doc_idx.jsonl")
    row_to_did = {i["row"]: i["id"] for i in d_idx}
    # embed queries (reuse bge-m3)
    from mhrag.embed import encode_texts
    queries = [r["query"] for r in annotated]
    log.info(f"encoding {len(queries)} queries (dense)")
    q_emb = encode_texts(queries, batch_size=64).astype(np.float32)
    # cosine via matmul (already normalized)
    sims = q_emb @ d_emb.T
    dense_r10 = []
    dense_mrr = []
    dense_g3 = []
    for i in range(len(annotated)):
        s = sims[i]
        top_idx = np.argpartition(-s, k_top)[:k_top]
        top_idx = top_idx[np.argsort(-s[top_idx])]
        ranked = [row_to_did[int(x)] for x in top_idx]
        dense_r10.append(recall_at_k(gold_any[i], ranked, k_top))
        dense_mrr.append(mrr(gold_any[i], ranked))
        if gold[i]:
            dense_g3.append(recall_at_k(gold[i], ranked, k_top))
    dense_r10_mean = float(np.mean(dense_r10)) if dense_r10 else 0.0
    dense_mrr_mean = float(np.mean(dense_mrr)) if dense_mrr else 0.0
    dense_g3_mean = float(np.mean(dense_g3)) if dense_g3 else 0.0
    log.info(f"DENSE R@{k_top}={dense_r10_mean:.3f} MRR={dense_mrr_mean:.3f} G3={dense_g3_mean:.3f}")

    # Free embedder before reader
    try:
        import mhrag.embed as _e
        _e._MODEL = None
    except Exception:
        pass
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # --- Reader: Qwen2.5-3B on a subsample ---
    reader_f1s = []
    reader_ems = []
    subsample = annotated[:reader_subsample]
    log.info(f"reader eval on {len(subsample)}")
    if bcfg.get("run_reader", True) and subsample:
        pem = ProgressEmitter("10_reader", total=len(subsample), every_n=10, every_s=10.0)
        for r in subsample:
            # Context = concat of gold doc clean_text (truncated)
            ctx_parts = []
            for sd in r.get("supporting_docs", []):
                if sd.get("grade", 0) >= 2:
                    d = doc_by_id.get(sd["doc_id"])
                    if d is not None:
                        ctx_parts.append(f"Title: {d.get('title','')}\n{d.get('clean_text','')[:3000]}")
            ctx = "\n\n".join(ctx_parts)[:12000]
            if not ctx:
                continue
            try:
                pred = reader_answer(r["query"], ctx, max_in=4096, max_out=128)
            except Exception as e:
                log.info(f"reader error: {e}")
                continue
            gold_short = r.get("short_answer", "")
            reader_f1s.append(f1(pred, gold_short))
            reader_ems.append(em(pred, gold_short))
            pem.tick()
        pem.done()
    reader_f1_mean = float(np.mean(reader_f1s)) if reader_f1s else 0.0
    reader_em_mean = float(np.mean(reader_ems)) if reader_ems else 0.0
    log.info(f"READER F1={reader_f1_mean:.3f} EM={reader_em_mean:.3f} N={len(reader_f1s)}")

    res = {
        "bm25": {"recall_at_10": bm25_r10_mean, "mrr": bm25_mrr_mean, "g3_recall": bm25_g3_mean,
                  "n": len(bm25_r10)},
        "dense": {"recall_at_10": dense_r10_mean, "mrr": dense_mrr_mean, "g3_recall": dense_g3_mean,
                   "n": len(dense_r10)},
        "reader": {"f1": reader_f1_mean, "em": reader_em_mean, "n": len(reader_f1s)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log.info(f"wrote {args.out}")
    phase_done("10_baselines", {"duration_s": int(time.time() - t0),
                                 "bm25_r10": bm25_r10_mean,
                                 "dense_r10": dense_r10_mean,
                                 "reader_f1": reader_f1_mean})


if __name__ == "__main__":
    main()
