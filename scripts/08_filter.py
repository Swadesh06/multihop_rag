#!/usr/bin/env python
"""Phase 8 -- filter battery + dedup + quality score.

Input: pilot_solv.jsonl (only decision=pass go in).
Apply filter battery (filters.apply_filter_battery), Jaccard dedup, semantic
dedup via bge-m3 cosine, quality scoring.

Output: pilot_filtered.jsonl.
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
import numpy as np

from mhrag.env import load_cfg, load_models_cfg, get_logger, DATA, OUT
from mhrag.progress import phase_start, phase_done, append_jsonl, read_jsonl
from mhrag.filters import apply_filter_battery, jaccard, composite_quality_score
from mhrag.embed import encode_texts


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--solv", default=str(OUT / "pilot_solv.jsonl"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--out", default=str(OUT / "pilot_filtered.jsonl"))
    return p.parse_args()


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("08_filter")
    phase_start("08_filter")
    t0 = time.time()

    fcfg = cfg["filters"]
    dcfg = cfg["dedup"]

    solv = read_jsonl(Path(args.solv))
    log.info(f"solvability records: {len(solv)}")

    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    cdf = pd.read_parquet(args.chunks)
    chunks_by_doc = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    # Filter 1: only decision=pass
    pass_rows = [r for r in solv if r.get("decision") == "pass"]
    log.info(f"pass solvability: {len(pass_rows)}")

    # Filter battery
    kept = []
    rej_counts = defaultdict(int)
    flag_counts = defaultdict(int)
    for r in pass_rows:
        rj = r.get("response_json") or {}
        ok, rej, flag = apply_filter_battery(rj, chunks_by_doc, fcfg)
        for x in rej:
            rej_counts[x] += 1
        for x in flag:
            flag_counts[x] += 1
        if ok:
            kept.append(r)
    log.info(f"after filter battery: kept={len(kept)} rej={sum(rej_counts.values())} flag={sum(flag_counts.values())}")
    log.info(f"reject reasons: {dict(rej_counts)}")
    log.info(f"flag reasons: {dict(flag_counts)}")

    # Jaccard dedup
    jt = float(dcfg.get("jaccard_threshold", 0.85))
    dedup_kept = []
    for r in kept:
        q_new = r["response_json"]["query"]
        dup = False
        for k in dedup_kept:
            q_old = k["response_json"]["query"]
            if jaccard(q_new, q_old) >= jt:
                dup = True
                break
        if not dup:
            dedup_kept.append(r)
    log.info(f"after Jaccard dedup: kept={len(dedup_kept)}")

    # Semantic dedup via embeddings
    ct = float(dcfg.get("embedding_cosine_threshold", 0.90))
    queries = [r["response_json"]["query"] for r in dedup_kept]
    if queries:
        log.info(f"semantic dedup: embedding {len(queries)} queries")
        embs = encode_texts(queries, batch_size=32).astype(np.float32)
        # pairwise cosine; since normalized, dot = cosine
        import torch
        Q = torch.from_numpy(embs).cuda()
        sims = (Q @ Q.T).cpu().numpy()
        # Cluster greedily
        used = [False] * len(dedup_kept)
        # Pre-compute quality scores for cluster winners
        for i in range(len(dedup_kept)):
            if used[i]:
                continue
            # gather cluster
            cluster = [i]
            for j in range(i + 1, len(dedup_kept)):
                if used[j]:
                    continue
                if sims[i, j] >= ct:
                    cluster.append(j)
            # mark all but the best-by-heuristic_score (use final_score)
            best = max(cluster, key=lambda k: dedup_kept[k].get("final_score") or 0.0)
            for k in cluster:
                if k != best:
                    used[k] = True
        final = [dedup_kept[i] for i in range(len(dedup_kept)) if not used[i]]
    else:
        final = dedup_kept
    log.info(f"after semantic dedup: kept={len(final)}")

    # Compute quality score for each surviving and keep only those >= min_quality (0.5 floor)
    filtered = []
    for r in final:
        rj = r["response_json"]
        jc = float(r.get("judge_confidence") or 0.65)
        sc = float(r.get("solvability_confidence") or 0.7)
        qs = composite_quality_score(rj, jc, sc, doc_by_id)
        r2 = dict(r)
        r2["quality_score"] = qs
        if qs >= 0.5:
            filtered.append(r2)
    filtered.sort(key=lambda x: -x["quality_score"])
    log.info(f"after quality floor (0.5): {len(filtered)}")

    # Write
    out = Path(args.out)
    if out.exists():
        out.unlink()  # overwrite -- this phase is idempotent
    append_jsonl(out, filtered)
    log.info(f"wrote {out} rows={len(filtered)}")

    # Phase progress
    phase_done("08_filter", {
        "n_accepted": len(filtered),
        "n_post_battery": len(kept),
        "n_post_jaccard": len(dedup_kept),
        "n_post_semantic": len(final),
        "reject_reasons": dict(rej_counts),
        "flag_reasons": dict(flag_counts),
        "duration_s": int(time.time() - t0),
    })


if __name__ == "__main__":
    main()
