#!/usr/bin/env python
"""Phase 4 -- embeddings.

Embed doc (title + first chunk) and all chunks with bge-m3 fp16.
Outputs:
  data/embeddings/pilot_doc_embeds.npy
  data/embeddings/pilot_chunk_embeds.npy
  data/embeddings/pilot_doc_idx.jsonl
  data/embeddings/pilot_chunk_idx.jsonl
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
import numpy as np

from mhrag.env import load_cfg, get_logger, DATA
from mhrag.progress import phase_start, phase_done, ProgressEmitter
from mhrag.embed import encode_texts


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--out_dir", default=str(DATA / "embeddings"))
    p.add_argument("--prefix", default="pilot")
    return p.parse_args()


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("04_embed")
    phase_start("04_embed")
    t0 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.docs)
    cdf = pd.read_parquet(args.chunks)
    log.info(f"docs={len(df)} chunks={len(cdf)}")

    # Doc embedding text: title + first ~2000 chars of clean_text
    doc_texts = []
    doc_ids = []
    for r in df.to_dict("records"):
        txt = (r.get("title") or "") + " " + (r.get("clean_text") or "")[:2000]
        doc_texts.append(txt)
        doc_ids.append(r["doc_id"])

    # Chunk embedding text: chunk_text
    chunk_texts = []
    chunk_ids = []
    for r in cdf.to_dict("records"):
        chunk_texts.append(r.get("chunk_text") or "")
        chunk_ids.append(r["chunk_id"])

    log.info(f"encoding docs N={len(doc_texts)}")
    d_emb = encode_texts(doc_texts, batch_size=32, show_progress=True)
    np.save(out_dir / f"{args.prefix}_doc_embeds.npy", d_emb)
    with open(out_dir / f"{args.prefix}_doc_idx.jsonl", "w") as f:
        for i, did in enumerate(doc_ids):
            f.write(json.dumps({"row": i, "id": did}) + "\n")
    log.info(f"wrote doc embeds shape={d_emb.shape} dtype={d_emb.dtype}")

    log.info(f"encoding chunks N={len(chunk_texts)}")
    c_emb = encode_texts(chunk_texts, batch_size=32, show_progress=True)
    np.save(out_dir / f"{args.prefix}_chunk_embeds.npy", c_emb)
    with open(out_dir / f"{args.prefix}_chunk_idx.jsonl", "w") as f:
        for i, cid in enumerate(chunk_ids):
            f.write(json.dumps({"row": i, "id": cid}) + "\n")
    log.info(f"wrote chunk embeds shape={c_emb.shape} dtype={c_emb.dtype}")

    dur = int(time.time() - t0)
    phase_done("04_embed", {"n_vecs_doc": len(doc_texts),
                             "n_vecs_chunk": len(chunk_texts),
                             "duration_s": dur})


if __name__ == "__main__":
    main()
