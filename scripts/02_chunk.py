#!/usr/bin/env python
"""Phase 2 -- chunking.

Read pilot_docs.parquet, sentence-split with blingfire (pysbd fallback), pack
into 150-300 word chunks with 30-50 word overlap, write pilot_chunks.parquet.
"""
from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd

from mhrag.env import load_cfg, get_logger, DATA
from mhrag.progress import phase_start, phase_done, ProgressEmitter, load as load_prog, save as save_prog
from mhrag.chunking import chunk_doc


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--out", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    return p.parse_args()


def _worker(doc: dict, size_min: int, size_max: int, o_min: int, o_max: int):
    return chunk_doc(doc, size_min, size_max, o_min, o_max)


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("02_chunk")
    phase_start("02_chunk")
    t0 = time.time()

    cc = cfg["chunking"]
    smin = int(cc["size_words_min"])
    smax = int(cc["size_words_max"])
    omin = int(cc["overlap_words_min"])
    omax = int(cc["overlap_words_max"])

    df = pd.read_parquet(args.docs)
    log.info(f"loaded {len(df)} docs from {args.docs}")
    docs = df.to_dict("records")

    nworkers = max(1, cpu_count() // 2)
    log.info(f"chunking with {nworkers} workers")

    fn = partial(_worker, size_min=smin, size_max=smax, o_min=omin, o_max=omax)
    pem = ProgressEmitter("02_chunk", total=len(docs), every_n=500, every_s=10.0)
    all_chunks = []
    with Pool(nworkers) as pool:
        for chunks in pool.imap(fn, docs, chunksize=16):
            all_chunks.extend(chunks)
            pem.tick()
    pem.done()

    log.info(f"total chunks={len(all_chunks)} (avg {len(all_chunks)/max(1,len(docs)):.1f}/doc)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cdf = pd.DataFrame(all_chunks)
    cdf.to_parquet(out, engine="pyarrow", index=False)
    log.info(f"wrote {out} rows={len(cdf)} size_mb={out.stat().st_size/1e6:.1f}")

    dur = int(time.time() - t0)
    phase_done("02_chunk", {"n_chunks": len(cdf), "duration_s": dur})


if __name__ == "__main__":
    main()
