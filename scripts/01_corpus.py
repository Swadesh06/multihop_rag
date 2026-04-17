#!/usr/bin/env python
"""Phase 1 -- corpus selection.

Stream HotpotQA wiki, apply 5-filter funnel, keep ~target_docs, write
data/corpus/pilot_docs.parquet.

Filters (per pipeline_description 3.2):
  F1 len(article) >= 500
  F2 len(links) >= 3
  F3 link_target_overlap >= 0.5 against candidate pool
  F4 domain clustering via seed categories
  F5 connectivity (drop isolated pool nodes)

Process: stream + partial read to build a candidate pool ~3x target, then apply
F3/F4/F5 and keep exactly `target_docs` documents.
"""
from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict, Counter

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
from datasets import load_dataset

from mhrag.env import load_cfg, get_logger, DATA
from mhrag.progress import phase_start, phase_done, ProgressEmitter, load as load_prog, save as save_prog
from mhrag.corpus import doc_passes_basic, extract_link_targets, row_to_doc


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--max_scan", type=int, default=200_000,
                   help="cap on rows scanned from the stream (safeguard)")
    return p.parse_args()


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("01_corpus")
    phase_start("01_corpus")
    t0 = time.time()
    corp = cfg["corpus"]
    target = int(corp["target_docs"])
    min_char = int(corp["min_char"])
    min_links = int(corp["min_links"])
    overlap_min = float(corp["link_target_overlap_min"])
    pool_target = target * 3

    log.info(f"loading hotpotqa-wiki stream -> pool_target={pool_target}, target={target}")
    ds = load_dataset("ParthMandaliya/hotpotqa-wiki", split="train", streaming=True)

    pem = ProgressEmitter("01_corpus_scan", total=pool_target, every_n=500, every_s=10.0)
    pool: list[dict] = []
    seen_titles: set[str] = set()

    scanned = 0
    kept = 0
    for row in ds:
        scanned += 1
        if scanned > args.max_scan:
            log.info(f"hit max_scan cap {args.max_scan}")
            break
        if not doc_passes_basic(row, min_char, min_links):
            continue
        title = (row.get("title") or "").strip()
        tl = title.lower()
        if tl in seen_titles:
            continue
        seen_titles.add(tl)
        pool.append(row)
        kept += 1
        pem.tick()
        if kept >= pool_target:
            break

    pem.done()
    log.info(f"[F1+F2] scanned={scanned} kept_pool={len(pool)}")

    # Build title set (lowercased) for F3
    pool_titles = {(r.get("title") or "").lower().strip() for r in pool}

    # F3: link target overlap
    pem3 = ProgressEmitter("01_corpus_F3", total=len(pool), every_n=500, every_s=10.0)
    f3_keep = []
    for r in pool:
        tgts = extract_link_targets(r)
        if not tgts:
            pem3.tick()
            continue
        have = sum(1 for t in tgts if t in pool_titles)
        ratio = have / len(tgts)
        if ratio >= overlap_min:
            f3_keep.append(r)
        pem3.tick()
    pem3.done()
    log.info(f"[F3] kept={len(f3_keep)} / pool={len(pool)} (overlap>={overlap_min})")

    # F4: domain clustering -- bias toward articles in seed-category neighborhoods.
    # Since we don't have Wikipedia categories in the corpus, approximate by taking
    # articles whose links include at least one seed-category token OR connected
    # via shared-link cluster density. For pilot scale we accept pool-relaxation.
    seeds = [s.lower() for s in corp.get("seed_categories", [])]
    seed_tokens = set()
    for s in seeds:
        for tok in s.split():
            if len(tok) >= 4:
                seed_tokens.add(tok)
    log.info(f"[F4] seed_tokens={sorted(seed_tokens)[:20]}")

    def seed_score(r: dict) -> int:
        tgts = extract_link_targets(r)
        text_title = (r.get("title") or "").lower()
        hit = 0
        for t in tgts:
            for tok in seed_tokens:
                if tok in t:
                    hit += 1
                    break
        for tok in seed_tokens:
            if tok in text_title:
                hit += 2
        return hit

    # Rank F3 candidates: seed_score desc, then #links desc (proxy for connectivity).
    scored = sorted(f3_keep, key=lambda r: (-seed_score(r), -len(r.get("links") or [])))

    # F5: connectivity proxy. Build inverted index on pool to compute degree.
    pool_ids = {(r.get("title") or "").lower().strip(): i for i, r in enumerate(scored)}
    inv: dict[str, set[int]] = defaultdict(set)
    for i, r in enumerate(scored):
        for l in (r.get("links") or []):
            t = unquote(l.get("href", "")).strip().lower()
            if t and t in pool_ids:
                inv[t].add(i)
    deg = Counter()
    for t, s in inv.items():
        if 2 <= len(s) <= 200:
            for i in s:
                deg[i] += 1
    # Drop zero-degree nodes; top up by seed_score
    kept_idx = [i for i in range(len(scored)) if deg[i] > 0]
    kept_idx.sort(key=lambda i: (-deg[i], -seed_score(scored[i])))
    log.info(f"[F5] connected={len(kept_idx)}")

    if len(kept_idx) < target:
        # top up from isolated docs
        extra = [i for i in range(len(scored)) if i not in set(kept_idx)]
        kept_idx.extend(extra[: target - len(kept_idx)])

    kept_idx = kept_idx[:target]
    log.info(f"[F5] final kept={len(kept_idx)}")

    # Convert to schema + write
    out_rows = []
    for seq, i in enumerate(kept_idx):
        d = row_to_doc(scored[i], seq)
        out_rows.append(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_parquet(out, engine="pyarrow", index=False)
    log.info(f"wrote {out} rows={len(df)} size_mb={out.stat().st_size/1e6:.1f}")

    dur = int(time.time() - t0)
    phase_done("01_corpus", {"n_docs": len(df), "duration_s": dur,
                             "scanned": scanned, "pool": len(pool),
                             "f3_keep": len(f3_keep)})
    # summary
    st = load_prog()
    st.setdefault("phases", {})["01_corpus"].update({"n_docs": len(df)})
    save_prog(st)
    log.info(f"done in {dur}s")


if __name__ == "__main__":
    main()
