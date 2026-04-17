#!/usr/bin/env python
"""Phase 1 -- corpus selection.

Stream HotpotQA wiki, apply 5-filter funnel, keep ~target_docs, write
data/corpus/pilot_docs.parquet.

Filters (per pipeline_description 3.2):
  F1 len(article) >= 500  (hard)
  F2 len(links) >= 3       (hard)
  F3 link_target_overlap_ratio   (used as SOFT rank signal, not hard cutoff --
                                    the "prioritize" wording makes sense only
                                    relative to pool size; a hard 0.5 cutoff at
                                    pool sizes << 5.49M kills most of the pool)
  F4 domain clustering via seed tokens (soft rank)
  F5 connectivity (drop isolated nodes; top up)

Strategy: scan a large pool (50K+), then rank the pool by composite (F3 soft +
F4 soft + n_links) and keep top `target_docs`. F5 verifies connectivity.
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
    p.add_argument("--pool_mult", type=int, default=20,
                   help="pool size = pool_mult * target_docs (bigger -> denser F3 overlap)")
    p.add_argument("--max_scan", type=int, default=800_000,
                   help="cap on rows scanned from the stream")
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
    overlap_min = float(corp["link_target_overlap_min"])  # used only as a ranking cue
    pool_target = max(target * args.pool_mult, target + 5000)

    log.info(f"loading hotpotqa-wiki stream -> pool_target={pool_target}, target={target}")
    ds = load_dataset("ParthMandaliya/hotpotqa-wiki", split="train", streaming=True)

    pem = ProgressEmitter("01_corpus_scan", total=pool_target, every_n=1000, every_s=10.0)
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

    # Build title set (lowercased) for F3 overlap calc
    pool_titles = {(r.get("title") or "").lower().strip() for r in pool}

    # Seed tokens for F4 (soft rank)
    seeds = [s.lower() for s in corp.get("seed_categories", [])]
    seed_tokens = set()
    for s in seeds:
        for tok in s.split():
            if len(tok) >= 4:
                seed_tokens.add(tok)
    log.info(f"[F4] seed_tokens={sorted(seed_tokens)}")

    # Compute composite scores for the entire pool
    pem3 = ProgressEmitter("01_corpus_score", total=len(pool), every_n=1000, every_s=10.0)
    scored: list[tuple[float, int, float, float, dict]] = []
    for r in pool:
        tgts = extract_link_targets(r)
        if not tgts:
            pem3.tick()
            continue
        # F3 overlap: fraction of targets also in pool_titles
        have = sum(1 for t in tgts if t in pool_titles)
        overlap = have / max(1, len(tgts))
        # F4: seed-token hits
        text_title = (r.get("title") or "").lower()
        f4 = 0
        for t in tgts[:40]:
            for tok in seed_tokens:
                if tok in t:
                    f4 += 1
                    break
        for tok in seed_tokens:
            if tok in text_title:
                f4 += 3
        # n_links (bounded) as connectivity proxy
        nl = min(len(tgts), 60)
        composite = 2.0 * overlap + 0.10 * f4 + 0.02 * nl
        # Small bump if overlap passes the spec threshold
        if overlap >= overlap_min:
            composite += 1.0
        scored.append((composite, overlap, f4, nl, r))
        pem3.tick()
    pem3.done()
    scored.sort(key=lambda x: -x[0])
    n_overlap_pass = sum(1 for s in scored if s[1] >= overlap_min)
    log.info(f"[F3 ranking] scored={len(scored)} n_overlap>={overlap_min}: {n_overlap_pass}")

    # Take top `max(target * 1.2, target + 500)` for F5 connectivity check
    pre_f5_n = int(target * 1.5)
    pre_f5 = [s[4] for s in scored[:pre_f5_n]]
    log.info(f"pre-F5 candidates={len(pre_f5)}")

    # F5: connectivity within the pre_f5 set
    title_idx_small = {(r.get("title") or "").lower().strip(): i for i, r in enumerate(pre_f5)}
    inv: dict[str, set[int]] = defaultdict(set)
    for i, r in enumerate(pre_f5):
        for l in (r.get("links") or []):
            t = unquote(l.get("href", "")).strip().lower()
            if not t or t.startswith("wikt:") or "#" in t:
                continue
            if t in title_idx_small:
                inv[t].add(i)
    deg = Counter()
    for t, s in inv.items():
        if 2 <= len(s) <= 400:
            for i in s:
                deg[i] += 1
    # Rank pre_f5 by degree; keep top `target`
    idx_ranked = sorted(range(len(pre_f5)), key=lambda i: (-deg[i], i))
    final_idx = idx_ranked[:target]
    n_zero = sum(1 for i in final_idx if deg[i] == 0)
    log.info(f"[F5] final kept={len(final_idx)} (zero-degree in final: {n_zero})")

    # Convert to schema + write
    out_rows = []
    for seq, i in enumerate(final_idx):
        d = row_to_doc(pre_f5[i], seq)
        out_rows.append(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_parquet(out, engine="pyarrow", index=False)
    log.info(f"wrote {out} rows={len(df)} size_mb={out.stat().st_size/1e6:.1f}")

    dur = int(time.time() - t0)
    phase_done("01_corpus", {"n_docs": len(df), "duration_s": dur,
                             "scanned": scanned, "pool": len(pool),
                             "pre_f5": len(pre_f5),
                             "n_overlap_pass_spec": n_overlap_pass,
                             "n_zero_degree_final": n_zero})
    log.info(f"done in {dur}s")


if __name__ == "__main__":
    main()
