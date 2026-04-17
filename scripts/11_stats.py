#!/usr/bin/env python
"""Phase 11 -- dataset stats + reference comparison vs wiki-multihop-qa-500k."""
from __future__ import annotations
import os, sys, time, argparse, json, random
from pathlib import Path

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd
import numpy as np

from mhrag.env import load_cfg, get_logger, DATA, OUT
from mhrag.progress import phase_start, phase_done, read_jsonl, ProgressEmitter
from mhrag.stats import distributions, answer_type


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--annotated", default=str(OUT / "pilot_annotated.jsonl"))
    p.add_argument("--out_json", default=str(OUT / "pilot_stats.json"))
    p.add_argument("--out_md", default=str(OUT / "pilot_stats.md"))
    p.add_argument("--ref_sample", type=int, default=5000)
    return p.parse_args()


def ref_distribution(sample_size: int, log) -> dict:
    from datasets import load_dataset
    try:
        ds = load_dataset("rajat5039/wiki-multihop-qa-500k", split="train", streaming=True)
    except Exception as e:
        log.info(f"could not load reference dataset: {e}")
        return {"n": 0}
    rows = []
    for i, r in enumerate(ds):
        if i >= sample_size:
            break
        q = r.get("question") or r.get("query") or ""
        sa = r.get("answer") or r.get("short_answer") or ""
        rows.append({
            "query": q,
            "short_answer": sa,
            "long_answer": "",
            "reasoning_type": "unknown",
            "difficulty": "unknown",
            "hop_count": r.get("hop_count", 2),
        })
    return distributions(rows)


def md_table(our: dict, ref: dict) -> str:
    lines = []
    lines.append("# Pilot dataset stats\n")
    lines.append(f"n (ours) = {our.get('n', 0)}, n (ref) = {ref.get('n', 0)}\n")

    def _row(lbl, ov, rv):
        return f"| {lbl} | {ov} | {rv} |"

    lines.append("## Summary\n")
    lines.append("| metric | ours | ref |")
    lines.append("|---|---|---|")
    lines.append(_row("q_len_mean", our.get("q_len_mean", "-"), ref.get("q_len_mean", "-")))
    lines.append(_row("sa_len_mean", our.get("sa_len_mean", "-"), ref.get("sa_len_mean", "-")))
    lines.append(_row("la_len_mean", our.get("la_len_mean", "-"), ref.get("la_len_mean", "-")))
    lines.append("")

    lines.append("## Reasoning type share\n")
    lines.append("| type | ours | ref |")
    lines.append("|---|---|---|")
    keys = sorted(set(list(our.get("reasoning_type_share", {}).keys()) + list(ref.get("reasoning_type_share", {}).keys())))
    for k in keys:
        lines.append(_row(k, our.get("reasoning_type_share", {}).get(k, 0),
                           ref.get("reasoning_type_share", {}).get(k, 0)))
    lines.append("")

    lines.append("## Difficulty share\n")
    lines.append("| diff | ours | ref |")
    lines.append("|---|---|---|")
    keys = sorted(set(list(our.get("difficulty_share", {}).keys()) + list(ref.get("difficulty_share", {}).keys())))
    for k in keys:
        lines.append(_row(k, our.get("difficulty_share", {}).get(k, 0),
                           ref.get("difficulty_share", {}).get(k, 0)))
    lines.append("")

    lines.append("## Answer type share\n")
    lines.append("| type | ours | ref |")
    lines.append("|---|---|---|")
    keys = sorted(set(list(our.get("answer_type_share", {}).keys()) + list(ref.get("answer_type_share", {}).keys())))
    for k in keys:
        lines.append(_row(k, our.get("answer_type_share", {}).get(k, 0),
                           ref.get("answer_type_share", {}).get(k, 0)))
    return "\n".join(lines) + "\n"


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    log = get_logger("11_stats")
    phase_start("11_stats")
    t0 = time.time()

    annotated = read_jsonl(Path(args.annotated))
    log.info(f"annotated: {len(annotated)}")
    our = distributions(annotated)

    ref_size = int(cfg.get("reference_dataset", {}).get("sample_size", args.ref_sample))
    log.info(f"loading reference sample n={ref_size}")
    ref = ref_distribution(ref_size, log)

    res = {"ours": our, "reference": ref}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    md = md_table(our, ref)
    with open(args.out_md, "w") as f:
        f.write(md)
    log.info(f"wrote {args.out_json} and {args.out_md}")
    phase_done("11_stats", {"duration_s": int(time.time() - t0)})


if __name__ == "__main__":
    main()
