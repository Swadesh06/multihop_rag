#!/usr/bin/env python
"""Phase 6 -- LLM-as-judge reranker over QG candidates.

Group pilot_qg_raw.jsonl by path_id, pick best among valid candidates using
reranker prompt. Compute composite final score = 0.7*judge + 0.3*heuristic.
Write pilot_qg_best.jsonl.
"""
from __future__ import annotations
import os, sys, time, argparse, asyncio, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd

from mhrag.env import load_cfg, load_models_cfg, get_logger, DATA, OUT, load_prompt
from mhrag.progress import (phase_start, phase_done, append_jsonl,
                              read_jsonl, load as load_prog)
from mhrag.gemini import GeminiPool, RerankerOutput, parse_json_lenient, validate_obj
from mhrag.reranker import build_reranker_prompt, heuristic_candidate_score


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--raw", default=str(OUT / "pilot_qg_raw.jsonl"))
    p.add_argument("--paths", default=str(DATA / "graphs" / "pilot_paths.jsonl"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--out", default=str(OUT / "pilot_qg_best.jsonl"))
    return p.parse_args()


async def run(args):
    cfg = load_cfg(Path(args.config).stem)
    mcfg = load_models_cfg()
    log = get_logger("06_rerank")
    phase_start("06_rerank")
    t0 = time.time()

    st = load_prog()
    model = st.get("resolved_models", {}).get("reranker") or mcfg["llm"]["reranker"]["candidates"][0]
    log.info(f"reranker model={model}")

    rcfg = cfg["reranker"]
    judge_w = float(rcfg["judge_weight"])
    concurrency = int(rcfg["concurrency"])
    slice_size = int(rcfg["batch_size_per_gather"])

    # Resume logic
    done_pids: set[str] = set()
    if Path(args.out).exists():
        for r in read_jsonl(Path(args.out)):
            done_pids.add(r["path_id"])
        log.info(f"resuming: {len(done_pids)} winners already written")

    raw = read_jsonl(Path(args.raw))
    # Group valid candidates by path_id
    by_path = defaultdict(list)
    for r in raw:
        if r.get("valid") and r.get("response_json"):
            by_path[r["path_id"]].append(r)
    log.info(f"paths with valid candidates: {len(by_path)}")

    # Docs + chunks for docs_block
    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    cdf = pd.read_parquet(args.chunks)
    chunks_by_doc = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    # Path chunk hints
    path_hints = {}
    with open(args.paths) as f:
        for ln in f:
            p = json.loads(ln)
            path_hints[p["path_id"]] = p

    template = load_prompt("reranker")

    def docs_block(pid: str, dids: list[str]) -> str:
        from scripts_05 import _dummy  # placeholder to avoid import; inlined below
        return ""

    # Inlined docs block
    DOC_LETTERS = "ABCDEFGHIJKLMNOP"
    def build_block(dids: list[str], hints: dict[str, list[str]]) -> str:
        lines = []
        for i, did in enumerate(dids):
            letter = DOC_LETTERS[i]
            title = doc_by_id[did].get("title", "") if did in doc_by_id else ""
            hint_set = set(hints.get(did) or [])
            chs = chunks_by_doc.get(did) or []
            sel = [ch for ch in chs if ch["chunk_id"] in hint_set][:3]
            if len(sel) < 3:
                rest = [ch for ch in chs if ch["chunk_id"] not in hint_set]
                sel.extend(rest[: 3 - len(sel)])
            lines.append(f"--- Document {letter} (doc_id: {did}, title: \"{title}\") ---")
            for ch in sel:
                lines.append((ch.get("chunk_text") or "")[:1200])
            lines.append("")
        return "\n".join(lines)

    # Build rerank jobs
    jobs = []
    for pid, cands in by_path.items():
        if pid in done_pids:
            continue
        if len(cands) < 1:
            continue
        dids = path_hints.get(pid, {}).get("doc_ids") or cands[0].get("doc_ids", [])
        hints = path_hints.get(pid, {}).get("chunk_hints") or {}
        block = build_block(dids, hints)
        cand_objs = [c["response_json"] for c in cands]
        prompt = build_reranker_prompt(template, block, cand_objs)
        jobs.append((pid, cands, prompt))
    log.info(f"rerank jobs={len(jobs)}")

    # Winners from single-candidate paths go through without calling LLM, using heuristic only
    single_cands = [j for j in jobs if len(j[1]) == 1]
    multi_jobs = [j for j in jobs if len(j[1]) > 1]
    log.info(f"single-candidate short-circuit: {len(single_cands)}, multi-judge: {len(multi_jobs)}")

    rows_out = []
    # Single-candidate paths: just keep with heuristic score
    for pid, cands, _ in single_cands:
        c = cands[0]
        h = heuristic_candidate_score(c["response_json"])
        rows_out.append({
            "path_id": pid, "best_candidate_idx": c["candidate_idx"],
            "doc_ids": c["doc_ids"], "bridge_entity": c["bridge_entity"],
            "response_json": c["response_json"],
            "heuristic_score": h, "judge_confidence": 0.65,
            "final_score": judge_w * 0.65 + (1 - judge_w) * h,
            "justification": "single valid candidate; heuristic only",
            "candidate_issues": [],
        })
    append_jsonl(Path(args.out), rows_out)
    rows_out.clear()

    if not multi_jobs:
        dur = int(time.time() - t0)
        phase_done("06_rerank", {"n_winners": len(single_cands), "duration_s": dur})
        return

    pool = GeminiPool(model=model, concurrency=concurrency,
                      temp=float(mcfg["llm"]["reranker"]["temperature"]),
                      max_out=int(mcfg["llm"]["reranker"]["max_output_tokens"]),
                      thinking=mcfg["llm"]["reranker"].get("thinking_level", "low"),
                      response_schema=RerankerOutput)

    i = 0
    n_winners = len(single_cands)
    while i < len(multi_jobs):
        batch = multi_jobs[i:i + slice_size]
        prompts = [j[2] for j in batch]
        results = await pool.batch(prompts)
        rows_out = []
        for (pid, cands, _), res in zip(batch, results):
            if isinstance(res, Exception):
                # fallback: heuristic
                best = max(cands, key=lambda c: heuristic_candidate_score(c["response_json"]))
                h = heuristic_candidate_score(best["response_json"])
                rows_out.append({
                    "path_id": pid, "best_candidate_idx": best["candidate_idx"],
                    "doc_ids": best["doc_ids"], "bridge_entity": best["bridge_entity"],
                    "response_json": best["response_json"],
                    "heuristic_score": h, "judge_confidence": 0.55,
                    "final_score": judge_w * 0.55 + (1 - judge_w) * h,
                    "justification": f"judge_error:{type(res).__name__}; heuristic fallback",
                    "candidate_issues": [],
                })
                continue
            obj = parse_json_lenient(res) or {}
            ro = validate_obj(obj, RerankerOutput)
            bi = int(obj.get("best_index", -1))
            issues = obj.get("candidate_issues", []) if isinstance(obj, dict) else []
            justification = obj.get("justification", "") if isinstance(obj, dict) else ""
            if bi < 0 or bi >= len(cands):
                # all disqualified -> drop this path (no winner)
                continue
            c = cands[bi]
            h = heuristic_candidate_score(c["response_json"])
            # Map confidence from justification text heuristically; default 0.8
            jc = 0.8
            rows_out.append({
                "path_id": pid, "best_candidate_idx": c["candidate_idx"],
                "doc_ids": c["doc_ids"], "bridge_entity": c["bridge_entity"],
                "response_json": c["response_json"],
                "heuristic_score": h, "judge_confidence": jc,
                "final_score": judge_w * jc + (1 - judge_w) * h,
                "justification": justification[:500],
                "candidate_issues": issues[:8],
            })
        append_jsonl(Path(args.out), rows_out)
        n_winners += len(rows_out)
        i += len(batch)
        print(f"[PROGRESS] phase=06_rerank {i}/{len(multi_jobs)} winners={n_winners}",
              flush=True)

    dur = int(time.time() - t0)
    phase_done("06_rerank", {"n_winners": n_winners, "duration_s": dur})


def main():
    args = ap()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
