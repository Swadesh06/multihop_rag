#!/usr/bin/env python
"""Phase 7 -- solvability gate.

For each Q in pilot_qg_best.jsonl, call solvability prompt per grade>=2
supporting doc. Decide pass / reject_single / reject_trivial.
Write pilot_solv.jsonl.
"""
from __future__ import annotations
import os, sys, time, argparse, asyncio, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd

from mhrag.env import load_cfg, load_models_cfg, get_logger, DATA, OUT, load_prompt
from mhrag.progress import (phase_start, phase_done, append_jsonl, read_jsonl, load as load_prog)
from mhrag.gemini import GeminiPool, SolvOutput, parse_json_lenient, validate_obj
from mhrag.solvability import build_solv_prompt, decide


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--best", default=str(OUT / "pilot_qg_best.jsonl"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--out", default=str(OUT / "pilot_solv.jsonl"))
    return p.parse_args()


async def run(args):
    cfg = load_cfg(Path(args.config).stem)
    mcfg = load_models_cfg()
    log = get_logger("07_solv")
    phase_start("07_solv")
    t0 = time.time()

    st = load_prog()
    model = st.get("resolved_models", {}).get("solvability") or mcfg["llm"]["solvability"]["candidates"][0]
    log.info(f"solvability model={model}")

    scfg = cfg["solvability"]
    min_grade = int(scfg.get("only_test_grade_ge", 2))
    concurrency = int(scfg["concurrency"])
    slice_size = int(scfg["batch_size_per_gather"])

    # Resume
    done_pids: set[str] = set()
    if Path(args.out).exists():
        for r in read_jsonl(Path(args.out)):
            done_pids.add(r["path_id"])
        log.info(f"resuming: {len(done_pids)} done")

    # Load inputs
    best = read_jsonl(Path(args.best))
    log.info(f"best winners loaded: {len(best)}")
    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    cdf = pd.read_parquet(args.chunks)
    chunks_by_doc = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    template = load_prompt("solvability")

    # Build jobs: one per (path, doc) where grade >= min_grade
    def doc_chunks_text(did: str, max_chars: int = 6000) -> str:
        chs = chunks_by_doc.get(did) or []
        out = []
        tot = 0
        for ch in chs:
            t = (ch.get("chunk_text") or "")
            out.append(t)
            tot += len(t)
            if tot >= max_chars:
                break
        return "\n".join(out)

    jobs = []  # (path_id, q_json, doc_id, prompt)
    for w in best:
        pid = w["path_id"]
        if pid in done_pids:
            continue
        rj = w.get("response_json") or {}
        q = rj.get("query", "")
        sa = rj.get("short_answer", "")
        sds = rj.get("supporting_docs") or []
        for sd in sds:
            g = int(sd.get("grade", 0))
            if g < min_grade:
                continue
            did = sd.get("doc_id", "")
            if did not in doc_by_id:
                continue
            title = doc_by_id[did].get("title", "")
            dtxt = doc_chunks_text(did)
            prompt = build_solv_prompt(template, q, sa, did, title, dtxt)
            jobs.append((pid, w, did, prompt))
    log.info(f"solvability jobs={len(jobs)}")

    if not jobs:
        phase_done("07_solv", {"n_pass": 0, "n_reject_single": 0, "n_reject_trivial": 0,
                                "duration_s": int(time.time() - t0)})
        return

    pool = GeminiPool(model=model, concurrency=concurrency,
                      temp=float(mcfg["llm"]["solvability"]["temperature"]),
                      max_out=int(mcfg["llm"]["solvability"]["max_output_tokens"]),
                      thinking=mcfg["llm"]["solvability"].get("thinking_level", "medium"),
                      response_schema=SolvOutput)

    # Buffer per-pid results until all that path's jobs have returned
    per_path: dict[str, list[dict]] = defaultdict(list)
    per_path_meta: dict[str, dict] = {}
    expected_per_path = defaultdict(int)
    for pid, w, did, _ in jobs:
        expected_per_path[pid] += 1
        per_path_meta[pid] = w

    # Run batched
    n_pass = n_rej_single = n_rej_triv = 0
    i = 0
    t_start = time.time()
    while i < len(jobs):
        batch = jobs[i:i + slice_size]
        prompts = [j[3] for j in batch]
        results = await pool.batch(prompts)
        for (pid, w, did, _), res in zip(batch, results):
            rec = {"doc_id": did}
            if isinstance(res, Exception):
                rec["error"] = f"{type(res).__name__}: {str(res)[:200]}"
                rec["solvable_from_this_document_alone"] = False
                rec["confidence"] = "low"
            else:
                obj = parse_json_lenient(res) or {}
                so = validate_obj(obj, SolvOutput)
                rec["solvable_from_this_document_alone"] = bool(
                    obj.get("solvable_from_this_document_alone", False))
                rec["confidence"] = obj.get("confidence", "medium")
                rec["all_facts_present"] = bool(obj.get("all_facts_present", False))
                rec["explanation"] = (obj.get("explanation") or "")[:400]
                rec["required_facts"] = obj.get("required_facts", [])[:10]
            per_path[pid].append(rec)
            # If all results for this pid received, finalize
            if len(per_path[pid]) == expected_per_path[pid]:
                w_rec = per_path_meta[pid]
                decision, sol_conf = decide(per_path[pid])
                out_rec = {
                    "path_id": pid, "doc_ids": w_rec.get("doc_ids"),
                    "response_json": w_rec.get("response_json"),
                    "best_candidate_idx": w_rec.get("best_candidate_idx"),
                    "heuristic_score": w_rec.get("heuristic_score"),
                    "judge_confidence": w_rec.get("judge_confidence"),
                    "final_score": w_rec.get("final_score"),
                    "solv_per_doc": per_path[pid],
                    "decision": decision,
                    "solvability_confidence": sol_conf,
                }
                append_jsonl(Path(args.out), [out_rec])
                if decision == "pass":
                    n_pass += 1
                elif decision == "reject_single_hop":
                    n_rej_single += 1
                else:
                    n_rej_triv += 1
        i += len(batch)
        el = time.time() - t_start
        rate = i / el if el else 0.0
        print(f"[PROGRESS] phase=07_solv {i}/{len(jobs)} elapsed={el:.1f}s rate={rate:.2f}/s "
              f"pass={n_pass} rej_single={n_rej_single} rej_triv={n_rej_triv}", flush=True)

    dur = int(time.time() - t0)
    phase_done("07_solv", {"n_pass": n_pass, "n_reject_single": n_rej_single,
                             "n_reject_trivial": n_rej_triv, "duration_s": dur})


def main():
    args = ap()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
