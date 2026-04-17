#!/usr/bin/env python
"""Phase 5 -- question generation (best-of-N).

For each path, run `candidates_per_path` concurrent QG calls. Parse+validate.
Append to outputs/pilot_qg_raw.jsonl, resuming on restart.
"""
from __future__ import annotations
import os, sys, time, argparse, asyncio, json, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

import pandas as pd

from mhrag.env import load_cfg, load_models_cfg, get_logger, DATA, OUT, load_prompt
from mhrag.progress import (phase_start, phase_done, ProgressEmitter,
                              append_jsonl, read_jsonl, load as load_prog)
from mhrag.gemini import GeminiPool, QGOutput, parse_json_lenient, validate_obj


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--paths", default=str(DATA / "graphs" / "pilot_paths.jsonl"))
    p.add_argument("--chunks", default=str(DATA / "chunks" / "pilot_chunks.parquet"))
    p.add_argument("--docs", default=str(DATA / "corpus" / "pilot_docs.parquet"))
    p.add_argument("--out", default=str(OUT / "pilot_qg_raw.jsonl"))
    return p.parse_args()


DOC_LETTERS = "ABCDEFGHIJKLMNOP"


def build_docs_block(path_doc_ids: list[str], chunks_by_doc: dict[str, list[dict]],
                     doc_by_id: dict[str, dict], path_chunk_hints: dict[str, list[str]],
                     max_chars_per_chunk: int = 1500, max_chunks_per_doc: int = 3) -> str:
    """Compose the DOCUMENTS block for the QG prompt."""
    lines = []
    for i, did in enumerate(path_doc_ids):
        letter = DOC_LETTERS[i]
        title = doc_by_id[did].get("title", "")
        hints = set(path_chunk_hints.get(did) or [])
        chs = chunks_by_doc.get(did) or []
        # prefer hinted chunks then earliest
        sel = [ch for ch in chs if ch["chunk_id"] in hints][:max_chunks_per_doc]
        if len(sel) < max_chunks_per_doc:
            rest = [ch for ch in chs if ch["chunk_id"] not in hints]
            sel.extend(rest[: max_chunks_per_doc - len(sel)])
        lines.append(f"--- Document {letter} (doc_id: {did}, title: \"{title}\") ---")
        for ch in sel:
            txt = (ch.get("chunk_text") or "")[:max_chars_per_chunk]
            lines.append(txt)
        lines.append("")
    return "\n".join(lines)


def validate_qg_record(rec: dict, doc_by_id: dict[str, dict]) -> tuple[bool, list[str]]:
    issues = []
    qo = validate_obj(rec, QGOutput)
    if not qo:
        return False, ["schema_invalid"]
    q = qo.query.strip()
    short = qo.short_answer.strip()
    if not q.endswith("?") or len(q.split()) < 5:
        issues.append("bad_question")
    if short.lower() in {"it", "they", "he", "she", "yes", "no"}:
        issues.append("pronoun_answer")
    if short and short.lower() in q.lower():
        # reject only verbatim word-boundary
        import re as _re
        if _re.search(r"\b" + _re.escape(short.lower()) + r"\b", q.lower()):
            issues.append("answer_in_question")
    bridge = qo.bridge_entity.strip()
    if bridge and bridge.lower() in q.lower() and len(bridge) >= 4:
        issues.append("bridge_in_question")
    # verbatim quoted spans: whitespace-normalized match (model may collapse newlines)
    import re as _re
    def _norm_ws(x: str) -> str:
        return _re.sub(r"\s+", " ", x or "").strip()
    bad_spans = []
    for did, spans in (qo.quoted_spans or {}).items():
        dc = doc_by_id.get(did)
        if not dc:
            bad_spans.append(f"unknown_doc:{did}")
            continue
        full_norm = _norm_ws(dc.get("clean_text") or "")
        for sp in spans or []:
            if not sp:
                continue
            if _norm_ws(sp) not in full_norm:
                bad_spans.append("span_not_in_doc")
                break
    if bad_spans:
        issues.extend(bad_spans)
    return len(issues) == 0, issues


async def run(args):
    cfg = load_cfg(Path(args.config).stem)
    mcfg = load_models_cfg()
    log = get_logger("05_qgen")
    phase_start("05_qgen")
    t0 = time.time()

    st = load_prog()
    model = st.get("resolved_models", {}).get("question_generation") or mcfg["llm"]["question_generation"]["candidates"][0]
    log.info(f"QG model={model}")

    # Resume: read existing outputs, skip completed (path_id, candidate_idx)
    done_keys: set[tuple[str, int]] = set()
    if Path(args.out).exists():
        prior = read_jsonl(Path(args.out))
        for r in prior:
            done_keys.add((r.get("path_id", ""), int(r.get("candidate_idx", -1))))
        log.info(f"resuming: {len(prior)} prior candidates loaded, {len(done_keys)} keys covered")

    # Load inputs
    docs_df = pd.read_parquet(args.docs)
    doc_by_id = {r["doc_id"]: r for r in docs_df.to_dict("records")}
    cdf = pd.read_parquet(args.chunks)
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in cdf.to_dict("records"):
        chunks_by_doc[r["doc_id"]].append(r)

    # Load paths
    paths = []
    with open(args.paths) as f:
        for ln in f:
            paths.append(json.loads(ln))
    log.info(f"paths={len(paths)}")

    qcfg = cfg["qg"]
    n_cands = int(qcfg["candidates_per_path"])
    concurrency = int(qcfg["concurrency"])
    slice_size = int(qcfg["batch_size_per_gather"])
    temp = float(qcfg["temperature"])
    top_p = qcfg.get("top_p")

    prompt_template = load_prompt("qg")

    # Build all (path, cand_idx) -> prompt pairs that aren't already done
    jobs = []
    for p in paths:
        pid = p["path_id"]
        dids = p["doc_ids"]
        bridge = p["bridge_entity"]
        dblock = build_docs_block(dids, chunks_by_doc, doc_by_id, p.get("chunk_hints") or {})
        prompt = prompt_template.replace("{n_docs}", str(len(dids))).replace(
            "{bridge_entity}", bridge).replace("{documents_block}", dblock)
        for ci in range(n_cands):
            if (pid, ci) in done_keys:
                continue
            jobs.append((pid, ci, dids, bridge, prompt))
    log.info(f"jobs={len(jobs)} (concurrency={concurrency}, slice={slice_size})")

    if not jobs:
        log.info("nothing to do")
        phase_done("05_qgen", {"n_calls": 0, "n_candidates_valid": 0,
                                "duration_s": int(time.time() - t0)})
        return

    pool = GeminiPool(model=model, concurrency=concurrency, temp=temp,
                      max_out=int(mcfg["llm"]["question_generation"]["max_output_tokens"]),
                      thinking=mcfg["llm"]["question_generation"].get("thinking_level", "minimal"),
                      response_schema=QGOutput,
                      top_p=top_p)

    n_valid_total = 0
    t_start = time.time()
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + slice_size]
        prompts = [j[4] for j in batch]
        results = await pool.batch(prompts)
        rows_out = []
        for (pid, ci, dids, bridge, _), res in zip(batch, results):
            rec_out = {"path_id": pid, "candidate_idx": ci, "doc_ids": dids,
                       "bridge_entity": bridge}
            if isinstance(res, Exception):
                rec_out["error"] = f"{type(res).__name__}: {str(res)[:200]}"
                rec_out["validation_flags"] = ["api_error"]
                rec_out["valid"] = False
                rows_out.append(rec_out)
                continue
            obj = parse_json_lenient(res)
            if obj is None:
                rec_out["raw"] = res[:2000]
                rec_out["validation_flags"] = ["json_invalid"]
                rec_out["valid"] = False
                rows_out.append(rec_out)
                continue
            ok, issues = validate_qg_record(obj, doc_by_id)
            rec_out["response_json"] = obj
            rec_out["validation_flags"] = issues
            rec_out["valid"] = ok
            if ok:
                n_valid_total += 1
            rows_out.append(rec_out)
        append_jsonl(Path(args.out), rows_out)
        i += len(batch)
        el = time.time() - t_start
        rate = i / el if el else 0.0
        print(f"[PROGRESS] phase=05_qgen {i}/{len(jobs)} elapsed={el:.1f}s rate={rate:.2f}/s valid={n_valid_total}",
              flush=True)

    dur = int(time.time() - t0)
    phase_done("05_qgen", {"n_calls": len(jobs), "n_candidates_valid": n_valid_total,
                             "duration_s": dur})


def main():
    args = ap()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
