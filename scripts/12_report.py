#!/usr/bin/env python
"""Phase 12 -- report + quality gates + escalation decision."""
from __future__ import annotations
import os, sys, time, argparse, json, subprocess
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/workspace/swadesh/multihop_rag/src")

from mhrag.env import load_cfg, load_models_cfg, get_logger, OUT, DATA, ROOT
from mhrag.progress import phase_start, phase_done, read_jsonl, load as load_prog, save as save_prog


def ap():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--annotated", default=str(OUT / "pilot_annotated.jsonl"))
    p.add_argument("--baselines", default=str(OUT / "pilot_baselines.json"))
    p.add_argument("--stats", default=str(OUT / "pilot_stats.json"))
    p.add_argument("--out_md", default=str(OUT / "pilot_report.md"))
    p.add_argument("--target_accepted", type=int, default=1000)
    return p.parse_args()


def evaluate_gates(annotated: list[dict], baselines: dict, target: int,
                    gates: dict) -> tuple[dict, bool]:
    """Return per-gate result + overall pass/fail."""
    n = len(annotated)
    qs = [r.get("generation_metadata", {}).get("quality_score") or 0 for r in annotated]
    avg_qs = (sum(qs) / len(qs)) if qs else 0.0

    rt_counter = Counter(r.get("reasoning_type", "unknown") for r in annotated)
    max_share = (max(rt_counter.values()) / len(annotated)) if annotated else 0.0

    # multihop_rate: share with hop_count >=2 AND solvability pass (annotated by construction)
    mh = sum(1 for r in annotated if int(r.get("hop_count", 2)) >= 2)
    mh_rate = mh / n if n else 0.0

    bm25_r10 = baselines.get("bm25", {}).get("recall_at_10", 0.0)
    reader_f1 = baselines.get("reader", {}).get("f1", 0.0)

    out = {
        "multihop_rate": {"value": mh_rate, "threshold": gates["multihop_rate_min"], "pass": mh_rate >= gates["multihop_rate_min"]},
        "avg_quality_score": {"value": avg_qs, "threshold": gates["avg_quality_score_min"], "pass": avg_qs >= gates["avg_quality_score_min"]},
        "max_reasoning_type_share": {"value": max_share, "threshold": gates["max_reasoning_type_share"], "pass": max_share <= gates["max_reasoning_type_share"]},
        "min_accepted_pct": {"value": (n / target) if target else 0.0, "threshold": gates["min_accepted_pct"], "pass": (n / target if target else 0) >= gates["min_accepted_pct"]},
        "bm25_recall_at_10_max": {"value": bm25_r10, "threshold": gates["bm25_recall_at_10_max"], "pass": bm25_r10 <= gates["bm25_recall_at_10_max"]},
        "reader_f1_max": {"value": reader_f1, "threshold": gates["reader_f1_max"], "pass": reader_f1 <= gates["reader_f1_max"]},
    }
    all_pass = all(g["pass"] for g in out.values())
    return out, all_pass


def build_report(annotated, baselines, stats, gates_result, all_pass, resolved, progress, run_id) -> str:
    # Pick 5 accepted samples and 3 rejects (sampled proxy: bottom quality scores)
    annotated_sorted = sorted(annotated, key=lambda r: -(r.get("generation_metadata", {}).get("quality_score") or 0.0))
    samples = annotated_sorted[:5]
    rejects = annotated_sorted[-3:] if len(annotated_sorted) >= 3 else []

    lines = []
    lines.append(f"# Multi-Hop RAG {run_id.title()} Report\n")
    lines.append("## Summary\n")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- start_ts: {progress.get('start_ts', '')}")
    lines.append(f"- resolved_models: {json.dumps(resolved)}")
    phases = progress.get("phases", {})
    lines.append(f"- phases completed: {sum(1 for p in phases.values() if p.get('status')=='done')}/{len(phases)}")
    lines.append("")

    lines.append("## Corpus\n")
    c1 = phases.get("01_corpus", {})
    lines.append(f"- kept docs: {c1.get('n_docs', 0)}")
    lines.append(f"- scan: {c1.get('scanned', 0)} -> pool: {c1.get('pool', 0)} -> F3: {c1.get('f3_keep', 0)} -> final: {c1.get('n_docs', 0)}")
    c2 = phases.get("02_chunk", {})
    lines.append(f"- chunks: {c2.get('n_chunks', 0)}")
    lines.append("")

    lines.append("## Graph\n")
    c3 = phases.get("03_graph", {})
    lines.append(f"- edges: {c3.get('n_edges', 0)}")
    lines.append(f"- paths sampled: {c3.get('n_paths', 0)}")
    lines.append(f"- reasoning type caps (picks): {c3.get('by_type', {})}")
    lines.append("")

    lines.append("## Generation yield\n")
    p5 = phases.get("05_qgen", {})
    p6 = phases.get("06_rerank", {})
    p7 = phases.get("07_solv", {})
    p8 = phases.get("08_filter", {})
    lines.append(f"- QG calls: {p5.get('n_calls', 0)}; schema-valid: {p5.get('n_candidates_valid', 0)}")
    lines.append(f"- Reranker winners: {p6.get('n_winners', 0)}")
    lines.append(f"- Solvability pass: {p7.get('n_pass', 0)} (rej_single={p7.get('n_reject_single',0)}, rej_trivial={p7.get('n_reject_trivial',0)})")
    lines.append(f"- Post-filter accepted: {p8.get('n_accepted', 0)}")
    lines.append("")

    lines.append("## Quality gates\n")
    lines.append("| Gate | Value | Threshold | Pass |")
    lines.append("|---|---|---|---|")
    for k, v in gates_result.items():
        lines.append(f"| {k} | {v['value']:.4f} | {v['threshold']} | {'yes' if v['pass'] else 'no'} |")
    lines.append("")

    lines.append("## Distribution comparison vs wiki-multihop-qa-500k\n")
    if stats and "ours" in stats:
        o = stats["ours"]
        r = stats.get("reference", {})
        lines.append("| metric | ours | ref |")
        lines.append("|---|---|---|")
        lines.append(f"| n | {o.get('n','')} | {r.get('n','')} |")
        lines.append(f"| q_len_mean | {o.get('q_len_mean','')} | {r.get('q_len_mean','')} |")
        lines.append(f"| sa_len_mean | {o.get('sa_len_mean','')} | {r.get('sa_len_mean','')} |")
        lines.append(f"| la_len_mean | {o.get('la_len_mean','')} | {r.get('la_len_mean','')} |")
        lines.append("")

    lines.append("## Baselines\n")
    b = baselines or {}
    lines.append(f"- BM25 Recall@10 = {b.get('bm25',{}).get('recall_at_10',0):.3f}, MRR = {b.get('bm25',{}).get('mrr',0):.3f}, G3 Recall = {b.get('bm25',{}).get('g3_recall',0):.3f}")
    lines.append(f"- Dense Recall@10 = {b.get('dense',{}).get('recall_at_10',0):.3f}, MRR = {b.get('dense',{}).get('mrr',0):.3f}, G3 Recall = {b.get('dense',{}).get('g3_recall',0):.3f}")
    lines.append(f"- Qwen2.5-3B Reader F1 = {b.get('reader',{}).get('f1',0):.3f} EM = {b.get('reader',{}).get('em',0):.3f} (n={b.get('reader',{}).get('n',0)})")
    lines.append("")

    lines.append("## Samples\n")
    for i, s in enumerate(samples):
        lines.append(f"### Accepted {i+1}\n")
        lines.append(f"- **query**: {s.get('query','')}")
        lines.append(f"- **short_answer**: {s.get('short_answer','')}")
        lines.append(f"- **long_answer**: {s.get('long_answer','')}")
        docs = [f"{d.get('doc_id','')}({d.get('grade','')})" for d in s.get('supporting_docs', [])]
        lines.append(f"- **supporting_docs**: {', '.join(docs)}")
        lines.append(f"- **reasoning_type**: {s.get('reasoning_type','')}  **difficulty**: {s.get('difficulty','')}")
        lines.append(f"- **bridge_entity**: {s.get('bridge_entity','')}")
        lines.append(f"- **reasoning_chain**: {s.get('reasoning_chain','')}")
        lines.append("")

    lines.append("## Escalation decision\n")
    if all_pass:
        lines.append("DECISION: escalate (all gates pass).\n")
    else:
        lines.append("DECISION: stop. Failing gates:\n")
        for k, v in gates_result.items():
            if not v["pass"]:
                lines.append(f"- {k}: value={v['value']:.4f}, threshold={v['threshold']}")
        lines.append("")
        lines.append("### Failure-mode analysis\n")
        lines.append("See run_progress.json for per-phase counts; iterate on the failing dimension (e.g., graph path enumeration for multihop_rate, corpus filters for baseline recall).\n")

    return "\n".join(lines) + "\n"


def main():
    args = ap()
    cfg = load_cfg(Path(args.config).stem)
    mcfg = load_models_cfg()
    log = get_logger("12_report")
    phase_start("12_report")
    t0 = time.time()

    annotated = read_jsonl(Path(args.annotated))
    baselines = {}
    if Path(args.baselines).exists():
        with open(args.baselines) as f:
            baselines = json.load(f)
    stats = {}
    if Path(args.stats).exists():
        with open(args.stats) as f:
            stats = json.load(f)

    run_id = cfg.get("run_id", "pilot")
    target = int(cfg.get("target_q_accepted", args.target_accepted))
    gates_cfg = mcfg["quality_gates"]
    gates_res, all_pass = evaluate_gates(annotated, baselines, target, gates_cfg)

    prog = load_prog()
    resolved = prog.get("resolved_models", {})
    md = build_report(annotated, baselines, stats, gates_res, all_pass, resolved, prog, run_id)
    Path(args.out_md).write_text(md)
    log.info(f"wrote {args.out_md}")

    # Update run_progress.json
    prog["quality_gates"] = gates_res
    prog["escalation_decision"] = "escalate" if all_pass else "stop"
    save_prog(prog)

    phase_done("12_report", {"duration_s": int(time.time() - t0),
                              "escalation_decision": prog["escalation_decision"]})

    if all_pass and run_id == "pilot":
        # Launch scale run via bash script
        log.info("all gates pass -> launching scale run")
        subprocess.Popen(["bash", str(ROOT / "scripts" / "run_scale.sh")])
    elif run_id == "pilot":
        log.info("gates failed -> stop")


if __name__ == "__main__":
    main()
