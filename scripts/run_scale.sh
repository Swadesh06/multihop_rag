#!/usr/bin/env bash
# Launch the scale-up run (phases 1-12 with configs/scale.yaml) in tmux.
# Only called by 12_report.py when pilot passes all gates.
set -euo pipefail

ROOT=/workspace/swadesh/multihop_rag
ACT="$ROOT/scripts/_activate.sh"
CFG="$ROOT/configs/scale.yaml"

# Scale reuses the same data/ paths for simplicity, but outputs/ use scale_ prefix.
# Since phase scripts accept --out / --prefix, we rename intermediates under
# outputs/ so pilot and scale coexist. Data artifacts (docs/chunks/graph/embed)
# are rebuilt into data/* (overwrite).

run_phase() {
  local ph="$1"; shift
  local sess="mhrag_scale_p${ph}"
  tmux new-session -d -s "$sess" "bash -c 'source $ACT && python $ROOT/scripts/${ph}.py --config $CFG $* > $ROOT/logs/scale_${ph}.log 2>&1; echo EXIT_CODE: \$? >> $ROOT/logs/scale_${ph}.log'"
  # Wait until session dies
  while tmux has-session -t "$sess" 2>/dev/null; do sleep 10; done
}

# scale-specific output paths
DATA=$ROOT/data
OUT=$ROOT/outputs

# Phase 1
run_phase 01_corpus \
  --out "$DATA/corpus/scale_docs.parquet"
# Phase 2
run_phase 02_chunk \
  --docs "$DATA/corpus/scale_docs.parquet" --out "$DATA/chunks/scale_chunks.parquet"
# Phase 3
run_phase 03_graph \
  --docs "$DATA/corpus/scale_docs.parquet" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --out_edges "$DATA/graphs/scale_edges.parquet" --out_paths "$DATA/graphs/scale_paths.jsonl"
# Phase 4
run_phase 04_embed \
  --docs "$DATA/corpus/scale_docs.parquet" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --prefix scale
# Phase 5
run_phase 05_qgen \
  --paths "$DATA/graphs/scale_paths.jsonl" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --docs "$DATA/corpus/scale_docs.parquet" --out "$OUT/scale_qg_raw.jsonl"
# Phase 6
run_phase 06_rerank \
  --raw "$OUT/scale_qg_raw.jsonl" --paths "$DATA/graphs/scale_paths.jsonl" \
  --docs "$DATA/corpus/scale_docs.parquet" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --out "$OUT/scale_qg_best.jsonl"
# Phase 7
run_phase 07_solv \
  --best "$OUT/scale_qg_best.jsonl" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --docs "$DATA/corpus/scale_docs.parquet" --out "$OUT/scale_solv.jsonl"
# Phase 8
run_phase 08_filter \
  --solv "$OUT/scale_solv.jsonl" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --docs "$DATA/corpus/scale_docs.parquet" --out "$OUT/scale_filtered.jsonl"
# Phase 9
run_phase 09_annotate \
  --filtered "$OUT/scale_filtered.jsonl" --chunks "$DATA/chunks/scale_chunks.parquet" \
  --docs "$DATA/corpus/scale_docs.parquet" --prefix scale --out "$OUT/scale_annotated.jsonl"
# Phase 10
run_phase 10_baselines \
  --annotated "$OUT/scale_annotated.jsonl" --docs "$DATA/corpus/scale_docs.parquet" \
  --chunks "$DATA/chunks/scale_chunks.parquet" --prefix scale --out "$OUT/scale_baselines.json"
# Phase 11
run_phase 11_stats \
  --annotated "$OUT/scale_annotated.jsonl" --out_json "$OUT/scale_stats.json" --out_md "$OUT/scale_stats.md"
# Phase 12 -- produce scale_report.md only; do NOT recurse into escalation
run_phase 12_report \
  --annotated "$OUT/scale_annotated.jsonl" --baselines "$OUT/scale_baselines.json" \
  --stats "$OUT/scale_stats.json" --out_md "$OUT/scale_report.md"

echo "scale complete" >> "$ROOT/logs/scale_done.log"
