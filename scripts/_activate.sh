#!/usr/bin/env bash
# Canonical activation block used by every tmux session + shell.
# Sourced, not executed. Idempotent.

source /workspace/swadesh/UMD/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/swadesh/btp_neurips_2026/conda_envs/btp

# SSH key fix for mfs workspace (see CLAUDE.md 2.6)
if [ ! -f /tmp/id_ed25519_gh ]; then
  install -m 600 /workspace/.ssh/id_ed25519 /tmp/id_ed25519_gh
fi

set -a
source /workspace/swadesh/multihop_rag/.env
set +a

export HF_HOME=/workspace/swadesh/multihop_rag
export TRANSFORMERS_CACHE=/workspace/swadesh/multihop_rag/hub
export HF_DATASETS_CACHE=/workspace/swadesh/multihop_rag/datasets
export TMPDIR=/workspace/swadesh/multihop_rag/.tmp
export PIP_CACHE_DIR=/workspace/swadesh/multihop_rag/.cache/pip
export PYTHONPATH=/workspace/swadesh/multihop_rag/src:${PYTHONPATH:-}

mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$PIP_CACHE_DIR"
cd /workspace/swadesh/multihop_rag
