#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="mhrag"
ENV_DIR="$REPO_DIR/conda_envs/$ENV_NAME"
ENV_YML="$REPO_DIR/environment.yml"

# Reuse UMD's miniconda installation (nothing written to UMD).
CONDA_DIR="/workspace/swadesh/UMD/miniconda3"

# All caches inside this project so nothing touches $HOME or UMD.
export PIP_CACHE_DIR="$REPO_DIR/.cache/pip"
export CONDA_PKGS_DIRS="$REPO_DIR/.cache/conda_pkgs"
export TMPDIR="$REPO_DIR/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR" "$REPO_DIR/conda_envs"

echo "=== multihop_rag conda env setup ==="
echo "  repo:      $REPO_DIR"
echo "  env name:  $ENV_NAME"
echo "  env dir:   $ENV_DIR"
echo "  conda:     $CONDA_DIR (shared from UMD; not modified)"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "ERROR: conda not found at $CONDA_DIR/bin/conda"
    echo "Run UMD/setup_conda.sh first so miniconda exists, or update CONDA_DIR in this script."
    exit 1
fi

echo "[1/3] Activating conda hook ..."
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"

if [ -d "$ENV_DIR/bin" ]; then
    echo "[2/3] Env already exists at $ENV_DIR -- skipping creation"
    echo "      (to recreate: rm -rf $ENV_DIR and rerun this script)"
else
    echo "[2/3] Creating $ENV_NAME environment ..."
    TMP_YML=$(mktemp "$TMPDIR/env_XXXXXX.yml")
    grep -v "^prefix:" "$ENV_YML" > "$TMP_YML"
    conda env create -f "$TMP_YML" --prefix "$ENV_DIR" -q
    rm -f "$TMP_YML"
fi

echo "[3/3] Verifying ..."
conda activate "$ENV_DIR"
echo "  which python:    $(which python)"
echo "  python:          $(python --version 2>&1)"
python - <<'PY'
import importlib, sys
mods = [
    ("torch", "torch.__version__"),
    ("datasets", "datasets.__version__"),
    ("transformers", "transformers.__version__"),
    ("sentence_transformers", "sentence_transformers.__version__"),
    ("huggingface_hub", "huggingface_hub.__version__"),
    ("google.genai", None),
    ("networkx", "networkx.__version__"),
    ("faiss", "faiss.__version__"),
    ("spacy", "spacy.__version__"),
    ("pyarrow", "pyarrow.__version__"),
]
for name, ver_expr in mods:
    try:
        m = importlib.import_module(name)
        v = eval(ver_expr) if ver_expr else "ok"
        print(f"  {name:22s} {v}")
    except Exception as e:
        print(f"  {name:22s} FAIL ({type(e).__name__}: {e})")

try:
    import torch
    print(f"  cuda available:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device:           {torch.cuda.get_device_name(0)}")
        print(f"  cuda capability:       {torch.cuda.get_device_capability(0)}")
        print(f"  torch cuda version:    {torch.version.cuda}")
except Exception as e:
    print(f"  torch.cuda check FAIL: {e}")
PY
conda deactivate

echo ""
echo "=== DONE ==="
echo "Activate: conda activate $ENV_DIR"
echo "Or register once: conda config --append envs_dirs $REPO_DIR/conda_envs"
echo "                  conda activate $ENV_NAME"
