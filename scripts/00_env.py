#!/usr/bin/env python
"""Phase 0: env bootstrap + model probe.

- Install missing pip packages (idempotent).
- Download NLTK punkt.
- Probe Gemini model candidates, write resolved_models.
- Verify CUDA + HotpotQA streaming.
- Write outputs/run_progress.json skeleton.
"""
from __future__ import annotations
import os, sys, subprocess, importlib, time, json
from pathlib import Path

ROOT = Path("/workspace/swadesh/multihop_rag")
sys.path.insert(0, str(ROOT / "src"))


# --- Pip install (idempotent) ---
PIP_MAP = {
    # pip_name -> import name
    "google-genai": "google.genai",
    "sentence-transformers": "sentence_transformers",
    "blingfire": "blingfire",
    "pysbd": "pysbd",
    "orjson": "orjson",
    "rank-bm25": "rank_bm25",
    "faiss-cpu": "faiss",
    "pymupdf": "fitz",
    "pdfplumber": "pdfplumber",
    "lxml": "lxml",
}


def pip_install(pkg: str):
    print(f"[ENV] installing {pkg}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


def ensure_pkgs():
    for pip_name, imp_name in PIP_MAP.items():
        try:
            importlib.import_module(imp_name)
            print(f"[ENV] {pip_name}: present", flush=True)
        except Exception:
            pip_install(pip_name)


def ensure_nltk_punkt():
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
            print("[ENV] nltk punkt: present", flush=True)
        except LookupError:
            nltk.download("punkt", quiet=True)
            print("[ENV] nltk punkt: downloaded", flush=True)
        # punkt_tab required for newer nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
    except Exception as e:
        print(f"[ENV] nltk not available: {e}", flush=True)


def probe_models(mcfg: dict) -> dict:
    from mhrag.gemini import resolve_candidates
    tasks = ["question_generation", "reranker", "solvability", "validation", "chunk_mapping"]
    resolved = {}
    for t in tasks:
        cands = mcfg["llm"][t]["candidates"]
        print(f"[PROBE] resolving {t} from {cands}", flush=True)
        m = resolve_candidates(cands)
        if not m:
            raise RuntimeError(f"no candidate model responded for {t}: {cands}")
        resolved[t] = m
        print(f"[PROBE] {t} = {m}", flush=True)
    return resolved


def verify_cuda():
    import torch
    print(f"[ENV] cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dev = torch.cuda.get_device_name(0)
    free, tot = torch.cuda.mem_get_info()
    print(f"[ENV] device={dev} vram_free={free/2**30:.1f}GB vram_tot={tot/2**30:.1f}GB", flush=True)


def verify_hf_stream():
    """Pull first 5 rows of HotpotQA wiki."""
    from datasets import load_dataset
    # streaming=True, no full download. This still hits HF to resolve config.
    t0 = time.time()
    ds = load_dataset("ParthMandaliya/hotpotqa-wiki", split="train", streaming=True)
    titles = []
    for i, row in enumerate(ds):
        titles.append(row.get("title", ""))
        if i >= 4:
            break
    el = time.time() - t0
    print(f"[ENV] hotpotqa-wiki streamed 5 rows in {el:.1f}s: {titles}", flush=True)


def main():
    t0 = time.time()
    ensure_pkgs()
    ensure_nltk_punkt()

    # Import after pkg install
    from mhrag.env import load_models_cfg
    from mhrag.progress import init_progress, phase_start, phase_done, load, save

    mcfg = load_models_cfg()

    verify_cuda()
    verify_hf_stream()

    print("[ENV] probing gemini models", flush=True)
    resolved = probe_models(mcfg)

    # Initialize run progress
    st = init_progress(run_id="pilot", resolved_models=resolved)
    st["resolved_models"] = resolved
    save(st)

    phase_start("00_env")
    phase_done("00_env", {"duration_s": int(time.time() - t0)})

    print(f"[ENV] done in {time.time()-t0:.1f}s", flush=True)
    print(f"[ENV] resolved_models={json.dumps(resolved)}", flush=True)


if __name__ == "__main__":
    main()
