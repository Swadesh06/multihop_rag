"""bge-m3 batched encoder (Phase 4)."""
from __future__ import annotations
import os, numpy as np, time
from typing import Iterable
from pathlib import Path


_MODEL = None
_DEV = None


def get_model(model_id: str = "BAAI/bge-m3", device: str = "cuda", fp16: bool = True):
    global _MODEL, _DEV
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device=device, cache_folder=os.environ.get("TRANSFORMERS_CACHE"))
    if fp16:
        m.half()
    _MODEL = m
    _DEV = device
    return m


def encode_texts(texts: list[str], model_id: str = "BAAI/bge-m3",
                 batch_size: int = 32, device: str = "cuda", fp16: bool = True,
                 show_progress: bool = False) -> np.ndarray:
    m = get_model(model_id=model_id, device=device, fp16=fp16)
    embs = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=show_progress)
    return embs.astype(np.float16)


def encode_to_file(texts: list[str], out_path: Path, idx_path: Path,
                   ids: list[str], model_id: str = "BAAI/bge-m3",
                   batch_size: int = 32, device: str = "cuda") -> None:
    import json
    emb = encode_texts(texts, model_id=model_id, batch_size=batch_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    with open(idx_path, "w") as f:
        for i, _id in enumerate(ids):
            f.write(json.dumps({"row": i, "id": _id}) + "\n")
