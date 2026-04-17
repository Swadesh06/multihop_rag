"""torch matmul top-k (exact NN)."""
from __future__ import annotations
import numpy as np
from typing import Tuple

try:
    import torch
except Exception:
    torch = None


def topk(queries: np.ndarray, corpus: np.ndarray, k: int = 10,
         device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """queries (M, D), corpus (N, D), both normalized. Returns (M, k) scores+idx."""
    assert torch is not None, "torch required"
    Q = torch.from_numpy(queries).to(device)
    V = torch.from_numpy(corpus).to(device)
    if Q.dtype != V.dtype:
        Q = Q.to(V.dtype)
    scores = Q @ V.T
    top_vals, top_idx = torch.topk(scores, k=min(k, V.shape[0]), dim=1)
    return top_vals.float().cpu().numpy(), top_idx.cpu().numpy()


def topk_chunked(queries: np.ndarray, corpus: np.ndarray, k: int = 10,
                 device: str = "cuda", chunk: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Chunked over corpus to avoid OOM."""
    assert torch is not None
    Q = torch.from_numpy(queries).to(device)
    n_c = corpus.shape[0]
    all_top_vals = []
    all_top_idx = []
    best_vals = None
    best_idx = None
    for i in range(0, n_c, chunk):
        V = torch.from_numpy(corpus[i:i + chunk]).to(device)
        if Q.dtype != V.dtype:
            Qc = Q.to(V.dtype)
        else:
            Qc = Q
        s = Qc @ V.T  # (M, chunk)
        tv, ti = torch.topk(s, k=min(k, s.shape[1]), dim=1)
        ti = ti + i
        if best_vals is None:
            best_vals = tv
            best_idx = ti
        else:
            cat_v = torch.cat([best_vals, tv], dim=1)
            cat_i = torch.cat([best_idx, ti], dim=1)
            tv2, idx2 = torch.topk(cat_v, k=min(k, cat_v.shape[1]), dim=1)
            best_vals = tv2
            best_idx = torch.gather(cat_i, 1, idx2)
    return best_vals.float().cpu().numpy(), best_idx.cpu().numpy()
