"""Baselines: BM25 + dense retrieval + Qwen2.5-3B reader (Phase 10)."""
from __future__ import annotations
import re, string, numpy as np, math
from collections import Counter
from typing import Any


# ---- BM25 ----

class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs_tokens)
        self.doc_lens = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if self.N else 0.0
        self.df = Counter()
        self.doc_tfs: list[dict[str, int]] = []
        for d in docs_tokens:
            tfs = Counter(d)
            self.doc_tfs.append(tfs)
            for t in tfs:
                self.df[t] += 1
        self.idf: dict[str, float] = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                                       for t, df in self.df.items()}

    def score(self, q_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in q_tokens:
            if t not in self.idf:
                continue
            idf_t = self.idf[t]
            for i, tfs in enumerate(self.doc_tfs):
                tf = tfs.get(t)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avg_len or 1.0))
                s[i] += idf_t * tf * (self.k1 + 1) / denom
        return s

    def topk(self, q_tokens: list[str], k: int = 10):
        s = self.score(q_tokens)
        idx = np.argsort(-s)[:k]
        return idx.tolist(), s[idx].tolist()


def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


# ---- retrieval metrics ----

def mrr(gold: set, ranked: list) -> float:
    for i, r in enumerate(ranked, 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(gold: set, ranked: list, k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for r in ranked[:k] if r in gold)
    return hit / len(gold)


# ---- SQuAD-style F1/EM ----

def normalize_ans(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1(pred: str, gold: str) -> float:
    pt = normalize_ans(pred).split()
    gt = normalize_ans(gold).split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return 2 * p * r / (p + r)


def em(pred: str, gold: str) -> float:
    return 1.0 if normalize_ans(pred) == normalize_ans(gold) else 0.0


# ---- Reader: Qwen2.5-3B-Instruct ----

_READER = None
_READER_TOK = None


def load_reader(model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
    global _READER, _READER_TOK
    if _READER is not None:
        return _READER, _READER_TOK
    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = os.environ.get("TRANSFORMERS_CACHE")
    _READER_TOK = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
    _READER = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True)
    _READER.eval()
    return _READER, _READER_TOK


def reader_answer(question: str, context: str, max_in: int = 4096, max_out: int = 128) -> str:
    import torch
    model, t = load_reader()
    sys_p = ("You are a careful question-answering assistant. Answer based only on the given "
             "context. Be concise; give the shortest exact-phrase answer (typically 1-6 words) "
             "that directly answers the question.")
    # Truncate context to fit
    ctx = t(context, add_special_tokens=False, truncation=True, max_length=max_in - 512)
    ctx_txt = t.decode(ctx.input_ids, skip_special_tokens=True)
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": f"Context:\n{ctx_txt}\n\nQuestion: {question}\n\nShort answer:"}]
    prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = t(prompt, return_tensors="pt", truncation=True, max_length=max_in).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_out, do_sample=False,
                             pad_token_id=t.eos_token_id)
    gen = out[0][enc.input_ids.shape[1]:]
    txt = t.decode(gen, skip_special_tokens=True).strip()
    # Keep first line
    txt = txt.split("\n")[0].strip()
    return txt
