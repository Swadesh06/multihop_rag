"""Progress helpers. run_progress.json read/update + [PROGRESS] log emitter."""
from __future__ import annotations
import json, os, time, sys
from pathlib import Path
from typing import Any
from .env import OUT

PFILE = OUT / "run_progress.json"


PHASES = [
    "00_env", "01_corpus", "02_chunk", "03_graph", "04_embed",
    "05_qgen", "06_rerank", "07_solv", "08_filter", "09_annotate",
    "10_baselines", "11_stats", "12_report",
]


def init_progress(run_id: str = "pilot", resolved_models: dict | None = None) -> dict:
    """Create skeleton if missing. Non-destructive if exists."""
    OUT.mkdir(parents=True, exist_ok=True)
    if PFILE.exists():
        st = load()
    else:
        st = {
            "pipeline_run": run_id,
            "start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resolved_models": resolved_models or {},
            "phases": {ph: {"status": "pending"} for ph in PHASES},
            "quality_gates": {},
            "escalation_decision": "pending",
        }
        save(st)
    if resolved_models and not st.get("resolved_models"):
        st["resolved_models"] = resolved_models
        save(st)
    return st


def load() -> dict:
    if not PFILE.exists():
        return {}
    with open(PFILE) as f:
        return json.load(f)


def save(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = PFILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, PFILE)


def phase_start(phase: str) -> None:
    st = load()
    st.setdefault("phases", {}).setdefault(phase, {})
    st["phases"][phase]["status"] = "running"
    st["phases"][phase]["start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(st)


def phase_done(phase: str, extra: dict | None = None) -> None:
    st = load()
    st.setdefault("phases", {}).setdefault(phase, {})
    st["phases"][phase]["status"] = "done"
    st["phases"][phase]["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = st["phases"][phase].get("start")
    if start:
        try:
            s = time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ"))
            e = time.mktime(time.strptime(st["phases"][phase]["end"], "%Y-%m-%dT%H:%M:%SZ"))
            st["phases"][phase]["duration_s"] = int(e - s)
        except Exception:
            pass
    if extra:
        st["phases"][phase].update(extra)
    save(st)


def phase_fail(phase: str, msg: str) -> None:
    st = load()
    st.setdefault("phases", {}).setdefault(phase, {})
    st["phases"][phase]["status"] = "failed"
    st["phases"][phase]["error"] = msg[:500]
    save(st)


class ProgressEmitter:
    """Print [PROGRESS] lines at a capped rate (every `every_n` items or `every_s`)."""
    def __init__(self, phase: str, total: int | None = None, every_n: int = 100, every_s: float = 10.0):
        self.phase = phase
        self.total = total
        self.every_n = every_n
        self.every_s = every_s
        self.n = 0
        self.t0 = time.time()
        self.tl = self.t0
        self.nl = 0

    def tick(self, inc: int = 1, force: bool = False) -> None:
        self.n += inc
        now = time.time()
        dn = self.n - self.nl
        dt = now - self.tl
        if not force and dn < self.every_n and dt < self.every_s:
            return
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        mem = 0.0
        try:
            import resource
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        except Exception:
            pass
        tot = self.total if self.total is not None else "?"
        print(f"[PROGRESS] phase={self.phase} {self.n}/{tot} elapsed={el:.1f}s rate={rate:.1f}/s mem_mb={mem:.0f}",
              flush=True)
        self.tl = now
        self.nl = self.n

    def done(self) -> None:
        self.tick(inc=0, force=True)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out
