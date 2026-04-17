"""Environment / config helpers. Load yaml, resolve paths, tiny logger."""
from __future__ import annotations
import os, sys, time, json, yaml, logging
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MHRAG_ROOT", "/workspace/swadesh/multihop_rag"))
LOGS = ROOT / "logs"
OUT = ROOT / "outputs"
DATA = ROOT / "data"
CFG = ROOT / "configs"
PROMPTS = ROOT / "prompts"


def load_yaml(p: Path) -> dict[str, Any]:
    with open(p) as f:
        return yaml.safe_load(f)


def load_cfg(pilot_or_scale: str) -> dict[str, Any]:
    """Load pilot.yaml or scale.yaml, resolve `extends:`."""
    p = CFG / f"{pilot_or_scale}.yaml"
    cfg = load_yaml(p)
    if "extends" in cfg:
        base = load_yaml(CFG / cfg["extends"])
        cfg = _deep_merge(base, cfg)
        cfg.pop("extends", None)
    return cfg


def load_models_cfg() -> dict[str, Any]:
    return load_yaml(CFG / "models.yaml")


def _deep_merge(a: dict, b: dict) -> dict:
    """b overrides a."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_prompt(name: str) -> str:
    with open(PROMPTS / f"{name}.txt") as f:
        return f.read()


def get_logger(phase: str) -> logging.Logger:
    """Phase logger -> stdout + logs/<phase>.log."""
    lg = logging.getLogger(f"mhrag.{phase}")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS / f"{phase}.log")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


def gemini_api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "")
    if not k:
        raise RuntimeError("GEMINI_API_KEY not set")
    return k
