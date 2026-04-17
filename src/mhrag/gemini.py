"""Async Gemini client with retry + structured output helpers."""
from __future__ import annotations
import asyncio, os, json, time, random
from typing import Any, Type
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import logging

# google-genai is the new SDK (required for Gemini 3.x)
try:
    from google import genai
    from google.genai import types as gtypes
    _GENAI_OK = True
except Exception as e:
    _GENAI_OK = False
    _GENAI_ERR = e

_LOG = logging.getLogger("mhrag.gemini")


def _client():
    if not _GENAI_OK:
        raise RuntimeError(f"google-genai import failed: {_GENAI_ERR}")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _thinking_cfg(level: str):
    budgets = {"minimal": 0, "low": 256, "medium": 1024, "high": 4096}
    if level in budgets:
        try:
            return gtypes.ThinkingConfig(thinking_budget=budgets[level])
        except Exception:
            return None
    return None


class _RetryableErr(Exception):
    pass


def _is_transient(e: Exception) -> bool:
    m = str(e).lower()
    return any(s in m for s in ["429", "500", "502", "503", "504", "timeout",
                                "rate", "quota", "resource_exhausted",
                                "unavailable", "deadline"])


class GeminiPool:
    """Async pool for one model + one task type.

    Use either .batch(prompts) -> list[str|Exception]
    or .one(prompt).
    """
    def __init__(self, model: str, concurrency: int = 16,
                 temp: float = 0.8, max_out: int = 1024,
                 thinking: str = "minimal",
                 response_schema: Type[BaseModel] | None = None,
                 json_mode: bool = False,
                 top_p: float | None = None):
        """response_schema enforces strict JSON schema (may fail on dict[str,X]
        because Gemini rejects additionalProperties). Set json_mode=True to get
        JSON mime without a strict schema; caller then validates with pydantic."""
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.response_schema = response_schema
        cfg_kwargs: dict[str, Any] = dict(
            temperature=temp, max_output_tokens=max_out,
            response_mime_type=(
                "application/json" if (response_schema or json_mode) else "text/plain"),
        )
        if response_schema is not None:
            try:
                # probe: build config with schema; if it trips, fall back to json_mode
                test_cfg = gtypes.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=1,
                    response_mime_type="application/json",
                    response_schema=response_schema)
                cfg_kwargs["response_schema"] = response_schema
            except Exception:
                self.response_schema = None  # schema invalid -> free JSON
        tc = _thinking_cfg(thinking)
        if tc is not None:
            cfg_kwargs["thinking_config"] = tc
        if top_p is not None:
            cfg_kwargs["top_p"] = top_p
        try:
            self.cfg = gtypes.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            # Retry without the schema if Gemini rejects it
            cfg_kwargs.pop("response_schema", None)
            self.response_schema = None
            self.cfg = gtypes.GenerateContentConfig(**cfg_kwargs)
        self._c = _client()

    async def _call(self, prompt: str) -> str:
        async with self.sem:
            last_err = None
            for attempt in range(1, 7):
                try:
                    resp = await self._c.aio.models.generate_content(
                        model=self.model, contents=prompt, config=self.cfg)
                    return resp.text or ""
                except ValueError as e:
                    msg = str(e)
                    if "additionalProperties" in msg or "not supported" in msg:
                        # Gemini rejected the schema -> strip it and retry in json_mode
                        _LOG.warning(f"gemini schema rejected ({msg[:120]}); retrying without schema")
                        cfg_dict = {
                            "temperature": self.cfg.temperature,
                            "max_output_tokens": self.cfg.max_output_tokens,
                            "response_mime_type": "application/json",
                        }
                        if self.cfg.thinking_config is not None:
                            cfg_dict["thinking_config"] = self.cfg.thinking_config
                        if self.cfg.top_p is not None:
                            cfg_dict["top_p"] = self.cfg.top_p
                        self.cfg = gtypes.GenerateContentConfig(**cfg_dict)
                        self.response_schema = None
                        continue
                    last_err = e
                    if not _is_transient(e) and attempt > 2:
                        raise
                    sleep_s = min(60.0, 2.0 * (2 ** (attempt - 1))) * (0.75 + 0.5 * random.random())
                    await asyncio.sleep(sleep_s)
                except Exception as e:
                    last_err = e
                    if not _is_transient(e) and attempt > 2:
                        raise
                    sleep_s = min(60.0, 2.0 * (2 ** (attempt - 1))) * (0.75 + 0.5 * random.random())
                    await asyncio.sleep(sleep_s)
            raise last_err or RuntimeError("unreachable")

    async def one(self, prompt: str) -> str:
        return await self._call(prompt)

    async def batch(self, prompts: list[str]) -> list[Any]:
        tasks = [asyncio.create_task(self._call(p)) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def run_batched(pool: GeminiPool, prompts: list[str], slice_size: int = 128,
                      phase: str = "gemini") -> list[Any]:
    """Slice a large prompt list into gather batches of `slice_size`."""
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), slice_size):
        chunk = prompts[i:i + slice_size]
        results = await pool.batch(chunk)
        out.extend(results)
        el = time.time() - t0
        rate = len(out) / el if el > 0 else 0.0
        print(f"[PROGRESS] phase={phase} {len(out)}/{len(prompts)} elapsed={el:.1f}s rate={rate:.2f}/s",
              flush=True)
    return out


def parse_json_lenient(txt: str) -> dict | None:
    """Try hard to get a JSON dict out of Gemini output (stripping ``` fences, prefix/suffix)."""
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        # strip fence
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(t)
    except Exception:
        pass
    # Grab the first {...} block
    i = t.find("{")
    j = t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


# ------ Pydantic schemas ------

class SupportingDoc(BaseModel):
    doc_id: str
    grade: int


class QGOutput(BaseModel):
    query: str
    short_answer: str
    long_answer: str
    answer_points: list[str]
    supporting_docs: list[SupportingDoc]
    reasoning_type: str
    difficulty: str
    reasoning_chain: str
    bridge_entity: str
    quoted_spans: dict[str, list[str]]


class RerankerOutput(BaseModel):
    best_index: int
    justification: str
    candidate_issues: list[dict] = Field(default_factory=list)


class SolvFact(BaseModel):
    fact: str
    present: bool
    evidence: str | None = None


class SolvOutput(BaseModel):
    required_facts: list[SolvFact]
    all_facts_present: bool
    solvable_from_this_document_alone: bool
    confidence: str
    explanation: str


class ChunkMappingOutput(BaseModel):
    chunk_id: str | None
    span_start_char_in_chunk: int | None
    span_end_char_in_chunk: int | None
    span_text: str | None
    confidence: str


def validate_obj(d: dict, schema: Type[BaseModel]) -> BaseModel | None:
    try:
        return schema(**d)
    except ValidationError:
        return None


def probe_model(model_id: str, timeout_s: float = 30.0) -> bool:
    """Sync 1-token probe. Returns True if model responds."""
    if not _GENAI_OK:
        return False
    try:
        c = _client()
        cfg = gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=1)
        resp = c.models.generate_content(model=model_id, contents="ok", config=cfg)
        return resp is not None
    except Exception as e:
        _LOG.info(f"probe {model_id} failed: {type(e).__name__}: {str(e)[:160]}")
        return False


def resolve_candidates(candidates: list[str]) -> str | None:
    for c in candidates:
        ok = probe_model(c)
        print(f"[PROBE] {c} -> {'ok' if ok else 'fail'}", flush=True)
        if ok:
            return c
    return None
