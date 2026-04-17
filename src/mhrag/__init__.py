"""Multi-hop RAG dataset construction pipeline.

Module layout mirrors the phases:
  env          -- config loading, dotenv, log helpers
  gemini       -- async Gemini client + retry + schemas
  corpus       -- streaming filter, schema conversion
  chunking     -- blingfire/pysbd + chunk schema
  graph        -- inverted index, IDF edges, path enumeration
  embed        -- bge-m3 batched encoder
  similarity   -- torch matmul top-k
  filters      -- filter battery + dedup
  reranker     -- LLM-as-judge driver
  solvability  -- solvability driver
  annotate     -- supporting chunks + distractors
  baselines    -- BM25 + dense retrieval + Qwen reader
  stats        -- distributions + reference comparison
  progress     -- run_progress.json helpers + [PROGRESS] emitter
"""
