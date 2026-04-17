# multihop_rag

A pipeline for constructing a high-quality multi-hop Retrieval-Augmented Generation (RAG) dataset.

## Overview

This project builds a RAG training/evaluation dataset where every example requires combining information from at least two distinct documents. Each example contains a natural-sounding question, a short answer, a long grounded answer with citations, supporting documents with graded relevance labels (0-3), adversarial distractor documents, atomic answer points, and an explicit reasoning chain.

### Primary corpus

- `ParthMandaliya/hotpotqa-wiki` -- 5.49 M Wikipedia articles with curated cross-document link annotations (October 2017 dump)

### Design principles

- **Link-based cross-document graph.** Wikipedia's human-curated `links` field is used directly instead of running NER. Documents are connected through shared link targets with IDF-weighted edges; multi-hop paths through this graph drive question generation.
- **Pilot-first scaling.** The pipeline runs first at ~1000-question scale to validate every stage, then escalates to ~5000 questions if quality gates pass.
- **Verbatim-span grounding.** Long answers must quote exact spans from source documents, enabling reliable evidence mapping.
- **LLM-as-judge reranking.** Best-of-N candidate questions are selected by a judge model rather than heuristic scoring.
- **Single-document solvability gate.** Every candidate question is tested against each supporting document alone to guarantee genuine multi-hop reasoning.

## Repository layout

- `pipeline_description` - canonical design document
- `configs/` - model IDs, pilot parameters, scale-up parameters
- `prompts/` - LLM prompt templates (question generation, reranker, solvability, chunk mapping, entity/relation extraction)
- `environment.yml` + `setup_conda.sh` - conda env build for the project
- `src/mhrag/` - pipeline modules (corpus, chunking, graph, embedding, filters, baselines, etc.)
- `scripts/` - entry point for each pipeline phase
- `data/` - intermediate artifacts (corpus, chunks, graph, embeddings) [gitignored]
- `outputs/` - generated dataset + reports [gitignored]

## Output format

Final dataset is written as JSONL where each line is one multi-hop example with the full schema described in `pipeline_description` Section 12. Key fields include `query`, `short_answer`, `long_answer`, `answer_points`, `supporting_docs` with relevance grades, `supporting_chunks` with character offsets, `reasoning_type`, `reasoning_chain`, `bridge_entity`, `difficulty`, and generation metadata.

## Environment

Python 3.10 conda environment with PyTorch (CUDA-enabled), Hugging Face `datasets` / `transformers` / `sentence-transformers`, `google-genai` for Gemini API, `networkx` for graph construction, `blingfire` + `pysbd` for sentence segmentation, `rank-bm25` for lexical retrieval, and `Qwen2.5-3B-Instruct` as the reader baseline. See `environment.yml` for the full spec and `setup_conda.sh` to build the environment.
