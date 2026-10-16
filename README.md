# second-brain

Adaptive Personal Knowledge Graph RAG: a local-first "second brain" that indexes your documents
into a vector index **and** a knowledge graph, retrieves with both, and answers with citations.

This repository is **Phase 0 + Phase 1** of the plan: hybrid graph + vector retrieval, single-pass
cited answers, and an evaluation harness. The agentic multi-hop loop + critique (Phase 2), the
feedback/adaptation loop (Phase 3) and UI polish (Phase 4) are not built yet; the schema and
interfaces already leave room for them (`queries`, `feedback`, `adapt_params` tables, per-edge
multiplier hook in the graph view, evidence `graph_path`).

```
files ─► scanner ─► parsers ─► heading-aware chunker ─► embed ─┬─► chunks + vectors + FTS5 (BM25)
        (sha256 diff) (md / pdf)                               │
                                  └─► LLM extractor ─► entity resolver ─► entities + relations
question ─► query analysis ─► dense ┐
                              BM25  ├─► weighted RRF ─► rerank ─► evidence (with graph paths) ─► cited answer
                              graph ┘  (Personalized PageRank over entities + chunks)
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) running locally.

```bash
cd second-brain
uv sync                                  # creates .venv and installs everything
ollama pull llama3.1:8b                  # chat / extraction model (set llm.model to taste)
ollama pull qwen3-embedding:0.6b         # embedding model (config/settings.yaml)
uv run brain doctor                      # checks Ollama, models, vector backend, optional parts
```

Optional extras: `uv sync --extra pdf` (Docling layout-aware PDF parsing) and
`uv sync --extra ml` (sentence-transformers: local embeddings and the cross-encoder reranker,
`BAAI/bge-reranker-v2-m3`). Without them PDFs use `pypdf` and reranking is a pass-through.

No embedding model yet? `config/settings.smoke.yaml` uses a deterministic hashing embedder so you
can exercise the whole pipeline with nothing but the chat model (retrieval quality is lexical-ish;
it verifies plumbing, not quality).

## Use

```bash
uv run brain ingest ./sample_corpus              # index a folder (incremental; re-runs are no-ops)
uv run brain ask "Which company did the author of RFC-12 work at before joining Northwind Labs?"
uv run brain ask "..." --trace                   # channel scores, ranks, linked entities, graph paths
uv run brain search "..." --preset dense         # retrieval only; presets form the ablation ladder
uv run brain stats
uv run brain entities --limit 20
uv run brain --config config/settings.smoke.yaml ingest sample_corpus   # download-free smoke test
```

`--no-extract` skips LLM entity extraction (vector + BM25 + explicit wikilink/tag edges only);
a later plain `ingest` fills in the graph. `--reset` deletes the database first (required after
changing the embedding model).

Retrieval presets: `dense`, `bm25`, `dense_bm25`, `dense_bm25_rerank`, `hybrid_graph_norerank`,
`hybrid_graph` (default: dense + BM25 + graph PPR, weighted RRF, rerank).

## Evaluate

```bash
uv run python eval/run_eval.py --golden eval/datasets/golden.sample.jsonl     # sample corpus
uv run python eval/run_eval.py                                                # your golden.jsonl
uv run python eval/run_eval.py --systems dense,hybrid_graph --no-router
```

Reports Recall@{1,5,10}, Hit@10, **AllHops@10** (every gold span retrieved: the multi-hop metric),
MRR, nDCG@10 and p50 latency per system, overall and per question type (the JSON report in
`eval/reports/` also has Recall/Hit/AllHops/nDCG at 1, 3, 5 and 10 and per-question span ranks). Gold evidence is anchored to *document + quote*, not chunk ids. The format is
documented at the top of `eval/datasets/golden.jsonl` (an empty template); start with ~30 questions.

## Layout

```
config/            settings.yaml, settings.smoke.yaml, ontology.yaml (closed entity/predicate vocab)
src/brain/
  ingest/          scanner, parsers/{markdown,pdf}, chunker, extractor, resolver, pipeline
  store/           schema.sql, db.py, vectors.py (sqlite-vec | numpy), graph.py (igraph PPR)
  llm/             base protocol, ollama client, embedders (ollama | sentence_transformers | hash)
  retrieval/       query_analysis, dense, lexical (FTS5), graph_ppr (entity linking + PPR),
                   fusion (weighted RRF), rerank, hybrid (Retriever + presets)
  agent/           generator.py: single-pass cited answers (Phase 2 adds planner/critic/verifier)
  prompts/         router, extractor, gleaning, adjudicate, generator, context (.md, editable)
  evalkit/         golden-set loader + metrics
eval/              run_eval.py, datasets/, reports/
sample_corpus/     12 small notes with wikilinks, tags, near-duplicate names and distractors
tests/             pytest suite (fake LLM, hash embedder; no network)
```

Design choices worth knowing:

* **One SQLite file** holds documents, chunks, FTS5, entities, relations, traces and (later)
  feedback. Vectors use **sqlite-vec** when this Python can load SQLite extensions and fall back to
  an in-memory numpy index otherwise (`brain doctor` tells you which). A database remembers which
  backend built it.
* **Graph retrieval is Personalized PageRank** over an entity + chunk graph: restart mass comes
  from entities linked in the question and the top dense chunks (HippoRAG 2 style), with a
  specificity/hub penalty. Every evidence item records the edges on its path (`graph_path`), which
  is both the explanation and (Phase 3) the unit of feedback credit.
* **Wikilinks and tags are free edges**: `[[Note]]` links and `#tags` become high-precision
  relations without an LLM. Document pseudo-nodes tie chunks of one file together.
* **Extraction is grounded**: entities must appear in the passage, relations need an exact-quote
  evidence span, unknown types/predicates fall back to `Concept` / `related_to`. Results are cached
  by (chunk text, model, prompt version).
* **Corrections survive re-ingest**: `overrides` (`must_link`, `cannot_link`, `drop_relation`) are
  honoured by the resolver and graph writer and never garbage-collected.
* Sensitivity labels (`sensitivity.rules`) are stored per document now and will gate cloud routing
  in Phase 2. Nothing in Phase 1 leaves your machine.

## Status: what has been exercised

* Test suite (`uv run pytest`): chunker, parsers, fusion, vector index (numpy **and** sqlite-vec),
  graph/PPR, metrics and an end-to-end ingest -> retrieve -> edit -> delete cycle with a fake LLM.
* Real run against Ollama (`llama3.1:8b`, hash embedder via `settings.smoke.yaml`): ingest of
  `sample_corpus` (12 notes -> 12 chunks, ~30 entities, ~55 relations), a correct cited multi-hop
  answer, and the eval harness over `golden.sample.jsonl`.
* PDFs through the `pypdf` fallback. **Docling, the sentence-transformers reranker/embedder and the
  default `qwen3-embedding` model have not been run here** (large downloads); their code paths are
  small and isolated, but treat them as untested.
* The 12-note sample corpus is a plumbing check: with so few chunks every system retrieves nearly
  everything, so its ablation numbers say nothing about retrieval quality. Build your own golden set.

## Notes and caveats

* **Why the package lives in `src/` as a separate, non-editable project.** If this folder is in
  iCloud Drive (`~/Documents`), macOS marks everything inside dot-directories such as `.venv` as
  *hidden* a few minutes after `uv sync`, and Python 3.12+ ignores hidden `.pth` files, which is how
  editable installs work. The result was `ModuleNotFoundError: No module named 'brain'`. A regular
  install has no `.pth`; `cache-keys` in `src/pyproject.toml` make uv reinstall on source changes,
  so day-to-day it behaves like an editable install.
* The default embedding model is not bundled: `ollama pull qwen3-embedding:0.6b`. If you change
  `embedding.model`, set matching `query_prefix` / `doc_prefix` (examples in `settings.yaml`).
* `llm.model` defaults to the model that was already installed (`llama3.1:8b`); entity extraction
  quality scales strongly with model size, so use the best 14–30B instruct model you can run.
* Not yet implemented: code parsing (tree-sitter), contextual prefix via LLM is implemented
  (`chunking.contextual_prefix: llm`) but off by default, community summaries / global search,
  cloud model routing, the agent loop, feedback adaptation.

## Tests

```bash
uv run pytest        # chunker, parsers, fusion, vectors, graph/PPR, metrics, end-to-end ingest
uv run ruff check src tests eval
```
