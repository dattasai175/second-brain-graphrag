"""Wires settings, database, embedder, LLM and reranker together (shared by CLI and eval)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings, load_settings
from .llm import Embedder, OllamaClient, build_embedder
from .retrieval.rerank import Reranker, build_reranker
from .store.db import Database


@dataclass
class Runtime:
    settings: Settings
    db: Database
    embedder: Embedder
    llm: OllamaClient
    reranker: Reranker

    def close(self) -> None:
        self.db.close()


def open_runtime(
    config: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    with_reranker: bool = True,
) -> Runtime:
    settings = load_settings(config)
    db = Database(db_path or settings.db_path)
    embedder = build_embedder(settings.embedding, settings.llm)
    llm = OllamaClient(settings.llm)
    reranker = (
        build_reranker(settings.rerank)
        if with_reranker
        else build_reranker(settings.rerank.model_copy(update={"backend": "none"}))
    )
    return Runtime(settings, db, embedder, llm, reranker)
