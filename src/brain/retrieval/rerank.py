"""Cross-encoder reranking, degrading gracefully to a pass-through when unavailable."""

from __future__ import annotations

import logging
from typing import Protocol

from ..config import RerankSettings

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str
    active: bool  # False => scores are not meaningful; callers keep the fused order

    def score(self, query: str, docs: list[str]) -> list[float]: ...


class PassthroughReranker:
    active = False

    def __init__(self, reason: str = "disabled"):
        self.name = f"passthrough ({reason})"

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [0.0] * len(docs)


class CrossEncoderReranker:  # pragma: no cover - needs the optional `ml` extra + model weights
    active = True

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model}"
        self._model = CrossEncoder(model)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self._model.predict([(query, d) for d in docs])]


def build_reranker(cfg: RerankSettings) -> Reranker:
    if cfg.backend == "none":
        return PassthroughReranker("backend: none")
    try:
        return CrossEncoderReranker(cfg.model)
    except ImportError:
        if cfg.backend == "cross_encoder":
            raise RuntimeError(
                "rerank.backend 'cross_encoder' needs sentence-transformers: uv sync --extra ml"
            ) from None
        return PassthroughReranker("sentence-transformers not installed; uv sync --extra ml")
    except Exception as e:  # model download/offline problems
        if cfg.backend == "cross_encoder":
            raise
        log.warning("reranker unavailable (%s); using fused order", e)
        return PassthroughReranker(f"load failed: {type(e).__name__}")
