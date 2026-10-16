"""Embedding backends behind one interface.

* ``ollama``               : local models via /api/embed (default; e.g. qwen3-embedding, bge-m3)
* ``sentence_transformers``: in-process (needs the ``ml`` extra)
* ``hash``                 : deterministic feature hashing; no model, for offline smoke tests
"""

from __future__ import annotations

import zlib
from typing import Protocol

import httpx
import numpy as np

from ..config import EmbeddingSettings, LLMSettings
from .base import LLMError, LLMUnavailable


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class OllamaEmbedder:
    def __init__(self, cfg: EmbeddingSettings, base_url: str, timeout_s: float = 300.0):
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"
        self.base_url = (cfg.base_url or base_url).rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed(["dimension probe"]).shape[1])
        return self._dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            resp = self._http.post(
                f"{self.base_url}/api/embed", json={"model": self.cfg.model, "input": texts}
            )
        except httpx.ConnectError as e:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url} for embeddings. Start it with `ollama serve`."
            ) from e
        if resp.status_code == 404:
            raise LLMUnavailable(
                f"Embedding model '{self.cfg.model}' is not installed in Ollama. Run: "
                f"ollama pull {self.cfg.model}\n(or set embedding.backend to 'hash' in the config "
                "for a download-free smoke test, see config/settings.smoke.yaml)"
            )
        if resp.status_code >= 400:
            raise LLMError(f"Ollama /api/embed failed ({resp.status_code}): {resp.text[:300]}")
        vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise LLMError(
                f"Unexpected embedding response shape {vecs.shape} for {len(texts)} inputs"
            )
        return vecs

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [self.cfg.doc_prefix + t for t in texts[i : i + self.cfg.batch_size]]
            out.append(self._embed(batch))
        return _l2(np.vstack(out))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._embed([self.cfg.query_prefix + text]))[0]


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingSettings):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise LLMUnavailable(
                "embedding.backend 'sentence_transformers' needs the extra: uv sync --extra ml"
            ) from e
        self.cfg = cfg
        self.name = f"st:{cfg.model}"
        self._model = SentenceTransformer(cfg.model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            [self.cfg.doc_prefix + t for t in texts],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
        )
        return _l2(vecs)

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover
        vec = self._model.encode([self.cfg.query_prefix + text], normalize_embeddings=True)
        return _l2(vec)[0]


class HashEmbedder:
    """Signed feature hashing over word unigrams/bigrams and character 3-grams.

    Deterministic, dependency-free, and good enough to verify plumbing. Not a semantic model.
    """

    def __init__(self, cfg: EmbeddingSettings):
        self.name = f"hash:{cfg.hash_dim}"
        self.dim = cfg.hash_dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = text.casefold()
        words = [w for w in "".join(c if c.isalnum() else " " for c in t).split() if w]
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"^{w}$"
            feats += [(f"c:{padded[i : i + 3]}", 0.3) for i in range(max(1, len(padded) - 2))]
        for feat, weight in feats:
            h = zlib.crc32(feat.encode("utf-8"))
            v[h % self.dim] += weight if (h >> 16) & 1 else -weight
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.vstack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


def build_embedder(cfg: EmbeddingSettings, llm_cfg: LLMSettings) -> Embedder:
    if cfg.backend == "ollama":
        return OllamaEmbedder(cfg, llm_cfg.base_url, llm_cfg.timeout_s)
    if cfg.backend == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg)
    if cfg.backend == "hash":
        return HashEmbedder(cfg)
    raise ValueError(f"Unknown embedding backend: {cfg.backend}")
