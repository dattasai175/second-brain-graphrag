"""Settings: YAML on disk -> validated pydantic models."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    extraction_model: str | None = None
    temperature: float = 0.0
    num_ctx: int = 8192
    timeout_s: float = 300.0


class EmbeddingSettings(BaseModel):
    backend: Literal["ollama", "sentence_transformers", "hash"] = "ollama"
    model: str = "qwen3-embedding:0.6b"
    base_url: str | None = None
    batch_size: int = 32
    query_prefix: str = ""
    doc_prefix: str = ""
    hash_dim: int = 512


class RerankSettings(BaseModel):
    backend: Literal["auto", "none", "cross_encoder"] = "auto"
    model: str = "BAAI/bge-reranker-v2-m3"


class ChunkingSettings(BaseModel):
    target_tokens: int = 600
    overlap_tokens: int = 60
    min_tokens: int = 80
    max_tokens: int = 900
    contextual_prefix: Literal["heading", "llm", "none"] = "heading"


class ExtractionSettings(BaseModel):
    enabled: bool = True
    gleaning: bool = True
    workers: int = 2
    adjudicate: bool = True
    candidate_threshold: float = 0.86
    auto_merge_threshold: float = 0.97
    max_chunk_chars: int = 6000


class GraphSettings(BaseModel):
    ppr_damping: float = 0.6
    chunk_seed_weight: float = 0.3
    mention_weight: float = 1.0
    doc_node_weight: float = 0.3
    hub_penalty: bool = True
    link_min_sim: float = 0.55
    max_seeds: int = 8


class RetrievalSettings(BaseModel):
    dense_k: int = 50
    bm25_k: int = 50
    graph_k: int = 50
    rerank_candidates: int = 40
    final_k: int = 8
    rrf_k: int = 60
    channel_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "lookup": {"dense": 1.0, "bm25": 1.0, "graph": 0.6},
            "multi_hop": {"dense": 1.0, "bm25": 0.7, "graph": 1.5},
            "global": {"dense": 1.0, "bm25": 0.7, "graph": 1.0},
        }
    )

    def weights_for(self, query_type: str) -> dict[str, float]:
        return self.channel_weights.get(query_type) or self.channel_weights.get(
            "lookup", {"dense": 1.0, "bm25": 1.0, "graph": 1.0}
        )


class SensitivityRule(BaseModel):
    glob: str
    label: str


class SensitivitySettings(BaseModel):
    default: str = "private"
    rules: list[SensitivityRule] = Field(default_factory=list)

    def label_for(self, rel_path: str) -> str:
        for rule in self.rules:
            if fnmatch.fnmatch(rel_path, rule.glob):
                return rule.label
        return self.default


class IngestSettings(BaseModel):
    extensions: list[str] = Field(default_factory=lambda: [".md", ".markdown", ".txt", ".pdf"])
    ignore: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".obsidian",
            ".brain",
            ".venv",
            "node_modules",
            "__pycache__",
        ]
    )


class Settings(BaseModel):
    data_dir: str = ".brain"
    ontology: str = "config/ontology.yaml"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    sensitivity: SensitivitySettings = Field(default_factory=SensitivitySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)

    # filled in by load_settings(); not part of the YAML
    project_root: Path = Path(".")

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_path / "brain.db"

    @property
    def ontology_path(self) -> Path:
        return self._resolve(self.ontology)

    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (self.project_root / path)


def _find_default_config() -> Path | None:
    env = os.environ.get("BRAIN_CONFIG")
    if env:
        return Path(env).expanduser()
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "config" / "settings.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_path = Path(path).expanduser() if path else _find_default_config()
    data: dict = {}
    root = Path.cwd()
    if cfg_path is not None:
        cfg_path = cfg_path.resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        data = yaml.safe_load(cfg_path.read_text()) or {}
        # config/settings.yaml -> project root is the folder holding config/
        root = cfg_path.parent.parent if cfg_path.parent.name == "config" else cfg_path.parent
    settings = Settings.model_validate(data)
    settings.project_root = root
    return settings
