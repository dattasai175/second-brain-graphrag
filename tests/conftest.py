from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from brain.config import EmbeddingSettings, Settings
from brain.ingest.extractor import ExtractedEntity, ExtractedRelation, Extraction
from brain.ingest.resolver import Verdict
from brain.llm.embeddings import HashEmbedder
from brain.retrieval.query_analysis import RouterOut
from brain.store.db import Database

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_corpus"

KNOWN = [
    "Priya Raman",
    "Marcus Lee",
    "Northwind Labs",
    "Helios Systems",
    "Helios Labs",
    "Project Atlas",
    "Apache Kafka",
    "RabbitMQ",
    "Cobalt Group",
]
RELATIONS = [  # (source, predicate, target, evidence phrase)
    ("Priya Raman", "reports_to", "Marcus Lee", "reports to Marcus Lee"),
    ("Priya Raman", "works_at", "Helios Systems", "Priya worked at Helios Systems"),
    ("Helios Systems", "related_to", "Cobalt Group", "acquired by Cobalt Group in 2024"),
    ("Project Atlas", "uses", "Apache Kafka", "Atlas is built on Apache Kafka"),
]


class FakeLLM:
    """Deterministic stand-in for Ollama: rule-based extraction, canned chat."""

    model = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, system: str, user: str, *, model=None, max_tokens=None) -> str:
        self.calls.append("chat")
        return "Per the notes [E1]."

    def structured(self, system: str, user: str, schema: type[BaseModel], *, model=None):
        self.calls.append(schema.__name__)
        if schema is Verdict:
            return Verdict(verdict="unsure")
        if schema is RouterOut:
            return RouterOut()
        passage = user.split('"""')[1] if '"""' in user else user
        ents = [
            ExtractedEntity(
                name=n, type="Organization" if "Labs" in n or "Systems" in n else "Person"
            )
            for n in KNOWN
            if n.lower() in passage.lower()
        ]
        rels = []
        for s, p, t, ev in RELATIONS:
            if ev.lower() in passage.lower().replace("[[", "").replace("]]", ""):
                rels.append(ExtractedRelation(source=s, predicate=p, target=t, evidence=ev))
        return Extraction(entities=ents, relations=rels)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(EmbeddingSettings(backend="hash", hash_dim=256))


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(tmp_path / "brain.db")
    yield d
    d.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.project_root = ROOT
    s.data_dir = str(tmp_path / "data")
    s.embedding.backend = "hash"
    s.embedding.hash_dim = 256
    s.extraction.gleaning = False
    s.extraction.workers = 1
    return s


@pytest.fixture
def sample_corpus() -> Path:
    return SAMPLE


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
