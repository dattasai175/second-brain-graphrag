"""Data contracts shared by retrieval, generation, the CLI and the eval harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    text: str
    type: str = "lookup"  # lookup | multi_hop | global | no_retrieval
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def keyword_query(self) -> str:
        return " ".join(self.keywords) if self.keywords else self.text


@dataclass
class Evidence:
    id: str  # "E1" within one trace
    chunk_id: int
    doc_id: int
    doc_path: str
    title: str
    section: str
    page: int | None
    text: str
    score: float = 0.0
    channels: dict[str, float] = field(default_factory=dict)  # raw per-channel scores
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank per channel
    graph_path: list[str] = field(default_factory=list)  # edge descriptors on the PPR path

    @property
    def source(self) -> str:
        loc = self.section or ""
        if self.page is not None:
            loc = f"{loc} p.{self.page}".strip()
        return f"{self.doc_path}" + (f" › {loc}" if loc else "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    plan: QueryPlan
    evidence: list[Evidence]
    preset: str
    timings: dict[str, float] = field(default_factory=dict)
    linked_entities: dict[str, float] = field(default_factory=dict)  # canonical name -> seed weight
    notes: list[str] = field(default_factory=list)
