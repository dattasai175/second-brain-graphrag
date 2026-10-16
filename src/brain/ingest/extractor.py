"""LLM entity/relation extraction against a closed ontology, with grounding checks and caching."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..config import ExtractionSettings
from ..llm.base import LLMClient, LLMError
from ..prompts import EXTRACTOR_VERSION
from ..prompts import load as load_prompt
from ..store.db import Database
from ..text import STOPWORDS, ngrams, normalize_name, tokenize

log = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    name: str
    type: str = "Concept"
    description: str = ""


class ExtractedRelation(BaseModel):
    source: str
    predicate: str = "related_to"
    target: str
    description: str = ""
    evidence: str = ""


class Extraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


@dataclass
class Ontology:
    entity_types: list[str]
    predicates: list[str]

    @classmethod
    def load(cls, path: Path) -> Ontology:
        if path.is_file():
            data = yaml.safe_load(path.read_text()) or {}
            return cls(
                entity_types=list(data.get("entity_types") or []) or DEFAULT_TYPES,
                predicates=list(data.get("predicates") or []) or DEFAULT_PREDICATES,
            )
        return cls(DEFAULT_TYPES, DEFAULT_PREDICATES)

    def canonical_type(self, raw: str) -> str:
        wanted = raw.strip().casefold()
        for t in self.entity_types:
            if t.casefold() == wanted:
                return t
        return "Concept" if "Concept" in self.entity_types else self.entity_types[0]

    def canonical_predicate(self, raw: str) -> tuple[str, bool]:
        """Return (predicate, was_known)."""
        p = re.sub(r"[^a-z0-9]+", "_", raw.strip().casefold()).strip("_")
        return (p, True) if p in self.predicates else ("related_to", False)


DEFAULT_TYPES = ["Person", "Organization", "Project", "Concept", "Tool", "Place", "Event", "Topic"]
DEFAULT_PREDICATES = ["works_at", "works_on", "authored", "uses", "part_of", "related_to"]


# --------------------------------------------------------------------- grounding
def _padded(text: str) -> str:
    return f" {normalize_name(text)} "


def name_in_text(name: str, text_norm_padded: str, text_tokens: set[str]) -> bool:
    n = normalize_name(name)
    if not n or n in STOPWORDS or n.isdigit():
        return False
    if f" {n} " in text_norm_padded:
        return True
    parts = n.split()
    return len(parts) >= 2 and all(p in text_tokens for p in parts)  # "Raman, Priya"


def quote_supported(quote: str, text_norm_padded: str) -> bool:
    q = normalize_name(quote)
    if not q:
        return False
    if f" {q} " in text_norm_padded:
        return True
    toks = q.split()
    if len(toks) < 4:
        return False
    grams = ngrams(toks, 3)
    have = ngrams(text_norm_padded.split(), 3)
    return len(grams & have) / max(1, len(grams)) >= 0.7


def mention_count(name: str, text: str) -> int:
    n = normalize_name(name)
    if not n:
        return 1
    return max(1, _padded(text).count(f" {n} "))


def ground(raw: Extraction, text: str, ontology: Ontology) -> Extraction:
    """Drop entities not present in the passage and relations without a quoted basis."""
    padded = _padded(text)
    toks = set(tokenize(text))
    entities: dict[str, ExtractedEntity] = {}
    for e in raw.entities:
        if not name_in_text(e.name, padded, toks):
            continue
        key = normalize_name(e.name)
        cleaned = ExtractedEntity(
            name=e.name.strip(),
            type=ontology.canonical_type(e.type),
            description=e.description.strip()[:240],
        )
        if key not in entities or len(cleaned.description) > len(entities[key].description):
            entities[key] = cleaned

    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for r in raw.relations:
        s_key, t_key = normalize_name(r.source), normalize_name(r.target)
        if s_key == t_key:
            continue
        for key, raw_name in ((s_key, r.source), (t_key, r.target)):
            if key not in entities and name_in_text(raw_name, padded, toks):
                entities[key] = ExtractedEntity(
                    name=raw_name.strip(), type=ontology.canonical_type("Concept")
                )
        if s_key not in entities or t_key not in entities:
            continue
        if not quote_supported(r.evidence, padded):
            continue
        pred, known = ontology.canonical_predicate(r.predicate)
        desc = r.description.strip()
        if not known:
            desc = f"{r.predicate.strip()}: {desc}".strip(": ")
        triple = (s_key, pred, t_key)
        if triple in seen:
            continue
        seen.add(triple)
        relations.append(
            ExtractedRelation(
                source=entities[s_key].name,
                predicate=pred,
                target=entities[t_key].name,
                description=desc[:240],
                evidence=r.evidence.strip()[:300],
            )
        )
    return Extraction(entities=list(entities.values()), relations=relations)


def _merge(a: Extraction, b: Extraction) -> Extraction:
    return Extraction(entities=a.entities + b.entities, relations=a.relations + b.relations)


# --------------------------------------------------------------------- extractor
class Extractor:
    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        cfg: ExtractionSettings,
        ontology: Ontology,
        model: str | None = None,
    ):
        self.llm = llm
        self.db = db
        self.cfg = cfg
        self.ontology = ontology
        self.model = model
        self._system = (
            load_prompt("extractor")
            .replace("{types}", ", ".join(ontology.entity_types))
            .replace("{predicates}", ", ".join(ontology.predicates))
        )
        self._gleaning = load_prompt("gleaning")
        self.failures = 0

    def _key(self, text: str, title: str, section: str) -> str:
        h = hashlib.sha256()
        for part in (
            EXTRACTOR_VERSION,
            self.model or self.llm.model,
            str(self.cfg.gleaning),
            ",".join(self.ontology.entity_types),
            ",".join(self.ontology.predicates),
            title,
            section,
            text,
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _call(self, text: str, title: str, section: str) -> Extraction:
        text = text[: self.cfg.max_chunk_chars]
        user = f'Document: {title}\nSection: {section or "-"}\n\nPassage:\n"""\n{text}\n"""'
        first = self.llm.structured(self._system, user, Extraction, model=self.model)
        if not self.cfg.gleaning:
            return first
        follow_up = f"{user}\n\nPrevious extraction:\n{first.model_dump_json()}\n\n{self._gleaning}"
        try:
            more = self.llm.structured(self._system, follow_up, Extraction, model=self.model)
        except LLMError:
            return first  # gleaning is a bonus; never lose the first pass
        return _merge(first, more)

    def extract_many(self, items: list[tuple[str, str, str]]) -> list[Extraction | None]:
        """items: (text, title, section). Returns grounded extractions; None where the LLM failed."""
        results: list[Extraction | None] = [None] * len(items)
        pending: list[int] = []
        keys = [self._key(*it) for it in items]
        for i, key in enumerate(keys):
            row = self.db.conn.execute(
                "SELECT payload FROM extraction_cache WHERE key = ?", (key,)
            ).fetchone()
            if row:
                results[i] = Extraction.model_validate_json(row[0])
            else:
                pending.append(i)

        def work(i: int) -> tuple[int, Extraction | None]:
            text, title, section = items[i]
            try:
                return i, ground(self._call(text, title, section), text, self.ontology)
            except LLMError as e:
                log.warning("extraction failed for chunk (%s): %s", title, e)
                return i, None

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as pool:
                for i, ext in pool.map(work, pending):
                    results[i] = ext
                    if ext is None:
                        self.failures += 1
                    else:
                        self.db.conn.execute(
                            "INSERT OR REPLACE INTO extraction_cache(key, payload) VALUES (?, ?)",
                            (keys[i], ext.model_dump_json()),
                        )
            self.db.conn.commit()
        return results
