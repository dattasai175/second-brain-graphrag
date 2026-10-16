"""Entity resolution: exact alias match -> user overrides -> embedding blocking -> LLM adjudication.

Conservative by design: a wrong merge corrupts the graph silently, a missed merge only costs
recall (and the user can fix it with a ``must_link`` override).
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from ..config import ExtractionSettings
from ..llm.base import LLMClient, LLMError
from ..llm.embeddings import Embedder
from ..prompts import load as load_prompt
from ..store.db import Database
from ..store.vectors import VectorIndex
from ..text import normalize_name
from .extractor import ExtractedEntity

log = logging.getLogger(__name__)


class Verdict(BaseModel):
    verdict: Literal["same", "different", "unsure"] = "unsure"
    reason: str = ""


def entity_key(norm_name: str, type_: str) -> str:
    return f"{norm_name}|{type_}"


class EntityResolver:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        index: VectorIndex,
        cfg: ExtractionSettings,
        llm: LLMClient | None = None,
        model: str | None = None,
    ):
        self.db = db
        self.embedder = embedder
        self.index = index
        self.cfg = cfg
        self.llm = llm
        self.model = model
        self._system = load_prompt("adjudicate")
        self.merged = 0
        self.created = 0
        self._load_overrides()

    # ---------------------------------------------------------------- overrides
    def _load_overrides(self) -> None:
        self._must: dict[str, set[str]] = {}
        self._cannot: set[frozenset[str]] = set()
        for r in self.db.conn.execute(
            "SELECT kind, a, b FROM overrides WHERE kind IN ('must_link', 'cannot_link')"
        ):
            if not r["b"]:
                continue
            if r["kind"] == "must_link":
                self._must.setdefault(r["a"], set()).add(r["b"])
                self._must.setdefault(r["b"], set()).add(r["a"])
            else:
                self._cannot.add(frozenset((r["a"], r["b"])))

    # ------------------------------------------------------------------ lookups
    def _by_alias(self, norm: str, type_: str) -> int | None:
        rows = self.db.conn.execute(
            """SELECT e.id, (SELECT count(*) FROM mentions m WHERE m.entity_id = e.id) AS n
               FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
               WHERE a.alias_norm = ? AND e.type = ? ORDER BY n DESC, e.id LIMIT 1""",
            (norm, type_),
        ).fetchone()
        return int(rows["id"]) if rows else None

    def _by_key(self, key: str) -> int | None:
        norm, _, type_ = key.rpartition("|")
        return self._by_alias(norm, type_)

    def _add_alias(self, entity_id: int, norm: str) -> None:
        if norm:
            self.db.conn.execute(
                "INSERT OR IGNORE INTO entity_aliases(alias_norm, entity_id) VALUES (?, ?)",
                (norm, entity_id),
            )

    def _enrich(self, entity_id: int, e: ExtractedEntity) -> None:
        if e.description:
            self.db.conn.execute(
                "UPDATE entities SET description = ? WHERE id = ? AND description = ''",
                (e.description, entity_id),
            )

    # --------------------------------------------------------------- resolution
    def resolve_many(self, items: list[ExtractedEntity]) -> dict[str, int]:
        """Resolve entities to ids. Returns {normalized_name: entity_id} for every input."""
        out: dict[str, int] = {}
        todo: dict[str, ExtractedEntity] = {}
        for e in items:
            norm = normalize_name(e.name)
            if not norm:
                continue
            cur = todo.get(norm)
            if cur is None or len(e.description) > len(cur.description):
                todo[norm] = e

        need_embedding: list[tuple[str, ExtractedEntity]] = []
        for norm, e in todo.items():
            eid = self._by_alias(norm, e.type)
            if eid is None:
                eid = self._via_must_link(norm, e)
            if eid is not None:
                self._enrich(eid, e)
                out[norm] = eid
            else:
                need_embedding.append((norm, e))

        if need_embedding:
            vecs = self.embedder.embed_documents(
                [f"{e.name} ({e.type}): {e.description}".strip() for _, e in need_embedding]
            )
            for (norm, e), vec in zip(need_embedding, vecs):
                out[norm] = self._match_or_create(norm, e, vec)
        return out

    def _via_must_link(self, norm: str, e: ExtractedEntity) -> int | None:
        for other in self._must.get(entity_key(norm, e.type), ()):
            eid = self._by_key(other)
            if eid is not None:
                self._add_alias(eid, norm)
                self.merged += 1
                return eid
        return None

    def _match_or_create(self, norm: str, e: ExtractedEntity, vec) -> int:
        key = entity_key(norm, e.type)
        hits = [
            (eid, sim)
            for eid, sim in self.index.search(vec, k=5)
            if sim >= self.cfg.candidate_threshold
        ]
        if hits:
            rows = {
                int(r["id"]): r
                for r in self.db.conn.execute(
                    f"SELECT id, canonical_name, norm_name, type, description FROM entities "
                    f"WHERE id IN ({','.join('?' * len(hits))})",
                    [h[0] for h in hits],
                )
            }
            for eid, sim in hits:
                row = rows.get(eid)
                if row is None or row["type"] != e.type:
                    continue
                if frozenset((key, entity_key(row["norm_name"], row["type"]))) in self._cannot:
                    continue
                if sim >= self.cfg.auto_merge_threshold or self._adjudicate(e, row) == "same":
                    self._add_alias(eid, norm)
                    self._enrich(eid, e)
                    self.merged += 1
                    return eid
        return self._create(norm, e, vec)

    def _adjudicate(self, e: ExtractedEntity, row) -> str:
        if not (self.cfg.adjudicate and self.llm is not None):
            return "unsure"
        user = (
            f"A: {e.name} ({e.type}) - {e.description or 'no description'}\n"
            f"B: {row['canonical_name']} ({row['type']}) - {row['description'] or 'no description'}"
        )
        try:
            return self.llm.structured(self._system, user, Verdict, model=self.model).verdict
        except LLMError as err:
            log.warning("adjudication failed (%s); keeping entities separate", err)
            return "unsure"

    def _create(self, norm: str, e: ExtractedEntity, vec) -> int:
        cur = self.db.conn.execute(
            "INSERT INTO entities(canonical_name, norm_name, type, description) VALUES (?, ?, ?, ?)",
            (e.name, norm, e.type, e.description),
        )
        eid = int(cur.lastrowid)
        self._add_alias(eid, norm)
        self.index.upsert([eid], vec[None, :])
        self.created += 1
        return eid
