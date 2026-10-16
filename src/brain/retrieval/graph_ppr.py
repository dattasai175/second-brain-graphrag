"""Graph channel: link the question to entities, then walk the graph with Personalized PageRank."""

from __future__ import annotations

import numpy as np

from ..config import GraphSettings
from ..llm.embeddings import Embedder
from ..store.db import Database
from ..store.graph import GraphStore, PPRResult, specificity
from ..store.vectors import VectorIndex
from ..text import STOPWORDS, normalize_name, tokenize
from ..types import QueryPlan

MAX_NGRAM = 5


class EntityLinker:
    """Question -> {entity_id: seed weight}.

    1. Exact alias match over every n-gram of the question (longest match wins, no overlaps).
    2. Entity strings from query analysis that didn't match exactly are embedded and matched
       against entity embeddings (cosine >= ``link_min_sim``).
    Seed weights are scaled by specificity so ubiquitous entities don't flood the walk.
    """

    def __init__(
        self,
        db: Database,
        graph: GraphStore,
        embedder: Embedder,
        entity_index: VectorIndex,
        cfg: GraphSettings,
    ):
        self.db = db
        self.graph = graph
        self.embedder = embedder
        self.entity_index = entity_index
        self.cfg = cfg

    def _alias_lookup(self, aliases: list[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for start in range(0, len(aliases), 500):
            part = aliases[start : start + 500]
            marks = ",".join("?" * len(part))
            for alias, eid in self.db.conn.execute(
                f"SELECT alias_norm, entity_id FROM entity_aliases WHERE alias_norm IN ({marks})",
                part,
            ):
                out.setdefault(alias, []).append(int(eid))
        return out

    def link(self, plan: QueryPlan) -> dict[int, float]:
        if self.db.conn.execute("SELECT 1 FROM entities LIMIT 1").fetchone() is None:
            return {}
        raw: dict[int, float] = {}  # entity_id -> match strength in (0, 1]

        # 1) n-gram alias matches over the question text
        tokens = [normalize_name(t) for t in tokenize(plan.text)]
        tokens = [t for t in tokens if t]
        windows: list[tuple[int, int, str]] = []  # (start, n, alias)
        for n in range(min(MAX_NGRAM, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                if n == 1 and (gram[0] in STOPWORDS or len(gram[0]) < 3):
                    continue
                if all(g in STOPWORDS for g in gram):
                    continue
                windows.append((i, n, " ".join(gram)))
        hits = self._alias_lookup(sorted({w[2] for w in windows}))
        covered: set[int] = set()
        matched_norms: set[str] = set()
        for start, n, alias in windows:  # already ordered longest first
            if alias not in hits or any(p in covered for p in range(start, start + n)):
                continue
            covered.update(range(start, start + n))
            matched_norms.add(alias)
            for eid in hits[alias]:
                raw[eid] = max(raw.get(eid, 0.0), 1.0)

        # 2) embedding match for extracted entity strings not already matched exactly
        surfaces = [
            s for s in plan.entities if normalize_name(s) and normalize_name(s) not in matched_norms
        ]
        exact = self._alias_lookup([normalize_name(s) for s in surfaces])
        remaining = []
        for s in surfaces:
            ids = exact.get(normalize_name(s))
            if ids:
                for eid in ids:
                    raw[eid] = max(raw.get(eid, 0.0), 1.0)
            else:
                remaining.append(s)
        if remaining and self.entity_index.count() > 0:
            vecs = self.embedder.embed_documents(remaining)
            for vec in vecs:
                for eid, sim in self.entity_index.search(np.asarray(vec), k=3):
                    if sim >= self.cfg.link_min_sim:
                        raw[eid] = max(raw.get(eid, 0.0), float(sim))

        if not raw:
            return {}
        df = self.graph.mention_count(list(raw))
        seeds = {
            eid: strength * (specificity(df.get(eid, 1)) if self.cfg.hub_penalty else 1.0)
            for eid, strength in raw.items()
        }
        top = sorted(seeds.items(), key=lambda kv: -kv[1])[: self.cfg.max_seeds]
        return dict(top)


def run_ppr(
    graph: GraphStore,
    entity_seeds: dict[int, float],
    dense_hits: list[tuple[int, float]],
    cfg: GraphSettings,
    edge_mult: dict[int, float] | None = None,
) -> PPRResult:
    """Restart mass: linked entities + the top dense chunks (HippoRAG 2 style)."""
    chunk_seeds = {cid: max(sim, 0.0) ** 2 for cid, sim in dense_hits[:10] if sim > 0}
    view = graph.view(edge_mult)
    return view.ppr(
        entity_seeds,
        chunk_seeds,
        damping=cfg.ppr_damping,
        chunk_seed_weight=cfg.chunk_seed_weight,
    )
