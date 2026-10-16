"""Entity/relation graph: SQLite is the source of truth, igraph is the in-memory traversal view.

The traversal graph has two vertex kinds (entities and chunk "passage" nodes) and three edge
kinds: entity--entity relations, chunk--entity mentions and chunk--document-node links.
Retrieval runs Personalized PageRank over it (HippoRAG 2 style) and reads the mass that lands
on chunk vertices.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import igraph as ig

from ..config import GraphSettings
from .db import Database


def specificity(df: int) -> float:
    """IDF-flavoured weight in (0, 1]: entities mentioned in many chunks count for less."""
    return 1.0 / math.log2(2 + max(df, 0))


@dataclass
class GraphView:
    g: ig.Graph
    ent_vid: dict[int, int]
    chunk_vid: dict[int, int]
    entity_ids: list[int]
    chunk_ids: list[int]
    version: int

    @classmethod
    def build(
        cls,
        db: Database,
        cfg: GraphSettings,
        edge_mult: dict[int, float] | None = None,
    ) -> GraphView:
        conn = db.conn
        entity_ids = [r[0] for r in conn.execute("SELECT id FROM entities ORDER BY id")]
        chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]
        ent_vid = {e: i for i, e in enumerate(entity_ids)}
        chunk_vid = {c: len(entity_ids) + i for i, c in enumerate(chunk_ids)}
        doc_entities = {
            r[0]
            for r in conn.execute("SELECT entity_id FROM documents WHERE entity_id IS NOT NULL")
        }
        df = {
            r[0]: r[1] for r in conn.execute("SELECT entity_id, count(*) FROM mentions GROUP BY 1")
        }

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        kinds: list[str] = []
        refs: list[int] = []

        for rid, src, dst, w in conn.execute("SELECT id, src_id, dst_id, weight FROM relations"):
            if src not in ent_vid or dst not in ent_vid:
                continue
            mult = (edge_mult or {}).get(rid, 1.0)
            edges.append((ent_vid[src], ent_vid[dst]))
            weights.append((1.0 + math.log(max(w, 1.0))) * mult)
            kinds.append("rel")
            refs.append(rid)

        for chunk_id, entity_id, count in conn.execute(
            "SELECT chunk_id, entity_id, count FROM mentions"
        ):
            if chunk_id not in chunk_vid or entity_id not in ent_vid:
                continue
            base = cfg.doc_node_weight if entity_id in doc_entities else cfg.mention_weight
            spec = specificity(df.get(entity_id, 1)) if cfg.hub_penalty else 1.0
            edges.append((chunk_vid[chunk_id], ent_vid[entity_id]))
            weights.append(base * spec * (1.0 + 0.3 * math.log(max(count, 1))))
            kinds.append("men")
            refs.append(entity_id)

        g = ig.Graph(n=len(entity_ids) + len(chunk_ids), edges=edges, directed=False)
        g.es["weight"] = weights
        g.es["cost"] = [1.0 / w if w > 0 else 1e9 for w in weights]
        g.es["kind"] = kinds
        g.es["ref"] = refs
        g.vs["kind"] = ["e"] * len(entity_ids) + ["c"] * len(chunk_ids)
        g.vs["ref"] = entity_ids + chunk_ids
        return cls(g, ent_vid, chunk_vid, entity_ids, chunk_ids, db.graph_version)

    # -------------------------------------------------------------------- PPR
    def ppr(
        self,
        entity_seeds: dict[int, float],
        chunk_seeds: dict[int, float],
        *,
        damping: float,
        chunk_seed_weight: float,
    ) -> PPRResult:
        n = self.g.vcount()
        ent = {self.ent_vid[e]: w for e, w in entity_seeds.items() if e in self.ent_vid and w > 0}
        chk = {
            self.chunk_vid[c]: w for c, w in chunk_seeds.items() if c in self.chunk_vid and w > 0
        }
        if n == 0 or (not ent and not chk) or self.g.ecount() == 0:
            return PPRResult({}, {}, self, {})
        reset = [0.0] * n
        ent_share = (1.0 - chunk_seed_weight) if (ent and chk) else 1.0
        chk_share = (chunk_seed_weight) if (ent and chk) else 1.0
        for group, share in ((ent, ent_share), (chk, chk_share)):
            total = sum(group.values())
            for v, w in group.items():
                reset[v] += share * w / total
        scores = self.g.personalized_pagerank(
            directed=False, damping=damping, reset=reset, weights="weight"
        )
        chunk_scores = {cid: scores[v] for cid, v in self.chunk_vid.items() if scores[v] > 0}
        entity_scores = {eid: scores[v] for eid, v in self.ent_vid.items() if scores[v] > 0}
        seed_weights = {**ent, **chk}
        return PPRResult(chunk_scores, entity_scores, self, seed_weights)


@dataclass
class PPRResult:
    chunk_scores: dict[int, float]
    entity_scores: dict[int, float]
    _view: GraphView
    _seeds: dict[int, float] = field(default_factory=dict)  # vertex id -> restart weight

    def top_chunks(self, n: int) -> list[tuple[int, float]]:
        return sorted(self.chunk_scores.items(), key=lambda kv: -kv[1])[:n]

    def path_to(self, chunk_id: int, max_sources: int = 4) -> list[str]:
        """Cheapest seed -> chunk path, as edge descriptors (``rel:<id>`` / ``men:<chunk>:<entity>``).

        Used for explanation now and for per-edge credit assignment in Phase 3.
        """
        v = self._view
        target = v.chunk_vid.get(chunk_id)
        if target is None or not self._seeds or target in self._seeds:
            return []
        sources = sorted(self._seeds, key=lambda s: -self._seeds[s])[:max_sources]
        best: tuple[float, list[int]] | None = None
        for s in sources:
            with warnings.catch_warnings():  # igraph warns when the target is unreachable
                warnings.simplefilter("ignore", RuntimeWarning)
                paths = v.g.get_shortest_paths(s, to=target, weights="cost", output="epath")
            if not paths or not paths[0]:
                continue
            cost = sum(v.g.es[e]["cost"] for e in paths[0])
            if best is None or cost < best[0]:
                best = (cost, paths[0])
        if best is None:
            return []
        out: list[str] = []
        for eid in best[1]:
            edge = v.g.es[eid]
            if edge["kind"] == "rel":
                out.append(f"rel:{edge['ref']}")
            else:
                a, b = edge.tuple
                chunk_v = a if v.g.vs[a]["kind"] == "c" else b
                out.append(f"men:{v.g.vs[chunk_v]['ref']}:{edge['ref']}")
        return out


class GraphStore:
    """Writes entities/relations/mentions and serves cached traversal views."""

    def __init__(self, db: Database, cfg: GraphSettings):
        self.db = db
        self.cfg = cfg
        self._view: GraphView | None = None

    # ------------------------------------------------------------------ writes
    def add_mention(self, chunk_id: int, entity_id: int, count: int = 1) -> None:
        self.db.conn.execute(
            "INSERT INTO mentions(chunk_id, entity_id, count) VALUES (?, ?, ?) "
            "ON CONFLICT(chunk_id, entity_id) DO UPDATE SET count = excluded.count",
            (chunk_id, entity_id, max(1, count)),
        )

    def entity_key(self, entity_id: int) -> str | None:
        row = self.db.conn.execute(
            "SELECT norm_name, type FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return f"{row['norm_name']}|{row['type']}" if row else None

    def is_dropped(self, src_id: int, dst_id: int, predicate: str) -> bool:
        a, b = self.entity_key(src_id), self.entity_key(dst_id)
        if a is None or b is None:
            return False
        row = self.db.conn.execute(
            "SELECT 1 FROM overrides WHERE kind = 'drop_relation' AND a = ? AND b = ? "
            "AND (predicate IS NULL OR predicate = ?) LIMIT 1",
            (a, b, predicate),
        ).fetchone()
        return row is not None

    def upsert_relation(
        self,
        src_id: int,
        dst_id: int,
        predicate: str,
        *,
        description: str = "",
        source: str = "extracted",
        chunk_id: int | None = None,
        quote: str = "",
        confidence: float = 1.0,
    ) -> int | None:
        if src_id == dst_id or self.is_dropped(src_id, dst_id, predicate):
            return None
        conn = self.db.conn
        conn.execute(
            """INSERT INTO relations(src_id, dst_id, predicate, description, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(src_id, predicate, dst_id) DO UPDATE SET
                 description = CASE WHEN length(excluded.description) > length(relations.description)
                                    THEN excluded.description ELSE relations.description END""",
            (src_id, dst_id, predicate, description, source, confidence),
        )
        rid = conn.execute(
            "SELECT id FROM relations WHERE src_id = ? AND predicate = ? AND dst_id = ?",
            (src_id, predicate, dst_id),
        ).fetchone()[0]
        if chunk_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO relation_evidence(relation_id, chunk_id, quote) VALUES (?, ?, ?)",
                (rid, chunk_id, quote),
            )
        self.refresh_weight(rid)
        return int(rid)

    def refresh_weight(self, relation_id: int) -> None:
        self.db.conn.execute(
            """UPDATE relations SET weight = MAX(1.0, (
                   SELECT count(*) FROM relation_evidence WHERE relation_id = relations.id))
               WHERE id = ?""",
            (relation_id,),
        )

    # ------------------------------------------------------------------- reads
    def view(self, edge_mult: dict[int, float] | None = None) -> GraphView:
        if edge_mult:  # learned multipliers: build fresh, never cache
            return GraphView.build(self.db, self.cfg, edge_mult)
        if self._view is None or self._view.version != self.db.graph_version:
            self._view = GraphView.build(self.db, self.cfg)
        return self._view

    def mention_count(self, entity_ids: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for start in range(0, len(entity_ids), 500):
            part = entity_ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for eid, n in self.db.conn.execute(
                f"SELECT entity_id, count(*) FROM mentions WHERE entity_id IN ({marks}) GROUP BY 1",
                part,
            ):
                out[int(eid)] = int(n)
        return out

    def entity_names(self, entity_ids: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for start in range(0, len(entity_ids), 500):
            part = entity_ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for eid, name in self.db.conn.execute(
                f"SELECT id, canonical_name FROM entities WHERE id IN ({marks})", part
            ):
                out[int(eid)] = name
        return out
