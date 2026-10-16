"""Hybrid retrieval: dense + BM25 + graph PPR -> weighted RRF -> rerank -> evidence with provenance."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import Settings
from ..llm.base import LLMClient
from ..llm.embeddings import Embedder
from ..store.db import Database
from ..store.graph import GraphStore
from ..types import Evidence, QueryPlan, RetrievalResult
from .dense import dense_search
from .fusion import ranks as rank_map
from .fusion import weighted_rrf
from .graph_ppr import EntityLinker, run_ppr
from .lexical import bm25_search, fts_match_expression
from .query_analysis import analyze_query
from .rerank import PassthroughReranker, Reranker


@dataclass(frozen=True)
class RetrievalConfig:
    """Which channels run. Named presets form the ablation ladder used by the eval harness."""

    name: str
    dense: bool = True
    bm25: bool = True
    graph: bool = True
    rerank: bool = True


PRESETS: dict[str, RetrievalConfig] = {
    c.name: c
    for c in [
        RetrievalConfig("dense", dense=True, bm25=False, graph=False, rerank=False),
        RetrievalConfig("bm25", dense=False, bm25=True, graph=False, rerank=False),
        RetrievalConfig("dense_bm25", dense=True, bm25=True, graph=False, rerank=False),
        RetrievalConfig("dense_bm25_rerank", dense=True, bm25=True, graph=False, rerank=True),
        RetrievalConfig("hybrid_graph_norerank", dense=True, bm25=True, graph=True, rerank=False),
        RetrievalConfig("hybrid_graph", dense=True, bm25=True, graph=True, rerank=True),
    ]
}


class Retriever:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.reranker: Reranker = reranker or PassthroughReranker("not configured")
        dim = db.stored_dim("chunks") or embedder.dim
        self.chunk_index = db.vector_index("chunks", dim)
        self.entity_index = db.vector_index("entities", dim)
        self.graph = GraphStore(db, settings.graph)
        self.linker = EntityLinker(db, self.graph, embedder, self.entity_index, settings.graph)

    def plan(self, question: str) -> QueryPlan:
        return analyze_query(question, self.llm)

    def retrieve(
        self,
        question: str,
        preset: str | RetrievalConfig = "hybrid_graph",
        *,
        k: int | None = None,
        plan: QueryPlan | None = None,
    ) -> RetrievalResult:
        rc = PRESETS[preset] if isinstance(preset, str) else preset
        cfg = self.settings.retrieval
        k = k or cfg.final_k
        timings: dict[str, float] = {}
        notes: list[str] = []
        t_start = time.perf_counter()

        def lap(name: str, t0: float) -> None:
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        plan = plan or self.plan(question)
        lap("analyze", t0)

        channels: dict[str, list[tuple[int, float]]] = {}
        qvec = None
        if rc.dense or rc.graph:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(question)
            lap("embed", t0)
        if rc.dense:
            t0 = time.perf_counter()
            channels["dense"] = dense_search(self.chunk_index, qvec, cfg.dense_k)
            lap("dense", t0)
        if rc.bm25:
            t0 = time.perf_counter()
            expr = fts_match_expression(question, " ".join(plan.keywords))
            channels["bm25"] = bm25_search(self.db, expr, cfg.bm25_k)
            lap("bm25", t0)

        ppr = None
        linked: dict[str, float] = {}
        if rc.graph:
            t0 = time.perf_counter()
            seeds = self.linker.link(plan)
            names = self.graph.entity_names(list(seeds))
            linked = {names.get(eid, str(eid)): round(w, 3) for eid, w in seeds.items()}
            dense_hits = channels.get("dense") or self.chunk_index.search(qvec, 10)
            ppr = run_ppr(self.graph, seeds, dense_hits, self.settings.graph)
            channels["graph"] = ppr.top_chunks(cfg.graph_k)
            lap("graph", t0)
            if not seeds:
                notes.append("no entities linked: graph channel seeded by dense hits only")

        t0 = time.perf_counter()
        weights = {
            ch: w for ch, w in cfg.weights_for(plan.type).items() if ch in channels and channels[ch]
        }
        fused = weighted_rrf({c: r for c, r in channels.items() if r}, weights, k=cfg.rrf_k)
        lap("fuse", t0)

        use_rerank = rc.rerank and self.reranker.active
        if rc.rerank and not self.reranker.active:
            notes.append(f"reranker inactive: {self.reranker.name}")
        pool = fused[: cfg.rerank_candidates if use_rerank else k]
        rows = self.db.fetch_chunks(cid for cid, _ in pool)
        pool = [(cid, s) for cid, s in pool if cid in rows]

        rerank_scores: dict[int, float] = {}
        if use_rerank and pool:
            t0 = time.perf_counter()
            docs = [f"{rows[cid]['context']}\n{rows[cid]['text']}" for cid, _ in pool]
            scores = self.reranker.score(question, docs)
            rerank_scores = {cid: s for (cid, _), s in zip(pool, scores)}
            pool = sorted(pool, key=lambda kv: -rerank_scores[kv[0]])
            lap("rerank", t0)
        pool = pool[:k]

        per_channel_rank = {c: rank_map(r) for c, r in channels.items()}
        per_channel_score = {c: dict(r) for c, r in channels.items()}
        evidence: list[Evidence] = []
        for n, (cid, fused_score) in enumerate(pool, start=1):
            row = rows[cid]
            ev = Evidence(
                id=f"E{n}",
                chunk_id=cid,
                doc_id=int(row["doc_id"]),
                doc_path=row["rel_path"],
                title=row["title"],
                section=row["section_path"],
                page=row["page"],
                text=row["text"],
                score=rerank_scores.get(cid, fused_score),
                channels={c: s[cid] for c, s in per_channel_score.items() if cid in s},
                ranks={c: r[cid] for c, r in per_channel_rank.items() if cid in r},
            )
            if cid in rerank_scores:
                ev.channels["rerank"] = rerank_scores[cid]
            if ppr is not None:
                ev.graph_path = ppr.path_to(cid)
            evidence.append(ev)

        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        return RetrievalResult(
            plan=plan,
            evidence=evidence,
            preset=rc.name,
            timings=timings,
            linked_entities=linked,
            notes=notes,
        )
