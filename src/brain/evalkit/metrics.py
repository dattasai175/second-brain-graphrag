"""Retrieval metrics against document+quote anchored gold spans."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..text import ngrams, normalize_name
from ..types import Evidence
from .golden import GoldItem, GoldSpan


def _path_matches(doc_path: str, span_path: str) -> bool:
    a = doc_path.replace("\\", "/").lstrip("./")
    b = span_path.replace("\\", "/").lstrip("./")
    return a == b or a.endswith("/" + b)


def span_matches(doc_path: str, text: str, span: GoldSpan) -> bool:
    """True if this chunk covers the gold span: same document and the quote is (mostly) in it."""
    if not _path_matches(doc_path, span.path):
        return False
    if not span.quote.strip():
        return True
    q, t = normalize_name(span.quote), normalize_name(text)
    if q in t:
        return True
    # quote straddling a chunk boundary / overlap: accept substantial trigram containment
    qg = ngrams(q.split(), 3)
    return bool(qg) and len(qg & ngrams(t.split(), 3)) / len(qg) >= 0.6


@dataclass
class QuestionScore:
    id: str
    type: str
    answerable: bool
    n_spans: int
    span_ranks: list[int | None]  # per gold span: 1-based rank of first matching chunk
    recall: dict[int, float] = field(default_factory=dict)  # @k
    hit: dict[int, float] = field(default_factory=dict)
    all_hops: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms: float = 0.0


def score_question(
    item: GoldItem,
    evidence: list[Evidence],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    latency_ms: float = 0.0,
) -> QuestionScore:
    n = len(item.gold)
    span_ranks: list[int | None] = []
    claimed: dict[int, int] = {}  # evidence rank -> span index (each span credited once)
    for si, span in enumerate(item.gold):
        rank = None
        for r, ev in enumerate(evidence, start=1):
            if span_matches(ev.doc_path, ev.text, span):
                rank = r
                break
        span_ranks.append(rank)
        if rank is not None:
            claimed.setdefault(rank, si)

    qs = QuestionScore(item.id, item.type, item.answerable, n, span_ranks, latency_ms=latency_ms)
    for k in ks:
        got = [r for r in span_ranks if r is not None and r <= k]
        qs.recall[k] = len(got) / n if n else 0.0
        qs.hit[k] = 1.0 if got else 0.0
        qs.all_hops[k] = 1.0 if n and len(got) == n else 0.0
        dcg = sum(1.0 / math.log2(r + 1) for r in claimed if r <= k)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n, k)))
        qs.ndcg[k] = dcg / ideal if ideal else 0.0
    found = [r for r in span_ranks if r is not None]
    qs.mrr = 1.0 / min(found) if found else 0.0
    return qs


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _p50(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def aggregate(scores: list[QuestionScore], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Overall + per question type means over answerable questions."""
    answerable = [s for s in scores if s.answerable]
    groups: dict[str, list[QuestionScore]] = defaultdict(list)
    groups["all"] = answerable
    for s in answerable:
        groups[s.type].append(s)
    out: dict[str, dict] = {}
    for name, group in groups.items():
        out[name] = {
            "n": len(group),
            **{f"recall@{k}": round(_mean([s.recall[k] for s in group]), 4) for k in ks},
            **{f"hit@{k}": round(_mean([s.hit[k] for s in group]), 4) for k in ks},
            **{f"all_hops@{k}": round(_mean([s.all_hops[k] for s in group]), 4) for k in ks},
            **{f"ndcg@{k}": round(_mean([s.ndcg[k] for s in group]), 4) for k in ks},
            "mrr": round(_mean([s.mrr for s in group]), 4),
            "latency_p50_ms": round(_p50([s.latency_ms for s in group]), 1),
        }
    return out
