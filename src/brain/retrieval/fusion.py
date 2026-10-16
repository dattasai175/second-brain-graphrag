"""Weighted reciprocal-rank fusion."""

from __future__ import annotations


def weighted_rrf(
    rankings: dict[str, list[tuple[int, float]]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse ranked lists: score(d) = sum_c w_c / (k + rank_c(d)). Rank is 1-based, list order.

    Channels absent from ``weights`` default to weight 1.0. Returns [(id, score)] best first.
    """
    fused: dict[int, float] = {}
    for channel, ranked in rankings.items():
        w = (weights or {}).get(channel, 1.0)
        if w <= 0:
            continue
        for rank, (item, _score) in enumerate(ranked, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def ranks(ranked: list[tuple[int, float]]) -> dict[int, int]:
    return {item: i for i, (item, _) in enumerate(ranked, start=1)}
