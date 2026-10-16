"""BM25 over chunk text via SQLite FTS5."""

from __future__ import annotations

from ..store.db import Database
from ..text import content_tokens


def fts_match_expression(*texts: str) -> str:
    """OR of quoted content tokens. Natural-language queries need OR (AND returns nothing)."""
    seen: list[str] = []
    for text in texts:
        for tok in content_tokens(text):
            if tok not in seen:
                seen.append(tok)
    return " OR ".join(f'"{t}"' for t in seen[:40])


def bm25_search(db: Database, expression: str, k: int) -> list[tuple[int, float]]:
    """Top-k (chunk_id, score) with higher = better. Empty expression -> []."""
    if not expression or k <= 0:
        return []
    rows = db.conn.execute(
        "SELECT rowid, bm25(fts_chunks) AS s FROM fts_chunks WHERE fts_chunks MATCH ? "
        "ORDER BY s LIMIT ?",
        (expression, k),
    ).fetchall()
    return [(int(r[0]), float(-r[1])) for r in rows]
