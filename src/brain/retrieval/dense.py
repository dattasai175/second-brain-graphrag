"""Dense (embedding) channel."""

from __future__ import annotations

import numpy as np

from ..store.vectors import VectorIndex


def dense_search(index: VectorIndex, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k (chunk_id, cosine similarity)."""
    return index.search(qvec, k)
