"""Vector index with two interchangeable backends.

* ``vec0`` (sqlite-vec): virtual table inside the same SQLite file. Needs a Python whose
  ``sqlite3`` module can load extensions (python-build-standalone / uv-managed, Homebrew;
  the python.org macOS build cannot).
* ``numpy``: embeddings kept as float32 BLOBs in a normal table and searched by brute force
  in memory. Always available; fine up to a few hundred thousand vectors.

Vectors are L2-normalised on the way in, so similarity is cosine everywhere.
"""

from __future__ import annotations

import sqlite3

import numpy as np


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class VectorIndex:
    """Cosine-similarity index keyed by integer ids (chunk ids or entity ids)."""

    def __init__(self, conn: sqlite3.Connection, name: str, dim: int, use_ext: bool):
        self.conn = conn
        self.name = name
        self.dim = dim
        self.use_ext = use_ext
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)  # numpy backend cache
        self._mat: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._n = 0
        self._loaded = False
        if use_ext:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{name} USING vec0(embedding float[{dim}])"
            )
        else:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS vecf_{name} "
                "(id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
            )

    # ------------------------------------------------------------------ writes
    def upsert(self, ids: list[int], vecs: np.ndarray) -> None:
        if not ids:
            return
        vecs = _normalize(vecs)
        if vecs.shape != (len(ids), self.dim):
            raise ValueError(f"vector shape {vecs.shape} != ({len(ids)}, {self.dim})")
        self.delete(ids)
        if self.use_ext:
            import sqlite_vec

            self.conn.executemany(
                f"INSERT INTO vec_{self.name}(rowid, embedding) VALUES (?, ?)",
                [(int(i), sqlite_vec.serialize_float32(v.tolist())) for i, v in zip(ids, vecs)],
            )
        else:
            self.conn.executemany(
                f"INSERT INTO vecf_{self.name}(id, embedding) VALUES (?, ?)",
                [(int(i), v.tobytes()) for i, v in zip(ids, vecs)],
            )
            if self._loaded:
                self._append(np.asarray(ids, dtype=np.int64), vecs)

    def delete(self, ids: list[int]) -> None:
        if not ids:
            return
        table = f"vec_{self.name}" if self.use_ext else f"vecf_{self.name}"
        col = "rowid" if self.use_ext else "id"
        self.conn.executemany(f"DELETE FROM {table} WHERE {col} = ?", [(int(i),) for i in ids])
        if not self.use_ext:
            self._loaded = False  # rebuilt lazily on next search

    # ------------------------------------------------------------------- reads
    def count(self) -> int:
        table = f"vec_{self.name}" if self.use_ext else f"vecf_{self.name}"
        return int(self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (id, cosine similarity), best first."""
        q = _normalize(qvec)[0]
        if k <= 0:
            return []
        if self.use_ext:
            import sqlite_vec

            rows = self.conn.execute(
                f"SELECT rowid, distance FROM vec_{self.name} "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(q.tolist()), int(k)),
            ).fetchall()
            # vec0 returns L2 distance; for unit vectors cos = 1 - d^2 / 2
            return [(int(r[0]), float(1.0 - (r[1] ** 2) / 2.0)) for r in rows]
        self._ensure_loaded()
        if self._n == 0:
            return []
        sims = self._mat[: self._n] @ q
        k = min(k, self._n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self._ids[i]), float(sims[i])) for i in top]

    def get(self, ids: list[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        if not ids:
            return out
        if self.use_ext:
            import sqlite_vec  # noqa: F401  (serialization is raw float32 bytes)

            for i in ids:
                row = self.conn.execute(
                    f"SELECT embedding FROM vec_{self.name} WHERE rowid = ?", (int(i),)
                ).fetchone()
                if row:
                    out[int(i)] = np.frombuffer(row[0], dtype=np.float32).copy()
        else:
            for i in ids:
                row = self.conn.execute(
                    f"SELECT embedding FROM vecf_{self.name} WHERE id = ?", (int(i),)
                ).fetchone()
                if row:
                    out[int(i)] = np.frombuffer(row[0], dtype=np.float32).copy()
        return out

    # ---------------------------------------------------- numpy backend internals
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rows = self.conn.execute(f"SELECT id, embedding FROM vecf_{self.name}").fetchall()
        self._n = len(rows)
        cap = max(64, self._n * 2)
        self._ids = np.zeros(cap, dtype=np.int64)
        self._mat = np.zeros((cap, self.dim), dtype=np.float32)
        for j, (i, blob) in enumerate(rows):
            self._ids[j] = i
            self._mat[j] = np.frombuffer(blob, dtype=np.float32)
        self._loaded = True

    def _append(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        need = self._n + len(ids)
        if need > len(self._ids):
            cap = max(need, len(self._ids) * 2)
            new_ids = np.zeros(cap, dtype=np.int64)
            new_mat = np.zeros((cap, self.dim), dtype=np.float32)
            new_ids[: self._n] = self._ids[: self._n]
            new_mat[: self._n] = self._mat[: self._n]
            self._ids, self._mat = new_ids, new_mat
        self._ids[self._n : need] = ids
        self._mat[self._n : need] = vecs
        self._n = need
