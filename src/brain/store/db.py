"""SQLite connection, schema bootstrap and small shared helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .vectors import VectorIndex

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA_PATH.read_text())
        self.vec_ext = self._try_load_vec_extension()
        self._indexes: dict[str, VectorIndex] = {}

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error."""
        with self.conn:
            yield self.conn

    # --------------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    @property
    def graph_version(self) -> int:
        return int(self.get_meta("graph_version", "0") or 0)

    def bump_graph_version(self) -> None:
        self.set_meta("graph_version", str(self.graph_version + 1))
        self.conn.commit()

    # ------------------------------------------------------------------ vectors
    def _try_load_vec_extension(self) -> bool:
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return True
        except (AttributeError, ImportError, sqlite3.OperationalError, sqlite3.NotSupportedError):
            return False

    def vector_index(self, name: str, dim: int) -> VectorIndex:
        """Open (creating if needed) the cosine index ``name``; guards backend + dimension."""
        backend = "vec0" if self.vec_ext else "numpy"
        stored_backend = self.get_meta("vec_backend")
        if stored_backend and stored_backend != backend:
            raise RuntimeError(
                f"This database was built with the '{stored_backend}' vector backend but this "
                f"Python provides '{backend}' (sqlite extension loading "
                f"{'available' if self.vec_ext else 'unavailable'}). Re-ingest with --reset, "
                "or run under the same interpreter that created it."
            )
        stored_dim = self.get_meta(f"vec_dim:{name}")
        if stored_dim and int(stored_dim) != dim:
            raise RuntimeError(
                f"Embedding dimension changed for '{name}' (database has {stored_dim}, "
                f"embedder produces {dim}). Re-ingest with --reset after changing the "
                "embedding model."
            )
        if name not in self._indexes:
            self._indexes[name] = VectorIndex(self.conn, name, dim, self.vec_ext)
            self.set_meta("vec_backend", backend)
            self.set_meta(f"vec_dim:{name}", str(dim))
            self.conn.commit()
        return self._indexes[name]

    def stored_dim(self, name: str) -> int | None:
        v = self.get_meta(f"vec_dim:{name}")
        return int(v) if v else None

    # ------------------------------------------------------------------ helpers
    def fetch_chunks(self, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = [int(i) for i in ids]
        out: dict[int, sqlite3.Row] = {}
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"""SELECT c.id, c.doc_id, c.ord, c.section_path, c.text, c.context, c.page,
                           d.path AS doc_path, d.rel_path, d.title
                    FROM chunks c JOIN documents d ON d.id = c.doc_id
                    WHERE c.id IN ({marks})""",
                part,
            ).fetchall()
            for r in rows:
                out[int(r["id"])] = r
        return out

    def counts(self) -> dict[str, int]:
        q = lambda sql: int(self.conn.execute(sql).fetchone()[0])  # noqa: E731
        return {
            "documents": q("SELECT count(*) FROM documents"),
            "chunks": q("SELECT count(*) FROM chunks"),
            "entities": q("SELECT count(*) FROM entities WHERE type != 'Document'"),
            "document_nodes": q("SELECT count(*) FROM entities WHERE type = 'Document'"),
            "relations": q("SELECT count(*) FROM relations"),
            "mentions": q("SELECT count(*) FROM mentions"),
            "queries": q("SELECT count(*) FROM queries"),
        }
