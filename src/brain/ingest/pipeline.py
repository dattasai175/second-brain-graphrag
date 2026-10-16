"""Incremental ingest: scan -> parse -> chunk -> embed -> extract -> resolve -> write.

Re-running over an unchanged folder is a no-op. Changed files are *retracted* (their chunks,
mentions and relation evidence are removed; relations/entities left without evidence are
garbage-collected) and re-ingested in a single transaction. User overrides are never touched.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..llm.base import LLMClient, LLMError, LLMUnavailable
from ..llm.embeddings import Embedder
from ..prompts import load as load_prompt
from ..store.db import Database
from ..store.graph import GraphStore
from ..text import normalize_name
from .chunker import Chunk, chunk_document
from .extractor import Extraction, Extractor, Ontology, mention_count
from .parsers import PARSER_VERSION, ParsedDocument, parse
from .resolver import EntityResolver
from .scanner import SourceFile, scan

log = logging.getLogger(__name__)

Progress = Callable[[str, str], None]  # (event, message)


@dataclass
class IngestReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0
    relations: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    extraction_failures: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        llm: LLMClient | None = None,
    ):
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.llm = llm
        dim = embedder.dim  # probes the backend: fails fast with a clear error if unavailable
        self.chunk_index = db.vector_index("chunks", dim)
        self.entity_index = db.vector_index("entities", dim)
        self.graph = GraphStore(db, settings.graph)
        self.ontology = Ontology.load(settings.ontology_path)
        ex = settings.extraction
        extraction_model = settings.llm.extraction_model
        self.extractor = (
            Extractor(llm, db, ex, self.ontology, model=extraction_model)
            if (llm is not None and ex.enabled)
            else None
        )
        self.resolver = EntityResolver(
            db, embedder, self.entity_index, ex, llm=llm, model=extraction_model
        )

    # ------------------------------------------------------------------- public
    def run(
        self,
        root: str | Path,
        *,
        force: bool = False,
        extract: bool = True,
        progress: Progress | None = None,
    ) -> IngestReport:
        t0 = time.time()
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"No such file or folder: {root}")
        base = root.parent if root.is_file() else root
        report = IngestReport()
        say = progress or (lambda event, msg: None)

        files = scan(root, self.settings.ingest.extensions, self.settings.ingest.ignore)
        report.scanned = len(files)
        existing = {
            r["path"]: r
            for r in self.db.conn.execute(
                "SELECT id, path, sha256, parser_ver, extracted FROM documents WHERE root = ?",
                (str(base),),
            )
        }
        want_extract = extract and self.extractor is not None

        for sf in files:
            row = existing.pop(str(sf.path), None)
            up_to_date = (
                row is not None
                and row["sha256"] == sf.sha256
                and row["parser_ver"] == PARSER_VERSION
                and (row["extracted"] == 1 or not want_extract)
            )
            if up_to_date and not force:
                report.unchanged += 1
                say("unchanged", sf.rel)
                continue
            try:
                self._ingest_file(sf, base, row, want_extract, report)
                report.added += row is None
                report.updated += row is not None
                say("ingested", sf.rel)
            except LLMUnavailable:
                raise
            except Exception as e:  # one bad file must not sink the run
                log.exception("failed to ingest %s", sf.path)
                report.failed.append((sf.rel, f"{type(e).__name__}: {e}"))
                say("failed", f"{sf.rel}: {e}")

        if root.is_dir():  # files that disappeared from disk
            for row in existing.values():
                with self.db.transaction():
                    self._retract(int(row["id"]))
                report.deleted += 1
                say("deleted", row["path"])

        self.db.bump_graph_version()
        report.entities_created = self.resolver.created
        report.entities_merged = self.resolver.merged
        report.extraction_failures = self.extractor.failures if self.extractor else 0
        report.seconds = time.time() - t0
        return report

    # ---------------------------------------------------------------- one file
    def _ingest_file(
        self,
        sf: SourceFile,
        base: Path,
        old_row,
        want_extract: bool,
        report: IngestReport,
    ) -> None:
        parsed = parse(sf.path)
        chunks = chunk_document(parsed, self.settings.chunking)
        contexts = [self._context_for(parsed, c) for c in chunks]
        vecs = self.embedder.embed_documents(
            [f"{ctx}\n{c.text}" for ctx, c in zip(contexts, chunks)]
        )

        extractions: list[Extraction | None] = [None] * len(chunks)
        fully_extracted = False
        if want_extract and chunks and self.extractor is not None:
            extractions = self.extractor.extract_many(
                [(c.text, parsed.title, c.section_path) for c in chunks]
            )
            fully_extracted = all(e is not None for e in extractions)

        with self.db.transaction() as conn:
            if old_row is not None:
                self._retract(int(old_row["id"]))
            doc_entity = self._document_entity(sf.path.stem, parsed.title)
            cur = conn.execute(
                """INSERT INTO documents(path, rel_path, root, title, sha256, parser_ver,
                                         sensitivity, mtime, entity_id, extracted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(sf.path),
                    sf.rel,
                    str(base),
                    parsed.title,
                    sf.sha256,
                    PARSER_VERSION,
                    self.settings.sensitivity.label_for(sf.rel),
                    sf.mtime,
                    doc_entity,
                    1 if (fully_extracted or not chunks) else 0,
                ),
            )
            doc_id = int(cur.lastrowid)

            chunk_ids: list[int] = []
            for c, ctx in zip(chunks, contexts):
                cur = conn.execute(
                    """INSERT INTO chunks(doc_id, ord, section_path, text, context, tokens, page,
                                          char_start, char_end)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc_id,
                        c.ord,
                        c.section_path,
                        c.text,
                        ctx,
                        c.tokens,
                        c.page,
                        c.char_start,
                        c.char_end,
                    ),
                )
                cid = int(cur.lastrowid)
                chunk_ids.append(cid)
                conn.execute(
                    "INSERT INTO fts_chunks(rowid, content) VALUES (?, ?)",
                    (cid, f"{ctx}\n{c.text}"),
                )
                self.graph.add_mention(cid, doc_entity)
            self.chunk_index.upsert(chunk_ids, vecs)
            report.chunks += len(chunks)

            self._explicit_edges(parsed, doc_entity, chunks, chunk_ids, report)
            self._extracted_edges(chunks, chunk_ids, extractions, report)

    # ---------------------------------------------------------------- helpers
    def _context_for(self, parsed: ParsedDocument, chunk: Chunk) -> str:
        mode = self.settings.chunking.contextual_prefix
        if mode == "none":
            return ""
        heading = parsed.title + (f" > {chunk.section_path}" if chunk.section_path else "")
        if mode == "llm" and self.llm is not None:
            try:
                blurb = self.llm.chat(
                    load_prompt("context"),
                    f"Document: {parsed.title}\nSection: {chunk.section_path or '-'}\n\n"
                    f"Passage:\n{chunk.text[:1500]}",
                    max_tokens=80,
                )
                return f"{heading}. {blurb}".strip()
            except LLMError:
                pass
        return heading

    def _simple_entity(self, name: str, type_: str, display: str | None = None) -> int:
        """Entity found/created by exact alias only (documents, tags): no embedding, no LLM."""
        conn = self.db.conn
        norm = normalize_name(name)
        row = conn.execute(
            """SELECT e.id FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
               WHERE a.alias_norm = ? AND e.type = ? LIMIT 1""",
            (norm, type_),
        ).fetchone()
        if row:
            return int(row[0])
        cur = conn.execute(
            "INSERT INTO entities(canonical_name, norm_name, type) VALUES (?, ?, ?)",
            (display or name, norm, type_),
        )
        eid = int(cur.lastrowid)
        conn.execute("INSERT OR IGNORE INTO entity_aliases VALUES (?, ?)", (norm, eid))
        return eid

    def _document_entity(self, stem: str, title: str) -> int:
        eid = self._simple_entity(stem, "Document", display=title or stem)
        self.db.conn.execute(
            "UPDATE entities SET canonical_name = ? WHERE id = ?", (title or stem, eid)
        )
        title_norm = normalize_name(title)
        if title_norm:
            self.db.conn.execute(
                "INSERT OR IGNORE INTO entity_aliases VALUES (?, ?)", (title_norm, eid)
            )
        return eid

    def _explicit_edges(
        self,
        parsed: ParsedDocument,
        doc_entity: int,
        chunks: list[Chunk],
        chunk_ids: list[int],
        report: IngestReport,
    ) -> None:
        """Wikilinks and tags are free, high-precision edges: no LLM involved."""
        if not chunks:
            return

        def chunk_with(needle: str) -> int:
            low = needle.casefold()
            for c, cid in zip(chunks, chunk_ids):
                if low in c.text.casefold():
                    return cid
            return chunk_ids[0]

        for target in parsed.wikilinks:
            if normalize_name(target) == normalize_name(parsed.path.stem):
                continue
            dst = self._simple_entity(target, "Document")
            rid = self.graph.upsert_relation(
                doc_entity,
                dst,
                "links_to",
                source="explicit",
                chunk_id=chunk_with(target),
                quote=f"[[{target}]]",
            )
            report.relations += rid is not None
        for tag in parsed.tags:
            dst = self._simple_entity(tag, "Tag", display=f"#{tag}")
            rid = self.graph.upsert_relation(
                doc_entity,
                dst,
                "tagged_with",
                source="explicit",
                chunk_id=chunk_with(f"#{tag}"),
                quote=f"#{tag}",
            )
            report.relations += rid is not None

    def _extracted_edges(
        self,
        chunks: list[Chunk],
        chunk_ids: list[int],
        extractions: list[Extraction | None],
        report: IngestReport,
    ) -> None:
        for chunk, cid, ext in zip(chunks, chunk_ids, extractions):
            if ext is None or not ext.entities:
                continue
            ids = self.resolver.resolve_many(ext.entities)
            for e in ext.entities:
                eid = ids.get(normalize_name(e.name))
                if eid is not None:
                    self.graph.add_mention(cid, eid, mention_count(e.name, chunk.text))
            for r in ext.relations:
                src = ids.get(normalize_name(r.source))
                dst = ids.get(normalize_name(r.target))
                if src is None or dst is None:
                    continue
                rid = self.graph.upsert_relation(
                    src,
                    dst,
                    r.predicate,
                    description=r.description,
                    source="extracted",
                    chunk_id=cid,
                    quote=r.evidence,
                )
                report.relations += rid is not None

    # --------------------------------------------------------------- retraction
    def _retract(self, doc_id: int) -> None:
        """Remove a document and everything derived only from it. Overrides are left alone."""
        conn = self.db.conn
        chunk_ids = [
            r[0] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        affected = [
            r[0]
            for r in conn.execute(
                """SELECT DISTINCT re.relation_id FROM relation_evidence re
                   JOIN chunks c ON c.id = re.chunk_id WHERE c.doc_id = ?""",
                (doc_id,),
            )
        ]
        self.chunk_index.delete(chunk_ids)
        conn.executemany("DELETE FROM fts_chunks WHERE rowid = ?", [(i,) for i in chunk_ids])
        conn.execute(
            "DELETE FROM documents WHERE id = ?", (doc_id,)
        )  # cascades chunks/mentions/evidence

        for rid in affected:
            left = conn.execute(
                "SELECT count(*) FROM relation_evidence WHERE relation_id = ?", (rid,)
            ).fetchone()[0]
            if left == 0:
                conn.execute("DELETE FROM relations WHERE id = ? AND source != 'user'", (rid,))
            else:
                self.graph.refresh_weight(rid)

        orphans = [
            r[0]
            for r in conn.execute(
                """SELECT id FROM entities
                   WHERE id NOT IN (SELECT entity_id FROM mentions)
                     AND id NOT IN (SELECT src_id FROM relations)
                     AND id NOT IN (SELECT dst_id FROM relations)
                     AND id NOT IN (SELECT entity_id FROM documents WHERE entity_id IS NOT NULL)"""
            )
        ]
        if orphans:
            self.entity_index.delete(orphans)
            conn.executemany("DELETE FROM entities WHERE id = ?", [(i,) for i in orphans])
