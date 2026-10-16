-- Second-brain schema. One SQLite file holds documents, chunks, FTS5 (BM25), the entity graph,
-- query traces and (from Phase 3) feedback. Vector tables (vec_chunks / vec_entities) are
-- created lazily by db.py because their dimension depends on the embedding model.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,          -- absolute, resolved
    rel_path    TEXT NOT NULL,                 -- relative to the ingest root (display / gold anchors)
    root        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    sha256      TEXT NOT NULL,
    parser_ver  TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT 'private',
    mtime       REAL,
    entity_id   INTEGER,                       -- pseudo-entity (type 'Document') used for explicit links
    extracted   INTEGER NOT NULL DEFAULT 0,    -- 1 once LLM entity extraction completed for every chunk
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,
    section_path TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    context      TEXT NOT NULL DEFAULT '',     -- contextual prefix (title > section, or an LLM blurb)
    tokens       INTEGER NOT NULL,
    page         INTEGER,
    char_start   INTEGER,
    char_end     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);

-- BM25 over context + text. rowid = chunks.id.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    norm_name      TEXT NOT NULL,
    type           TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(norm_name, type);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_norm TEXT    NOT NULL,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (alias_norm, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases(entity_id);

-- chunk <-> entity edges
CREATE TABLE IF NOT EXISTS mentions (
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (chunk_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);

CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY,
    src_id      INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id      INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate   TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    weight      REAL    NOT NULL DEFAULT 1.0,   -- number of supporting evidence rows
    confidence  REAL    NOT NULL DEFAULT 1.0,
    valid_from  TEXT,
    valid_to    TEXT,
    source      TEXT    NOT NULL DEFAULT 'extracted',   -- explicit | extracted | user
    UNIQUE (src_id, predicate, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_id);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_id);

CREATE TABLE IF NOT EXISTS relation_evidence (
    relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id)    ON DELETE CASCADE,
    quote       TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (relation_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_relevidence_chunk ON relation_evidence(chunk_id);

-- User corrections. Keys are "<normalized name>|<type>" so they survive re-extraction.
CREATE TABLE IF NOT EXISTS overrides (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('must_link', 'cannot_link', 'drop_relation')),
    a          TEXT NOT NULL,
    b          TEXT,
    predicate  TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Extraction results keyed by sha256(prompt version | model | title | section | text).
CREATE TABLE IF NOT EXISTS extraction_cache (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Query traces (Phase 2 renders them; Phase 3 credits feedback against them).
CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL DEFAULT (datetime('now')),
    text       TEXT NOT NULL,
    route      TEXT,
    trace_json TEXT NOT NULL DEFAULT '{}',
    answer     TEXT,
    model_info TEXT
);

-- Phase 3 tables are created now so the schema stays stable.
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    query_id  INTEGER REFERENCES queries(id) ON DELETE SET NULL,
    kind      TEXT NOT NULL,
    target_id TEXT,
    value     REAL,
    note      TEXT
);

CREATE TABLE IF NOT EXISTS adapt_params (
    version     INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    params_json TEXT NOT NULL,
    eval_report TEXT
);
