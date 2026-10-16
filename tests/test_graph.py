from brain.config import GraphSettings
from brain.store.graph import GraphStore


def _entity(db, name, type_="Person"):
    cur = db.conn.execute(
        "INSERT INTO entities(canonical_name, norm_name, type) VALUES (?, ?, ?)",
        (name, name.lower(), type_),
    )
    return cur.lastrowid


def _doc_chunk(db, n):
    db.conn.execute(
        "INSERT OR IGNORE INTO documents(id, path, rel_path, root, sha256, parser_ver) VALUES (1,'/p','p','/','x','1')"
    )
    cur = db.conn.execute(
        "INSERT INTO chunks(doc_id, ord, text, tokens) VALUES (1, ?, ?, 5)", (n, f"chunk {n}")
    )
    return cur.lastrowid


def test_ppr_reaches_chunk_through_relation(db):
    g = GraphStore(db, GraphSettings())
    a, b = _entity(db, "Alice"), _entity(db, "Bob")
    c_a, c_b, c_far = _doc_chunk(db, 0), _doc_chunk(db, 1), _doc_chunk(db, 2)
    g.add_mention(c_a, a)
    g.add_mention(c_b, b)
    rid = g.upsert_relation(a, b, "works_with", chunk_id=c_a, quote="Alice works with Bob")
    db.bump_graph_version()

    res = g.view().ppr({a: 1.0}, {}, damping=0.6, chunk_seed_weight=0.3)
    ranked = [c for c, _ in res.top_chunks(5)]
    assert ranked[0] == c_a
    assert c_b in ranked and c_far not in ranked
    path = res.path_to(c_b)
    assert f"rel:{rid}" in path and path[-1] == f"men:{c_b}:{b}"


def test_dropped_relation_override_blocks_upsert(db):
    g = GraphStore(db, GraphSettings())
    a, b = _entity(db, "Alice"), _entity(db, "Bob")
    db.conn.execute(
        "INSERT INTO overrides(kind, a, b) VALUES ('drop_relation', 'alice|Person', 'bob|Person')"
    )
    assert g.upsert_relation(a, b, "knows") is None
    assert g.upsert_relation(a, a, "knows") is None
