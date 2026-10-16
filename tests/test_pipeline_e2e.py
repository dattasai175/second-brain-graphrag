import shutil

from brain.ingest import IngestPipeline
from brain.retrieval import Retriever
from brain.store.db import Database


def build(settings, db, embedder, llm):
    return IngestPipeline(settings, db, embedder, llm)


def test_ingest_retrieve_incremental(tmp_path, settings, embedder, fake_llm, sample_corpus):
    corpus = tmp_path / "corpus"
    shutil.copytree(sample_corpus, corpus)
    db = Database(tmp_path / "brain.db")
    pipe = build(settings, db, embedder, fake_llm)

    r1 = pipe.run(corpus)
    assert r1.added == 12 and r1.failed == [] and r1.chunks >= 12
    counts = db.counts()
    assert counts["entities"] > 0 and counts["relations"] > 0
    # explicit wikilink edge: Atlas note -> Priya Raman note
    assert (
        db.conn.execute(
            "SELECT count(*) FROM relations WHERE predicate = 'links_to' AND source = 'explicit'"
        ).fetchone()[0]
        >= 8
    )

    # unchanged re-run is a no-op
    r2 = pipe.run(corpus)
    assert r2.unchanged == 12 and r2.added == 0 and r2.updated == 0

    retr = Retriever(settings, db, embedder, None, None)
    res = retr.retrieve("Why was Kafka chosen over RabbitMQ?", "hybrid_graph", k=3)
    assert res.evidence and res.evidence[0].doc_path.endswith("RFC-12 Event Streaming.md")

    # multi-hop: the graph channel should surface the Helios note for "previous employer's acquirer"
    res = retr.retrieve(
        "Which company acquired the previous employer of the owner of Project Atlas?",
        "hybrid_graph",
        k=5,
    )
    paths = [e.doc_path for e in res.evidence]
    assert any(p.endswith("Helios Systems.md") for p in paths), paths
    assert res.linked_entities

    # edit one file -> exactly one update; stale chunks gone
    note = corpus / "companies" / "Helios Systems.md"
    note.write_text(note.read_text() + "\nThey also opened an office in Berlin.\n")
    r3 = pipe.run(corpus)
    assert r3.updated == 1 and r3.unchanged == 11
    assert (
        db.conn.execute("SELECT count(*) FROM chunks WHERE text LIKE '%Berlin%'").fetchone()[0] == 1
    )

    # delete a file -> retracted, no orphan chunks/FTS rows
    (corpus / "misc" / "Sourdough.md").unlink()
    r4 = pipe.run(corpus)
    assert r4.deleted == 1
    assert (
        db.conn.execute("SELECT count(*) FROM chunks WHERE text LIKE '%hydration%'").fetchone()[0]
        == 0
    )
    assert (
        db.conn.execute(
            "SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH 'hydration'"
        ).fetchone()[0]
        == 0
    )
    assert pipe.chunk_index.count() == db.counts()["chunks"]


def test_no_extract_mode_then_extract_later(tmp_path, settings, embedder, fake_llm, sample_corpus):
    corpus = tmp_path / "corpus"
    shutil.copytree(sample_corpus, corpus)
    db = Database(tmp_path / "brain.db")
    pipe = build(settings, db, embedder, fake_llm)
    pipe.run(corpus, extract=False)
    assert "Extraction" not in fake_llm.calls
    before = db.counts()["entities"]
    r = pipe.run(corpus)  # docs were never extracted -> picked up now
    assert r.updated == 12 and db.counts()["entities"] > before
