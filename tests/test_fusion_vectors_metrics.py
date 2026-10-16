import numpy as np
import pytest

from brain.evalkit.golden import GoldItem, GoldSpan
from brain.evalkit.metrics import score_question, span_matches
from brain.retrieval.fusion import weighted_rrf
from brain.store.db import Database
from brain.store.vectors import VectorIndex
from brain.types import Evidence


def test_rrf_prefers_items_found_by_multiple_channels():
    fused = weighted_rrf(
        {"a": [(1, 0), (2, 0), (3, 0)], "b": [(3, 0), (2, 0)]}, {"a": 1, "b": 1}, k=60
    )
    assert [i for i, _ in fused][0] in (2, 3)
    assert fused[-1][0] == 1


def test_rrf_weights_shift_ranking():
    rankings = {"a": [(1, 0), (2, 0)], "b": [(2, 0), (1, 0)]}
    assert weighted_rrf(rankings, {"a": 5, "b": 1})[0][0] == 1
    assert weighted_rrf(rankings, {"a": 1, "b": 5})[0][0] == 2
    assert weighted_rrf(rankings, {"a": 0, "b": 1})[0][0] == 2


@pytest.mark.parametrize("use_ext", [False, True])
def test_vector_index_roundtrip(tmp_path, use_ext):
    db = Database(tmp_path / "v.db")
    if use_ext and not db.vec_ext:
        pytest.skip("sqlite extension loading unavailable in this Python")
    idx = VectorIndex(db.conn, "t", 4, use_ext)
    vecs = np.eye(4, dtype=np.float32)
    idx.upsert([10, 11, 12, 13], vecs)
    hits = idx.search(np.array([1, 0.1, 0, 0], dtype=np.float32), 2)
    assert hits[0][0] == 10 and hits[0][1] > 0.9
    idx.delete([10])
    assert 10 not in [i for i, _ in idx.search(np.array([1, 0, 0, 0], dtype=np.float32), 4)]
    idx.upsert([11], np.array([[0, 0, 0, 1]], dtype=np.float32))  # overwrite
    assert idx.search(np.array([0, 0, 0, 1], dtype=np.float32), 1)[0][0] in (11, 13)
    assert idx.count() == 3


def ev(path, text, i):
    return Evidence(
        id=f"E{i}", chunk_id=i, doc_id=1, doc_path=path, title="", section="", page=None, text=text
    )


def test_span_matching_and_scores():
    item = GoldItem(
        id="q",
        question="?",
        type="multi_hop",
        gold=[
            GoldSpan(path="a/x.md", quote="alpha beta gamma"),
            GoldSpan(path="b/y.md", quote="delta"),
        ],
    )
    assert span_matches("/root/a/x.md", "Some ALPHA, beta gamma text", item.gold[0])
    assert not span_matches("/root/a/z.md", "alpha beta gamma", item.gold[0])
    s = score_question(
        item,
        [ev("a/x.md", "alpha beta gamma", 1), ev("c.md", "nope", 2), ev("b/y.md", "delta", 3)],
        ks=(1, 3),
    )
    assert s.span_ranks == [1, 3]
    assert s.recall[1] == 0.5 and s.recall[3] == 1.0
    assert s.all_hops[1] == 0.0 and s.all_hops[3] == 1.0
    assert s.mrr == 1.0
    assert 0 < s.ndcg[3] <= 1.0
