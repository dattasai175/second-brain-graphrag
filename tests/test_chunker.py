from pathlib import Path

from brain.config import ChunkingSettings
from brain.ingest.chunker import chunk_document
from brain.ingest.parsers.markdown import ParsedDocument, parse_markdown_text


def doc(text: str) -> ParsedDocument:
    return parse_markdown_text(text, Path("x.md"))


CFG = ChunkingSettings(target_tokens=60, overlap_tokens=10, min_tokens=15, max_tokens=90)


def test_sections_keep_heading_path():
    body = "\n\n".join(f"Sentence number {i} of this paragraph is padding text." for i in range(30))
    text = f"# Top\n\n## Alpha\n\n{body}\n\n## Beta\n\n{body}\n"
    chunks = chunk_document(doc(text), CFG)
    paths = {c.section_path for c in chunks}
    assert "Top > Alpha" in paths and "Top > Beta" in paths
    assert all(c.tokens <= CFG.max_tokens + 30 for c in chunks)  # overlap seed can add a little


def test_fenced_code_is_not_split_or_scanned_for_headings():
    text = "# Title\n\nIntro paragraph with enough words to matter here.\n\n```python\n# not a heading\n\nprint('x')\n```\n\nAfter code."
    chunks = chunk_document(doc(text), ChunkingSettings(target_tokens=200, min_tokens=5))
    joined = "\n".join(c.text for c in chunks)
    assert "# not a heading\n\nprint('x')" in joined
    assert all("not a heading" not in c.section_path for c in chunks)


def test_tiny_sections_are_carried_forward_with_their_heading():
    text = "# Doc\n\n## Tiny\n\nShort.\n\n## Real\n\n" + " ".join(["word"] * 60)
    chunks = chunk_document(doc(text), CFG)
    assert len(chunks) == 1
    assert "Tiny" in chunks[0].text and chunks[0].section_path.endswith("Real")


def test_oversize_paragraph_is_split():
    text = "# Big\n\n" + " ".join(f"This is sentence {i}." for i in range(400))
    chunks = chunk_document(doc(text), CFG)
    assert len(chunks) > 3
    assert all(c.tokens <= CFG.max_tokens + 30 for c in chunks)


def test_pdf_pages_keep_page_numbers():
    d = ParsedDocument(
        path=Path("x.pdf"),
        title="x",
        text="",
        pages=[(1, "First page text. " * 20), (2, "Second page text. " * 20)],
    )
    chunks = chunk_document(d, CFG)
    assert {c.page for c in chunks} == {1, 2}


def test_empty_document_has_no_chunks():
    assert chunk_document(doc(""), CFG) == []
