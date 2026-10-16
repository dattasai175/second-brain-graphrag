from pathlib import Path

from brain.ingest.parsers.markdown import parse_markdown_text


def test_frontmatter_title_tags_and_links():
    raw = """---
title: My Note
tags: [alpha, beta]
---
# Heading
See [[Other Note|alias]] and [[folder/Third#section]], plus #inline-tag and [link](sub/Fourth.md).
`[[ignored]]` and
```
[[also ignored]] #nope
```
"""
    d = parse_markdown_text(raw, Path("n.md"))
    assert d.title == "My Note"
    assert d.wikilinks == ["Other Note", "Third", "Fourth"]
    assert set(d.tags) == {"inline-tag", "alpha", "beta"}
    assert d.text.startswith("# Heading")


def test_title_falls_back_to_h1_then_stem():
    assert parse_markdown_text("# Hello\n\ntext", Path("a.md")).title == "Hello"
    assert parse_markdown_text("just text", Path("a-note.md")).title == "a-note"
