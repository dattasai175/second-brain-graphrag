"""Prompt templates live as Markdown next to this file so they can be edited and diffed.

Bump ``EXTRACTOR_VERSION`` whenever extractor.md/gleaning.md change meaningfully: it is part of
the extraction cache key, so old cached extractions are not reused with a different prompt.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

EXTRACTOR_VERSION = "1"


@cache
def load(name: str) -> str:
    return (Path(__file__).with_name(f"{name}.md")).read_text().strip()
