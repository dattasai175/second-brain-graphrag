"""Parsers turn a file into a ``ParsedDocument`` (markdown-ish text + explicit structure)."""

from __future__ import annotations

from pathlib import Path

from .markdown import ParsedDocument, parse_markdown
from .pdf import parse_pdf

PARSER_VERSION = "1"

__all__ = ["PARSER_VERSION", "ParsedDocument", "parse"]


def parse(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    return parse_markdown(path)
