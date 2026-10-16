"""PDF parser: Docling (layout-aware markdown, tables) when installed, pypdf page text otherwise."""

from __future__ import annotations

import logging
from pathlib import Path

from .markdown import ParsedDocument

log = logging.getLogger(__name__)
_docling_converter = None


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_with_docling(path: Path) -> ParsedDocument:
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter

        _docling_converter = DocumentConverter()
    result = _docling_converter.convert(str(path))
    markdown = result.document.export_to_markdown()
    return ParsedDocument(
        path=path, title=path.stem, text=markdown, wikilinks=[], tags=[], parser="docling"
    )


def _parse_with_pypdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    title = path.stem
    meta_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    if meta_title and str(meta_title).strip():
        title = str(meta_title).strip()
    return ParsedDocument(
        path=path,
        title=title,
        text="\n\n".join(t for _, t in pages),
        pages=pages,
        parser="pypdf",
    )


def parse_pdf(path: Path) -> ParsedDocument:
    if _docling_available():
        try:
            return _parse_with_docling(path)
        except Exception as e:  # docling can fail on odd PDFs; don't lose the file
            log.warning("Docling failed on %s (%s); falling back to pypdf", path.name, e)
    return _parse_with_pypdf(path)
