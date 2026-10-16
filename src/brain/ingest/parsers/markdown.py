"""Markdown / plain-text parser. Extracts the explicit structure that becomes free graph edges:
wikilinks (``[[Note]]``), local markdown links, tags and frontmatter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FENCE_BLOCK = re.compile(r"(^|\n)(```|~~~).*?\n\2[^\n]*(?=\n|$)", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_WIKILINK = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")
_MDLINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+\.(?:md|markdown))(?:#[^)]*)?\)", re.I)
_TAG = re.compile(r"(?<![\w/&#])#([A-Za-z][\w/-]*)")
_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.M)
_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)


@dataclass
class ParsedDocument:
    path: Path
    title: str
    text: str  # markdown body, frontmatter removed
    wikilinks: list[str] = field(default_factory=list)  # target note stems, in order, deduped
    tags: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    pages: list[tuple[int, str]] | None = None  # PDF fallback parser: (1-based page, text)
    parser: str = "markdown"


def split_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(data, dict):
        return {}, raw
    return data, raw[m.end() :]


def _strip_code(text: str) -> str:
    return _INLINE_CODE.sub("", _FENCE_BLOCK.sub("\n", text))


def _stem(target: str) -> str:
    name = target.strip().replace("\\", "/").split("/")[-1]
    for suffix in (".md", ".markdown"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def extract_links_and_tags(
    text: str, frontmatter: dict | None = None
) -> tuple[list[str], list[str]]:
    plain = _strip_code(text)
    links: list[str] = []
    for m in _WIKILINK.finditer(plain):
        links.append(_stem(m.group(1)))
    for m in _MDLINK.finditer(plain):
        if "://" not in m.group(1):
            links.append(_stem(m.group(1)))
    tags = [m.group(1) for m in _TAG.finditer(plain)]
    fm_tags = (frontmatter or {}).get("tags") or (frontmatter or {}).get("tag") or []
    if isinstance(fm_tags, str):
        fm_tags = [t.strip() for t in re.split(r"[,\s]+", fm_tags) if t.strip()]
    tags += [str(t).lstrip("#") for t in fm_tags]

    def dedupe(items: list[str]) -> list[str]:
        seen, out = set(), []
        for it in items:
            key = it.casefold()
            if it and key not in seen:
                seen.add(key)
                out.append(it)
        return out

    return dedupe(links), dedupe(tags)


def parse_markdown_text(raw: str, path: Path) -> ParsedDocument:
    fm, body = split_frontmatter(raw)
    links, tags = extract_links_and_tags(body, fm)
    title = fm.get("title")
    if not title:
        m = _H1.search(_strip_code(body))
        title = m.group(1).strip() if m else path.stem
    return ParsedDocument(
        path=path,
        title=str(title),
        text=body,
        wikilinks=links,
        tags=tags,
        frontmatter=fm,
    )


def parse_markdown(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return parse_markdown_text(raw, path)
