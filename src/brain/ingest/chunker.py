"""Heading-aware chunker.

Structure first, size second:
  1. split the document into sections by markdown headings (fenced code is never split or scanned);
  2. split sections into paragraphs, and oversize paragraphs into sentences / word windows;
  3. pack paragraphs greedily up to ``target_tokens`` with a small sentence overlap;
  4. sections smaller than ``min_tokens`` are carried (heading included) into the next section
     rather than becoming near-empty chunks.
The heading path is kept in ``section_path`` and re-attached as retrieval context by the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ChunkingSettings
from ..text import approx_tokens
from .parsers.markdown import ParsedDocument

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    ord: int
    text: str
    section_path: str
    tokens: int
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class _Unit:
    text: str
    start: int
    end: int
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = approx_tokens(self.text)


@dataclass
class _Section:
    path: str = ""
    heading_line: str = ""
    page: int | None = None
    units: list[_Unit] = field(default_factory=list)


# ------------------------------------------------------------------ sectioning
def _sections_from_markdown(text: str) -> list[_Section]:
    sections: list[_Section] = [_Section()]
    stack: list[tuple[int, str]] = []
    par: list[str] = []
    par_start = 0
    in_fence: str | None = None
    offset = 0

    def flush(end: int) -> None:
        nonlocal par
        if par:
            body = "".join(par).strip()
            if body:
                sections[-1].units.append(_Unit(body, par_start, end))
        par = []

    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        fence = _FENCE.match(bare)
        if fence:
            marker = fence.group(1)
            if in_fence is None:
                in_fence = marker
            elif in_fence == marker:
                in_fence = None
        heading = None if in_fence or fence else _HEADING.match(bare)
        if heading:
            flush(offset)
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(
                _Section(path=" > ".join(t for _, t in stack), heading_line=bare.strip())
            )
        elif not bare.strip() and in_fence is None:
            flush(offset)
        else:
            if not par:
                par_start = offset
            par.append(line)
        offset += len(line)
    flush(offset)
    return [s for s in sections if s.units or s.heading_line]


def _sections_from_pages(pages: list[tuple[int, str]]) -> list[_Section]:
    sections: list[_Section] = []
    for page_no, raw in pages:
        # PDF text wraps lines mid-sentence: join single newlines, keep blank-line paragraphs
        joined = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
        units = [_Unit(p.strip(), -1, -1) for p in re.split(r"\n\s*\n", joined) if p.strip()]
        if units:
            sections.append(_Section(path="", page=page_no, units=units))
    return sections


# --------------------------------------------------------------------- packing
def _split_oversize(unit: _Unit, cfg: ChunkingSettings) -> list[_Unit]:
    """Break a paragraph larger than max_tokens into sentence groups (then word windows)."""
    if unit.tokens <= cfg.max_tokens:
        return [unit]
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in (s for s in _SENTENCE_SPLIT.split(unit.text) if s.strip()):
        t = approx_tokens(sent)
        if t > cfg.max_tokens:  # one giant "sentence": cut by words
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_tokens = [], 0
            words = sent.split()
            step = max(1, int(cfg.target_tokens / 1.3))
            pieces += [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
            continue
        if cur and cur_tokens + t > cfg.target_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sent)
        cur_tokens += t
    if cur:
        pieces.append(" ".join(cur))
    out, pos = [], 0
    for piece in pieces:
        if unit.start >= 0:
            found = unit.text.find(piece[:40], pos)
            start = unit.start + (found if found >= 0 else pos)
            pos = max(pos, found + len(piece)) if found >= 0 else pos
            out.append(_Unit(piece, start, start + len(piece)))
        else:
            out.append(_Unit(piece, -1, -1))
    return out


def _tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    taken: list[str] = []
    total = 0
    for s in reversed(sentences):
        taken.append(s)
        total += approx_tokens(s)
        if total >= overlap_tokens:
            break
    seed = " ".join(reversed(taken))
    return "" if seed.strip() == text.strip() else seed


def _pack(units: list[_Unit], cfg: ChunkingSettings) -> list[tuple[str, int, int, int]]:
    """Greedy pack -> (text, start, end, new_tokens). Overlap seeds don't count as new content."""
    expanded: list[_Unit] = []
    for u in units:
        expanded += _split_oversize(u, cfg)

    chunks: list[tuple[str, int, int, int]] = []
    cur: list[_Unit] = []  # new content units
    seed = ""
    tokens = 0

    def emit() -> None:
        body = "\n\n".join(u.text for u in cur)
        text = f"{seed}\n\n{body}" if seed else body
        start = cur[0].start if cur[0].start >= 0 else -1
        end = cur[-1].end if cur[-1].end >= 0 else -1
        chunks.append((text, start, end, sum(u.tokens for u in cur)))

    for u in expanded:
        over_target = tokens + u.tokens > cfg.target_tokens
        # a runt (< min_tokens) may overshoot the target rather than become a near-empty chunk
        runt_can_absorb = tokens < cfg.min_tokens and tokens + u.tokens <= cfg.max_tokens
        if cur and over_target and not runt_can_absorb:
            emit()
            seed = _tail(chunks[-1][0], cfg.overlap_tokens)
            cur, tokens = [], approx_tokens(seed) if seed else 0
        cur.append(u)
        tokens += u.tokens
    if cur:
        emit()

    # fold a tiny trailing chunk into its predecessor when it fits
    if len(chunks) >= 2 and chunks[-1][3] < cfg.min_tokens:
        prev, last = chunks[-2], chunks[-1]
        if approx_tokens(prev[0]) + last[3] <= cfg.max_tokens:
            # drop the overlap seed of `last`: it is already the tail of `prev`
            tail_body = last[0].split("\n\n", 1)[-1] if "\n\n" in last[0] else last[0]
            chunks[-2:] = [(f"{prev[0]}\n\n{tail_body}", prev[1], last[2], prev[3] + last[3])]
    return chunks


def chunk_document(doc: ParsedDocument, cfg: ChunkingSettings) -> list[Chunk]:
    sections = (
        _sections_from_pages(doc.pages)
        if doc.pages is not None
        else _sections_from_markdown(doc.text)
    )
    out: list[Chunk] = []
    pending: list[_Unit] = []
    for idx, sec in enumerate(sections):
        is_last = idx == len(sections) - 1
        body_tokens = sum(u.tokens for u in sec.units)
        small = body_tokens + sum(u.tokens for u in pending) < cfg.min_tokens
        head = (
            [_Unit(sec.heading_line, sec.units[0].start if sec.units else -1, -1)]
            if sec.heading_line and (small or pending)
            else []
        )
        if small and not is_last and sec.page is None:
            pending += head + sec.units
            continue
        units = pending + head + sec.units
        pending = []
        if not units:
            continue
        for text, start, end, _ in _pack(units, cfg):
            out.append(
                Chunk(
                    ord=len(out),
                    text=text.strip(),
                    section_path=sec.path,
                    tokens=approx_tokens(text),
                    page=sec.page,
                    char_start=start if start >= 0 else None,
                    char_end=end if end >= 0 else None,
                )
            )
    return [c for c in out if c.text]
