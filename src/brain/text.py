"""Small text utilities shared across ingest and retrieval."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+", re.UNICODE)
_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be but by did do does for from had has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to us was we were
    what when where which who whom why will with would you your about after also been being can
    could should some such than very""".split()
)


def normalize_name(s: str) -> str:
    """Canonical key for entity names: NFKC, casefolded, punctuation collapsed to spaces."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return _NON_ALNUM.sub(" ", s).strip()


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _WORD.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def approx_tokens(text: str) -> int:
    """Cheap LLM-token estimate (~1.3 tokens per word); good enough for chunk sizing."""
    return max(1, round(len(_WORD.findall(text)) * 1.3))


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
